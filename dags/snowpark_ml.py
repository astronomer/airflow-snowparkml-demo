from datetime import datetime
from airflow.decorators import dag, task, task_group
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from airflow.models import Variable
from astro import sql as aql 
from astro.files import File 
from astro.sql.table import Table 

Variable.set('_SNOWFLAKE_CONN_ID', 'snowflake_default')
Variable.set('raw_table_name', 'TAXI_RAW')
Variable.set('taxi_table_name', 'TAXI_DATA')
Variable.set('feature_table_name', 'TAXI_FEATURE')
Variable.set('pred_table_name', 'TAXI_PRED')

_SNOWFLAKE_CONN_ID = Variable.get('_SNOWFLAKE_CONN_ID')
_SNOWPARK_BIN = '/home/astro/.venv/snowpark_env/bin/python'

ingest_files=['yellow_tripdata_sample_2019_01.csv', 'yellow_tripdata_sample_2019_02.csv']

@dag(dag_id='snowpark_ml_dag', schedule_interval=None, start_date=datetime(2023, 3, 25))
def snowml_demo():

	@task.external_python(task_id="setup_registry", python=_SNOWPARK_BIN )
	def setup_registry() -> str:
		from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
		from snowflake.snowpark import Session
		from snowflake.ml.registry import model_registry

		hook = SnowflakeHook(_SNOWFLAKE_CONN_ID) #, role='sysadmin')
		snowpark_session = Session.builder.configs(hook._get_conn_params()).create()

		create_result = model_registry.create_model_registry(session=snowpark_session, database_name='MODEL_REGISTRY')

		registry = model_registry.ModelRegistry(session=snowpark_session, name='MODEL_REGISTRY')
		
		return registry._name

	@task_group()
	def load():
	
		for source in ingest_files:
			aql.load_file(task_id=f'load_{source}',
				input_file = File(f'include/{source}'), 
				output_table = Table(name=Variable.get('raw_table_name'), conn_id=_SNOWFLAKE_CONN_ID),
				if_exists='replace'
			)

	@task.external_python(task_id="transform_data", python=_SNOWPARK_BIN)
	def transform() -> str:

		from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
		from airflow.models import Variable
		from snowflake.snowpark import Session
		from snowflake.snowpark import functions as F
		from snowflake.snowpark import types as T

		hook = SnowflakeHook(_SNOWFLAKE_CONN_ID)
		conn_params = hook._get_conn_params()
		snowpark_session = Session.builder.configs(conn_params).create()

		taxi_table_name = Variable.get('taxi_table_name')
		
		taxidf = snowpark_session.table(Variable.get('raw_table_name')) \
					.with_column('TRIP_DURATION_SEC',
		  						 F.datediff('seconds',
		      								F.col('PICKUP_DATETIME'), F.col('DROPOFF_DATETIME')))\
					.with_column('HOUR', 
		  						F.date_part('hour', F.col('PICKUP_DATETIME').cast(T.TimestampType())))\
					.select(\
						F.col('PICKUP_LOCATION_ID').cast(T.StringType()).alias('PICKUP_LOCATION_ID'), \
						F.col('DROPOFF_LOCATION_ID').cast(T.StringType()).alias('DROPOFF_LOCATION_ID'), \
						F.col('HOUR'), \
						F.col('TRIP_DISTANCE'), \
						F.col('TRIP_DURATION_SEC'))\
						.write.mode('overwrite').save_as_table(taxi_table_name)
		
		snowpark_session.close()

		return taxi_table_name

	@task.external_python(task_id="feature_eng", python=_SNOWPARK_BIN)
	def feature_engineering(taxi_table_name:str) -> str:

		from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
		from airflow.models import Variable
		from snowflake.snowpark import Session
		from snowflake.snowpark import functions as F
		from snowflake.snowpark import types as T
		from snowflake.ml.preprocessing import MaxAbsScaler, OneHotEncoder
		from snowflake.ml.framework import pipeline

		hook = SnowflakeHook(_SNOWFLAKE_CONN_ID)
		conn_params = hook._get_conn_params()
		snowpark_session = Session.builder.configs(conn_params).create()

		feature_table_name = Variable.get('feature_table_name')
		
		taxidf = snowpark_session.table(taxi_table_name)\
					.with_column('HOUR_OF_DAY', F.col('HOUR').cast(T.StringType()))

		feature_pipeline = pipeline.Pipeline([
			('ohe', OneHotEncoder(
						input_cols=['PICKUP_LOCATION_ID', 'DROPOFF_LOCATION_ID', 'HOUR_OF_DAY'], 
						output_cols=['PICKUP_LOCATION_ID', 'DROPOFF_LOCATION_ID', 'HOUR_OF_DAY'])),
			('mas', MaxAbsScaler(
						input_cols=['TRIP_DISTANCE'], 
						output_cols=['TRIP_DISTANCE_SCALED']))
		])

		feature_pipeline.fit_transform(taxidf)\
						.write.mode('overwrite').save_as_table(feature_table_name)
		
		snowpark_session.close()

		return feature_table_name

	@task.external_python(task_id="train", python=_SNOWPARK_BIN )
	def train(feature_table_name:str, registry_name:str) -> str:
		from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
		from airflow.models import Variable
		from snowflake.snowpark import Session
		from sklearn.linear_model import LinearRegression
		from sklearn.model_selection import train_test_split
		from sklearn.metrics import mean_squared_error
		from snowflake.ml.registry import model_registry

		hook = SnowflakeHook(_SNOWFLAKE_CONN_ID)
		conn_params = hook._get_conn_params()
		snowpark_session = Session.builder.configs(conn_params).create()

		registry = model_registry.ModelRegistry(session=snowpark_session, name=registry_name)
	
		df = snowpark_session.table(Variable.get('feature_table_name')).to_pandas()
		X = df.drop(['PICKUP_LOCATION_ID','DROPOFF_LOCATION_ID', 'HOUR_OF_DAY', 'HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE'], axis=1)
		y = df[['TRIP_DURATION_SEC']]

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

		lr = LinearRegression().fit(X_train, y_train)

		test_pred = lr.predict(X_test).reshape(-1)

		model_id = registry.log_model(
			model=lr, 
			name='trip_duration_estimator', 
			tags={'stage': 'dev', 'model_type': 'LinearRegression'})
		
		registry.set_metric(id=model_id, name='mse', value=mean_squared_error(test_pred, y_test))
		
		for feature in ['HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE']:
			registry.set_metric(id=model_id, name=feature+'_std', value=df[feature].std())

		registry.set_metric(id=model_id, name='test_pred_std', value=test_pred.std())

		snowpark_session.close()
		
		return model_id

	@task.external_python(task_id="predict", python=_SNOWPARK_BIN)
	def predict(feature_table_name:str, registry_name:str, model_id:str) -> str:
		from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
		from airflow.models import Variable
		from snowflake.snowpark import Session
		from snowflake.ml.registry import model_registry
		
		hook = SnowflakeHook(_SNOWFLAKE_CONN_ID)
		conn_params = hook._get_conn_params()
		snowpark_session = Session.builder.configs(conn_params).create()
		
		pred_table_name = Variable.get('pred_table_name')
		
		registry = model_registry.ModelRegistry(session=snowpark_session, name=registry_name)

		metrics = registry.get_metrics(id=model_id)
		lr = registry.load_model(id=model_id)

		df = snowpark_session.table(Variable.get('feature_table_name')).to_pandas()
		X = df.drop(['PICKUP_LOCATION_ID','DROPOFF_LOCATION_ID', 'HOUR_OF_DAY', 'HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE'], axis=1)
				
		#look for drift 
		for feature in ['HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE']:
			drift = df[feature].std() - metrics[feature+'_std']

		df['PREDICTED_DURATION'] = lr.predict(X).astype(int)

		write_columns = ['PICKUP_LOCATION_ID', 'DROPOFF_LOCATION_ID', 'HOUR_OF_DAY', 'PREDICTED_DURATION', 'TRIP_DURATION_SEC']

		snowpark_session.write_pandas(
			df[write_columns], 
			table_name=pred_table_name,
			auto_create_table=True,
			overwrite=True
		)

		snowpark_session.close()

		return pred_table_name

	registry_name = setup_registry()
	taxi_table_name = transform()
	feature_table_name = feature_engineering(taxi_table_name)
	model_id = train(feature_table_name=feature_table_name, registry_name=registry_name) 
	pred_table_name = predict(feature_table_name=feature_table_name, registry_name=registry_name, model_id=model_id)

	load() >> taxi_table_name

snowml_demo = snowml_demo()




# from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
# your_role = 'MICHAELGREGORY'
# SnowflakeHook(role='sysadmin').run(f'''
# 	CREATE DATABASE IF NOT EXISTS MODEL_REGISTRY;
# 	GRANT USAGE ON DATABASE MODEL_REGISTRY TO ROLE {your_role} ;
# 	CREATE SCHEMA IF NOT EXISTS MODEL_REGISTRY.PUBLIC; 
# 	GRANT USAGE, CREATE TABLE ON SCHEMA MODEL_REGISTRY.PUBLIC TO ROLE {your_role};
# ''')