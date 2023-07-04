"""
### Use the Snowpark Provider in a Machine Learning workflow

This DAG demonstrates a simple machine learning pipeline using the Snowpark decorators, 
the Snowflake XCOM backend, and the Snowpark ML model registry.  

The Snowpark provider is in a dev status and not yet in the pypi registry. 
Instead the provider is available via a wheel file in the linked 
repository.
"""

from datetime import datetime
from airflow.decorators import dag, task, task_group
from astro import sql as aql
from astro.files import File 
from astro.sql.table import Table 
import os
from astronomer.providers.snowflake.utils.snowpark_helpers import SnowparkTable

demo_database = os.environ['DEMO_DATABASE']
demo_schema = os.environ['DEMO_SCHEMA']

@dag(default_args={
         "temp_data_output": "table",
         "temp_data_db": demo_database,
         "temp_data_schema": demo_schema,
         "temp_data_overwrite": True,
         "database": demo_database,
         "schema": demo_schema
         },
     schedule_interval=None, 
     start_date=datetime(2023, 4, 1))
def snowpark_ml_demo():

    _SNOWFLAKE_CONN_ID = 'snowflake_default'

    ingest_files=['yellow_tripdata_sample_2019_01.csv', 'yellow_tripdata_sample_2019_02.csv']
    raw_table = Table(name='TAXI_RAW', metadata={'database':demo_database, 'schema':demo_schema}, conn_id=_SNOWFLAKE_CONN_ID)
    
    @task.snowpark_python(snowflake_conn_id=_SNOWFLAKE_CONN_ID)
    def check_model_registry(demo_database:str, demo_schema:str) -> dict: 
        from snowflake.ml.registry import model_registry

        assert model_registry.create_model_registry(session=snowpark_session, 
                                                    database_name=demo_database, 
                                                    schema_name=demo_schema)
        
        return {'database': demo_database, 'schema': demo_schema}

    @task_group()
    def load():

        for source in ingest_files:
            aql.load_file(task_id=f'load_{source}',
                input_file = File(f'include/data/{source}'), 
                output_table = raw_table,
                if_exists='replace'
            )
        
    @task.snowpark_python()
    def transform(raw_table:SnowparkTable) -> SnowparkTable:

        return raw_table.with_column('TRIP_DURATION_SEC',
                                     F.datediff('seconds', F.col('PICKUP_DATETIME'), F.col('DROPOFF_DATETIME')))\
                        .with_column('HOUR', F.date_part('hour', F.col('PICKUP_DATETIME').cast(T.TimestampType())))\
                        .select(F.col('PICKUP_LOCATION_ID').cast(T.StringType()).alias('PICKUP_LOCATION_ID'),
                                F.col('DROPOFF_LOCATION_ID').cast(T.StringType()).alias('DROPOFF_LOCATION_ID'),
                                F.col('HOUR'), 
                                F.col('TRIP_DISTANCE'), 
                                F.col('TRIP_DURATION_SEC'))

    @task.snowpark_python()
    def feature_engineering(taxidf:SnowparkTable) -> str:
        from snowflake.ml.modeling.preprocessing import MaxAbsScaler, OneHotEncoder
        from snowflake.ml.modeling.pipeline import Pipeline
        import pickle
        
        featuredf = taxidf.with_column('HOUR_OF_DAY', F.col('HOUR').cast(T.StringType()))

        feature_pipeline = Pipeline([
            ('ohe', OneHotEncoder(
                        input_cols=['PICKUP_LOCATION_ID', 'DROPOFF_LOCATION_ID', 'HOUR_OF_DAY'], 
                        output_cols=['PICKUP_LOCATION_ID', 'DROPOFF_LOCATION_ID', 'HOUR_OF_DAY'])),
            ('mas', MaxAbsScaler(
                        input_cols=['TRIP_DISTANCE'], 
                        output_cols=['TRIP_DISTANCE_SCALED']))
        ])
        #We could return the table object that gets created by fit_transform() which would be serialized
        #by the Snowpark operator and passed to later tasks.  This is ideal if idempotency is necessary.  
        # However, since it is one-hot encoded this is a very sparse table with >400 columns.
        ###return feature_pipeline.fit_transform(taxidf)

        #Instead we will pass the pipeline and run the fit_transform in later steps.  This is just for demo
        #purposes and is not ideal if the underlying raw table is changing.  Because we are using the 
        #Snowflake XCOM backend we can return a binary object like a pickle version of the pipeline.

        return (featuredf, pickle.dumps(feature_pipeline))

    @task.snowpark_python()
    def train(feature_pipeline:tuple, snowpark_model_registry:dict) -> dict:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from snowflake.ml.registry import model_registry
        from uuid import uuid4
        import pickle

        model_version=uuid4().urn
        model_name='trip_duration_estimator'

        registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                database_name=snowpark_model_registry['database'], 
                                                schema_name=snowpark_model_registry['schema'])
        
        featuredf = pickle.loads(feature_pipeline[1]).fit_transform(feature_pipeline[0])
        df = featuredf.to_pandas()
        X = df.drop(['PICKUP_LOCATION_ID','DROPOFF_LOCATION_ID', 'HOUR_OF_DAY', 'HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE'], axis=1)
        y = df[['TRIP_DURATION_SEC']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        lr = LinearRegression().fit(X_train, y_train)

        test_pred = lr.predict(X_test).reshape(-1)

        model_id = registry.log_model(
            model=lr, 
            model_version=model_version,
            model_name=model_name, 
            sample_input_data=X_test.iloc[0].values.reshape(1,-1),
            tags={'stage': 'dev', 'model_type': 'LinearRegression'})
        
        registry.set_metric(model_name=model_name, 
                            model_version=model_version, 
                            metric_name='mse', 
                            metric_value=mean_squared_error(test_pred, y_test))
        
        for feature in ['HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE']:
            registry.set_metric(model_name=model_name, 
                                model_version=model_version, 
                                metric_name=feature+'_std', 
                                metric_value=df[feature].std())

        registry.set_metric(model_name=model_name, 
                            model_version=model_version, 
                            metric_name='test_pred_std', 
                            metric_value=test_pred.std())
        
        return {'id': model_id, 'name': model_name, 'version': model_version}
            
    @task.snowpark_python()
    def predict(feature_pipeline:tuple, snowpark_model_registry:dict, model:dict) -> str:
        from snowflake.ml.registry import model_registry
        import pickle

        pred_table_name = 'TAXI_PRED' 
                        
        registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                database_name=snowpark_model_registry['database'], 
                                                schema_name=snowpark_model_registry['schema'])

        metrics = registry.get_metrics(model_name=model['name'], model_version=model['version'])
        lr = registry.load_model(model_name=model['name'], model_version=model['version'])

        featuredf = pickle.loads(feature_pipeline[1]).fit_transform(feature_pipeline[0])
        df = featuredf.to_pandas()
        X = df.drop(['PICKUP_LOCATION_ID','DROPOFF_LOCATION_ID', 'HOUR_OF_DAY', 'HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE'], axis=1)
                
        #look for drift 
        for feature in ['HOUR', 'TRIP_DURATION_SEC', 'TRIP_DISTANCE']:
            drift = df[feature].std() - metrics[feature+'_std']
            assert drift < 1, "Model drift greater than 1.  Do something."

        df['PREDICTED_DURATION'] = lr.predict(X).astype(int)

        write_columns = ['PICKUP_LOCATION_ID', 'DROPOFF_LOCATION_ID', 'HOUR_OF_DAY', 'PREDICTED_DURATION', 'TRIP_DURATION_SEC']

        snowpark_session.write_pandas(
            df[write_columns], 
            table_name=pred_table_name,
            auto_create_table=True,
            overwrite=True
        )

        return SnowparkTable(name=pred_table_name)
    
    _snowpark_model_registry = check_model_registry(demo_database, demo_schema)

    _rawdf = load() 

    _taxidf = transform(raw_table = raw_table)

    _feature_pipeline = feature_engineering(_taxidf)

    _model = train(_feature_pipeline, _snowpark_model_registry)

    _pred = predict(_feature_pipeline, _snowpark_model_registry, _model)
    
    _rawdf >> _taxidf 

snowpark_ml_demo()

def test():
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
    from snowflake.snowpark import Session as SnowparkSession
    from snowflake.snowpark import functions as F, types as T
    conn_params = SnowflakeHook('snowflake_default')._get_conn_params()
    snowpark_session = SnowparkSession.builder.configs(conn_params).create()