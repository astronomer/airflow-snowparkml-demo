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

# This demo assumes the use of newly created objects in a Snowflake Trial.  
# If using an existing snowflake account update the following.  If changing 
# the demo_xcom_stage or demo_xcom_table update the environment variables in 
# the docker-compose.override.yml file.
snowflake_objects = {'demo_database': 'DEMO',
                     'demo_schema': 'DEMO',
                     'demo_warehouse': 'COMPUTE_WH',
                     'demo_xcom_stage': 'XCOM_STAGE',
                     'demo_xcom_table': 'XCOM_TABLE',
                     'demo_snowpark_wh': 'SNOWPARK_WH'
}

_SNOWFLAKE_CONN_ID = "snowflake_default"

@dag(default_args={
         "snowflake_conn_id": _SNOWFLAKE_CONN_ID,
         "temp_data_output": "table",
         "temp_data_db": snowflake_objects['demo_database'],
         "temp_data_schema": snowflake_objects['demo_schema'],
         "temp_data_overwrite": True,
         "database": snowflake_objects['demo_database'],
         "schema": snowflake_objects['demo_schema']
         },
     schedule_interval=None, 
     start_date=datetime(2023, 4, 1))
def snowpark_ml_demo():
    """
    This DAG demonstrates a simple workflow using the Snowpark operators of the Snowflake provider
    to ingest data, perform feature engineering, train a model and generate predictions.  The DAG 
    also includes examples for using setup and teardown tasks (introduced in Airflow 2.7) to manage 
    and cleanup Snowflake objects as well as the use of the Snowflake XCOM backend.

    The Snowpark provider is in a dev status and not yet in the pypi registry. Instead the provider 
    is available via a wheel file in the linked repository.
    """

    ingest_files=['yellow_tripdata_sample_2019_01.csv', 'yellow_tripdata_sample_2019_02.csv']
    raw_table = Table(name='TAXI_RAW', metadata={'database':snowflake_objects['demo_database'], 
                                                 'schema':snowflake_objects['demo_schema']}, 
                                                 conn_id=_SNOWFLAKE_CONN_ID)

    @task.snowpark_python()
    def create_snowflake_objects(snowflake_objects:dict):
        """
        The Astronomer provider for Snowpark adds a `snowpark_python` task decorator which executes 
        the Snowpark Python code in the decorated callable function. The provider also includes a 
        traditional operator `SnowparkPythonOperator` though this demo only shows the taskflow API.

        The decorator (and operator) automatically instantiates a `snowpark_session`.  Snowflake 
        credentials are automatically and securely passed using the Airflow Connections so users do 
        not need to pass Snowflake credentials as function arguments.  For this demo the 
        `snowflake_conn_id` parameter is defined in the `default_args` above.
        
        [AIP-52](https://cwiki.apache.org/confluence/display/AIRFLOW/AIP-52+Setup+and+teardown+tasks) 
        introduced the notion of 
        [setup and teardown tasks](https://docs.astronomer.io/learn/airflow-setup-teardown)

        Because we assume that the demo will be run with a brand new Snowflake trial 
        account this task creates Snowflake objects (databases, schemas, stages, etc.) prior to 
        running any tasks.
        """

        snowpark_session.sql(f"""CREATE WAREHOUSE IF NOT EXISTS \
                                {snowflake_objects['demo_snowpark_wh']} WITH
                                 WAREHOUSE_SIZE = 'MEDIUM'
                                 WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED';""").collect()
        
        snowpark_session.sql(f"""CREATE DATABASE IF NOT EXISTS \
                                {snowflake_objects['demo_database']};""").collect()

        snowpark_session.sql(f"""CREATE SCHEMA IF NOT EXISTS \
                                {snowflake_objects['demo_database']}.\
                                {snowflake_objects['demo_schema']};""").collect()

        snowpark_session.sql(f"""CREATE STAGE IF NOT EXISTS \
                                {snowflake_objects['demo_database']}.\
                                {snowflake_objects['demo_schema']}.\
                                {snowflake_objects['demo_xcom_stage']} 
                                    DIRECTORY = (ENABLE = TRUE)
                                    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
                             """).collect()
        
        snowpark_session.sql(f"""CREATE TABLE IF NOT EXISTS \
                             {snowflake_objects['demo_database']}.\
                            {snowflake_objects['demo_schema']}.\
                            {snowflake_objects['demo_xcom_table']}
                                    ( 
                                        dag_id varchar NOT NULL, 
                                        task_id varchar NOT NULL, 
                                        run_id varchar NOT NULL,
                                        multi_index integer NOT NULL,
                                        key varchar NOT NULL,
                                        value_type varchar NOT NULL,
                                        value varchar NOT NULL
                                 ); 
                              """).collect()
    
    @task.snowpark_virtualenv(python_version='3.8', requirements=['snowflake-ml-python==1.0.9'])
    def check_model_registry(snowflake_objects:dict) -> dict: 
        """
        Snowpark ML provides a model registry leveraging tables, views and stages to 
        track model state as well as model artefacts. 

        If the model registry objects have not yet been created in Snowflake this task 
        will create them and return a dictionary with the database and schema where they 
        exist.

        Snowpark Python currently supports Python 3.8, 3.9, and 3.10.  If the version of 
        Python used to run Airflow is different it may be necessary to use a Python virtual 
        environment for Snowpark tasks.  
            
        This task shows an example of using the `snowpark_virtualenv` decorator. It, and the corresponding 
        `SnowparkVirtualenvOperator`, creates a python virtual environment with any `requirements` executable 
        similar to the [PythonVirtualenvOperator]
        (https://registry.astronomer.io/providers/apache-airflow/versions/latest/modules/pythonvirtualenvoperator)).  
            
        Astronomer has created a [Docker Buildkit](https://github.com/astronomer/astro-provider-venv) 
        to simplify building virtual environments with Astro CLI.  See the `Dockerfile` for 
        details.
        """
        from snowflake.ml.registry import model_registry

        assert model_registry.create_model_registry(session=snowpark_session, 
                                                    database_name=snowflake_objects['demo_database'], 
                                                    schema_name=snowflake_objects['demo_schema'])
        
        snowpark_model_registry = {'database': snowflake_objects['demo_database'], 
                                   'schema': snowflake_objects['demo_schema']}

        return snowpark_model_registry

    @task_group()
    def load():
        """
        The [Astro Python SDK](https://docs.astronomer.io/learn/astro-python-sdk-etl)
        is a good way to easily load data to Snowflake with just a few lines of code.

        Here we use dynamically generated tasks in a task group.  A task will be created 
        to load each file in the `data_sources` list.
        """

        for source in ingest_files:
            aql.load_file(task_id=f'load_{source}',
                input_file = File(f'include/data/{source}'), 
                output_table = raw_table,
                if_exists='replace'
            )
        
    @task.snowpark_python()
    def transform(raw_table:SnowparkTable) -> SnowparkTable:
        """
        Historically users would write SQL code for data transformations in Snowflake.  
        For example, the SQL code in the `include/sql/jaffle_shop.sql` file can be 
        [orchestrated as a task](https://docs.astronomer.io/learn/airflow-snowflake) 
        in Airflow with something like the following:  

        _customers = SnowflakeOperator(task_id="jaffle_shop",
                                        sql=Path('include/sql/jaffle_shop.sql).read_text(),
                                        params={"table_name": CUSTOMERS})
        
        Alternatively, users can use the Snowpark Dataframe API for simplicity, 
        readability and extensibility.

        Best practices for pipeline orchestration dictate the need to build 'atomic' 
        and 'idempotent' tasks.  However, Snowpark sessions and session-residenct objects, 
        such as Snowpark DataFrames, are not serializable and cannot easily be passed between 
        tasks. 
        
        The Astronomer provider for Snowpark includes a `SnowparkTable` dataclass which 
        can be serialized and deserialized.

        Any SnowparkTable objects passed as arguments are automatically instantiated as 
        Snowpark dataframes.

        Any Snowpark dataframe objects returned from the task are automatically serialized as tables 
        based on the decorator parameters `temp_data_output`, `temp_data_schema`, `temp_data_overwrite`, 
        and `temp_data_table_prefix`.  For this demo these parameters are set as `default_args` at the 
        top of this file.  Alternatively, these can be set for each task or overriden per-task.

        The Snowpark `Functions` and `Types` classes have been automatically imported as `F` and 
        `T` respectively.
        """
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
        """
        In this task we perform feature engineering with Snowpark python. We could return the table 
        object that gets created by fit_transform() which would be serialized by the Snowpark 
        operator and passed to later tasks.  
        
        return feature_pipeline.fit_transform(taxidf)

        This is ideal if idempotency is necessary.  However, since it is one-hot encoded this is a 
        very sparse table with >400 columns.  Instead we will pass the pipeline and run the 
        fit_transform in later steps.  This is just for demo purposes and is not ideal if the 
        underlying raw table is changing.  

        Normally Airflow's cross-communication (XCOM) system is not designed for passing 
        large or complex (ie. non-json-serializable) data between tasks as it relies on 
        storing data in a database (usually postgres or mysql). Airflow 2.5+ has added 
        support for [passing pandas dataframe](https://github.com/apache/airflow/pull/30390). 
        However, if the data is very large or, in this case passing a pickle object, the 
        task will fail as it is unable to serialize the object.
        
        Additionally, for data governance reasons it may be not be advisable to pass 
        sensitive or regulated data between tasks to avoid storing this data in a the
        XCOM database.
        
        For these reasons the Astronomer provider for Snowpark includes a [custom XCOM 
        backend](https://docs.astronomer.io/learn/xcom-backend-tutorial) which saves 
        XCOM data passed between tasks in either Snowflake tables or stages. 
        With the Snowflake XCOM backend small, JSON-serializable data is stored in a single 
        table and large or non-serializable data is stored in a stage.  This allows passing 
        arbitrarily large or complex data between tasks and ensures that all data, including 
        intermediate datasets, stay inside the secure, goverened boundary of Snowflake.

        The 'AIRFLOW__CORE__XCOM_SNOWFLAKE_TABLE', 'AIRFLOW__CORE__XCOM_SNOWFLAKE_STAGE', 
        and 'AIRFLOW__CORE__XCOM_SNOWFLAKE_CONN_NAME' environment variables in the 
        `docker-compose.override.yml` file specify where this data is stored.

        This task returns a binary object from the pickle dump and passes that to downstream
        tasks.
        """
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
                
        return (featuredf, pickle.dumps(feature_pipeline))

    @task.snowpark_python()
    def train(feature_pipeline:tuple, snowpark_model_registry:dict, snowflake_objects:dict) -> dict:
        """
        The Snowpark ML framework has many options for feature engineering and model training.  See the 
        [Quickstart](https://quickstarts.snowflake.com/guide/getting_started_with_dataengineering_ml_using_snowpark_python/index.html#0
        for details.  

        In this task we train a simple linear regression model using the upstream feature 
        engineering pipeline.

        In the Airflow UI we can see that the upstream `feature_engineering` task returned two 
        URIs in the XCOM.

        ie.: 

        (snowflake://org-acct?&table=DEMO.DEMO.XCOM_TABLE&key=snowpark_ml_demo/feature_engineering/manual__2023-10-01T10:47:24.888705+00:00/0/return_value,
        snowflake://org-acct?&stage=DEMO.DEMO.XCOM_STAGE&key=snowpark_ml_demo/feature_engineering/manual__2023-10-01T10:47:24.888705+00:00/1/return_value.bin)

        The actual data was serialized by the Snowflake XCOM backend.  The feature dataframe was stored as a 
        table and the pickled pipeline was stored as a binary file on a Snowflake stage.  These objects are 
        automatically deserialized when we pass them into this downstream task.

        """
        from snowflake.ml.modeling.linear_model import LinearRegression
        from snowflake.ml.registry import model_registry
        import pickle
        from uuid import uuid4

        model_version=uuid4().urn
        model_name='trip_duration_estimator'

        registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                database_name=snowpark_model_registry['database'], 
                                                schema_name=snowpark_model_registry['schema'])

        featuredf = pickle.loads(feature_pipeline[1]).fit_transform(feature_pipeline[0])

        traindf, testdf = featuredf.random_split(weights=[0.7, 0.3], seed=42)

        model = LinearRegression(label_cols=["TRIP_DURATION_SEC"],
                                 output_cols=["TRIP_DURATION_SEC_PRED"])

        snowpark_session.use_warehouse(snowflake_objects['demo_snowpark_wh'])
        
        model = model.fit(traindf)

        model_id = registry.log_model(model=model, 
                                      model_version=model_version,
                                      model_name=model_name, 
                                      tags={'stage': 'dev', 'model_type': 'LinearRegression'})
        
        snowpark_session.use_warehouse(snowflake_objects['demo_warehouse'])
        try:
            snowpark_session.sql(f"""ALTER WAREHOUSE \
                                    {snowflake_objects['demo_snowpark_wh']} \
                                    SUSPEND;""").collect()
        except: 
            pass

        train_score = model.score(traindf)
        print(f'Training dataset r-squared value is {train_score}')

        model_id.set_metric(metric_name='train_score', 
                            metric_value=train_score)

        test_score = model.score(testdf)
        print(f'Test dataset r-squared value is {test_score}')
        
        model_id.set_metric(metric_name='test_score', 
                            metric_value=test_score)
        
        return {'name': model_name, 'version': model_version}
            
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

        preddf = lr.predict(featuredf)

        write_columns = ['PICKUP_LOCATION_ID', 'DROPOFF_LOCATION_ID', 'HOUR_OF_DAY', 'TRIP_DURATION_SEC_PRED']

        preddf = preddf[write_columns]

        preddf.write.save_as_table(pred_table_name)

        return pred_table_name
    
    @task.snowpark_ext_python(python='/home/astro/.venv/snowpark/bin/python')
    def cleanup_temp_tables(snowflake_objects:dict, **context):
        """
        This task will be run as an Airflow 2.7 teardown task.  The task deletes 
        the intermediate, temporary data passed between Snowpark tasks. In production 
        it may be best to keep intermediate tables as they provide useful 
        audting data.  For dev/test it may be beneficial to reduce objects and noise.

        The `temp_data_dict` is instantiated by default in the task namespace based
        on the decorator args or `default_args`.  Likewise, all of the variables 
        needed to construct the temporary data URI (e.g. `dag_id`, `ts_nodash`, etc.)
        are also instantiated.  This allows us to cleanup temporary data after the 
        DAG run.  

        In the future this may be added as another operator for the Snowpark provider.  
        Here it shows a good use of teardown tasks.
        """
        
        snowpark_session.database = temp_data_dict['temp_data_db'] \
                                        or snowflake_objects['demo_database']
        snowpark_session.schema = temp_data_dict['temp_data_schema'] \
                                        or snowflake_objects['demo_schema']

        if temp_data_dict['temp_data_output'] == 'table':
            xcom_table_string=f"{temp_data_dict['temp_data_table_prefix']}{dag_id}__%__{ts_nodash}__%".upper()

            xcom_table_list = snowpark_session.table('information_schema.tables')\
                                        .select('table_name')\
                                        .where(F.col('table_name').like(xcom_table_string))\
                                        .to_pandas()['TABLE_NAME'].to_list()

            print(f'Removing tables {xcom_table_list}')

            for table in xcom_table_list:
                    try:
                        snowpark_session.table(table).drop_table()
                    except:
                        pass
        elif temp_data_dict['temp_data_output'] == 'stage':
            
            xcom_stage_string = f"{dag_id.lower()}/.*/{run_id.split('+')[0]}.*/"

            print(f'Removing files based on {xcom_stage_string}')
            
            xcom_file_list = snowpark_session.sql(f"""REMOVE @{temp_data_dict['temp_data_stage']} 
                                                      PATTERN='{xcom_stage_string}'""").collect()
   
            
    #To use the setup and teardown tasks we instantiate the setup task
    _create_snowflake_objects = create_snowflake_objects(snowflake_objects).as_setup() 

    #Then wrap the rest of the tasks in a with statement specifying the teardown and setup tasks
    with cleanup_temp_tables(snowflake_objects).as_teardown(setups=_create_snowflake_objects):
    
        _snowpark_model_registry = check_model_registry(snowflake_objects)

        _rawdf = load() 

        _taxidf = transform(raw_table = raw_table)

        _feature_pipeline = feature_engineering(_taxidf)

        _model = train(_feature_pipeline, _snowpark_model_registry, snowflake_objects)

        _pred = predict(_feature_pipeline, _snowpark_model_registry, _model)
        
        _rawdf >> _taxidf 

snowpark_ml_demo()

def test():
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
    from snowflake.snowpark import Session as SnowparkSession
    from snowflake.snowpark import functions as F, types as T
    conn_params = SnowflakeHook('snowflake_default')._get_conn_params()
    snowpark_session = SnowparkSession.builder.configs(conn_params).create()