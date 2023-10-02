"""
### Use the Snowpark Provider in an advanced Machine Learning workflow

This DAG demonstrates an end-to-end application workflow using OpenAI embeddings with a
Weaviate vector database as well as Snowpark decorators, the Snowflake XCOM backend and 
the Snowpark ML model registry.  The Astro CLI can easily be adapted to include additional 
Docker-based services.  This demo includes services for Weaviate and streamlit.

The Snowpark provider is in a dev status and not yet in the pypi registry. 
Instead the provider is available via a wheel file in the linked 
repository.

See [README_CA.md](https://github.com/astronomer/airflow-snowparkml-demo/blob/main/README_CA.md) for setup instructions.
"""

import datetime 
import os
import pandas as pd
import numpy as np
from textwrap import dedent

from astro import sql as aql 
from astro.files import File 
from astro.sql.table import Table, Metadata
from airflow.decorators import dag, task, task_group, setup, teardown
from weaviate.util import generate_uuid5
from astronomer.providers.snowflake.utils.snowpark_helpers import SnowparkTable
from weaviate_provider.operators.weaviate import WeaviateRestoreOperator

# This demo assumes the use of newly created objects in a Snowflake Trial.  
# If using an existing snowflake account update the following.  If changing 
# the demo_xcom_stage or demo_xcom_table update the environment variables in 
# the docker-compose.override.yml file.
snowflake_objects = {'demo_database': 'DEMO',
                     'demo_schema': 'DEMO',
                     'demo_xcom_stage': 'XCOM_STAGE',
                     'demo_xcom_table': 'XCOM_TABLE',
                     'demo_snowpark_wh': 'DEMO'
}

_SNOWFLAKE_CONN_ID = "snowflake_default"

restore_data_uri = 'https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/sissy-g-toys-demo/data'
calls_directory_stage = 'call_stage'
data_sources = ['ad_spend', 'sessions', 'customers', 'payments', 'subscription_periods', 'customer_conversions', 'orders']
twitter_sources = ['twitter_comments', 'comment_training']
weaviate_class_objects = {'CommentTraining': {'count': 1987}, 'CustomerComment': {'count': 12638}, 'CustomerCall': {'count': 43}}

default_args={
    "temp_data_output": 'table',
    "temp_data_schema": snowflake_objects['demo_schema'],
    "temp_data_overwrite": True,
    "temp_data_table_prefix": "XCOM_",
    "snowflake_conn_id": _SNOWFLAKE_CONN_ID,
    "weaviate_conn_id": "weaviate_default",

}

@dag(schedule=None, start_date=datetime.datetime(2023, 1, 1), catchup=False, default_args=default_args)
def customer_analytics():
    """
    ### Use the Snowpark Provider in an advanced Machine Learning workflow

    This DAG demonstrates an end-to-end application workflow using OpenAI embeddings with a
    Weaviate vector database as well as Snowpark decorators, the Snowflake XCOM backend and 
    the Snowpark ML model registry.  The Astro CLI can easily be adapted to include additional 
    Docker-based services.  This demo includes services for Weaviate and streamlit.

    The Snowpark provider is in a dev status and not yet in the pypi registry. 
    Instead the provider is available via a wheel file in the linked 
    repository.

    See [README_CA.md](https://github.com/astronomer/airflow-snowparkml-demo/blob/main/README_CA.md) for setup instructions.
    """
    @task.snowpark_python()
    def create_snowflake_objects(snowflake_objects:dict, calls_directory_stage:str):
        """
        The Astronomer provider for Snowpark adds a `snowpark_python` task decorator which executes 
        the Snowpark Python code in the decorated callable function. The provider also includes a 
        traditional operator `SnowparkPythonOperator` though this demo only shows the taskflow API.

        The decorator (and operator) automatically instantiates a `snowpark_session`.  Snowflake 
        credentials are automatically and securely passed using the Airflow Connections so users do 
        not need to pass Snowflake credentials as function arguments.  For this demo the 
        `snowflake_conn_id` parameter is defined in the `default_args` above.
        
        [AIP-52](https://cwiki.apache.org/confluence/display/AIRFLOW/AIP-52+Setup+and+teardown+tasks) 
        introduced the notion of [setup and teardown tasks](https://docs.astronomer.io/learn/airflow-setup-teardown)

        Because we assume that the demo will be run with a brand new Snowflake trial 
        account this task creates Snowflake objects (databases, schemas, stages, etc.) prior to 
        running any tasks.
        """

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
        
        snowpark_session.sql(f"""CREATE OR REPLACE STAGE \
                                {snowflake_objects['demo_database']}.\
                                {snowflake_objects['demo_schema']}.\
                                {calls_directory_stage} 
                                        DIRECTORY = (ENABLE = TRUE) 
                                        ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
                                """).collect()
            
    @task_group()
    def enter():
        """
        Using an `enter()` task group allows us to group together tasks that should be run to setup 
        state for the rest of the DAG.  Functionally this is very similar to setup tasks but allows 
        some additional flexibility in dependency mapping.
        """

        @task()
        def download_weaviate_backup() -> str:
            """
            [Weaviate](http://www.weaviate.io) is a vector database which allows us to store a 
            vectorized representation of unstructured data like twitter tweets or audio calls.
            In this demo we use the [OpenAI  embeddings](https://platform.openai.com/docs/guides/embeddings/embeddings) 
            model to build the vectors.  With the vectors we can do sentiment classification 
            based on cosine similarity with a labeled dataset.  

            This demo uses a version of Weaviate running locally in a Docker container.  See the 
            `docker-compose.override.yml` file for details. The Astro CLI will start this container 
            alongside the Airflow webserver, trigger, scheduler and database.

            In order to speed up the demo process the data has already been ingested into weaviate 
            and vectorized.  The data was then backed up and stored in the cloud for easy restore.            
            
            This task will download the backup.zip and make it available in a docker mounted 
            filesystem for the weaviate restore task.  Normally this would be in an cloud storage.
            """
            import urllib
            import zipfile

            weaviate_restore_uri = f'{restore_data_uri}/weaviate-backup/backup.zip'

            zip_path, _ = urllib.request.urlretrieve(weaviate_restore_uri)
            with zipfile.ZipFile(zip_path, "r") as f:
                f.extractall('/usr/local/airflow/include/weaviate/data/backups')

        @task.snowpark_python()
        def check_model_registry(snowflake_objects:dict) -> dict:
            """
            Snowpark ML provides a model registry leveraging tables, views and stages to 
            track model state as well as model artefacts. 

            If the model registry objects have not yet been created in Snowflake this task 
            will create them and return a dictionary with the database and schema where they 
            exist.
            """
            from snowflake.ml.registry import model_registry

            assert model_registry.create_model_registry(session=snowpark_session, 
                                                        database_name=snowflake_objects['demo_database'], 
                                                        schema_name=snowflake_objects['demo_schema'])
            
            snowpark_model_registry = {'database': snowflake_objects['demo_database'], 
                                    'schema': snowflake_objects['demo_schema']}

            return snowpark_model_registry
        
        _snowpark_model_registry = check_model_registry(snowflake_objects)
        
        _restore_weaviate = WeaviateRestoreOperator(task_id='restore_weaviate',
                                                    backend='filesystem', 
                                                    id='backup',
                                                    include=list(weaviate_class_objects.keys()),
                                                    replace_existing=True)
        
        _restore_weaviate.doc_md = dedent(
            """
            ### Restoring Demo Data  
            In order to speed up the demo process the data has already been ingested into weaviate 
            and vectorized.  The data was then backed up and stored in the cloud for easy restore.

            This task restores the pre-vectorized demo data using the backup.zip file downloaded 
            in the `download_weaviate_backup` task.  

            Upstream tasks will try to import to weaviate will but will be `skipped` since they 
            already exist.  For any new data Weaviate will use OpenAI embeddings to vectorize 
            and import data.
            """
        )

        download_weaviate_backup() >> _restore_weaviate

        return _snowpark_model_registry, _restore_weaviate
    
    @task_group()
    def structured_data():
        """
        This demo shows the ability to build with structured, semi-strcutured and unstructured data 
        in Snowflake.  

        This section extracts, transforms and load structured data from an S3 bucket.
        """

        @task_group()
        def load_structured_data():
            """
            The [Astro Python SDK](https://docs.astronomer.io/learn/astro-python-sdk-etl)
            is a good way to easily load data to a database with very few lines of code.

            Here we use dynamically generated tasks in a task group.  A task will be created 
            to load each file in the `data_sources` list.
            """
            for source in data_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"{restore_data_uri}/{source}.csv"), 
                    output_table = Table(name=f'STG_{source.upper()}', 
                                         conn_id=_SNOWFLAKE_CONN_ID)
                )

        @task_group()
        def transform_structured():

            @task.snowpark_python()
            def jaffle_shop(customers_df:SnowparkTable, orders_df:SnowparkTable, payments_df:SnowparkTable):
                """
                Historically users must write SQL code for data transformations in Snowflake.  
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

                The Snowpark `Functions` and `Types` have been automatically imported as `F` and 
                `T` respectively.
                """
                
                customer_orders_df = orders_df.group_by('customer_id').agg(F.min('order_date').alias('first_order'),
                                                                           F.max('order_date').alias('most_recent_order'),
                                                                           F.count('order_id').alias('number_of_orders'))
                
                customer_payments_df = payments_df.join(orders_df, how='left', on='order_id')\
                                                  .group_by('customer_id')\
                                                  .agg((F.sum('amount') / 100).alias('total_amount'))
                
                customers = customers_df.join(customer_orders_df, how='left', on='customer_id')\
                                        .join(customer_payments_df, how='left', on='customer_id')\
                                        .rename('total_amount', 'customer_lifetime_value')
                
                payment_types = ['credit_card', 'coupon', 'bank_transfer', 'gift_card']
                
                orders = payments_df.drop('payment_id')\
                                    .pivot('payment_method', payment_types )\
                                    .agg(F.sum('amount'))\
                                    .group_by('order_id')\
                                    .agg({f"'{x}'": "sum" for x in payment_types})\
                                    .rename({f"SUM('{x.upper()}')": x+'_amount' for x in payment_types})\
                                    .join(payments_df.group_by('order_id')\
                                                     .agg(F.sum('amount').alias('total_amount')), on='order_id')\
                                    .join(orders_df, on='order_id')

                return customers

            @task.snowpark_virtualenv(python_version='3.8', requirements=['snowflake-snowpark-python>=1.8'])
            def mrr_playbook(subscription_df:SnowparkTable):
                """
                Snowpark Python currently supports Python 3.8, 3.9, and 3.10.  If the version of 
                Python used to run Airflow is different it may be necessary to use a Python virtual 
                environment for Snowpark tasks.  
                  
                The `snowpark_virtualenv` decorator and `SnowparkVirtualenvOperator` allow users 
                to specify a different python version similar to the 
                [PythonVirtualenvOperator](https://registry.astronomer.io/providers/apache-airflow/versions/latest/modules/pythonvirtualenvoperator).  
                Python executables for the specified version must be installed on the executor.  
                Additional packages can be installed in the virtual environment by specifying a
                `requirements` parameter as a list of strings.
                  
                Astronomer has created a [Docker Buildkit](https://github.com/astronomer/astro-provider-venv) 
                to simplify building virtual environments with Astro CLI.  See the `Dockerfile` for 
                details.
                """
                from snowflake.snowpark import Window
                from datetime import date

                day_count = date.today() - date(2018,1,1)
                months = snowpark_session.generator(F.seq4(), rowcount=day_count.days)\
                                         .with_column('date_month', F.date_trunc('month', 
                                                                            F.date_add(F.to_date(F.lit('2018-01-01')), 
                                                                                       F.row_number().over(Window.order_by('SEQ4(0)')))))\
                                         .select('date_month').distinct().sort('date_month', ascending=True)

                subscription_periods = subscription_df.with_column('start_date', F.to_date('start_date'))\
                                                      .with_column('end_date', F.to_date('end_date'))
                
                customers = subscription_periods.group_by('customer_id').agg(F.date_trunc('month', F.min('start_date')).alias('date_month_start'),
                                                                             F.date_trunc('month', F.max('end_date')).alias('date_month_end'))
                
                customer_months = customers.join(months, how='inner', on=(months['date_month'] >= customers['date_month_start']) & 
                                                                         ( months['date_month'] < customers['date_month_end']))\
                                           .select(['customer_id', 'date_month'])
                
                customer_revenue_by_month = customer_months.join(subscription_periods, 
                                                                how='left',
                                                                rsuffix='_',
                                                                on=(customer_months.customer_id == subscription_periods.customer_id) & 
                                                                    (customer_months.date_month >= subscription_periods.start_date) & 
                                                                    ((customer_months.date_month < subscription_periods.end_date) |
                                                                        (subscription_periods.end_date.is_null())))\
                                                            .fillna(subset=['monthly_amount'], value=0)\
                                                            .select(F.col('date_month'), F.col('customer_id'), F.col('monthly_amount').alias('mrr'))\
                                                            .with_column('is_active', F.col('mrr')>0)\
                                                            .with_column('first_active_month', 
                                                                         F.when(F.col('is_active'), 
                                                                            F.min(F.col('date_month')).over(Window.partition_by('customer_id'))))\
                                                            .with_column('last_active_month', 
                                                                         F.when(F.col('is_active'), 
                                                                            F.max(F.col('date_month')).over(Window.partition_by('customer_id'))))\
                                                            .with_column('is_first_month', F.col('first_active_month') == F.col('date_month'))\
                                                            .with_column('is_last_month', F.col('last_active_month') == F.col('date_month'))
                                                            
                customer_churn_month = customer_revenue_by_month.where('is_last_month')\
                                                                .select(F.add_months(F.col('date_month'), 1),
                                                                        'customer_id',
                                                                        F.to_decimal('mrr', 38, 2),
                                                                        F.lit(False).alias('is_active'),
                                                                        'first_active_month',
                                                                        'last_active_month',
                                                                        F.lit(False).alias('is_first_month'),
                                                                        F.lit(False).alias('is_last_month'))
                
                customer_date_window = Window.partition_by('customer_id').order_by('date_month')

                mrr = customer_revenue_by_month.union_all(customer_churn_month)\
                                               .with_column('id', F.md5(F.col('customer_id')))\
                                               .with_column('previous_month_is_active', 
                                                            F.lag('is_active', default_value=False).over(customer_date_window))\
                                               .with_column('previous_month_mrr', 
                                                            F.lag('mrr', default_value=0).over(customer_date_window))\
                                               .with_column('mrr_change', F.col('mrr') - F.col('previous_month_mrr'))\
                                               .with_column('change_category', 
                                                            F.when(F.col('is_first_month'), 'new')\
                                                             .when(F.not_(F.col('is_active') & F.col('previous_month_is_active')), 'churn')\
                                                             .when(F.col('is_active') & F.not_(F.col('previous_month_is_active')), 'reactivation')\
                                                             .when(F.col('mrr_change') > 0, 'upgrade')\
                                                             .when(F.col('mrr_change') < 0, 'downgrade')
                                                            )\
                                               .with_column('renewal_amount', F.least(F.col('mrr'), F.col('previous_month_mrr')))

                return mrr
            
            @task.snowpark_ext_python(python='/home/astro/.venv/snowpark/bin/python')
            def attribution_playbook(customer_conversions_df:SnowparkTable, sessions_df:SnowparkTable):
                """
                Snowpark Python currently supports Python 3.8, 3.9, and 3.10.  If the version of 
                Python used to run Airflow is different it may be necessary to use a Python virtual 
                environment for Snowpark tasks.  
                  
                The `snowpark_ext_python` decorator and `SnowparkExternalPythonOperator` allow users 
                to specify a different python executable similar to the 
                [ExternalPythonOperator](https://registry.astronomer.io/providers/apache-airflow/versions/latest/modules/externalpythonoperator).  
                  
                Astronomer has created a [Docker Buildkit](https://github.com/astronomer/astro-provider-venv) 
                to simplify building virtual environments with Astro CLI.  See the `Dockerfile` for 
                details.
                """
                from snowflake.snowpark import Window

                customer_window = Window.partition_by('customer_id')

                attribution_touches = sessions_df.join(customer_conversions_df, on='customer_id')\
                                                .filter((F.col('started_at') <= F.col('converted_at')) & 
                                                        (F.col('started_at') >= F.date_add(F.col('converted_at'), -30)))\
                                                .with_column('total_sessions', F.count('customer_id')\
                                                                                .over(customer_window))\
                                                .with_column('session_index', F.row_number()\
                                                                                .over(customer_window\
                                                                                .order_by('started_at')))\
                                                .with_column('first_touch_points', 
                                                            F.when(F.col('session_index') == 1, 1)\
                                                            .otherwise(0))\
                                                .with_column('last_touch_points', 
                                                            F.when(F.col('session_index') == F.col('total_sessions'), 1)\
                                                            .otherwise(0))\
                                                .with_column('forty_twenty_forty_points', 
                                                            F.when(F.col('total_sessions') == 1, 1)\
                                                            .when(F.col('total_sessions') == 2, .5)\
                                                            .when(F.col('session_index') == 1, .4)\
                                                            .when(F.col('session_index') == F.col('total_sessions'), .4)\
                                                            .otherwise(F.lit(0.2) / (F.col('total_sessions') - 2)))\
                                                .with_column('linear_points', F.lit(1) / F.col('total_sessions'))\
                                                .with_column('first_touch_revenue', 
                                                             F.col('revenue') * F.col('first_touch_points'))\
                                                .with_column('last_touch_revenue', 
                                                             F.col('revenue') * F.col('last_touch_points'))\
                                                .with_column('forty_twenty_forty_revenue', 
                                                             F.col('revenue') * F.col('forty_twenty_forty_points'))\
                                                .with_column('linear_revenue', 
                                                             F.col('revenue') * (1 / F.col('total_sessions')))
                return attribution_touches

            _customers = jaffle_shop(customers_df=SnowparkTable('stg_customers'),
                                     orders_df=SnowparkTable('stg_orders'),
                                     payments_df=SnowparkTable('stg_payments'))
            
            _mrr = mrr_playbook(subscription_df=SnowparkTable('stg_subscription_periods'))

            _attribution_touches = attribution_playbook(customer_conversions_df=SnowparkTable('stg_customer_conversions'), 
                                                        sessions_df=SnowparkTable('stg_sessions'))
            
            return _attribution_touches, _mrr, _customers

        _structured_data = load_structured_data()
        _attribution_touches, _mrr, _customers = transform_structured()
        _structured_data >> [_attribution_touches, _mrr, _customers]

        return _attribution_touches, _mrr, _customers

    @task_group()
    def unstructured_data():
        
        @task_group()
        def load_unstructured_data():
            
            @task.snowpark_python()
            def load_support_calls_to_stage(restore_data_uri:str, calls_directory_stage:str) -> str:
                import zipfile
                import io
                import tempfile
                import requests

                with tempfile.TemporaryDirectory() as td:
                    calls_zipfile = requests.get(f'{restore_data_uri}/customer_calls.zip').content 
                    buffer = io.BytesIO(calls_zipfile)
                    z = zipfile.ZipFile(buffer)
                    z.extractall(td)

                    snowpark_session.file.put(local_file_name=f"file://{td}/customer_calls/*",
                                              stage_location=f"@{calls_directory_stage}",
                                              source_compression=None,
                                              auto_compress=False,
                                              overwrite=True)

                snowpark_session.sql(f"ALTER STAGE {calls_directory_stage} REFRESH;").collect()

                return calls_directory_stage

            _calls_directory_stage = load_support_calls_to_stage(restore_data_uri=restore_data_uri, 
                                                                 calls_directory_stage=calls_directory_stage)
            
            _stg_comment_table = aql.load_file(task_id='load_twitter_comments',
                                               input_file = File(f'{restore_data_uri}/twitter_comments.parquet'),
                                               output_table = Table(name='STG_TWITTER_COMMENTS', 
                                                                    conn_id=_SNOWFLAKE_CONN_ID),
                                               use_native_support=False)

            _stg_training_table = aql.load_file(task_id='load_comment_training',
                                                input_file = File(f'{restore_data_uri}/comment_training.parquet'), 
                                                output_table = Table(name='STG_COMMENT_TRAINING', 
                                                                     conn_id=_SNOWFLAKE_CONN_ID),
                                                use_native_support=False)

            return _calls_directory_stage, _stg_comment_table, _stg_training_table
        
        _calls_directory_stage, _stg_comment_table, _stg_training_table = load_unstructured_data()
        
        whisper_requirements = [
            'numpy',
            'torch==2.0.0',
            'tqdm',
            'more-itertools==9.1.0',
            'transformers==4.27.4',
            'ffmpeg-python==0.2.0',
            'openai-whisper==v20230314']
        
        @task.snowpark_virtualenv(requirements=whisper_requirements)
        def transcribe_calls(calls_directory_stage:str):
            import requests
            import tempfile
            from pathlib import Path            
            import os
            import whisper

            model = whisper.load_model('tiny.en', download_root=os.getcwd())
            
            calls_df = snowpark_session.sql(f"""SELECT *, 
                                                       get_presigned_url(@{calls_directory_stage}, 
                                                            LIST_DIR_TABLE.RELATIVE_PATH) as presigned_url 
                                                FROM DIRECTORY( @{calls_directory_stage})""")
            calls_df = calls_df.to_pandas()

            #Extract customer_id from file name
            calls_df['CUSTOMER_ID']= calls_df['RELATIVE_PATH'].apply(lambda x: x.split('-')[0])

            with tempfile.TemporaryDirectory() as tmpdirname:
                
                calls_df.apply(lambda x: Path(tmpdirname)\
                                        .joinpath(x.RELATIVE_PATH)\
                                        .write_bytes(requests.get(x.PRESIGNED_URL).content), axis=1)
                
                calls_df['TRANSCRIPT'] = calls_df.apply(lambda x: model.transcribe(Path(tmpdirname)
                                                                            .joinpath(x.RELATIVE_PATH).as_posix())['text'], axis=1)

            return snowpark_session.create_dataframe(calls_df[['CUSTOMER_ID', 'RELATIVE_PATH', 'TRANSCRIPT']])

        _stg_calls_table = transcribe_calls(calls_directory_stage=_calls_directory_stage)

        @task_group()
        def generate_embeddings(): 

            @task.snowpark_python()
            def get_training_pandas(stg_training_table:SnowparkTable):
                """
                The `weaviate_import` decorator below takes a pandas dataframe for input. This 
                task pulls the data from Snowflake tables and passes it to the import task as 
                a pandas dataframe.

                Normally Airflow's cross-communication (XCOM) system is not designed for passing 
                large or complex (ie. non-json-serializable) data between tasks as it relies on 
                storing data in a database (usually postgres or mysql). Airflow 2.5+ has added 
                support for [passing pandas dataframe](https://github.com/apache/airflow/pull/30390). 
                However, if the data is very large or if passing other types of data this will fail. 
                
                Additionally, for data governance reasons it may be not be advisable to pass 
                sensitive or regulated data between tasks to avoid storing this data in a the
                XCOM database.
                
                For these reasons the Astronomer provider for Snowpark includes a [custom XCOM 
                backend](https://docs.astronomer.io/learn/xcom-backend-tutorial) which saves 
                XCOM data passed between tasks in either Snowflake tables or stages. 
                Small, JSON-serializable data is stored in a single table and large or 
                non-serializable data is stored in a stage.  This allows passing arbitrarily 
                large or complex data between tasks and ensures that all data, including 
                intermediate datasets, stay inside the secure, goverened boundary of Snowflake.

                The 'AIRFLOW__CORE__XCOM_SNOWFLAKE_TABLE', 'AIRFLOW__CORE__XCOM_SNOWFLAKE_STAGE', 
                and 'AIRFLOW__CORE__XCOM_SNOWFLAKE_CONN_NAME' settings in the 
                `docker-compose.override.yml` file specify where this data is stored.

                This task returns a pandas dataframe for downstream import by passing it through 
                the custom XCOM backend.
                """
                return stg_training_table.to_pandas()
            
            @task.snowpark_python()
            def get_comment_pandas(stg_comment_table:SnowparkTable):
                return stg_comment_table.to_pandas()
            
            @task.snowpark_python()
            def get_calls_pandas(stg_calls_table:SnowparkTable):
                return stg_calls_table.to_pandas()
            
            @task.weaviate_import()
            def generate_training_embeddings(stg_training_table:pd.DataFrame):
                """
                Astronomer's [Weaviate provider for Airflow](https://registry.astronomer.io/providers/airflow-provider-weaviate/versions/latest) 
                provides the `weaviate_import` decorator or `WeaviateImportDataOperator` to import 
                data to Weaviate.  
                
                The decorator version includes a pre_execute phase which executes a decorated 
                python callable function just before passing the data for import.

                This task takes the pandas dataframe fetched previously by passing 
                it through the custom XCOM backend for Snowflake, performs some simple 
                transformations and must return a dictionary with all required parameters 
                for the import operator.

                Because we restored weaviate from pre-built embeddings this task should "skip" 
                re-importing each row.
                """

                df = stg_training_table
                df.rename({'REVIEW_TEXT': 'rEVIEW_TEXT', 'LABEL': 'lABEL'}, axis=1, inplace=True)

                df['lABEL'] = df['lABEL'].apply(str)

                #openai works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['rEVIEW_TEXT'] = df['rEVIEW_TEXT'].apply(lambda x: x.replace("\n",""))
                df['UUID'] = df.apply(lambda x: generate_uuid5(x.to_dict(), 'CommentTraining'), axis=1)

                return {"data": df, 
                        "class_name": 'CommentTraining', 
                        "uuid_column": "UUID", 
                        "batch_size": 1000, 
                        "error_threshold": 0}
            
            @task.weaviate_import()
            def generate_twitter_embeddings(stg_comment_table:pd.DataFrame):
                """
                Because we restored weaviate from pre-built embeddings this task should "skip" 
                re-importing each row.
                """

                df = stg_comment_table
                df.rename({'CUSTOMER_ID': 'cUSTOMER_ID', 'REVIEW_TEXT': 'rEVIEW_TEXT', 'DATE': 'dATE'}, axis=1, inplace=True)

                df['cUSTOMER_ID'] = df['cUSTOMER_ID'].apply(str)
                df['dATE'] = pd.to_datetime(df['dATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

                #openai works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['rEVIEW_TEXT'] = df['rEVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

                df['UUID'] = df.apply(lambda x: generate_uuid5(x.to_dict(), 'CustomerComment'), axis=1)

                return {"data": df, 
                        "class_name": 'CustomerComment', 
                        "uuid_column": "UUID", 
                        "batch_size": 1000, 
                        "error_threshold": 0}

            @task.weaviate_import()
            def generate_call_embeddings(stg_calls_table:pd.DataFrame):
                """
                Because we restored weaviate from pre-built embeddings this task should "skip" 
                re-importing each row.
                """
                
                df = stg_calls_table
                df.rename({'CUSTOMER_ID': 'cUSTOMER_ID', 'TRANSCRIPT': 'tRANSCRIPT', 'RELATIVE_PATH': 'rELATIVE_PATH'}, axis=1, inplace=True)

                df['cUSTOMER_ID'] = df['cUSTOMER_ID'].apply(str)

                #openai works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['tRANSCRIPT'] = df['tRANSCRIPT'].apply(lambda x: x.replace("\n",""))

                df['UUID'] = df.apply(lambda x: generate_uuid5(x.to_dict(), 'CustomerCall'), axis=1)

                return {"data": df, 
                        "class_name": 'CustomerCall', 
                        "uuid_column": "UUID", 
                        "batch_size": 1000, 
                        "error_threshold": 0}

            _training_table = get_training_pandas(stg_training_table=_stg_training_table)
            _training_table = generate_training_embeddings(stg_training_table=_training_table)
            
            _comment_table = get_comment_pandas(stg_comment_table=_stg_comment_table)
            _comment_table = generate_twitter_embeddings(stg_comment_table=_comment_table)
            
            _calls_table = get_calls_pandas(stg_calls_table=_stg_calls_table)
            _calls_table = generate_call_embeddings(stg_calls_table=_calls_table)

            return _training_table, _comment_table, _calls_table
        
        _training_table, _comment_table, _calls_table = generate_embeddings()

        return _training_table, _comment_table, _calls_table
    
    @task.snowpark_virtualenv(requirements=['lightgbm==3.3.5', 'scikit-learn==1.2.2', 'astro_provider_snowflake'])
    def train_sentiment_classifier(class_name:str, snowpark_model_registry:dict):
        """
        The Snowpark ML framework has many options for feature engineering and model training.  See the 
        [Quickstart](https://quickstarts.snowflake.com/guide/getting_started_with_dataengineering_ml_using_snowpark_python/index.html#0
        for details.  

        In this demo we show how to use the Snowpark ML model registry with a bring-your-own model.

        This task fetches embeddings for a labled dataset using the Weaviate hook from the 
        Weaviate provider for Airflow.  Labled and embeddings are used to train a simple sentiment 
        classifier.  No optimization or model evaluation steps are performed as they are outside 
        the scope of the demo.
        """
        from snowflake.ml.registry import model_registry
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split 
        from lightgbm import LGBMClassifier
        from uuid import uuid1
        from weaviate_provider.hooks.weaviate import WeaviateHook

        registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                database_name=snowpark_model_registry['database'], 
                                                schema_name=snowpark_model_registry['schema'])
        
        weaviate_client = WeaviateHook('weaviate_default').get_conn()

        df = pd.DataFrame(weaviate_client.data_object.get(with_vector=True, class_name=class_name)['objects'])
        df = pd.concat([pd.json_normalize(df['properties']), df['vector']], axis=1)

        model_version = uuid1().urn
        model_name='sentiment_classifier'

        X_train, X_test, y_train, y_test = train_test_split(df['vector'], df['lABEL'], test_size=.3, random_state=1883)
        X_train = np.array(X_train.values.tolist())
        y_train = np.array(y_train.values.tolist())
        X_test = np.array(X_test.values.tolist())
        y_test = np.array(y_test.values.tolist())
        
        model = LGBMClassifier(random_state=42)
        model.fit(X=X_train, y=y_train, eval_set=(X_test, y_test))

        model_id = registry.log_model(
            model=model, 
            model_name=model_name, 
            model_version=model_version, 
            sample_input_data=X_test[0].reshape(1,-1),
            tags={'stage': 'dev', 'model_type': 'lightgbm.LGBMClassifier'})
        
        return {'name': model_id.get_name(), 'version':model_id.get_version()}

    @task_group()
    def score_sentiment():

        @task.snowpark_virtualenv(requirements=['lightgbm==3.3.5', 'astro_provider_snowflake'], retries=2, retry_delay=datetime.timedelta(seconds=5))
        def call_sentiment(class_name:str, snowpark_model_registry:dict, model:dict) -> SnowparkTable:
            """
            Airflow task retries are a good way to gracefully deal with race conditions or other 
            transient errors in external/dependent systems.

            At the time this demo was built there is a race condition or concurrency issue with 
            the `load_model()` function in the Snowpark ML model registry.  Task failures may 
            occur with either of the two scoring tasks and by setting `retries=2` we can simply 
            rerun the task 5 seconds later and the DAG will complete.
            """
            from snowflake.ml.registry import model_registry
            import numpy as np
            import pandas as pd
            from weaviate_provider.hooks.weaviate import WeaviateHook
            weaviate_client = WeaviateHook('weaviate_default').get_conn()

            df = pd.DataFrame(weaviate_client.data_object.get(with_vector=True, class_name=class_name)['objects'])
            df = pd.concat([pd.json_normalize(df['properties']), df['vector']], axis=1)

            registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                    database_name=snowpark_model_registry['database'], 
                                                    schema_name=snowpark_model_registry['schema'])
            
            metrics = registry.get_metrics(model_name=model['name'], model_version=model['version'])
            model = registry.load_model(model_name=model['name'], model_version=model['version'])
            
            df['sentiment'] = model.predict_proba(np.stack(df['vector'].values))[:,1]


            return snowpark_session.create_dataframe(df.rename(columns=str.upper))

        @task.snowpark_virtualenv(requirements=['lightgbm==3.3.5', 'astro_provider_snowflake'], retries=2, retry_delay=datetime.timedelta(seconds=5))
        def twitter_sentiment(class_name:str, snowpark_model_registry:dict, model:dict) -> SnowparkTable:
            """
            Airflow task retries are a good way to gracefully deal with race conditions or other 
            transient errors in external/dependent systems.

            At the time this demo was built there is a race condition or concurrency issue with 
            the `load_model()` function in the Snowpark ML model registry.  Task failures may 
            occur with either of the two scoring tasks and by setting `retries=2` we can simply 
            rerun the task 5 seconds later and the DAG will complete.
            """

            from snowflake.ml.registry import model_registry
            import numpy as np
            import pandas as pd
            from weaviate_provider.hooks.weaviate import WeaviateHook
            weaviate_client = WeaviateHook('weaviate_default').get_conn()

            df = pd.DataFrame(weaviate_client.data_object.get(with_vector=True, class_name=class_name)['objects'])
            df = pd.concat([pd.json_normalize(df['properties']), df['vector']], axis=1)
            
            registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                    database_name=snowpark_model_registry['database'], 
                                                    schema_name=snowpark_model_registry['schema'])
            
            metrics = registry.get_metrics(model_name=model['name'], model_version=model['version'])
            model = registry.load_model(model_name=model['name'], model_version=model['version'])
            
            df['sentiment'] = model.predict_proba(np.stack(df['vector'].values))[:,1]

            return snowpark_session.create_dataframe(df.rename(columns=str.upper))
        
        _pred_calls_table = call_sentiment(class_name='CustomerCall',
                                           snowpark_model_registry=_snowpark_model_registry, 
                                           model=_model)
        
        _pred_comment_table = twitter_sentiment(class_name='CustomerComment',
                                                snowpark_model_registry=_snowpark_model_registry, 
                                                model=_model)

        return _pred_calls_table, _pred_comment_table

    @task_group()
    def exit():
        """
        The exit() task group is a good place to perform any tasks which consolidate or cleanup
        from other tasks in the DAG.
        """

        @task.snowpark_python()
        def create_presentation_tables(attribution_df:SnowparkTable, 
                                       mrr_df:SnowparkTable, 
                                       customers_df:SnowparkTable,
                                       pred_calls_table:SnowparkTable, 
                                       pred_comment_table:SnowparkTable):
            """
            This task consolidates all of the structured and unstructured data results to create
            tables for the presentation layer running in the Streamlit app.
            
            Because the app needs to know the name for tables we write them specifically here 
            with `save_as_table` rather than passing through xcom or using the Snowpark return 
            processing.
            """
            customers_df = customers_df.with_column('CLV', 
                                                    F.round(F.col('CUSTOMER_LIFETIME_VALUE'), 2))

            sentiment_df =  pred_calls_table.group_by(F.col('CUSTOMER_ID'))\
                                            .agg(F.avg('SENTIMENT').alias('CALLS_SENTIMENT'))\
                                            .join(pred_comment_table.group_by(F.col('CUSTOMER_ID'))\
                                                    .agg(F.avg('SENTIMENT').alias('COMMENTS_SENTIMENT')), 
                                                on='cUSTOMER_ID',
                                                how='right')\
                                            .fillna(0, subset=['CALLS_SENTIMENT'])\
                                            .with_column('SENTIMENT_SCORE', 
                                                         F.round((F.col('CALLS_SENTIMENT') \
                                                                  + F.col('COMMENTS_SENTIMENT'))/2, 4))\
                                            .with_column('SENTIMENT_BUCKET', 
                                                         F.call_builtin('WIDTH_BUCKET',
                                                                         F.col('SENTIMENT_SCORE'), 0, 1, 10))
                                    
            sentiment_df.write.save_as_table('PRES_SENTIMENT', mode='overwrite')
            
            ad_spend_df = attribution_df.select(['UTM_MEDIUM', 'REVENUE'])\
                                        .dropna()\
                                        .group_by(F.col('UTM_MEDIUM'))\
                                        .sum(F.col('REVENUE'))\
                                        .rename('SUM(REVENUE)', 'Revenue')\
                                        .rename('UTM_MEDIUM', 'Medium')\
                                        .write.save_as_table('PRES_AD_SPEND', mode='overwrite')
            
            clv_df = customers_df.dropna(subset=['CLV'])\
                                 .join(sentiment_df, 'CUSTOMER_ID', how='left')\
                                 .sort(F.col('CLV'), ascending=False)\
                                 .with_column('NAME', 
                                              F.concat(F.col('FIRST_NAME'), 
                                                       F.lit(' '), 
                                                       F.col('LAST_NAME')))\
                                 .select(['CUSTOMER_ID', 
                                          'NAME', 
                                          'FIRST_ORDER', 
                                          'MOST_RECENT_ORDER', 
                                          'NUMBER_OF_ORDERS', 
                                          'CLV', 
                                          'SENTIMENT_SCORE'])\
                                 .write.save_as_table('PRES_CLV', mode='overwrite')
            
            churn_df = customers_df.select(['CUSTOMER_ID', 'FIRST_NAME', 'LAST_NAME', 'CLV'])\
                                   .join(mrr_df.select(['CUSTOMER_ID', 
                                                        'FIRST_ACTIVE_MONTH', 
                                                        'LAST_ACTIVE_MONTH', 
                                                        'CHANGE_CATEGORY']), 
                                        on='CUSTOMER_ID', 
                                        how='right')\
                                   .join(sentiment_df, 'CUSTOMER_ID', how='left')\
                                   .dropna(subset=['CLV'])\
                                   .filter(F.col('CHANGE_CATEGORY') == 'churn')\
                                   .sort(F.col('LAST_ACTIVE_MONTH'), ascending=False)\
                                   .with_column('NAME', 
                                                F.concat(F.col('FIRST_NAME'), 
                                                         F.lit(' '), 
                                                         F.col('LAST_NAME')))\
                                   .select(['CUSTOMER_ID', 
                                            'NAME', 
                                            'CLV', 
                                            'LAST_ACTIVE_MONTH', 
                                            'SENTIMENT_SCORE'])\
                                   .write.save_as_table('PRES_CHURN', mode='overwrite')
            
            pred_calls_table.write.save_as_table('PRED_CUSTOMER_CALLS', mode='overwrite')
            pred_comment_table.write.save_as_table('PRED_TWITTER_COMMENTS', mode='overwrite')
            attribution_df.write.save_as_table('ATTRIBUTION_TOUCHES', mode='overwrite')
        
        create_presentation_tables(attribution_df=_attribution_touches, 
                                   mrr_df=_mrr, 
                                   customers_df=_customers,
                                   pred_calls_table=_pred_calls_table, 
                                   pred_comment_table=_pred_comment_table)
        
    @task.snowpark_python()
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
    
    _create_snowflake_objects = create_snowflake_objects(snowflake_objects, calls_directory_stage).as_setup() 

    with cleanup_temp_tables(snowflake_objects).as_teardown(setups=_create_snowflake_objects):
        _snowpark_model_registry, _restore_weaviate = enter()

        _attribution_touches, _mrr, _customers = structured_data()
        
        _training_table, _comment_table, _calls_table = unstructured_data()

        _model = train_sentiment_classifier(class_name='CommentTraining',
                                            snowpark_model_registry=_snowpark_model_registry)
        
        _pred_calls_table, _pred_comment_table = score_sentiment()
        
        _exit = exit()
        
        _restore_weaviate >> [_training_table, _comment_table, _calls_table] >> _model

customer_analytics()

def test():
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
    from snowflake.snowpark import Session as SnowparkSession
    from snowflake.snowpark import functions as F, types as T
    conn_params = SnowflakeHook('snowflake_default')._get_conn_params()
    snowpark_session = SnowparkSession.builder.configs(conn_params).create()