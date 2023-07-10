"""
### Use the Snowpark Provider in an advanced Machine Learning workflow

This DAG demonstrates an end-to-end application workflow using OpenAI embeddings with a
Weaviate vector database as well as Snowpark decorators, the Snowflake XCOM backend and 
the Snowpark ML model registry.  The Astro CLI can easily be adapted to include additional 
Docker-based services.  This demo includes services for Minio, Weaviate and streamlit.

The Snowpark provider is in a dev status and not yet in the pypi registry. 
Instead the provider is available via a wheel file in the linked 
repository.

See [README_CA.md](https://github.com/astronomer/airflow-snowparkml-demo/blob/main/README_CA.md) for setup instructions.
"""

from datetime import datetime 
import os
from pathlib import Path
import json

from astro import sql as aql 
from astro.files import File 
from astro.sql.table import Table 
from airflow.decorators import dag, task, task_group
from cosmos.task_group import DbtTaskGroup
from astronomer.providers.snowflake.utils.snowpark_helpers import SnowparkTable
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator


_SNOWFLAKE_CONN_ID = 'snowflake_default'

demo_database = os.environ['DEMO_DATABASE']
demo_schema = os.environ['DEMO_SCHEMA']

restore_data_uri = 'https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/sissy-g-toys-demo/data'
calls_directory_stage = 'call_stage'
weaviate_backup_bucket = 'weaviate-backup'
data_sources = ['ad_spend', 'sessions', 'customers', 'payments', 'subscription_periods', 'customer_conversions', 'orders']
twitter_sources = ['twitter_comments', 'comment_training']
weaviate_class_objects = {'CommentTraining': {'count': 1987}, 'CustomerComment': {'count': 12638}, 'CustomerCall': {'count': 43}}
minio_conn = json.loads(os.environ['AIRFLOW_CONN_MINIO_DEFAULT'])
weaviate_conn = json.loads(os.environ['AIRFLOW_CONN_WEAVIATE_DEFAULT'])

_DBT_BIN = '/home/astro/.venv/dbt/bin/dbt'

default_args={
    "temp_data_output": 'table',
    "temp_data_schema": demo_schema,
    "temp_data_overwrite": True
}

@dag(schedule=None, start_date=datetime(2023, 1, 1), catchup=False, default_args=default_args)
def customer_analytics():
    
    @task_group()
    def enter():

        @task()
        def create_weaviate_backup_bucket() -> str:
            import minio

            minio_client = minio.Minio(
                endpoint=minio_conn['extra']['endpoint_url'],
                access_key=minio_conn['extra']['aws_access_key'],
                secret_key=minio_conn['extra']['aws_secret_access_key'],
                secure=False
            )
            minio_client.list_buckets()

            try:
                minio_client.make_bucket(weaviate_backup_bucket)
            except Exception as e:
                if e.code == 'BucketAlreadyOwnedByYou':
                    print(e.code)

            minio_client.list_buckets()

            return weaviate_backup_bucket
        
        @task()
        def restore_weaviate(weaviate_backup_bucket:str, replace_existing=False):
            """
            This task exists only to speedup the demo. By restoring prefetched embeddings to weaviate the later tasks 
            will skip embeddings and only make calls to openai for data it hasn't yet embedded.
            """
            import weaviate 
            import tempfile
            import urllib
            import zipfile
            import os
            import warnings
            import minio

            weaviate_restore_uri = f'{restore_data_uri}/weaviate-backup/backup.zip'

            weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                              additional_headers=weaviate_conn['extra']['headers'])
            assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

            minio_client = minio.Minio(
                endpoint=minio_conn['extra']['endpoint_url'],
                access_key=minio_conn['extra']['aws_access_key'],
                secret_key=minio_conn['extra']['aws_secret_access_key'],
                secure=False
            )

            if replace_existing:
                weaviate_client.schema.delete_all()
            
            else:
                existing_classes = [classes['class'] for classes in weaviate_client.schema.get()['classes']]
                class_collision = set.intersection(set(existing_classes), set(weaviate_class_objects.keys()))
                if class_collision:
                    warnings.warn(f'Class objects {class_collision} already exist and replace_existing={replace_existing}. Skipping restore.')
                    response = 'skipped'

                    return weaviate_conn
            
            with tempfile.TemporaryDirectory() as td:
                zip_path, _ = urllib.request.urlretrieve(weaviate_restore_uri)
                with zipfile.ZipFile(zip_path, "r") as f:
                    f.extractall(td)

                for root, dirs, files in os.walk(td, topdown=False):
                    for name in files:
                        filename = os.path.join(root, name)

                        minio_client.fput_object(
                            object_name=os.path.relpath(filename, td),
                            file_path=filename,
                            bucket_name=weaviate_backup_bucket,
                        )

            response = weaviate_client.backup.restore(
                    backup_id='backup',
                    backend="s3",
                    include_classes=list(weaviate_class_objects.keys()),
                    wait_for_completion=True,
                )

            #check backup counts
            for class_object in weaviate_class_objects.keys():
                expected_count = weaviate_class_objects[class_object]['count']
                response = weaviate_client.query.aggregate(class_name=class_object).with_meta_count().do()               
                count = response["data"]["Aggregate"][class_object][0]["meta"]["count"]
                assert count == expected_count, f"Class {class_object} check failed. Expected {expected_count} objects.  Found {count}"
            
            return weaviate_conn
            
        @task.snowpark_python(snowflake_conn_id=_SNOWFLAKE_CONN_ID)
        def check_model_registry(demo_database:str, demo_schema:str) -> dict: 
            from snowflake.ml.registry import model_registry

            assert model_registry.create_model_registry(session=snowpark_session, 
                                                        database_name=demo_database, 
                                                        schema_name=demo_schema)
            
            return {'database': demo_database, 'schema': demo_schema}

        _weaviate_backup_bucket = create_weaviate_backup_bucket()
        _weaviate_conn = restore_weaviate(_weaviate_backup_bucket)
        _snowpark_model_registry = check_model_registry(demo_database, demo_schema)

        return _weaviate_backup_bucket, _weaviate_conn, _snowpark_model_registry
    
    @task_group()
    def structured_data():

        @task_group()
        def load_structured_data():
            for source in data_sources:
                aql.load_file(task_id=f'load_{source}',
                    input_file = File(f"{restore_data_uri}/{source}.csv"), 
                    output_table = Table(name=f'STG_{source.upper()}', conn_id=_SNOWFLAKE_CONN_ID)
                )

        @task_group()
        def data_quality_checks():
            expectations_dir = Path('include/great_expectations').joinpath(f'expectations')
            for project in [x[0] for x in os.walk(expectations_dir)][1:]:
                for expectation in os.listdir(project):
                    project = project.split('/')[-1]
                    expectation = expectation.split('.')[0]
                    GreatExpectationsOperator(
                        task_id=f"ge_{project}_{expectation}",
                        data_context_root_dir='include/great_expectations',
                        conn_id=_SNOWFLAKE_CONN_ID,
                        expectation_suite_name=f"{project}.{expectation}",
                        data_asset_name=f"STG_{expectation.upper()}",
                        fail_task_on_validation_failure=False,
                        return_json_dict=True,
                    )
            
        @task_group()
        def transform_structured():
            jaffle_shop = DbtTaskGroup(
                dbt_project_name="jaffle_shop",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_SNOWFLAKE_CONN_ID,
                dbt_args={"dbt_executable_path": _DBT_BIN},
                profile_args={
                    "schema": demo_schema,
                },
                test_behavior="after_all",
            )
            
            attribution_playbook = DbtTaskGroup(
                dbt_project_name="attribution_playbook",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_SNOWFLAKE_CONN_ID,
                dbt_args={"dbt_executable_path": _DBT_BIN},
                profile_args={
                    "schema": demo_schema,
                },
            )

            mrr_playbook = DbtTaskGroup(
                dbt_project_name="mrr_playbook",
                dbt_root_path="/usr/local/airflow/include/dbt",
                conn_id=_SNOWFLAKE_CONN_ID,
                dbt_args={"dbt_executable_path": _DBT_BIN},
                profile_args={
                    "schema": demo_schema,
                },
            )

        load_structured_data() >> \
            data_quality_checks() >> \
                transform_structured()

    @task_group()
    def unstructured_data(weaviate_conn:dict):
        
        @task_group()
        def load_unstructured_data():
            
            @task.snowpark_python(snowflake_conn_id=_SNOWFLAKE_CONN_ID)
            def load_support_calls_to_stage(restore_data_uri:str, calls_directory_stage:str) -> str:
                import zipfile
                import io
                import tempfile
                import requests

                snowpark_session.sql(f"""CREATE OR REPLACE STAGE {calls_directory_stage} 
                                        DIRECTORY = (ENABLE = TRUE) 
                                        ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
                                    """).collect()

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

            _calls_directory_stage = load_support_calls_to_stage(restore_data_uri=restore_data_uri, calls_directory_stage=calls_directory_stage)
            
            _stg_comment_table = aql.load_file(task_id='load_twitter_comments',
                input_file = File(f'{restore_data_uri}/twitter_comments.parquet'),
                output_table = Table(name='STG_TWITTER_COMMENTS', conn_id=_SNOWFLAKE_CONN_ID),
                use_native_support=False,
            )

            _stg_training_table = aql.load_file(task_id='load_comment_training',
                input_file = File(f'{restore_data_uri}/comment_training.parquet'), 
                output_table = Table(name='STG_COMMENT_TRAINING', conn_id=_SNOWFLAKE_CONN_ID),
                use_native_support=False,
            )

            return _calls_directory_stage, _stg_comment_table, _stg_training_table
        
        _calls_directory_stage, _stg_comment_table, _stg_training_table = load_unstructured_data()
        
        @task.snowpark_virtualenv(requirements=['numpy','torch==2.0.0','tqdm','more-itertools==9.1.0','transformers==4.27.4','ffmpeg-python==0.2.0','openai-whisper==v20230314'], snowflake_conn_id=_SNOWFLAKE_CONN_ID)
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
        def generate_embeddings(weaviate_conn:dict): 
            
            @task.snowpark_virtualenv(requirements=['weaviate-client==3.15.4'], snowflake_conn_id=_SNOWFLAKE_CONN_ID)
            def generate_training_embeddings(weaviate_conn:dict, stg_training_table:SnowparkTable):
                import weaviate
                from weaviate.util import generate_uuid5
                from time import sleep
                import numpy as np

                df = stg_training_table.to_pandas()

                df['LABEL'] = df['LABEL'].apply(str)

                #openai works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))

                weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                                  additional_headers=weaviate_conn['extra']['headers'])
                assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

                class_obj = {
                    "class": "CommentTraining",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                            "name": "rEVIEW_TEXT",
                            "dataType": ["text"]
                        },
                        {
                            "name": "lABEL",
                            "dataType": ["string"],
                            "moduleConfig": {"text2vec-openai": {"skip": True}}
                        }
                    ]
                }
                try:
                    weaviate_client.schema.create_class(class_obj)
                except Exception as e:
                    if isinstance(e, weaviate.UnexpectedStatusCodeException) and "already used as a name for an Object class" in e.message:                                
                        print("schema exists.")
                    else:
                        raise e
                    
                # #For openai subscription without rate limits go the fast route
                # uuids=[]
                # with client.batch as batch:
                #     batch.batch_size=100
                #     for properties in df.to_dict(orient='index').items():
                #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
                #         uuids.append(uuid)

                #For openai with rate limit go the VERY slow route
                #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
                uuids = []
                for row_id, row in df.T.items():
                    data_object = {'rEVIEW_TEXT': row[0], 'lABEL': row[1]}
                    uuid = generate_uuid5(data_object, class_obj['class'])
                    sleep_backoff=.5
                    success = False
                    while not success:
                        try:
                            if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                                print(f'UUID {uuid} exists.  Skipping.')
                            else:
                                uuid = weaviate_client.data_object.create(
                                    data_object=data_object, 
                                    uuid=uuid, 
                                    class_name=class_obj['class'])
                                print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
                                sleep(sleep_backoff)
                            success=True
                            uuids.append(uuid)
                        except Exception as e:
                            if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
                                sleep_backoff+=1
                                print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
                                sleep(sleep_backoff)
                            else:
                                raise(e)
                
                df['UUID']=uuids

                return snowpark_session.create_dataframe(df)

            @task.snowpark_virtualenv(requirements=['weaviate-client==3.15.4'], snowflake_conn_id=_SNOWFLAKE_CONN_ID)
            def generate_twitter_embeddings(weaviate_conn:dict, stg_comment_table:SnowparkTable):
                import weaviate
                from weaviate.util import generate_uuid5
                from time import sleep
                import numpy as np
                import pandas as pd
                                
                df = stg_comment_table.to_pandas()

                df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)
                df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime("%Y-%m-%dT%H:%M:%S-00:00")

                #openai embeddings works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['REVIEW_TEXT'] = df['REVIEW_TEXT'].apply(lambda x: x.replace("\n",""))
                
                weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                                  additional_headers=weaviate_conn['extra']['headers'])
                assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

                class_obj = {
                    "class": "CustomerComment",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                        "name": "CUSTOMER_ID",
                        "dataType": ["string"],
                        "moduleConfig": {"text2vec-openai": {"skip": True}}
                        },
                        {
                        "name": "DATE",
                        "dataType": ["date"],
                        "moduleConfig": {"text2vec-openai": {"skip": True}}
                        },
                        {
                        "name": "REVIEW_TEXT",
                        "dataType": ["text"]
                        }
                    ]
                }
                            
                try:
                    weaviate_client.schema.create_class(class_obj)
                except Exception as e:
                    if isinstance(e, weaviate.UnexpectedStatusCodeException) and \
                            "already used as a name for an Object class" in e.message:                                
                        print("schema exists.")
                    else:
                        raise e
                                
                # #For openai subscription without rate limits go the fast route
                # uuids=[]
                # with client.batch as batch:
                #     batch.batch_size=100
                #     for properties in df.to_dict(orient='index').items():
                #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
                #         uuids.append(uuid)

                #For openai with rate limit go the VERY slow route
                #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
                uuids = []
                for row_id, row in df.T.items():
                    data_object = {'cUSTOMER_ID': row[0], 'dATE': row[1], 'rEVIEW_TEXT': row[2]}
                    uuid = generate_uuid5(data_object, class_obj['class'])
                    sleep_backoff=.5
                    success = False
                    while not success:
                        try:
                            if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                                print(f'UUID {uuid} exists.  Skipping.')
                            else:
                                uuid = weaviate_client.data_object.create(
                                            data_object=data_object, 
                                            uuid=uuid, 
                                            class_name=class_obj['class']
                                        )
                                            
                                print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
                                sleep(sleep_backoff)
                            success=True
                            uuids.append(uuid)
                        except Exception as e:
                            if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
                                sleep_backoff+=1
                                print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
                                sleep(sleep_backoff)
                            else:
                                raise(e)

                df['UUID']=uuids
                df['DATE'] = pd.to_datetime(df['DATE'])

                return snowpark_session.create_dataframe(df)
            
            @task.snowpark_virtualenv(requirements=['weaviate-client==3.15.4'], snowflake_conn_id=_SNOWFLAKE_CONN_ID)
            def generate_call_embeddings(weaviate_conn:dict, stg_calls_table:SnowparkTable):
                import weaviate
                from weaviate.util import generate_uuid5
                from time import sleep
                import numpy as np

                df = stg_calls_table.to_pandas()

                df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(str)

                #openai embeddings works best without empty lines or new lines
                df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
                df['TRANSCRIPT'] = df['TRANSCRIPT'].apply(lambda x: x.replace("\n",""))

                weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                                  additional_headers=weaviate_conn['extra']['headers'])
                assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

                class_obj = {
                    "class": "CustomerCall",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                        "name": "CUSTOMER_ID",
                        "dataType": ["string"],
                        "moduleConfig": {"text2vec-openai": {"skip": True}}
                        },
                        {
                        "name": "RELATIVE_PATH",
                        "dataType": ["string"],
                        "moduleConfig": {"text2vec-openai": {"skip": True}}
                        },
                        {
                        "name": "TRANSCRIPT",
                        "dataType": ["text"]
                        }
                    ]
                }
                
                try:
                    weaviate_client.schema.create_class(class_obj)
                except Exception as e:
                    if isinstance(e, weaviate.UnexpectedStatusCodeException) and \
                            "already used as a name for an Object class" in e.message:                                
                        print("schema exists.")
                    
                # #For openai subscription without rate limits go the fast route
                # uuids=[]
                # with client.batch as batch:
                #     batch.batch_size=100
                #     for properties in df.to_dict(orient='index').items():
                #         uuid=client.batch.add_data_object(properties[1], class_obj['class'])
                #         uuids.append(uuid)

                #For openai with rate limit go the VERY slow route
                #Because we restored weaviate from pre-built embeddings this shouldn't be too long.
                uuids = []
                for row_id, row in df.T.items():
                    data_object = {'cUSTOMER_ID': row[0], 'rELATIVE_PATH': row[1], 'tRANSCRIPT': row[2]}
                    uuid = generate_uuid5(data_object, class_obj['class'])
                    sleep_backoff=.5
                    success = False
                    while not success:
                        try:
                            if weaviate_client.data_object.exists(uuid=uuid, class_name=class_obj['class']):
                                print(f'UUID {uuid} exists.  Skipping.')
                            else:
                                uuid = weaviate_client.data_object.create(
                                            data_object=data_object, 
                                            uuid=uuid, 
                                            class_name=class_obj['class']
                                        )   
                                print(f'Added row {row_id} with uuid {uuid}, sleeping for {sleep_backoff} seconds.')
                                sleep(sleep_backoff)
                            success=True
                            uuids.append(uuid)
                        except Exception as e:
                            if isinstance(e, weaviate.UnexpectedStatusCodeException) and "Rate limit reached" in e.message:                                
                                sleep_backoff+=1
                                print(f'Rate limit reached. Sleeping {sleep_backoff} seconds.')
                                sleep(sleep_backoff)
                            else:
                                raise(e)
                
                df['UUID']=uuids
                
                return snowpark_session.create_dataframe(df)
                            
            _training_table = generate_training_embeddings(weaviate_conn=weaviate_conn, stg_training_table=_stg_training_table)
            _comment_table = generate_twitter_embeddings(weaviate_conn=weaviate_conn, stg_comment_table=_stg_comment_table)
            _calls_table = generate_call_embeddings(weaviate_conn=weaviate_conn, stg_calls_table=_stg_calls_table)

            return _training_table, _comment_table, _calls_table

        _training_table, _comment_table, _calls_table = generate_embeddings(weaviate_conn=weaviate_conn) 
            
        return _training_table, _comment_table, _calls_table
        
    @task.snowpark_virtualenv(requirements=['lightgbm==3.3.5', 'scikit-learn==1.2.2', 'weaviate-client==3.15.4'], snowflake_conn_id=_SNOWFLAKE_CONN_ID)
    def train_sentiment_classifier(training_table:SnowparkTable, weaviate_conn:dict, snowpark_model_registry:dict):
        from snowflake.ml.registry import model_registry
        import numpy as np
        from sklearn.model_selection import train_test_split 
        from lightgbm import LGBMClassifier
        import weaviate
        from uuid import uuid1

        model_version = uuid1().urn
        model_name='sentiment_classifier'

        weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                          additional_headers=weaviate_conn['extra']['headers'])
        
        weaviate_status = weaviate_client.cluster.get_nodes_status()[0]['status'] 
        assert weaviate_status == 'HEALTHY' and weaviate_client.is_live()
        
        registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                database_name=snowpark_model_registry['database'], 
                                                schema_name=snowpark_model_registry['schema'])
        
        df = training_table.with_column('LABEL', F.col('LABEL').astype('int'))\
                      .with_column('REVIEW_TEXT', F.regexp_replace(F.col('REVIEW_TEXT'), F.lit('^\s*$')))\
                      .filter(F.col('REVIEW_TEXT') != '')\
                      .with_column('REVIEW_TEXT', F.regexp_replace(F.col('REVIEW_TEXT'), F.lit('\n'), F.lit(' ')))\
                      .to_pandas()

        #read from pre-existing embeddings in weaviate
        df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CommentTraining', 
                                                                          uuid=x.UUID, with_vector=True)['vector'], 
                                                                          axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(df['VECTOR'], df['LABEL'], test_size=.3, random_state=1883)
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
        
        return {'id': model_id, 'name': model_name, 'version':model_version}

    @task_group()
    def score_sentiment():

        @task.snowpark_virtualenv(requirements=['lightgbm==3.3.5', 'weaviate-client==3.15.4'], snowflake_conn_id=_SNOWFLAKE_CONN_ID)
        def call_sentiment(calls_table:SnowparkTable, weaviate_conn:dict, snowpark_model_registry:dict, model:dict):
            from snowflake.ml.registry import model_registry
            import weaviate
            import numpy as np

            registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                    database_name=snowpark_model_registry['database'], 
                                                    schema_name=snowpark_model_registry['schema'])
            
            metrics = registry.get_metrics(model_name=model['name'], model_version=model['version'])
            model = registry.load_model(model_name=model['name'], model_version=model['version'])

            df = calls_table.to_pandas()

            weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                              additional_headers=weaviate_conn['extra']['headers'])
            
            weaviate_status = weaviate_client.cluster.get_nodes_status()[0]['status'] 
            assert weaviate_status == 'HEALTHY' and weaviate_client.is_live()
            
            df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CustomerCall', 
                                                                              uuid=x.UUID, with_vector=True)['vector'], 
                                                                              axis=1)
            
            df['SENTIMENT'] = model.predict_proba(np.stack(df['VECTOR'].values))[:,1]

            return snowpark_session.create_dataframe(df)

        @task.snowpark_virtualenv(requirements=['lightgbm==3.3.5', 'weaviate-client==3.15.4'], snowflake_conn_id=_SNOWFLAKE_CONN_ID)
        def twitter_sentiment(comment_table:SnowparkTable, weaviate_conn:dict, snowpark_model_registry:dict, model:dict):
            from snowflake.ml.registry import model_registry
            import weaviate
            import numpy as np
            
            registry = model_registry.ModelRegistry(session=snowpark_session, 
                                                    database_name=snowpark_model_registry['database'], 
                                                    schema_name=snowpark_model_registry['schema'])
            
            metrics = registry.get_metrics(model_name=model['name'], model_version=model['version'])
            model = registry.load_model(model_name=model['name'], model_version=model['version'])
            
            df = comment_table.to_pandas()

            weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                              additional_headers=weaviate_conn['extra']['headers'])
            
            weaviate_status = weaviate_client.cluster.get_nodes_status()[0]['status'] 
            assert weaviate_status == 'HEALTHY' and weaviate_client.is_live()
            
            df['VECTOR'] = df.apply(lambda x: weaviate_client.data_object.get(class_name='CustomerComment', 
                                                                              uuid=x.UUID, with_vector=True)['vector'], 
                                                                              axis=1)
            
            df['SENTIMENT'] = model.predict_proba(np.stack(df['VECTOR'].values))[:,1]

            return snowpark_session.create_dataframe(df)
        
        _pred_calls_table = call_sentiment(calls_table=_calls_table,
                                           weaviate_conn=weaviate_conn, 
                                           snowpark_model_registry=_snowpark_model_registry, 
                                           model=_model)
        _pred_comment_table = twitter_sentiment(comment_table=_comment_table, 
                                                weaviate_conn=weaviate_conn, 
                                                snowpark_model_registry=_snowpark_model_registry, 
                                                model=_model)

        return _pred_calls_table, _pred_comment_table

    @task_group()
    def exit():

        @task.snowpark_python(snowflake_conn_id=_SNOWFLAKE_CONN_ID)
        def create_presentation_tables(pred_calls_table:SnowparkTable, pred_comment_table:SnowparkTable):
            
            attribution_df = snowpark_session.table('ATTRIBUTION_TOUCHES')
            rev_df = snowpark_session.table('MRR').drop(['ID'])
            customers_df = snowpark_session.table('CUSTOMERS')\
                                .with_column('CLV', F.round(F.col('CUSTOMER_LIFETIME_VALUE'), 2))

            sentiment_df =  pred_calls_table.group_by(F.col('CUSTOMER_ID'))\
                                            .agg(F.avg('SENTIMENT').alias('CALLS_SENTIMENT'))\
                                            .join(pred_comment_table.group_by(F.col('CUSTOMER_ID'))\
                                                    .agg(F.avg('SENTIMENT').alias('COMMENTS_SENTIMENT')), 
                                                on='CUSTOMER_ID',
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
                                   .join(rev_df.select(['CUSTOMER_ID', 
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

            return 'success'
        
        @task()
        def backup_weaviate(replace_existing=False):
            import weaviate 
            import minio

            weaviate_client = weaviate.Client(url=weaviate_conn['extra']['endpoint_url'], 
                                              additional_headers=weaviate_conn['extra']['headers'])
            assert weaviate_client.cluster.get_nodes_status()[0]['status'] == 'HEALTHY' and weaviate_client.is_live()

            if replace_existing:
                minio_client = minio.Minio(
                    endpoint=minio_conn['extra']['endpoint_url'],
                    access_key=minio_conn['extra']['aws_access_key'],
                    secret_key=minio_conn['extra']['aws_secret_access_key'],
                    secure=False
                )
                for obj in minio_client.list_objects(bucket_name=weaviate_backup_bucket, prefix='backup', recursive=True):
                    minio_client.remove_object(bucket_name=weaviate_backup_bucket, object_name=obj.object_name)
                

            response = weaviate_client.backup.create(
                    backup_id='backup',
                    backend="s3",
                    include_classes=list(weaviate_class_objects.keys()),
                    wait_for_completion=True,
                )
            return weaviate_conn
                            
        create_presentation_tables(pred_calls_table=_pred_calls_table, 
                                    pred_comment_table=_pred_comment_table)

        backup_weaviate(replace_existing=True)
        
    _weaviate_backup_bucket, _weaviate_conn, _snowpark_model_registry = enter()

    _structured_data = structured_data()
    
    _training_table, _comment_table, _calls_table = unstructured_data(weaviate_conn=_weaviate_conn,)
    
    _model = train_sentiment_classifier(training_table=_training_table,
                                           weaviate_conn=_weaviate_conn,
                                           snowpark_model_registry=_snowpark_model_registry)
    
    _pred_calls_table, _pred_comment_table = score_sentiment()
    
    _exit = exit()
    
    _structured_data >> _model >> _exit

customer_analytics()

def test():
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
    from snowflake.snowpark import Session as SnowparkSession
    from snowflake.snowpark import functions as F, types as T
    conn_params = SnowflakeHook('snowflake_default')._get_conn_params()
    snowpark_session = SnowparkSession.builder.configs(conn_params).create()
