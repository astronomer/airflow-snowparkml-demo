from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from snowflake.snowpark import Session as SnowparkSession
from snowflake.snowpark import functions as F, types as T
import argparse

def cleanup(snowflake_conn_id:str, database:str, schema:str):
    
    hook = SnowflakeHook(snowflake_conn_id=snowflake_conn_id)
    conn_params = hook._get_conn_params()
    snowpark_session = SnowparkSession.builder.configs(conn_params).create()

    tables = ['_SYSTEM_REGISTRY_METADATA', '_SYSTEM_REGISRTRY_DEPLOYMENTS', '_SYSTEM_REGISTRY_MODELS',
              'PRES_CHURN', 'PRED_TWITTER_COMMENTS', 'PRES_CLV', 'PRES_AD_SPEND', 'PRES_SENTIMENT', 'PRED_CUSTOMER_CALLS',
              'CUSTOMERS', 'ORDERS', 'STG_AD_SPEND', 'STG_COMMENT_TRAINING', 'STG_CUSTOMERS', 'STG_CUSTOMER_CONVERSIONS', 'STG_ORDERS', 'STG_PAYMENTS', 'STG_SESSIONS', 'STG_SUBSCRIPTION_PERIODS', 'STG_TWITTER_COMMENTS', 
              'TAXI_PRED', 'TAXI_RAW']
    views = ['_SYSTEM_REGISRTRY_DEPLOYMENTS_VIEW', 
             '_SYSTEM_REGISTRY_METADATA_LAST_DESCRIPTION', 
             '_SYSTEM_REGISTRY_METADATA_LAST_METRICS',
             '_SYSTEM_REGISTRY_METADATA_LAST_REGISTRATION',
             '_SYSTEM_REGISTRY_METADATA_LAST_TAGS',
             '_SYSTEM_REGISTRY_MODELS_VIEW',
             'ATTRIBUTION_TOUCHES', 'CUSTOMER_CHURN_MONTH', 'CUSTOMER_REVENUE_BY_MONTH', 'MRR', 'UTIL_MONTHS']
    stages = ['xcom_stage', 'call_stage']

    xcom_tables = snowpark_session.table('information_schema.tables')\
                                  .select('table_name')\
                                  .where(F.col('table_name').like('XCOM_%'))\
                                  .to_pandas()['TABLE_NAME'].to_list()

    snowml_stages = snowpark_session.table('information_schema.stages')\
                                  .select('stage_name')\
                                  .where(F.col('stage_name').like('SNOWML_MODEL_%'))\
                                  .to_pandas()['STAGE_NAME'].to_list()

    print(f"WARNING: DANGER ZONE!  This will drop the \ntables {tables}, \nviews {views} \nand stages {stages} in schema {database}.{schema}")
    prompt = input("Are you sure you want to do this?: N/y: ")

    if prompt.upper() == 'Y':
        prompt = input("Are you REALLY sure?: N/y: ")

        if prompt.upper() == 'Y':

            for table in tables+xcom_tables:
                # hook.run(f"DROP TABLE IF EXISTS {table};")
                try:
                    snowpark_session.table(table).drop_table()
                except:
                    pass

            for view in views:
                hook.run(f"DROP VIEW IF EXISTS {view};")

            for stage in stages+snowml_stages:
                hook.run(f"DROP STAGE IF EXISTS {stage};")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cleanup Snowflake tables and views craeted in the demo.',
        allow_abbrev=False)

    parser.add_argument('--conn_id', 
                        dest='snowflake_conn_id', 
                        help="Airflow connection name. Default: 'snowflake_default'",
                        default='snowflake_default')
    parser.add_argument('--database', 
                        dest='database',
                        help="Database name to create. Default: 'demo'")
    parser.add_argument('--schema', 
                        dest='schema',
                        help="Schema name to create for the demo data. Default: 'demo'")

    args = parser.parse_args()

    cleanup(snowflake_conn_id=args.snowflake_conn_id, 
            database=args.database,
            schema=args.schema)