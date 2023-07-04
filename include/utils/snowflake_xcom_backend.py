from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import argparse

def setup(hook:SnowflakeHook, database:str='DEMO', schema='XCOM', stage='XCOM_STAGE', table='XCOM_TABLE', user_role=''):
    
    if not user_role:
        user_role = hook._get_conn_params()['role']

#CREATE DATABASE IF NOT EXISTS {database};
        # CREATE SCHEMA IF NOT EXISTS {database}.{schema};
    hook.run(f'''
        
        CREATE OR REPLACE STAGE {database}.{schema}.{stage} DIRECTORY = (ENABLE = TRUE) ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
        CREATE OR REPLACE TABLE {database}.{schema}.{table} 
                                ( 
                                    dag_id varchar NOT NULL, 
                                    task_id varchar NOT NULL, 
                                    run_id varchar NOT NULL,
                                    multi_index integer NOT NULL,
                                    key varchar NOT NULL,
                                    value_type varchar NOT NULL,
                                    value varchar NOT NULL
                                ); 
        GRANT USAGE ON DATABASE {database} TO ROLE {user_role};
        GRANT USAGE ON SCHEMA {database}.{schema} TO ROLE {user_role};
        GRANT CREATE TABLE ON SCHEMA {database}.{schema} TO ROLE {user_role};
        GRANT SELECT, INSERT, UPDATE ON TABLE {database}.{schema}.{table} TO ROLE {user_role};
        GRANT READ, WRITE ON STAGE {database}.{schema}.{stage} TO ROLE {user_role};
    ''')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Setup  Airflow custom XCOM backend on Snowflake table/stage.',
        allow_abbrev=False)

    parser.add_argument('--database', dest='database', default='')
    parser.add_argument('--schema', dest='schema', default='')
    parser.add_argument('--table', dest='table', default='XCOM_TABLE')
    parser.add_argument('--stage', dest='stage', default='XCOM_STAGE')
    parser.add_argument('--admin_role', dest='admin_role', default='sysadmin')
    parser.add_argument('--user_role', dest='user_role', default='')
    parser.add_argument('--conn_id', dest='snowflake_conn_id', default='snowflake_default')

    args = parser.parse_args()

    assert args.database

    hook = SnowflakeHook(snowflake_conn_id=args.snowflake_conn_id) #, role=args.admin_role)

    setup(
        hook, 
        database=args.database,
        schema=args.schema,
        stage=args.stage,
        table=args.table,
        user_role=args.user_role,
    )