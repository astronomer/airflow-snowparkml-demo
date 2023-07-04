from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from snowflake_xcom_backend import setup as xcom_setup
import argparse

def setup(snowflake_conn_id:str, admin_role:str, database:str = 'demo', schema:str='demo'):
    
    hook = SnowflakeHook(snowflake_conn_id=snowflake_conn_id)

    user_role = hook._get_conn_params()['role']

    hook.role = admin_role

    hook.run(f"""CREATE DATABASE IF NOT EXISTS {database};
                 CREATE SCHEMA IF NOT EXISTS {database}.{schema};
                 GRANT USAGE ON DATABASE {database} TO ROLE {user_role};
                 GRANT USAGE ON SCHEMA {database}.{schema} TO ROLE {user_role};
                 GRANT CREATE SCHEMA ON DATABASE {database} TO ROLE {user_role};
                 GRANT CREATE TABLE ON SCHEMA {database}.{schema} TO ROLE {user_role};
                 GRANT CREATE VIEW ON SCHEMA {database}.{schema} TO ROLE {user_role};
                 GRANT CREATE STAGE ON SCHEMA {database}.{schema} TO ROLE {user_role};
            """)
    
    xcom_setup(hook, database='DEMO', schema='XCOM', user_role=user_role)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Setup Snowflake demo database and grant user access.',
        allow_abbrev=False)

    parser.add_argument('--conn_id', 
                        dest='snowflake_conn_id', 
                        help="Airflow connection name. Default: 'snowflake_default'",
                        default='snowflake_default')
    parser.add_argument('--admin_role', 
                        dest='admin_role', 
                        help="Name of admin role with create database privileges.  Default: 'sysadmin'", 
                        default='sysadmin')
    parser.add_argument('--database', 
                        dest='database',
                        help="Database name to create. Default: 'demo'",
                        default='demo')
    parser.add_argument('--schema', 
                        dest='schema',
                        help="Schema name to create for the demo data. Default: 'demo'",
                        default='demo')

    args = parser.parse_args()

    setup(
        snowflake_conn_id=args.snowflake_conn_id, 
        admin_role=args.admin_role,
        database=args.database,
        schema=args.schema,
    )