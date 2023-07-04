[Snowpark ML](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index) (in public preview) is a python framework for Machine Learning workloads with [Snowpark](https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html).  Currently Snowpark ML provides a model registry (storing ML tracking data and models in Snowflake tables and stages), feature engineering primitives similar to scikit-learn (ie. LabelEncoder, OneHotEncoder, etc.) and support for training and deploying [certain model types](https://docs.snowflake.com/en/developer-guide/snowpark-ml/snowpark-ml-modeling#snowpark-ml-modeling-classes) as well as deployments as user-defined functions (UDFs).

This guide demonstrates how to use Apache Airflow to orchestrate a machine learning pipeline leveraging Snowpark ML for feature engineering and model tracking. While Snowpark ML has its own support for models similar to scikit-learn this code demonstrates a "bring-your-own" model approach showing the use of open-source scikit-learn along with Snowpark ML model registry and model serving in an Airflow task rather than Snowpark UDF.

This demo also shows the use of the Snowflake XCOM backend which supports security and governance by serializing all task in/output to Snowflake tables and stages while storing in the Airflow XCOM table a URI pointer to the data.

## Prerequisites  
  
- Astro [CLI](https://docs.astronomer.io/astro/cli/get-started)
- Docker Desktop
- Git
- Snowflake account (or a [trial account](https://signup.snowflake.com/))
  
## Setup  
  
1. Install Astronomer's [Astro CLI](https://github.com/astronomer/astro-cli).  The Astro CLI is an Apache 2.0 licensed, open-source tool for building Airflow instances and is the fastest and easiest way to be up and running with Airflow in minutes. Open a terminal window and run:

For MacOS  
```bash
brew install astro
```
  
For Linux
```bash
curl -sSL install.astronomer.io | sudo bash -s
```

2. Clone this repository:
```bash
git clone https://github.com/astronomer/airflow-snowparkml-demo
cd airflow-snowparkml-demo
```

3.  Setup shell environment variables for the demo.  Update values in brackets `<>` and run the commands in the terminal where you will be running the demo.

- Export database and schema to be used for this demo.  If these objects do not yet exist they can be created in a later step.
```bash
export DEMO_DATABASE='<DB_NAME>'
export DEMO_SCHEMA='<SCHEMA_NAME>'
```

-Export Snowflake account credentials as environment variables.
```bash
export AIRFLOW_CONN_SNOWFLAKE_DEFAULT='{"conn_type": "snowflake", "login": "<USER_NAME>", "password": "<PASSWORD>", "schema": "${DEMO_SCHEMA}", "extra": {"account": "<ORG_NAME>-<ACCOUNT_NAME>", "warehouse": "<WAREHOUSE_NAME>", "database": "${DEMO_DATABASE}", "region": "<REGION_NAME>", "role": "<USER_ROLE>", "authenticator": "snowflake", "session_parameters": null, "application": "AIRFLOW"}}'
```

4.  Start Apache Airflow:
    ```sh
    astro dev start
    ```  
5. Setup Snowflake objects for the demo.  
  
If using an existing database and schema in Snowflake skip to step 6.  Otherwise run the following script to create a database, schema and tables needed for the demo.
  
Note, this must be run as a user with admin priveleges.  Alternatively use an existing database and schema or look at the setup scripts and have a Snowflake administrator create these objects and grant permissions.
  
```bash
astro dev bash -s
```

```bash
python include/utils/setup_snowflake.py \
  --conn_id 'snowflake_default' \
  --admin_role 'sysadmin' \
  --database $DEMO_DATABASE \
  --schema $DEMO_SCHEMA
exit
```  
  
6. Setup the table and stage to be used as the Snowflake XCOM backend.
```bash
astro dev bash -s
```
  
```bash
python include/utils/snowflake_xcom_backend.py \
  --conn_id 'snowflake_default' \
  --database $DEMO_DATABASE \
  --schema $DEMO_SCHEMA
exit
```

7. Run the Snowpark ML Demo DAG
```bash
astro dev run dags unpause snowpark_ml_demo
astro dev run dags trigger snowpark_ml_demo
```

8. Connect to the Local [Airflow UI](http://localhost:8080/dags/snowpark_ml_demo/grid) and login with **Admin/Admin**  

For a more advanced example see the [Customer Analytics Demo](./README_CA.md)