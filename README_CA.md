[Snowpark ML](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index) (in public preview) is a python framework for Machine Learning workloads with [Snowpark](https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html).  Currently Snowpark ML provides a model registry (storing ML tracking data and models in Snowflake tables and stages), feature engineering primitives similar to scikit-learn (ie. LabelEncoder, OneHotEncoder, etc.) and support for training and deploying [certain model types](https://docs.snowflake.com/en/developer-guide/snowpark-ml/snowpark-ml-modeling#snowpark-ml-modeling-classes) as well as deployments as user-defined functions (UDFs).

This guide demonstrates how to use Apache Airflow to orchestrate a machine learning pipeline leveraging Snowpark ML for feature engineering and model tracking. While Snowpark ML has its own support for models similar to scikit-learn this code demonstrates a "bring-your-own" model approach showing the use of open-source scikit-learn along with Snowpark ML model registry and model serving in an Airflow task rather than Snowpark UDF.

This demo also shows the use of the Snowflake XCOM backend which supports security and governance by serializing all task in/output to Snowflake tables and stages while storing in the Airflow XCOM table a URI pointer to the data.

This workflow includes:
- sourcing structured, unstructured and semistructured data from different systems
- ingest with Astronomer's [python SDK for Airflow](https://github.com/astronomer/astro-sdk)
- data quality checks with [Great Expectations](https://greatexpectations.io/)
- transformations and tests in [DBT](https://www.getdbt.com/), 
- audio file transcription with [OpenAI Whisper](https://github.com/openai/whisper)
- natural language embeddings with [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- vector search and named-entity recognition with [Weaviate](https://weaviate.io/)
- sentiment classification with [Keras](https://keras.io/)  
- ML model management with [Snowflake ML](https://docs.snowflake.com/LIMITEDACCESS/snowflake-ml-modeling)
  
All of the above are presented in a [Streamlit](http://www.streamlit.io) applicaiton.  
  
## Prerequisites  
  
- Docker Desktop or similar Docker services running locally with the docker CLI installed.  
- Astronomer account or [Trial Account](https://www.astronomer.io/try-astro/) (optional)
- Snowflake account or [Trial Account](https://signup.snowflake.com/)
- OpenAI account or [Trial Account](https://platform.openai.com/signup)
    
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
export OPENAI_APIKEY='<OPENAI_APIKEY>'
```

-Export Snowflake account credentials as environment variables.
```bash
export AIRFLOW_CONN_SNOWFLAKE_DEFAULT='{"conn_type": "snowflake", "login": "<USER_NAME>", "password": "<PASSWORD>", "schema": "${DEMO_SCHEMA}", "extra": {"account": "<ORG_NAME>-<ACCOUNT_NAME>", "warehouse": "<WAREHOUSE_NAME>", "database": "${DEMO_DATABASE}", "region": "<REGION_NAME>", "role": "<USER_ROLE>", "authenticator": "snowflake", "session_parameters": null, "application": "AIRFLOW"}}'
```

4.  The Astro CLI uses Docker Compose to create local development resources in addition to the Airflow services.  To show this, uncomment the `minio`, `streamlit` and `weaviate` sections of the `docker-compose.override.yml` file to enable these services:

- [minio](https://min.io/): Object storage which is used for ingest staging as well as stateful backups for other services.  As an alternative to S3 or GCS this service is controlled by the user and provides a local option for simpler development.
- [weaviate](https://weaviate.io/): A vector database 
- [streamlit](http://www.streamlit.io): A web application framework for building data-centric apps.



4.  Start Apache Airflow:
    ```sh
    astro dev restart
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
astro dev run dags unpause customer_analytics
astro dev run dags trigger customer_analytics
```

Follow the status of the DAG run in the [Airflow UI](http://localhost:8080/dags/customer_analytics/grid) (username: `admin`, password: `admin`)


8. After the DAG completes open the [streamlit application](http://localhost:8501)

Other service UIs are available at the the following:
- Airflow: [http://localhost:8080](http://localhost:8080) Username:Password is admin:admin
- Minio: [http://localhost:9000](http://localhost:9000) Username:Password is minioadmin:minioadmin
- Weaviate: [https://console.weaviate.io/](https://link.weaviate.io/3UD9H8z) Enter localhost:8081 in the "Self-hosted Weaviate" field.
