[Snowpark ML](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index) (in public preview) is a python framework for Machine Learning workloads with [Snowpark](https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html).  Currently Snowpark ML provides a model registry (storing ML tracking data and models in Snowflake tables and stages), feature engineering primitives similar to scikit-learn (ie. LabelEncoder, OneHotEncoder, etc.) and support for training and deploying [certain model types](https://docs.snowflake.com/en/developer-guide/snowpark-ml/snowpark-ml-modeling#snowpark-ml-modeling-classes) as well as deployments as user-defined functions (UDFs).

This guide demonstrates how to use Apache Airflow to orchestrate a machine learning pipeline leveraging Snowpark ML for feature engineering and model tracking. While Snowpark ML has its own support for models similar to scikit-learn this code demonstrates a "bring-your-own" model approach showing the use of open-source scikit-learn along with Snowpark ML model registry and model serving in an Airflow task rather than Snowpark UDF.

This demo also shows the use of the Snowflake XCOM backend which supports security and governance by serializing all task in/output to Snowflake tables and stages while storing in the Airflow XCOM table a URI pointer to the data.

This workflow includes:
- sourcing structured, unstructured and semistructured data from different systems
- extract, transform and load with [Snowpark Python provider for Airflow](https://github.com/astronomer/astro-provider-snowflake)
- ingest with Astronomer's [python SDK for Airflow](https://github.com/astronomer/astro-sdk)
- audio file transcription with [OpenAI Whisper](https://github.com/openai/whisper)
- natural language embeddings with [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) and the [Weaviate provider for Airflow](https://github.com/astronomer/airflow-provider-weaviate)
- vector search with [Weaviate](https://weaviate.io/)
- sentiment classification with [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)  
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

3. Open the `.env` file in an editor and update the following variables with you account information
This demo assumes the use of a new Snowflake trial account with admin privileges.  A database named 'DEMO' and schema named 'DEMO' will be created in the DAG.  Running this demo without admin privileges or with existing database/schema will require further updates to the `.env` file.
  
- AIRFLOW_CONN_SNOWFLAKE_DEFAULT  
  -- login  
  -- password  
  -- account **  
- OPENAI_APIKEY  
  
** The Snowflake `account` field of the connection should use the new `ORG_NAME-ACCOUNT_NAME` format as per [Snowflake Account Identifier policies](https://docs.snowflake.com/en/user-guide/admin-account-identifier).  The ORG and ACCOUNT names can be found in the confirmation email or in the Snowflake login link (ie. `https://xxxxxxx-yyy11111.snowflakecomputing.com/console/login`)
Do not specify a `region` when using this format for accounts.
  
NOTE: Database and Schema names should be CAPITALIZED due to a bug in Snowpark ML.
  
4.  Start Apache Airflow:
    ```sh
    astro dev restart
    ```  
  A browser window should open to [http://localhost:8080](http://localhost:8080) 
  Login with 
    username: `admin` 
    password: `admin`  

7. Run the Customer Analytics Demo DAG
```bash
astro dev run dags unpause customer_analytics
astro dev run dags trigger customer_analytics
```

Follow the status of the DAG run in the [Airflow UI](http://localhost:8080/dags/customer_analytics/grid) (username: `admin`, password: `admin`)


8. After the DAG completes open the [streamlit application](http://localhost:8501)

Other service UIs are available at the the following:
- Airflow: [http://localhost:8080](http://localhost:8080) Username:Password is admin:admin
- Weaviate: [https://console.weaviate.io/](https://link.weaviate.io/3UD9H8z) Enter localhost:8081 in the "Self-hosted Weaviate" field.
