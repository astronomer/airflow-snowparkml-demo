# syntax=quay.io/astronomer/airflow-extensions:latest

FROM quay.io/astronomer/astro-runtime:8.6.0-base

COPY /include/astro_provider_snowflake-0.0.0-py3-none-any.whl /tmp
COPY /include/airflow_provider_weaviate-1.0.0-py3-none-any.whl /tmp

PYENV 3.8 snowpark requirements-snowpark.txt

#seed packages for testing snowpark_ext_python operator
RUN source /home/astro/.venv/snowpark/bin/activate && \
    /home/astro/.venv/snowpark/bin/pip install '/tmp/astro_provider_snowflake-0.0.0-py3-none-any.whl[snowpark]'