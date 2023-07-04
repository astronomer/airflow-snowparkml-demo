# syntax=quay.io/astronomer/airflow-extensions:latest

FROM quay.io/astronomer/astro-runtime:8.6.0-base

COPY /include/astro_provider_snowflake-0.0.0-py3-none-any.whl /tmp

PYENV 3.9 dbt include/dbt/requirements-dbt.txt