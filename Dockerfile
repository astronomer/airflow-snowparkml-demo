# syntax=quay.io/astronomer/airflow-extensions:latest

FROM quay.io/astronomer/astro-runtime:9.1.0-python-3.10

COPY /include/astro_provider_snowflake-0.0.1-py3-none-any.whl /tmp
COPY /include/airflow_provider_weaviate-0.0.1-py3-none-any.whl /tmp

PYENV 3.8 snowpark requirements-snowpark.txt

#seed packages for testing snowpark_ext_python operator
COPY requirements-snowpark.txt /tmp
RUN python3.8 -m pip install -r /tmp/requirements-snowpark.txt
# RUN /usr/local/bin/python3.8 -m pip install '/tmp/astro_provider_snowflake-0.0.0-py3-none-any.whl[snowpark]'