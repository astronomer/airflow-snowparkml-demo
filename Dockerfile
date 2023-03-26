# syntax=quay.io/astronomer/airflow-extensions:v1.0.0-alpha.3
FROM quay.io/astronomer/astro-runtime:7.4.1-base
COPY include/0.2.1/snowflake_ml_python-0.2.1-py3-none-any.whl /tmp
PYENV 3.8 snowpark_env requirements-snowpark.txt


# FROM quay.io/astronomer/astro-runtime:7.4.1

# #Python for Snowpark via ExternalPythonOperator
# COPY requirements-snowpark.txt /tmp
# COPY include/0.2.1/snowflake_ml_python-0.2.1-py3-none-any.whl /tmp
# RUN arch=$(arch | sed s/aarch64/aarch64/ | sed s/x86_64/x86_64/) && \
#     wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${arch}.sh -O ~/miniconda.sh && \
#     /bin/bash /home/astro/miniconda.sh -b && \
#     /home/astro/miniconda3/bin/conda init bash && \
#     mkdir -p /home/astro/.venv/snowpark_env && \
#     /home/astro/miniconda3/bin/conda create -qyp /home/astro/.venv/snowpark_env python=3.8 && \
#     /home/astro/.venv/snowpark_env/bin/pip install -r /tmp/requirements-snowpark.txt

# ENV PATH=~/miniconda3/bin:$PATH
#____________________________________

# FROM quay.io/astronomer/astro-runtime:7.4.1

# COPY requirements-snowpark.txt /tmp
# COPY include/0.2.1/snowflake_ml_python-0.2.1-py3-none-any.whl /tmp
# USER root
# COPY --link --from=python:3.8-slim /usr/local/bin/*3.8* /usr/local/bin/
# COPY --link --from=python:3.8-slim /usr/local/include/python3.8* /usr/local/include/python3.8
# COPY --link --from=python:3.8-slim /usr/local/lib/pkgconfig/*3.8* /usr/local/lib/pkgconfig/
# COPY --link --from=python:3.8-slim /usr/local/lib/*3.8*.so* /usr/local/lib/
# COPY --link --from=python:3.8-slim /usr/local/lib/python3.8 /usr/local/lib/python3.8
# RUN /sbin/ldconfig /usr/local/lib && \
#     ln -s /usr/local/include/python3.8 /usr/local/include/python3.8

# USER astro
# RUN mkdir -p /home/astro/.venv/snowpark_env && \
#     /usr/local/bin/python3.8 -m venv --system-site-packages /home/astro/.venv/snowpark_env && \
#     /home/astro/.venv/snowpark_env/bin/pip install -r /tmp/requirements-snowpark.txt

# # build with buildx
# # docker buildx build --load -t airflow-snowpark:latest .
# # astro dev start -i airflow-snowpark:latest
#____________________________________
# FROM quay.io/astronomer/astro-runtime:7.4.1

# ENV PYENV_ROOT="/home/astro/.venv" 
# ENV PATH=${PYENV_ROOT}/bin:${PATH}

# COPY requirements-snowpark.txt /tmp

# RUN curl https://pyenv.run | bash  && \
#     eval "$(pyenv init -)" && \
#     pyenv install 3.8 && \
#     pyenv virtualenv 3.8 snowpark_env && \
#     pyenv activate snowpark_env && \
#     pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r /tmp/requirements-snowpark.txt
#____________________________________
