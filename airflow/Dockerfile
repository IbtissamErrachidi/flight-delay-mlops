FROM apache/airflow:3.1.5

# Copier les requirements
COPY requirements-docker.txt /requirements-docker.txt

# Installer les dépendances en tant que user airflow 
USER airflow
RUN pip install --no-cache-dir -r /requirements-docker.txt
