from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="flight_delay_dvc_pipeline",
    start_date=datetime(2025, 12, 22),
    schedule=None,  # run manuel
    catchup=False,
    tags=["mlops", "dvc", "flight-delay"],
) as dag:


    run_dvc_pipeline = BashOperator(
        task_id="run_dvc_repro",
        bash_command="cd /opt/airflow/project && dvc repro",
    )

    run_dvc_pipeline
