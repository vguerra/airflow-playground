import tempfile

# from airflow.hooks.S3_hook

from airflow.operators import DummyOperator, PythonOperator
# from airflow.operators.bash_operator import BashOperator
from airflow import DAG


DATASET_URL="https://iasd-data-in-the-cloud.s3.eu-west-3.amazonaws.com/petrol_consumption.csv"

args = {
    'owner': 'vguerra'
}

with DAG('Petrol_dag', default_args=args, schedule_interval='@once') as dag:
    start_task = DummyOperator(task_id='dummy_start')

    debug_task = PythonOperator(
        task_id='debug',
        python_callable= lambda _: print("Debuging task")
    )

    start_task >> debug_task