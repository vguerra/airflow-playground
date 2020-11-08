import tempfile
import logging
import os
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from airflow.hooks.S3_hook import S3Hook
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import DAG


LOGGER = logging.getLogger("airflow.petroleum")
DATASET_URL="https://iasd-data-in-the-cloud.s3.eu-west-3.amazonaws.com/petrol_consumption.csv"
DATASET_BUCKET_NAME='iasd-data-in-the-cloud'
DATASET_FILENAME='petrol_consumption.csv'
MODEL_BUCKET_NAME='vm-guerra-moran'

###
#  Utility functions used by AirFlow operators.
###
def _download_dataset(bucket_name, filename, output_path):
    """
    Downloads a file from the specified bucket.

    Arguments:
    bucket_name: str. AWS S3 bucket name to get the file from.
    filename: str. Name of the file to download.
    output_path: str. Path where the file should be put.
    """
    hook = S3Hook(aws_conn_id='aws_default')
    LOGGER.info(f"Downloading {filename} from {bucket_name}")
    # LOGGER.info(f"exists key? {hook.check_for_key(filename, bucket_name=bucket_name)}")
    key_obj = hook.get_key(filename, bucket_name=bucket_name)
    key_obj.download_file(output_path)
    LOGGER.info(f"Done, dataset is at {output_path}")


def _upload_model(bucket_name, file_path, key_name):
    """
    Uploads a file to a given S3 bucket under a specific key.

    Arguments:
        bucket_name: str, Name of the S3 bucket.
        file_path: str, Local path to the file to be uploaded.
        key_name: str, key under which the file will be uploaded.
    """
    LOGGER.info(f"Uploading {file_path} to S3 bucket {bucket_name} at path {key_name}")
    hook = S3Hook(aws_conn_id='aws_default')
    hook.load_file(
        file_path,
        key_name,
        bucket_name=bucket_name,
        replace=True
    )
    LOGGER.info("Done with upload.")

def _clean_files(*files):
    """
    Cleans up a list of files.

    Arguments:
        files: List of strings. Each represeting the path to a file to be deleted.
    """
    for file_to_remove in files:
        LOGGER.info(f"Removing: {file_to_remove}")
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)


# Most of the code adapter from: https://github.com/faouzelfassi/pipelining/blob/master/model.py
def _train_model(dataset_path, model_path):
    """
    Trains a model given a dataset and persist it to the file system.

    Arguments:
        dataset_path: str, path to the dataset.
        model_path: str, path where the resulting model will be dumped.

    """
    LOGGER.info(f"dataset for training: {dataset_path}")
    LOGGER.info(f"output model to: {model_path}")

    dataset = pd.read_csv(dataset_path)
    attributes = dataset.iloc[:, 0:4].values
    labels = dataset.iloc[:, 4].values

    # Divide the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        attributes, labels, test_size=0.2, random_state=0
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train the model
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Evaluate the performance of the model
    LOGGER.info(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    LOGGER.info(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    LOGGER.info(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    # saving model to the desired path
    pickle.dump(regressor, open(model_path, 'wb'))


###
# General configuration
##
args = {
    'owner': 'vm.guerramoran',
    'start_date': days_ago(1)
}

###
# DAG definition
###
with DAG('petroleum',
         description="""
         DAG that downloads a dataset and trains a regression model to predict pretrol consumption.
         The dataset is downloaded from a S3 bucket. The resulting model is as well uploaded to S3.
         """,
         default_args=args,
         max_active_runs=1,
         concurrency=10,
         schedule_interval="@daily"
) as petroleum:

    output_model_filename_prefix = "{{ ds }}"
    dataset_path = str(Path(tempfile.gettempdir(), DATASET_FILENAME))
    # dataset_path = str(Path("/tmp/", DATASET_FILENAME))
    model_path = str(Path(tempfile.gettempdir(), f"{output_model_filename_prefix}_model.sav"))
    upload_key = str(Path(output_model_filename_prefix, "model.sav"))

    start_task = DummyOperator(
        task_id='start_placeholder',
        dag=petroleum
    )

    download_dataset_task = PythonOperator(
        task_id='download_dataset',
        python_callable=_download_dataset,
        op_args = [DATASET_BUCKET_NAME, DATASET_FILENAME, dataset_path],
        dag=petroleum
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable = _train_model,
        op_args = [dataset_path, model_path],
        dag=petroleum
    )

    upload_model_task = PythonOperator(
        task_id='upload_model',
        python_callable = _upload_model,
        op_args = [MODEL_BUCKET_NAME, model_path, upload_key],
        dag=petroleum
    )

    # TODO: define upload task here
    cleanup_task = PythonOperator(
        task_id='cleanup',
        python_callable = _clean_files,
        op_args = [dataset_path, model_path],
        dag=petroleum
    )

    start_task >>  download_dataset_task >> train_model_task >> upload_model_task >> cleanup_task
