# airflow-playground

We define an AirFlow DAG that will:

* Download a dataset on petrol consumption.
* Train a model to predict the target of the dataset and persist it locally.
* Uploads the model to AWS S3 so that other services can use it.
* Clean up all temporary artifacts generated during the process. 

The DAG definition can be found in [petrol_dag.py](dags/petrol_dag.py) file.
