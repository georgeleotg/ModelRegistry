# Databricks notebook source
# ruff: noqa
%pip install .. 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC We’ll define the train and the test sets. The train set now contains 12 months of data, ending on 2018-11-30, while the test set will contain data exclusively from December 2018. Additionally, we’ll use the latest version of the hotel_booking table:
# MAGIC %md
# MAGIC

# COMMAND ----------



import json
from datetime import datetime

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.data.data_loader import DataLoader
from hotel_booking.models.lightgbm_model import LightGBMModel
from hotel_booking.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()
cfg = ProjectConfig.from_yaml(config_path="../project_config.yml")

spark = SparkSession.builder.getOrCreate()
data_loader = DataLoader(spark=spark, config=cfg)
X_train, y_train, X_test, y_test = data_loader.split()

# COMMAND ----------

model = LightGBMModel(config=cfg)
model.train(X_train=X_train, y_train=y_train)

# COMMAND ----------

# set the MLflow experiment, start an MLflow run, and log parameters:
mlflow.set_experiment("/Shared/hotel-booking-training")
tags = Tags(**{"git_sha": "1234567890abcd", "branch": "main"})
run = mlflow.start_run(
    run_name=f"lightgbm-training-{datetime.now().strftime('%Y-%m-%d')}",
    description="LightGBM model training",
    tags=tags.to_dict(),
)
run_id = run.info.run_id
mlflow.log_params(cfg.parameters.model_dump())

# COMMAND ----------

# MAGIC %md
# MAGIC Now we want to proceed with logging the model, and to be able to register it later in Unity Catalog, we need to provide a model signature that defines the structure of the model input and the model output. To explore more detailed information about model signatures in MLflow as well as additional methods to define them, refer to the [documentation](https://mlflow.org/docs/latest/ml/model/signatures/). However, the easiest way to do it is using the infer_signature function from mlflow.models module:

# COMMAND ----------

signature = infer_signature(
    model_input=X_test, model_output=model.pipeline.predict(X_test)
)

# COMMAND ----------

# MAGIC %md
# MAGIC To ensure full traceability and reproducibility, we must log the model input. We pass a pyspark dataframe and a specific SQL query that also contains the delta table version:
# MAGIC
# MAGIC

# COMMAND ----------

training = mlflow.data.from_spark(
    df=data_loader.train_set_spark, sql=data_loader.train_query
)
testing = mlflow.data.from_spark(
    df=data_loader.test_set_spark, sql=data_loader.test_query
)
mlflow.log_input(training, context="training")
mlflow.log_input(testing, context="testing")

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the model can finally be logged. Here we also provide an input example, which can be used as an example for model serving inference later. Notice that we have not computed and logged any model metrics yet. This time, we’ll use MLflow Evaluation to compute the metrics. The mlflow.models.evaluate() function takes the model uri, evaluation data, target, model type (“regressor” and “classifier” are supported values), and evaluators as inputs. The mlflow.models.evaluate() function uses an active run to log metrics (or starts a new run if no active run is available):

# COMMAND ----------

model_info = mlflow.sklearn.log_model(
    sk_model=model.pipeline,
    name="lightgbm-pipeline",
    signature=signature,
    input_example=X_test[0:1],
)
eval_data = X_test.copy()
eval_data[cfg.target] = y_test

# This will log the evaluation metrics
result = mlflow.models.evaluate(
    model_info.model_uri,
    eval_data,
    targets=cfg.target,
    model_type="regressor",
    evaluators=["default"],
)
mlflow.end_run()

# COMMAND ----------

result.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC We can now view our logged model that belongs to the hotel-booking-training experiment in the Models tab (Experiments)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieving the model

# COMMAND ----------

# MAGIC %md
# MAGIC The LoggedModel object was introduced starting with MLflow 3. It allows users to interact with the logged model outside of the experiment run.

# COMMAND ----------

logged_model = mlflow.get_logged_model(model_info.model_id)
model = mlflow.sklearn.load_model(f"models:/{model_info.model_id}")

# COMMAND ----------

logged_model_dict = logged_model.to_dictionary()
logged_model_dict["metrics"] = [x.__dict__ for x in logged_model_dict["metrics"]]
with open("../demo_artifacts/logged_model.json", "w") as json_file:
    json.dump(logged_model_dict, json_file, indent=4)

# COMMAND ----------

display(logged_model_dict)

# COMMAND ----------

logged_model.params

# COMMAND ----------

logged_model.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC While logging the model, we’ve also logged the input train and test datasets. We can load the logged datasets at any point later as long as the table version used for logging is within the retention period:

# COMMAND ----------

run = mlflow.get_run(run_id)

# COMMAND ----------

inputs = run.inputs.dataset_inputs
training_input = next(
    (x for x in inputs if x.tags and x.tags[0].value == "training"),
    None,
)
training_source = mlflow.data.get_source(training_input)
training_source.load()

# COMMAND ----------

testing_input = next(
    (x for x in inputs if x.tags and x.tags[0].value == "testing"),
    None,
)
testing_source = mlflow.data.get_source(testing_input)
testing_source.load()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registering the model

# COMMAND ----------

# MAGIC %md
# MAGIC To use the logged model for model serving on Databricks, or make it accessible by other teams, the model must be registered in Unity Catalog.
# MAGIC
# MAGIC It’s possible to do it by providing the registered_model argument to the log_model function. In that case, the ModelInfo object will also have a registered_model_version attribute. However, we may not want to register the model right away without evaluation against the latest registered version of the model. Since no model is registered yet, let’s now register a model using the mlflow.register_model() function.

# COMMAND ----------

model_name = (
    f"{cfg.catalog}.{cfg.schema}.hotel_booking_basic"
)
registered_model = mlflow.register_model(
    model_uri=logged_model.model_uri,
    name=model_name,
    tags=tags.to_dict(),
)

# COMMAND ----------

tags

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# client = MlflowClient()

# job_id = "1234567890abcdef"  # Example job ID; will fail if the job does not exist
# client.create_registered_model(model_name, deployment_job_id=job_id)

# COMMAND ----------

# MAGIC %md
# MAGIC After the model is registered, an alias can be assigned referring to a particular version of a registered model. An alias can be used in a model URI simplifying the way we can access the registered model.
# MAGIC
# MAGIC We find it convenient to assign the “latest-model” tag to the most recently registered model. We can’t use the alias “latest” directly as it’s reserved, nor can we use “latest” in the model URI:
# MAGIC
# MAGIC

# COMMAND ----------

# latest alias is reserved, so we cannot use it
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)

model = mlflow.sklearn.load_model(f"models:/{model_name}@latest-model")

# COMMAND ----------

# MAGIC %md
# MAGIC It’s a good idea to use aliases, as model registry search functionality is not widely supported on Unity Catalog. For instance, it’s only possible to search for model versions by name:

# COMMAND ----------

# only searching by name is supported
v = mlflow.search_model_versions(filter_string=f"name='{model_name}'")
print(v[0].__dict__)

# COMMAND ----------


# not supported
# v = mlflow.search_model_versions(filter_string="tags.git_sha='1234567890abcd'")

# COMMAND ----------

#we can get model version using alias
model_version = client.get_model_version_by_alias(alias="latest-model", name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Wrapping the model using pyfunc
# MAGIC Model signature in MLflow defines the way different interfaces interact with the model. For instance, it defines the payload of the endpoint if the model gets served using Databricks model serving (we’ll go into detail in Chapter 4).
# MAGIC
# MAGIC We previously registered a sklearn pipeline. If we deploy it behind an endpoint, we will get an output in the format: {“Predictions”: [120.1234567]}. We may want to have a custom output, and a pyfunc model flavour comes in useful.
# MAGIC
# MAGIC There are other examples when pyfunc becomes indispensable:
# MAGIC
# MAGIC when we want to use an ensemble of models trained separately, and serve it behind an endpoint;
# MAGIC
# MAGIC model serving requires specific context (for instance, a file that defines what predictions must be adjusted);
# MAGIC
# MAGIC we need to access other systems (for example, a vector database) to return predictions.
# MAGIC
# MAGIC Essentially, we are using pyfunc as a wrapper. Keeping the definition of the payload separate from the model itself is quite useful: we can easily adjust the pyfunc wrapper definition without touching the registered model itself. In a certain sense, it’s very similar to the functionality of a FastAPI wrapper.
# MAGIC
# MAGIC Under the models module of the hotel_booking package we defined the HotelBookingModelWrapper class. It loads a sklearn pipeline as part of the load_context method and modifies the predictions (here, we are adding a 5% commission on top of the price, and rounding the total price down to two decimals)
# MAGIC The HotelBookingModelWrapper class has the log_register_model method. It requires the ModelInfo object from the model we are going to wrap, the model name we’ll use to register the pyfunc model, experiment name, tags, and code_paths.

# COMMAND ----------

git_sha = "3e8ae31"   
branch  = "main"                

from hotel_booking.config import Tags
tags = Tags(**{"git_sha": git_sha, "branch": branch})

# COMMAND ----------

model_name = (
    f"{cfg.catalog}.{cfg.schema}.hotel_booking_basic2"
)
registered_model = mlflow.register_model(
    model_uri=logged_model.model_uri,
    name=model_name,
    tags=tags.to_dict(),
)

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)
