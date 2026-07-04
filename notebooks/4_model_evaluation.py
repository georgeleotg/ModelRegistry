# Databricks notebook source
# ruff: noqa
%pip install .. 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC We can’t evaluate the impact of updated data and compare the new model to the old one without actually introducing new data. Let’s simulate the arrival of new data by generating a synthetic dataset containing a full month of hotel bookings.
# MAGIC
# MAGIC DataProcessor class definition  includes the generate_synthetic_df method. Then we can provide the number of observations to generate. It takes an optional max_date argument referring to the timestamp, after which the data must be generated. If not provided, max_date is taken from the hotel_booking table. We use the svd package and GaussianCopulaSynthesizer to generate the data with similar distribution as the provided dataset. 

# COMMAND ----------



import pandas as pd
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig, Tags
from hotel_booking.data import DataLoader, DataProcessor

from hotel_booking.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()
spark = SparkSession.builder.getOrCreate()

cfg = ProjectConfig.from_yaml("../project_config.yml")

# Load and process the data
df = pd.read_csv("../data/booking.csv")
data_processor = DataProcessor(df=df, config=cfg, spark=spark)
data_processor.preprocess()
data_processor.generate_synthetic_df(n=1000, max_date=None)
data_processor.df["arrival_month"] = data_processor.df["arrival_month"].astype("int32")
data_processor.save_to_catalog()


# COMMAND ----------

# MAGIC %md
# MAGIC Then we load the data (where the test set includes the newly generated month of data, and the train set 12 months before) and train the model:

# COMMAND ----------

from hotel_booking.models.lightgbm_model import LightGBMModel

data_loader = DataLoader(spark=spark, config=cfg)
X_train, y_train, X_test, y_test = data_loader.split()

model = LightGBMModel(config=cfg)

model.train(X_train=X_train,
            y_train=y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC The code for model logging is now also added to the LightGBMModel class. This is how it can be used:

# COMMAND ----------

tags=Tags(**{"git_sha": "d294dca", "branch": "main"})

model_info = model.log_model(
    experiment_name="/Shared/hotel-booking-training",
    tags=tags,
    X_test=X_test,
    y_test=y_test,
    train_set_spark=data_loader.train_set_spark,
    train_query=data_loader.train_query,
    test_set_spark=data_loader.test_set_spark,
    test_query=data_loader.test_query
    )

# COMMAND ----------

metrics_new = model.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC Let’s now evaluate the currently registered latest model against the new test set:

# COMMAND ----------

import mlflow

sklearn_model_name = f"{cfg.catalog}.{cfg.schema}.hotel_booking_basic"
model_uri = f"models:/{sklearn_model_name}@latest-model"
eval_data = X_test.copy()
eval_data[cfg.target] = y_test

result = mlflow.models.evaluate(
        model_uri,
        eval_data,
        targets=cfg.target,
        model_type="regressor",
        evaluators=["default"],
    )
metrics_old = result.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Previously, we were evaluating model performance based on the RMSE. If this metric for the new model is lower than for the old model, we can proceed with registering the model:

# COMMAND ----------

print(metrics_new['root_mean_squared_error'], metrics_old['root_mean_squared_error'])

# COMMAND ----------

if metrics_new['root_mean_squared_error'] < metrics_old['root_mean_squared_error']:
    model.register_model(model_name=sklearn_model_name, tags=tags)

# COMMAND ----------

#    def register_model(self: "LightGBMModel", model_name: str, tags: Tags) -> None:
#        """Register the model in MLflow Model Registry."""
#        client = MlflowClient()
#        registered_model = mlflow.register_model(
#                model_uri=self.model_info.model_uri,
#                name=model_name,
#                tags=tags.to_dict(),
#            )
#        client.set_registered_model_alias(
#            name=model_name,
#            alias="latest-model",
#            version=registered_model.version,
#        )
#        return registered_model.version

# COMMAND ----------


