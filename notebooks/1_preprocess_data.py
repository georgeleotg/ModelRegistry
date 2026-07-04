# Databricks notebook source
# MAGIC %pip install ..

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys, mlflow, lightgbm, sklearn, pandas
print("Python:", sys.version)
print("mlflow:", mlflow.__version__)


# COMMAND ----------

import sys
from pathlib import Path

# Busca hacia arriba la carpeta que contiene src/hotel_booking
root = Path.cwd()
for _ in range(4):
    if (root / "src" / "hotel_booking").exists():
        break
    root = root.parent

src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hotel_booking
print("hotel_booking OK desde:", src_path)

# COMMAND ----------

# ruff: noqa
from pyspark.sql import SparkSession

from hotel_booking.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

cfg = ProjectConfig.from_yaml("../project_config.yml")
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

# Create catalog and schema if they do not exist
# spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")


# COMMAND ----------

# Load and process the data
from datetime import datetime

import pandas as pd

df = pd.read_csv("../data/booking.csv")
print(df.head())
df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
df["date_of_reservation"] = df["date_of_reservation"].apply(
    lambda x: "3/1/2018" if x == "2018-2-29" else x
)
df["date_of_reservation"] = df["date_of_reservation"].apply(
    lambda x: datetime.strptime(x, "%m/%d/%Y")
)
df["arrival_date"] = df["date_of_reservation"] + pd.to_timedelta(
    df["lead_time"], unit="d"
)
df["arrival_month"] = df["arrival_date"].dt.month

dst_table = f"{catalog}.{schema}.hotel_booking"

spark.createDataFrame(df).write.mode("overwrite").saveAsTable(f"{dst_table}")

spark.sql(
  f"""ALTER TABLE {dst_table}
   SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"""
)
