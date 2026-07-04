# Databricks notebook source
# ruff: noqa
%pip install .. 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


import json
import os

import mlflow

from hotel_booking.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()
mlflow.get_tracking_uri()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Experiment
# MAGIC MLflow Experiment can be created by using mlflow.create_experiment() or mlflow.set_experiment() functions. The mlflow.set_experiment() function will activate the existing experiment by name or will create a new one if such an experiment does not exist yet. It’s possible to add tags to your experiment by passing it as a parameter to mlflow.create_experiment command, or, if the experiment already exists, using mlflow.set_experiment_tags command after the experiment got activated
# MAGIC
# MAGIC

# COMMAND ----------

experiment = mlflow.set_experiment(experiment_name="/Shared/demo_model_reg")
mlflow.set_experiment_tags(
    {"repository_name": "RosePY/testmlops2"}
)

print(experiment)

# COMMAND ----------

# MAGIC %md
# MAGIC We can then retrieve an experiment by experiment id using the mlflow.get_experiment() command:

# COMMAND ----------

mlflow.get_experiment(experiment.experiment_id)

# COMMAND ----------



# COMMAND ----------

display(experiment.__dict__)

# COMMAND ----------

# dump class attributes in a json file for visualization
if not os.path.exists("../demo_artifacts"):
    os.mkdir("../demo_artifacts")
with open("../demo_artifacts/mlflow_experiment.json", "w") as json_file:
    json.dump(experiment.__dict__, json_file, indent=4)

# COMMAND ----------

# MAGIC %md
# MAGIC The MLflow experiment can also be found by name or by tag. The search_experiments() function outputs a list of instances of an Experiment class containing information about the experiment, such as experiment_id, creation time, artifact location, and tags created by Databricks and by the users:

# COMMAND ----------

# search for experiment
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='RosePY/testmlops2'"
)
print(experiments)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Run

# COMMAND ----------

# MAGIC %md
# MAGIC An MLflow run can be created by running mlflow.start_run() command. Let’s start a run and log some metrics and parameters:

# COMMAND ----------

# start a run
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": "1234567890abcd", "branch": "main"},
    description="demo run",
) as run:
    run_id = run.info.run_id
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})

# COMMAND ----------

run_info = mlflow.get_run(run_id=run_id)
print(run_info)

# COMMAND ----------

run_info_dict = run_info.to_dictionary()
with open("../demo_artifacts/run_info.json", "w") as json_file:
    json.dump(run_info_dict, json_file, indent=4)

# COMMAND ----------

run_info = mlflow.get_run(run_id=run_id)
display(run_info.to_dictionary())

# COMMAND ----------

print(run_info_dict["data"]["metrics"])

# COMMAND ----------

print(run_info_dict["data"]["params"])

# COMMAND ----------

# search for runs
from time import time

time_hour_ago = int((time() - 3600) * 1000)

runs = mlflow.search_runs(
    search_all_experiments=True,  # or experiment_ids=[], or experiment_names=[]
    order_by=["start_time DESC"],
    filter_string="status='FINISHED' AND "
    f"start_time>{time_hour_ago} AND "
    "run_name LIKE '%demo-run%' AND "
    "metrics.metric3>0",
)

# COMMAND ----------

runs

# COMMAND ----------

mlflow.start_run(run_id=run_id)
mlflow.log_metric(key="metric3", value=3.0)
# dynamically log metric (trainings epochs)
for i in range(0, 3):
    mlflow.log_metric(key="metric1", value=3.0 + i / 2, step=i)
#Un artefacto es cualquier archivo que quieras adjuntar al run.
#mlflow.log_artifact("../file")
mlflow.log_text("hello, MLflow!", "hello.txt")
mlflow.log_dict({"k": "v"}, "dict_example.json")
#sube una carpeta entera, no un solo archivo. Toma todo lo que hay en ../demo_artifacts
mlflow.log_artifacts("../demo_artifacts", artifact_path="demo_artifacts")

# COMMAND ----------

# MAGIC %md
# MAGIC MLflow experiment tracking supports logging images and figures. You can even log images dynamically, for example, after each training epoch. Since we’re not using a “with” block, remember to end the run once you’ve finished logging.

# COMMAND ----------

# log figure
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot([0, 1], [2, 3])
mlflow.log_figure(fig, "figure.png")

for i in range(0, 3):
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    mlflow.log_image(image, key="demo_image", step=i)

mlflow.end_run()

# COMMAND ----------

# load objects
artifact_uri = runs.artifact_uri[0]
dict_example = mlflow.artifacts.load_dict(f"{artifact_uri}/dict_example.json")
figure = mlflow.artifacts.load_image(f"{artifact_uri}/figure.png")
text = mlflow.artifacts.load_text(f"{artifact_uri}/hello.txt")

# COMMAND ----------

# download artifacts
if not os.path.exists("../downloaded_artifacts"):
    os.mkdir("../downloaded_artifacts")
mlflow.artifacts.download_artifacts(
    artifact_uri=f"{artifact_uri}/demo_artifacts", dst_path="../downloaded_artifacts"
)

# REvisar en UI, matraz a la derecha

# COMMAND ----------

# # nested runs: useful for hyperparameter tuning
# with mlflow.start_run(run_name="top_level_run") as run:
#     for i in range(1, 5):
#         with mlflow.start_run(run_name=f"subrun_{str(i)}", nested=True) as subrun:
#             mlflow.log_metrics({"m1": 5.1 + i, "m2": 2 * i, "m3": 3 + 1.5 * i})
