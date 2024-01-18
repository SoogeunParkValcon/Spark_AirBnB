# Databricks notebook source
# DBTITLE 0,--i18n-04aa5a94-e0d3-4bec-a9b5-a0590c33a257
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Airbnb Amsterdam: Model Registry
# MAGIC
# MAGIC MLflow Model Registry is a **collaborative hub where teams can share ML models, work together from experimentation** to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance.  This lesson explores how to manage models using the MLflow model registry.
# MAGIC
# MAGIC This demo notebook will use **scikit-learn on the Airbnb dataset** (this will then be the first time that we use ML not from ```spark```), but in the lab you will use MLlib.
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC * Explain MLflow Model Registry components
# MAGIC * Register a model to MLflow Model Registry
# MAGIC * Manage model lifecycle programmatically
# MAGIC * Manage model lifecycle using MLflow UI
# MAGIC * Archive and delete models from MLflow Model Registry
# MAGIC

# COMMAND ----------

# DBTITLE 0,--i18n-5802ff47-58b5-4789-973d-2fb855bf347a
# MAGIC %md-sandbox 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Model Registry
# MAGIC
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides **model lineage** (which MLflow Experiment and Run produced the model), model versioning, stage transitions (e.g. from staging to production), annotations (e.g. with comments, tags), and deployment management (e.g. which production jobs have requested a specific model version).
# MAGIC
# MAGIC MLflow Model Registry has the following features:
# MAGIC
# MAGIC * **Central Repository:** Register MLflow models with the MLflow Model Registry. A registered model has a unique name, version, stage, and other metadata.
# MAGIC * **Model Versioning:** Automatically keep track of versions for registered models when updated.
# MAGIC * **Model Stage:** Assigned preset or custom stages to each model version, like “Staging” and “Production” to represent the lifecycle of a model.
# MAGIC * **Model Stage Transitions:** Record new registration events or changes as activities that automatically log users, changes, and additional metadata such as comments.
# MAGIC * **CI/CD Workflow Integration:** Record stage transitions, request, review and approve changes as part of CI/CD pipelines for better control and governance.
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See <a href="https://mlflow.org/docs/latest/model-registry.html" target="_blank">the MLflow docs</a> for more details on the model registry.

# COMMAND ----------

# DBTITLE 0,--i18n-7f34f7da-b5d2-42af-b24d-54e1730db95f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Registering a Model
# MAGIC
# MAGIC The following workflow will work with either the UI or in pure Python.  This notebook will use pure Python.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Explore the UI throughout this lesson by clicking the "Models" tab on the left-hand side of the screen.

# COMMAND ----------

# DBTITLE 0,--i18n-cbc59424-e45b-4179-a586-8c14a66a61a1
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Train a model and log it to MLflow using <a href="https://mlflow.org/docs/latest/tracking.html#automatic-logging" target="_blank">autologging</a>. Autologging allows you to log metrics, parameters, and models without the need for explicit log statements.
# MAGIC
# MAGIC There are a few ways to use autologging:
# MAGIC
# MAGIC   1. Call **`mlflow.autolog()`** before your training code. This will enable autologging for each supported library you have installed as soon as you import it.
# MAGIC   > this captures all of the metrics apparently. It's like log metrics in the previous notebook.
# MAGIC
# MAGIC   2. Enable autologging at the workspace level from the admin console
# MAGIC
# MAGIC   3. Use library-specific autolog calls for each library you use in your code. (e.g. **`mlflow.spark.autolog()`**)
# MAGIC   > So not just SparkML but other libraries can be used.
# MAGIC
# MAGIC Here we are only using numeric features for simplicity of building the random forest.

# COMMAND ----------

# MAGIC %md
# MAGIC Load dataset:

# COMMAND ----------

import os
absolute_dir_path = os.path.abspath("./data")
absolute_dir_path = "file:" + absolute_dir_path

# load data
airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results")

# changing into pandas.df:
df = airbnb_df.toPandas()


# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# convert ineteger fields to double in case of missing values, which would cause error
int_cols = df.select_dtypes(include='int').columns
df[int_cols] = df[int_cols].astype('float')

df_numeric = df.select_dtypes(include=["int", "float"])

df_numeric.head()

# COMMAND ----------


X_train, X_test, y_train, y_test = train_test_split(df_numeric.drop(["price"], axis=1), df_numeric[["price"]].values.ravel(), random_state=42)


# COMMAND ----------

X_train.columns

# COMMAND ----------

with mlflow.start_run(run_name="LR Model") as run:

    # autolog: I have to specify which ML library I'm using. For example, here: mlflow.sklern.autolog.
    mlflow.sklearn.autolog(log_input_examples=True, 
                           log_model_signatures=True, 
                           log_models=True)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    signature = infer_signature(X_train, lr.predict(X_train))

# COMMAND ----------

# MAGIC %md
# MAGIC Because of ```autolog()```, many metrics are automatically computed.

# COMMAND ----------

# DBTITLE 0,--i18n-1322cac5-9638-4cc9-b050-3545958f3936
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create a unique model name so you don't clash with other workspace users. 
# MAGIC
# MAGIC Note that a registered model name must be a non-empty UTF-8 string and cannot contain forward slashes(/), periods(.), or colons(:).

# COMMAND ----------

model_name = "sklearn-lr_soogeuniewhatever11"
print(f"Model Name: {model_name}")

# COMMAND ----------

# DBTITLE 0,--i18n-0777e3f5-ba7c-41c4-a477-9f0a5a809664
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Register the model.

# COMMAND ----------

run.info.run_id

# COMMAND ----------

run_id = run.info.run_id # this is from the run above

model_uri = f"runs:/{run_id}/model"

# this registers the model
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# DBTITLE 0,--i18n-22756858-ff7f-4392-826f-f401a81230c4
# MAGIC %md-sandbox 
# MAGIC
# MAGIC ## Model Registery in MLflow UI
# MAGIC
# MAGIC **Open the *Models* tab on the left of the screen to explore the registered model.**  Note the following:<br><br>
# MAGIC
# MAGIC * It logged who trained the model and what code was used
# MAGIC * It logged a history of actions taken on this model
# MAGIC * It logged this model as a first version
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/registered_model_v3.png" style="height: 600px; margin: 20px"/></div>

# COMMAND ----------

# DBTITLE 0,--i18n-481cba23-661f-4de7-a1d8-06b6be8c57d3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Check the status.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient


# Always need to create a client for checking the model...
client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

mlflow.search_runs(run.info.experiment_id)

# COMMAND ----------

# DBTITLE 0,--i18n-10556266-2903-4afc-8af9-3213d244aa21
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now add a model description

# COMMAND ----------

model_details

# COMMAND ----------

client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

# COMMAND ----------

# DBTITLE 0,--i18n-5abeafb2-fd60-4b0d-bf52-79320c10d402
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Add a version-specific description.

# COMMAND ----------

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using OLS linear regression with sklearn."
)


# COMMAND ----------

# DBTITLE 0,--i18n-aaac467f-3a52-4428-a119-8286cb0ac158
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Deploying a Model
# MAGIC
# MAGIC The MLflow Model Registry defines several model stages: **`None`**, **`Staging`**, **`Production`**, and **`Archived`**. Each stage has a unique meaning. For example, **`Staging`** is meant for model testing, while **`Production`** is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# DBTITLE 0,--i18n-dff93671-f891-4779-9e41-a0960739516f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now that you've learned about stage transitions, transition the model to the **`Production`** stage.

# COMMAND ----------

import time

time.sleep(10) # In case the registration is still pending

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------

# DBTITLE 0,--i18n-4dc7e8b7-da38-4ce1-a238-39cad74d97c5
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Fetch the model's current status, which we just changed.

# COMMAND ----------

model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# DBTITLE 0,--i18n-ba563293-bb74-4318-9618-a1dcf86ec7a3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Fetch the latest model using a **`pyfunc`**.  Loading the model in this way allows us to use the model regardless of the package that was used to train it.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You can load a specific version of the model too.

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# DBTITLE 0,--i18n-e1bb8ae5-6cf3-42c2-aebd-bde925a9ef30
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Apply the model.

# COMMAND ----------

model_version_1.predict(X_test)

# COMMAND ----------

# DBTITLE 0,--i18n-75a9c277-0115-4cef-b4aa-dd69a0a5d8a0
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Deploying a New Model Version
# MAGIC
# MAGIC The MLflow Model Registry enables you to create multiple model versions corresponding to a single registered model. By performing stage transitions, you can seamlessly integrate new model versions into your staging or production environments.

# COMMAND ----------

# DBTITLE 0,--i18n-2ef7acd0-422a-4449-ad27-3a26f217ab15
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create a new model version and register that model when it's logged.

# COMMAND ----------

from sklearn.linear_model import Ridge

with mlflow.start_run(run_name="LR Ridge Model") as run:
    alpha = .9
    ridge_regression = Ridge(alpha=alpha)
    ridge_regression.fit(X_train, y_train)

    # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
    # function to register the model with the MLflow Model Registry. This automatically
    # creates a new model version

    mlflow.sklearn.log_model(
        sk_model=ridge_regression,
        artifact_path="sklearn-ridge-model",
        registered_model_name=model_name, 
        # by adding this this last argument, the new model goes under the registered model! 
    )
    # because here, the same model_name is used, this is added in the model registry and in the same experiment!

    mlflow.log_params(ridge_regression.get_params())
    mlflow.log_metric("mse", mean_squared_error(y_test, ridge_regression.predict(X_test)))

# COMMAND ----------

# DBTITLE 0,--i18n-dc1dd6b4-9e9e-45be-93c4-5500a10191ed
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Put the new model into staging.

# COMMAND ----------

model_details.name

# COMMAND ----------

import time

time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=2,
    stage="Production"
)

# COMMAND ----------

model_version_details = client.get_model_version(
    name=model_details.name,
    version=2
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# DBTITLE 0,--i18n-fe857eeb-6119-4927-ad79-77eaa7bffe3a
# MAGIC %md-sandbox 
# MAGIC
# MAGIC ### Review Model Using the UI
# MAGIC
# MAGIC
# MAGIC Check the UI to see the new model version.
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/model_version_v3.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# DBTITLE 0,--i18n-6f568dd2-0413-4b78-baf6-23debb8a5118
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Use the search functionality to grab the latest model version.

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

model_version_infos

# COMMAND ----------

[model_version_info.version for model_version_info in model_version_infos]

# COMMAND ----------

# DBTITLE 0,--i18n-4fb5d7c9-b0c0-49d5-a313-ac95da7e0f91
# MAGIC %md 
# MAGIC
# MAGIC ### Update a Deployed Model
# MAGIC
# MAGIC Add a description to this new version.

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=2,
    description=f"Hello. It is possible to change the description of the model, although it is in production already."
)

# COMMAND ----------

# DBTITLE 0,--i18n-10adff21-8116-4a01-a309-ce5a7d233fcf
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Since this model is now in staging, you can execute an automated CI/CD pipeline against it to test it before going into production.  Once that is completed, you can push that model into production.

# COMMAND ----------

new_model_version

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production", 
    archive_existing_versions=True # Archive existing model in production 
)

# COMMAND ----------

# DBTITLE 0,--i18n-e3caaf08-a721-425b-8765-050c757d1d2e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Delete version 1.  
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You cannot delete a model that is not first archived.

# COMMAND ----------

client.delete_model_version(
    name=model_name,
    version=1
)

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

client.delete_model_version(
  name = model_name,
  version = 2
)

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=3,
    stage="Archived"
)

# COMMAND ----------

# DBTITLE 0,--i18n-0eb4929d-648b-4ae6-bca3-aff8af50f15f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now delete the entire registered model.

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------

# DBTITLE 0,--i18n-6fe495ec-f481-4181-a006-bea55a6cef09
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Review
# MAGIC **Question:** How does MLflow tracking differ from the model registry?  
# MAGIC **Answer:** Tracking is meant for experimentation and development.  The model registry is designed to **take a model from tracking and put it through staging and into production**.  This is often the point that a data engineer or a machine learning engineer takes responsibility for the deployment process.
# MAGIC
# MAGIC **Question:** Why do I need a model registry?  
# MAGIC **Answer:** Just as MLflow tracking provides end-to-end reproducibility for the machine learning training process, a model registry provides reproducibility and governance for the deployment process.  Since production systems are mission critical, components can be isolated with ACL's so only specific individuals can alter production models.  Version control and CI/CD workflow integration is also a critical dimension of deploying models into production.
# MAGIC
# MAGIC **Question:** What can I do programmatically versus using the UI?  
# MAGIC **Answer:** Most operations can be done using the UI or in pure Python.  A model must be tracked using Python, but from that point on everything can be done either way.  For instance, a model logged using the MLflow tracking API can then be registered using the UI and can then be pushed into production.
