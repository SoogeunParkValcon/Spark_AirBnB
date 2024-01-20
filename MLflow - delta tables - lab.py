# Databricks notebook source
# DBTITLE 0,--i18n-2cf41655-1957-4aa5-a4f0-b9ec55ea213b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # MLflow Lab
# MAGIC
# MAGIC In this lab we will explore the path to moving models to production with MLflow using the following steps:
# MAGIC
# MAGIC 1. Load in Airbnb dataset, and save both training dataset and test dataset as Delta tables
# MAGIC 1. Train an MLlib linear regression model using all the listing features and tracking parameters, metrics artifacts and Delta table version to MLflow
# MAGIC 1. Register this initial model and move it to staging using MLflow Model Registry
# MAGIC 1. Add a new column, **`log_price`** to both our train and test table and update the corresponding Delta tables
# MAGIC 1. Train a second MLlib linear regression model, this time using **`log_price`** as our target and training on all features, tracking to MLflow 
# MAGIC 1. Compare the performance of the different runs by looking at the underlying data versions for both models
# MAGIC 1. Move the better performing model to production in MLflow model registry
# MAGIC
# MAGIC
# MAGIC By the end of this lab, you should be able to;
# MAGIC * Create Delta tables from existing data
# MAGIC * Explain Delta history feature and history retention policy
# MAGIC * Track a model fit process with MLflow
# MAGIC * Register a model with MLflow Model Registry
# MAGIC * Manage MLflow model lifecycle
# MAGIC * Select best model and move it production with MLflow

# COMMAND ----------

# DBTITLE 0,--i18n-197ad07c-dead-4444-82de-67353d81dcb0
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##  Step 1. Creating Delta Tables

# COMMAND ----------

# DBTITLE 0,--i18n-8e9e809d-4142-43d8-b361-830099a02d06
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Data versioning is an advantage of using Delta Lake, which preserves previous versions of datasets so that you can restore later.
# MAGIC
# MAGIC Let's split our dataset into train and test datasets, and writing them out in Delta format. You can read more at the Delta Lake <a href="https://docs.delta.io/latest/index.html" target="_blank">documentation</a>.

# COMMAND ----------


import os
absolute_dir_path = os.path.abspath("./data")
absolute_dir_path = "file:" + absolute_dir_path

# load data
airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results")


# COMMAND ----------

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC We want to save the data as delta table format:

# COMMAND ----------

train_delta_path = f"{absolute_dir_path}/train.delta"
test_delta_path = f"{absolute_dir_path}/test.delta"

# In case paths already exists
dbutils.fs.rm(train_delta_path, True)
dbutils.fs.rm(test_delta_path, True)

train_df.write.mode("overwrite").format("delta").save(train_delta_path)
test_df.write.mode("overwrite").format("delta").save(test_delta_path)

# COMMAND ----------

# DBTITLE 0,--i18n-ead09bc6-c6f2-4dfa-bc9c-ddf41accc8f8
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's now read in our train and test Delta tables, specifying that we want the first version of these tables. This <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">blog post</a> has a great example of how to read in a Delta table at a given version.

# COMMAND ----------

# `versionAsOf` is an option in Delta Lake which allows you to read data from a specific version of a Delta table by specifying the version number. In this case, `data_version` is set to 0 which indicates that we want to read the original version of the Delta table.

data_version = 0
train_delta = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)

test_delta = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)


# COMMAND ----------

# DBTITLE 0,--i18n-2bf375c9-fb36-47a3-b973-82fa805e8b22
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Review Delta Table History
# MAGIC All the transactions for this table are stored within this table including the initial set of insertions, update, delete, merge, and inserts.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`"))

# COMMAND ----------

print(train_delta_path)

# COMMAND ----------

# DBTITLE 0,--i18n-d3d7251e-c070-4d54-844f-ec6880079e5b
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC By default Delta tables <a href="https://docs.databricks.com/delta/delta-batch.html#data-retention" target="_blank">keep a commit history of 30 days</a>. This retention period can be adjusted by setting **`delta.logRetentionDuration`**, which will determine how far back in time you can go. Note that setting this can result in storage costs to go up. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Be aware that versioning with Delta in this manner may not be feasible as a long term solution. The retention period of Delta tables can be increased, but with that comes additional costs to storage. Alternative methods of data versioning when training models and tracking to MLflow is to save copies of the datasets, either as an MLflow artifact (for a small dataset), or save to a separate distributed location and record the location of the underlying dataset as a tag in MLflow

# COMMAND ----------

# DBTITLE 0,--i18n-ffe159a7-e5dd-49fd-9059-e399237005a7
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 2. Log Initial Run to MLflow
# MAGIC
# MAGIC Let's first log a run to MLflow where we use all features. We use the same approach with RFormula as before. This time however, let's also log:
# MAGIC
# MAGIC 1. version of our data 
# MAGIC 2. the data path to MLflow.

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name="lr_model") as run:

    mlflow.log_param("data_path", train_delta_path)  
    mlflow.log_param("data_version", data_version) # 0 for now
    mlflow.log_param("label", "price_all_features") # price_all_features is a string for now
    mlflow.log_param("totally irrelevant stuff", "yeah dude this is irrelevant, I can put whatever I want in the log_param() function.")

    # Create pipeline
    r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")
    lr = LinearRegression(labelCol="price", featuresCol="features")
    pipeline = Pipeline(stages = [r_formula, lr])
    model = pipeline.fit(train_delta)

    # Log pipeline
    # TODO: Log model: model
    mlflow.spark.log_model(model, "model") # the string is just the name we give


    # Create predictions and metrics
    pred_df = model.transform(test_delta)
    regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
    rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

    # Log metrics
    # TODO: Log RMSE
    # TODO: Log R2
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 0,--i18n-3bac0fef-149d-4c25-9db1-94fbcd63ba13
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 3. Register Model and Move to Staging Using MLflow Model Registry
# MAGIC
# MAGIC We are happy with the performance of the above model and want to move it to staging. Let's create the model and register it to the MLflow model registry.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Make sure the path to **`model_uri`** matches the subdirectory (the second argument to **`mlflow.log_model()`**) included above.

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
# run_id is from above

suffix = "whatevermansoogeun"
model_name = f"mllib-lr_{suffix}"
print(f"Model Name: {model_name}\n")

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# DBTITLE 0,--i18n-78b33d27-0815-4d31-80a0-5e110aa96224
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Transition model to staging.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
# always initialize the client in order to do modify the registered model and stuff

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# COMMAND ----------

# DBTITLE 0,--i18n-b5f74e40-1806-46ab-9dd0-97b82d8f297e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Add a model description using <a href="https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.update_registered_model" target="_blank">update_registered_model</a>.

# COMMAND ----------

# let's add in a description 
client.update_registered_model(
  name = model_name,
  description = "This model registered uses Airbnb data and rformula. We're doing spark LR. I'm Soogeun. "
)

# COMMAND ----------

wait_for_model(model_details.name, 1, stage="Staging")

# COMMAND ----------

# DBTITLE 0,--i18n-03dff1c0-5c7b-473f-83ec-4a8283427280
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##  Step 4. Feature Engineering: Evolve Data Schema
# MAGIC
# MAGIC We now want to do some feature engineering with the aim of improving model performance; we can use Delta Lake to track older versions of the dataset. 
# MAGIC
# MAGIC We will add **`log_price`** as a new column and update our Delta table with it.

# COMMAND ----------

from pyspark.sql.functions import col, log, exp

# Create a new log_price column for both train and test datasets
train_new = train_delta.withColumn("log_price", log(col("price")))
test_new = test_delta.withColumn("log_price", log(col("price")))

# COMMAND ----------

# DBTITLE 0,--i18n-565313ed-2bca-4cc6-af87-1c0d509c0a69
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Save the updated DataFrames to **`train_delta_path`** and **`test_delta_path`**, respectively, passing the **`mergeSchema`** option to safely evolve its schema. 
# MAGIC
# MAGIC Take a look at this <a href="https://databricks.com/blog/2019/09/24/diving-into-delta-lake-schema-enforcement-evolution.html" target="_blank">blog</a> on Delta Lake for more information about **`mergeSchema`**.

# COMMAND ----------

# using the same path "train_delta_path", and we can just overwrite.
# However, the overwriting only works if the schema (things like the same column names and stuff) is the same, by default
# we need to put the arguments that allows different schema. 

train_new.write.mode("overwrite").option("mergeSchema", "true").save(train_delta_path)
test_new.write.mode("overwrite").option("mergeSchema", "true").save(test_delta_path)

# COMMAND ----------

# DBTITLE 0,--i18n-735a36b6-7510-4e4f-9df8-4ae51f9f87dc
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can have a look at the original data schema and the new schema:

# COMMAND ----------

set(train_new.schema.fields) ^ set(train_delta.schema.fields)

# COMMAND ----------

# DBTITLE 0,--i18n-0c7c986b-1346-4ff1-a4e2-ee190891a5bf
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's review the Delta history of our **`train_delta`** table and load in the most recent versions of our train and test Delta tables.
# MAGIC > So basically, the delta_path keeps the history of data.. this is pretty cool!

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC > So now we have:
# MAGIC   1. New spark.Dataframe made
# MAGIC   2. the new spark.DataFrame overwritten on the delta lake path
# MAGIC
# MAGIC
# MAGIC > and now, we read in this delta lake path into our spark notebook:

# COMMAND ----------

data_version = 1
train_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)  
test_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)



# COMMAND ----------

# DBTITLE 0,--i18n-f29c99ca-b92c-4f74-8bf5-c74070a8cd50
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 5. Use **`log_price`** as Target and Track Run with MLflow
# MAGIC
# MAGIC Retrain the model on the updated data and compare its performance to the original, logging results to MLflow.

# COMMAND ----------

with mlflow.start_run(run_name = "lr_log_model_soogeun") as run:
  mlflow.log_param("label", "log_price")
  mlflow.log_param("data_version", data_version)
  mlflow.log_param("data_path", train_delta_path)
  mlflow.log_param("Info", "This is a model where log(price) is used as the label for prediction.")

  # pipeline build
  rformula_stuff = RFormula(formula = "log_price ~ .-price",
                            featuresCol= "features",
                            labelCol= "log_price",
                            handleInvalid= "skip")
  
  log_lr = LinearRegression(labelCol="log_price", featuresCol="features")

  pipetimez = Pipeline(stages=[rformula_stuff, log_lr])

  pipetimez_fitted = pipetimez.fit(train_delta_new)
  
  # logging the model:
  mlflow.spark.log_model(spark_model = pipetimez_fitted,
                          artifact_path= "log-model-soogeun",
                          registered_model_name=model_name
                          # I have to put this registered_model_name so that it goes in the registered model!!
                          )

  # predicting the new data:
  pred_df_log = pipetimez_fitted.transform(test_delta_new)

  # turning the log price feature into exponents:
  pred_df_exp = test_delta_new.withColumn("prediction_exp", exp(col("log_price")))
  
  # metric:
  evaltime = RegressionEvaluator(predictionCol="prediction_exp", labelCol="price", metricName = "rmse")

  mlflow.log_metric("rmse", evaltime.evaluate(pred_df_exp))
  mlflow.log_metric("r2", evaltime.setMetricName("r2").evaluate(pred_df_exp))

  run_id = run.info.run_id



# COMMAND ----------

with mlflow.start_run(run_name="lr_log_model") as run:
    # Log parameters
    mlflow.log_param("label", "log-price")
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_path", train_delta_path)    

    # Create pipeline
    r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip")  
    lr = LinearRegression(labelCol="log_price", predictionCol="log_prediction")
    pipeline = Pipeline(stages = [r_formula, lr])
    pipeline_model = pipeline.fit(train_delta_new)

    # Log model and update the registered model
    mlflow.spark.log_model(
        spark_model=pipeline_model,
        artifact_path="log-model",
        registered_model_name=model_name
    )  
    # because we specify here the "registered_model_name", this "run" goes directly under the registered model

    # Create predictions and metrics
    pred_df = pipeline_model.transform(test_delta)
    exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)  

    run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 0,--i18n-e5bd7bfb-f445-44b5-a272-c6ae2849ac9f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Step 6. Compare Performance Across Runs by Looking at Delta Table Versions 
# MAGIC
# MAGIC Use MLflow's <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs" target="_blank">**`mlflow.search_runs`**</a> API to identify runs according to the version of data the run was trained on. Let's compare our runs according to our data versions.
# MAGIC
# MAGIC Filter based on **`params.data_path`** and **`params.data_version`**.

# COMMAND ----------

# searching and sorting the runs by using the API
data_version = 0

mlflow.search_runs(filter_string = f"params.data_version = {data_version}")
# this filter_string always has to be a string!

# COMMAND ----------

# now getting only the runs where data_version is 1 (which is after the log transformation!)
data_version = 1

mlflow.search_runs(filter_string = f"params.data_version = {data_version}")

# COMMAND ----------

# DBTITLE 0,--i18n-fd0fc3ae-7c2e-4d7d-90da-0b6e6b830496
# MAGIC %md 
# MAGIC
# MAGIC **Question:** Which version of the data produced the best model?

# COMMAND ----------

# DBTITLE 0,--i18n-3056bfcc-7623-4410-8b1b-82cba24ae3dd
# MAGIC %md 
# MAGIC
# MAGIC ## Step 7. Move the Best Performing Model to Production Using MLflow Model Registry
# MAGIC
# MAGIC Get the most recent model version and move it to production.

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
# this just gets the biggest model version

# COMMAND ----------

new_model_version 

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="This model version was built using a MLlib Linear Regression model with all features and log_price as predictor."
)

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=new_model_version)
model_version_details.status

# COMMAND ----------

# Move Model into Production
client.transition_model_version_stage(
  name = model_name,
  version = new_model_version,
  stage = "Archived"
)

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Archived"
)

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md 
# MAGIC
# MAGIC ## Classroom Cleanup
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
