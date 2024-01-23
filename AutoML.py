# Databricks notebook source
# DBTITLE 0,--i18n-2630af5a-38e6-482e-87f1-1a1633438bb6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # AutoML
# MAGIC
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/automl.html" target="_blank">Databricks AutoML</a> helps you automatically build machine learning models both through a UI and programmatically. It prepares the dataset for model training and then performs and records a set of trials (using HyperOpt), creating, tuning, and evaluating multiple models. 
# MAGIC
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Utilize AutoML to automatically train and tune a model
# MAGIC * Explain input and output parameters in AutoML UI
# MAGIC * Interpret the results of an AutoML run
# MAGIC * Search models in AutoML and pick the best model

# COMMAND ----------


import os
absolute_dir_path = os.path.abspath("./data")
absolute_dir_path = "file:" + absolute_dir_path

# load data
airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results")

train_df, val_df, test_df = airbnb_df.randomSplit([0.70, 0.15, 0.15], seed = 42)



# COMMAND ----------

display(train_df.limit(5))

# COMMAND ----------

# DBTITLE 0,--i18n-7aa84cf3-1b6c-4ba4-9249-00359ee8d70a
# MAGIC %md 
# MAGIC
# MAGIC ## Run an Experiment with AutoML
# MAGIC
# MAGIC Currently, AutoML uses a combination of XGBoost, lightgbm and sklearn (only single node models) but optimizes the hyperparameters within each.

# COMMAND ----------

# DBTITLE 0,--i18n-1b5c8a94-3ac2-4977-bfe4-51a97d83ebd9
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC We can now use AutoML to search for the optimal <a href="https://docs.databricks.com/applications/machine-learning/automl.html#regression" target="_blank">regression</a> model. 
# MAGIC
# MAGIC Required parameters:
# MAGIC * **`dataset`** - Input Spark or pandas DataFrame that contains training features and targets. If using a Spark DataFrame, it will convert it to a Pandas DataFrame under the hood by calling .toPandas() - just be careful you don't OOM!
# MAGIC * **`target_col`** - Column name of the target labels
# MAGIC
# MAGIC We will also specify these optional parameters:
# MAGIC * **`primary_metric`** - Primary metric to select the best model. Each trial will compute several metrics, but this one determines which model is selected from all the trials. One of **`r2`** (default, R squared), **`mse`** (mean squared error), **`rmse`** (root mean squared error), **`mae`** (mean absolute error) for regression problems.
# MAGIC * **`timeout_minutes`** - The maximum time to wait for the AutoML trials to complete. **`timeout_minutes=None`** will run the trials without any timeout restrictions

# COMMAND ----------

from databricks import automl

summary = automl.regress(train_df, 
                         target_col="price", 
                         primary_metric="rmse", 
                         timeout_minutes=5, 
                         max_trials=10)
# this is then, either 5 minutes, or 10 trials..

# by running the "regress", we get all of the methods that are available within AutoML. Such as decision tree, LR, ... 

# COMMAND ----------

# DBTITLE 0,--i18n-57d884c6-2099-4f34-b840-a4e873308ffe
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC  
# MAGIC
# MAGIC After running the previous cell, you will notice two notebooks and an MLflow experiment:
# MAGIC * **`Data exploration notebook`** - we can see a Profiling Report which organizes the input columns and discusses values, frequency and other information
# MAGIC * **`Best trial notebook`** - shows the source code for reproducing the best trial conducted by AutoML
# MAGIC * **`MLflow experiment`** - contains high level information, such as the root artifact location, experiment ID, and experiment tags. The list of trials contains detailed summaries of each trial, such as the notebook and model location, training parameters, and overall metrics.
# MAGIC
# MAGIC Dig into these notebooks and the MLflow experiment - what do you find?
# MAGIC
# MAGIC Additionally, AutoML shows a short list of metrics from the best run of the model.

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# DBTITLE 0,--i18n-3c0cd1ec-8965-4af3-896d-c30938033abf
# MAGIC %md 
# MAGIC
# MAGIC ## Use the Best Model for Inference
# MAGIC
# MAGIC Now we can test the model that we got from AutoML against our test data. We'll be using <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf" target="_blank">mlflow.pyfunc.spark_udf</a> to register our model as a UDF and apply it in parallel to our test data.

# COMMAND ----------

# Load the best trial as an MLflow Model
import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("price").columns))
display(pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE on test dataset: {rmse:.3f}")

# COMMAND ----------

print(regression_evaluator.setMetricName("r2").evaluate(pred_df))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
