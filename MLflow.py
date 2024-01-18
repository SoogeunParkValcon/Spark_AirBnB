# Databricks notebook source
# DBTITLE 0,--i18n-b27f81af-5fb6-4526-b531-e438c0fda55e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Airbnb Amsterdam: Tracking models with MLflow
# MAGIC
# MAGIC <a href="https://mlflow.org/docs/latest/concepts.html" target="_blank">MLflow</a> seeks to address these three core issues:
# MAGIC
# MAGIC * It’s difficult to keep track of experiments
# MAGIC * It’s difficult to reproduce code
# MAGIC * There’s no standard way to package and deploy models
# MAGIC
# MAGIC In the past, when examining a problem, you would have to manually keep track of the many models you created, as well as their associated parameters and metrics. This can quickly become tedious and take up valuable time, which is where MLflow comes in.
# MAGIC
# MAGIC _MLflow is pre-installed on the Databricks Runtime for ML._
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC * Utilize MLflow to track experiments and log metrics
# MAGIC * Query and view past runs programmatically
# MAGIC * Search and view past runs using MLflow UI
# MAGIC * Save and reload models using MLflow

# COMMAND ----------

# DBTITLE 0,--i18n-b7c8a0e0-649e-4814-8310-ae6225a57489
# MAGIC %md 
# MAGIC
# MAGIC ## MLflow Architecture
# MAGIC
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC MLFlow can be interacted using UI or API!

# COMMAND ----------

# DBTITLE 0,--i18n-c1a29688-f50a-48cf-9163-ebcc381dfe38
# MAGIC %md 
# MAGIC
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Let's start by loading in our SF Airbnb Dataset.

# COMMAND ----------

import os
absolute_dir_path = os.path.abspath("./data")
absolute_dir_path = "file:" + absolute_dir_path

print(absolute_dir_path)

# COMMAND ----------

airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results")

# COMMAND ----------

# MAGIC %md
# MAGIC Filtering out rows with ```price > 10000```:

# COMMAND ----------

airbnb_df = airbnb_df.filter(airbnb_df.price < 10000)

# COMMAND ----------

# MAGIC %md
# MAGIC Train-val-test split:

# COMMAND ----------

train_df, val_df, test_df = airbnb_df.randomSplit([0.7, 0.15, 0.15], seed =42)

# COMMAND ----------

# DBTITLE 0,--i18n-9ab8c080-9012-4f38-8b01-3846c1531a80
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## MLflow Tracking
# MAGIC
# MAGIC MLflow Tracking is a logging API specific for machine learning and agnostic to libraries and environments that do the training.  It is organized around the concept of **runs**, which are executions of data science code.  Runs are aggregated into **experiments** where many runs can be a part of a given experiment and an MLflow server can host many experiments.
# MAGIC
# MAGIC You can use <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment" target="_blank">mlflow.set_experiment()</a> to set an experiment, but if you do not specify an experiment, it will automatically be scoped to this notebook.

# COMMAND ----------

# DBTITLE 0,--i18n-82786653-4926-4790-b867-c8ccb208b451
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Track Runs
# MAGIC
# MAGIC Each run can record the following information:<br><br>
# MAGIC
# MAGIC - **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC - **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC - **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and data files
# MAGIC - **Source:** The code that originally ran the experiment
# MAGIC
# MAGIC **NOTE**: For Spark models, MLflow can only log PipelineModels.

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="LR-Single-Feature") as run:

    # The following chunk does the pipeline for LR
    # This we already did before. Using only "number_of_reviews" as the predictor
    vec_assembler = VectorAssembler(inputCols=["number_of_reviews"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="price")
    pipeline = Pipeline(stages=[vec_assembler, lr])
    pipeline_model = pipeline.fit(train_df)

    # Logging the parameters: tracking the parameters using MLflow
    # I think here, we are manually providing the strings? 
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "number_of_reviews")

    # Log model: logging the model built above
    mlflow.spark.log_model(pipeline_model, "model", 
                           input_example=train_df.limit(5).toPandas()) 
    # We save the model, and also a sample of data that went in the model 

    # Evaluate predictions
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regression_evaluator.evaluate(pred_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC The experiment is the **Parent** of all these runs...
# MAGIC
# MAGIC In there, everything gets saved. Dependencies, parameters...

# COMMAND ----------

# DBTITLE 0,--i18n-44bc7cac-de4a-47e7-bfff-6d2eb58172cd
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC There, all done! Let's go through the other two linear regression models and then compare our runs. 
# MAGIC
# MAGIC **Question**: Does anyone remember the RMSE of the other runs?
# MAGIC
# MAGIC Next let's build our linear regression model but use all of our features.

# COMMAND ----------

from pyspark.ml.feature import RFormula

# COMMAND ----------

with mlflow.start_run(run_name = "LR-All-Features-Soogeun") as run:
  
  r_formula = RFormula(formula = "price ~ .",
                       featuresCol= "features",
                       labelCol = "price",
                       handleInvalid= "skip")
  
  lr_model = LinearRegression(featuresCol="features",
                              labelCol="price")
  
  pipeline = Pipeline(stages = [r_formula, lr_model])

  pipeline_model = pipeline.fit(train_df)

  mlflow.spark.log_model(pipeline_model, "model", 
                         input_example=train_df.limit(5).toPandas())
  
  mlflow.log_param("label", "price")
  mlflow.log_param("features", "all of the features mannnn")

  # applying the pipeline to get the metrics:
  pred_df = pipeline_model.transform(test_df)

  evaluator_is_here = RegressionEvaluator(predictionCol="prediction", labelCol = "price", metricName="r2")
  r2_result = evaluator_is_here.evaluate(pred_df)
  rmse_result = evaluator_is_here.setMetricName("rmse").evaluate(pred_df)

  mlflow.log_metric("r2haha", r2_result)
  mlflow.log_metric("rmse", rmse_result)

# COMMAND ----------

from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name="LR-All-Features") as run:
    # Create pipeline

    # the R formula uses all of the features!
    r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")
    lr = LinearRegression(labelCol="price", featuresCol="features")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log pipeline
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas())

    # Log parameter
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "all_features")

    # Create predictions and metrics
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
    rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

    # Log both metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

# COMMAND ----------

# DBTITLE 0,--i18n-70188282-8d26-427d-b374-954e9a058000
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, we will use Linear Regression to predict the log of the price, due to its log normal distribution. 
# MAGIC
# MAGIC We'll also practice logging artifacts to keep a visual of our log normal histogram.

# COMMAND ----------

# DBTITLE 0,--i18n-66785d5e-e1a7-4896-a8a9-5bfcd18acc5c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC That's it! Now, let's use MLflow to easily look over our work and compare model performance. You can either query past runs programmatically or use the MLflow UI.

# COMMAND ----------

# DBTITLE 0,--i18n-0b1a68e1-bd5d-4f78-a452-90c7ebcdef39
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Querying Past Runs Programmatically.. Above this was via UI
# MAGIC
# MAGIC You can query past runs programmatically in order to use this data back in Python.  The pathway to doing this is an **`MlflowClient`** object.
# MAGIC
# MAGIC > this is without using the UI, but with code.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

display(client.search_experiments()) 
# this would show us all of the experiments

# COMMAND ----------

# DBTITLE 0,--i18n-dcd771b2-d4ed-4e9c-81e5-5a3f8380981f
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC You can also use `search_runs` <a href="https://mlflow.org/docs/latest/search-syntax.html" target="_blank">(documentation)</a> to find all runs for a given experiment.

# COMMAND ----------

# "run" was just now executed:

experiment_id = run.info.experiment_id

print(experiment_id)

# COMMAND ----------

experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)
# here you can see the metrics.. r2 and rmse..

# COMMAND ----------

# DBTITLE 0,--i18n-68990866-b084-40c1-beee-5c747a36b918
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Pull the last run and look at metrics.

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
len(runs)

# COMMAND ----------

runs[0].info.run_id

# COMMAND ----------

# DBTITLE 0,--i18n-cfbbd060-6380-444f-ba88-248e10a56559
# MAGIC %md 
# MAGIC
# MAGIC ### View past runs using MLflow UI
# MAGIC
# MAGIC Go to the **"Experiments"** page and click on the experiment that you want to view. 
# MAGIC
# MAGIC Examine the following experiment details using the UI:<br><br>
# MAGIC * The **`Experiment ID`**
# MAGIC * The artifact location.  This is where the artifacts are stored in DBFS.
# MAGIC
# MAGIC #### Table View
# MAGIC
# MAGIC You can customize the table view which lists all runs for the experiment. For example, you can show/hide `rmse` or `features` columns.
# MAGIC
# MAGIC Following details can be found on the Experiment list page:
# MAGIC * **Run Name**: This is the run name is used while logging the run. Click on the name to view details of the run. See steps below for more details about run page.
# MAGIC * **Duration**: This shows the elapsed time for each run.
# MAGIC * **Source**: This is the notebook that created this run.
# MAGIC * **Model**: This column shows the model type.
# MAGIC
# MAGIC
# MAGIC After clicking on the time of the run, take a look at the following:<br><br>
# MAGIC * The Run ID will match what we printed above
# MAGIC * The model that we saved, as well as the Conda environment and the **`MLmodel`** file.
# MAGIC * `scikit-learn` models will have a pickled version of the model. As in this demo we are using `sparkml` no pickled version of the model is logged. 
# MAGIC
# MAGIC Note that you can add notes under the "Description" tab to help keep track of important information about your models. 
# MAGIC
# MAGIC Also, click on the run for the log normal distribution and see that the histogram is saved in "Artifacts".
# MAGIC
# MAGIC
# MAGIC #### Chart View
# MAGIC
# MAGIC Chart view allows you to compare runs by features and evaluation metric. You can use various charts, such as bar chart or scatter plot chart, to visually compare experiment runs.

# COMMAND ----------

# DBTITLE 0,--i18n-63ca7584-2a86-421b-a57e-13d48db8a75d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Load Saved Model
# MAGIC
# MAGIC Let's practice <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html" target="_blank">loading</a> our logged log-normal model.

# COMMAND ----------

model_path = f"runs:/{run.info.run_id}/log-model"
loaded_model = mlflow.spark.load_model(model_path)

# wow very cool.. we can use the run_id and load the model.. 
display(loaded_model.transform(test_df))

# this didn't work for me, but let's pretend it did!
