# Databricks notebook source
# DBTITLE 0,--i18n-60a5d18a-6438-4ee3-9097-5145dc31d938
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Airbnb Amsterdam: SparkML Regression 2
# MAGIC
# MAGIC In this notebook we will be adding additional features to our model, as well as discuss how to handle categorical features.
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Encode categorical variables using One-Hot-Encoder method
# MAGIC * Create a Spark ML Pipeline to fit a model
# MAGIC * Evaluate a model’s performance
# MAGIC * Save and load a model using Spark ML Pipeline

# COMMAND ----------

# DBTITLE 0,--i18n-b44be11f-203c-4ea4-bc3e-20e696cabb0e
# MAGIC %md
# MAGIC ## Load Dataset

# COMMAND ----------

import os
absolute_dir_path = os.path.abspath("./data")
absolute_dir_path = "file:" + absolute_dir_path

# load data
airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results")

# COMMAND ----------

# DBTITLE 0,--i18n-f8b3c675-f8ce-4339-865e-9c64f05291a6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC Let's use the same 80/20 split with the same seed as the previous notebook so we can compare our results apples to apples (unless you changed the cluster config!)

# COMMAND ----------

train_df, val_df, test_df = airbnb_df.randomSplit(weights = [0.7, 0.15, 0.15], seed = 42)

# COMMAND ----------

# DBTITLE 0,--i18n-09003d63-70c1-4fb7-a4b7-306101a88ae3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Categorical Variables
# MAGIC
# MAGIC There are a few ways to handle categorical features:
# MAGIC * Assign them a numeric value
# MAGIC * Create "dummy" variables (also known as One Hot Encoding)
# MAGIC * Generate embeddings (mainly used for textual data)
# MAGIC
# MAGIC ### One Hot Encoder
# MAGIC Here, we are going to One Hot Encode (OHE) our categorical variables. Spark doesn't have a **`dummies`** function, and 
# MAGIC
# MAGIC ##### OHE is a two-step process. First, we need to use <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer" target="_blank">StringIndexer</a> to map a string column of labels to an ML column of label indices.
# MAGIC
# MAGIC Then, we can apply the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html?highlight=onehotencoder#pyspark.ml.feature.OneHotEncoder" target="_blank">OneHotEncoder</a> to the output of the StringIndexer.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

df = spark.createDataFrame([(0, "a"), (2, "b"), (3, "c"), (4, "c"), (5, "c")],
                           ["id", "category"])


# COMMAND ----------

display(df)

# COMMAND ----------

indexer_model = StringIndexer(inputCol="category", outputCol= "category_index")

df_indexed = indexer_model.fit(df).transform(df)

display(df_indexed)

# COMMAND ----------

print(df)

# COMMAND ----------

train_df.dtypes

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
# extract all of our string columns 

index_output_cols = [x + "Index" for x in categorical_cols]
# create new column names that will be the the resulting column names after StringIndexer

ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
# handleInvalid = "skip" just doesn't perform any execution when labels that are unseen during the fit phase
# this prevents from exceptions from occuring

ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)
# For the input of OneHotEncoder, we use index_output_cols, which is numbers. I guess OHE cannot handle string groups. 



# COMMAND ----------

ohe_output_cols

# COMMAND ----------

index_output_cols

# COMMAND ----------

# MAGIC %md
# MAGIC See above: it didn't work and that's because the input data was string, instead of numeric.

# COMMAND ----------

# DBTITLE 0,--i18n-dedd7980-1c27-4f35-9d94-b0f1a1f92839
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC Now we can combine our OHE categorical features with our numeric features.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
# numeric_cols contain the columns that are numeric, except for the outcome feature "price"

assembler_inputs = ohe_output_cols + numeric_cols

vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

print(assembler_inputs)

# COMMAND ----------

print(type(assembler_inputs))

# COMMAND ----------

SI_df = string_indexer.fit(train_df).transform(train_df)

# COMMAND ----------

display(SI_df.limit(5))

# COMMAND ----------

OHE_df = ohe_encoder.fit(SI_df).transform(SI_df)

# COMMAND ----------

display(OHE_df.limit(5))

# COMMAND ----------

vec_df = vec_assembler.transform(OHE_df)

# COMMAND ----------

vec_df.columns

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr_model = LinearRegression(labelCol = "price", featuresCol= "features")

# COMMAND ----------

lr_model.fit(vec_df)

# COMMAND ----------

# DBTITLE 0,--i18n-fb06fb9b-5dac-46df-aff3-ddee6dc88125
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC Now that we have all of our features, let's build a linear regression model.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="price", featuresCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-a7aabdd1-b384-45fc-bff2-f385cc7fe4ac
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC Let's put all these stages in a Pipeline. A <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html?highlight=pipeline#pyspark.ml.Pipeline" target="_blank">Pipeline</a> is a way of organizing all of our transformers and estimators.
# MAGIC
# MAGIC This way, we don't have to worry about remembering the same ordering of transformations to apply to our test dataset.

# COMMAND ----------

from pyspark.ml import Pipeline

# defining pipeline stages
stages = [string_indexer, ohe_encoder, vec_assembler, lr]
# index -> encode -> assemble -> regression

pipeline = Pipeline(stages=stages)

# and it can just be fitted with the pipeline.. this is how simple it can be!
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC This takes advantage of spark's optimized computation to do this pipeline combined together.

# COMMAND ----------

pipeline_model.stages[0].explainParams

# COMMAND ----------

pipeline_model.stages[1].explainParams

# COMMAND ----------

pipeline_model.stages[2].explainParams

# COMMAND ----------

pipeline_model.stages[3].explainParams

# COMMAND ----------

# we got a lot of parameters, because we did one-hot encoding
print(pipeline_model.stages[3].coefficients)

# COMMAND ----------

# DBTITLE 0,--i18n-c7420125-24be-464f-b609-1bb4e765d4ff
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Saving Models
# MAGIC
# MAGIC We can save our models to persistent storage (e.g. DBFS) in case our cluster goes down so we don't have to recompute our results.
# MAGIC
# MAGIC > Models have write and save methods

# COMMAND ----------

model_dir = "file:" + os.path.abspath("./") + "/models/"


# COMMAND ----------

pipeline_model.write().overwrite().save(model_dir)

# COMMAND ----------

# DBTITLE 0,--i18n-15f4623d-d99a-42d6-bee8-d7c4f79fdecb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Loading models
# MAGIC
# MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
# MAGIC
# MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic `PipelineModel` back in.

# COMMAND ----------

from pyspark.ml import PipelineModel

saved_pipeline_model = PipelineModel.load(model_dir)
# model loading didn't work. But that's okay, we will just work with the model trained in this session 

# COMMAND ----------

# DBTITLE 0,--i18n-1303ef7d-1a57-4573-8afe-561f7730eb33
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply the Model to Test Set

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

display(pred_df.select("features", "price", "prediction").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Because our dataset is dummy coded, our vectors are sparse: it only takes the value, and the index that it is non-zero.

# COMMAND ----------

display(pred_df.select("price", "prediction").limit(5))

# COMMAND ----------

# DBTITLE 0,--i18n-9497f680-1c61-4bf1-8ab4-e36af502268d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate the Model - using metrics like R2 or RMSE
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) How is our R2 doing?

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

r2_eval = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName = "r2")

r2_eval.evaluate(pred_df)

# COMMAND ----------

val_pred_df = pipeline_model.transform(val_df)

# COMMAND ----------

r2_eval.evaluate(val_pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)

# this changes the metric name without re-defining the RegressionEvaluator
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# DBTITLE 0,--i18n-cc0618e0-59d9-4a6d-bb90-a7945da1457e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC As you can see, our RMSE decreased when compared to the model without one-hot encoding that we trained in the previous notebook, and the R2 increased as well!
