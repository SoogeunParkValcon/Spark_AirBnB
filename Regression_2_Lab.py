# Databricks notebook source
# DBTITLE 0,--i18n-4e1b9835-762c-42f2-9ff8-75164cb1a702
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Airbnb Amsterdam: Regression 2 - lab
# MAGIC
# MAGIC Alright! We're making progress. Still not a great RMSE or R2, but better than the baseline or just using a single feature.
# MAGIC
# MAGIC In the lab, you will see how to improve our performance even more.
# MAGIC
# MAGIC By the end of this lab, you should be able to;
# MAGIC
# MAGIC * Use RFormula to simplify the process of using StringIndexer, OneHotEncoder, and VectorAssembler
# MAGIC * Transform data into log scale to fit a model
# MAGIC * Convert log scale predictions to appropriate form for model evaluation

# COMMAND ----------

# DBTITLE 0,--i18n-1500312a-d027-42d0-a787-0dea4f8d7d03
# MAGIC %md
# MAGIC
# MAGIC ## Load Dataset and Train Model

# COMMAND ----------

1+1

# COMMAND ----------

import os
absolute_dir_path = os.path.abspath("./data")
absolute_dir_path = "file:" + absolute_dir_path

# load data
airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results")

# COMMAND ----------

# random split:

train_df, val_df, test_df = airbnb_df.randomSplit([0.7, 0.15, 0.15], seed = 42)

# COMMAND ----------

# DBTITLE 0,--i18n-a427d25c-591f-4899-866a-14064eff40e3
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## RFormula
# MAGIC
# MAGIC Instead of manually specifying which columns are categorical to the StringIndexer and OneHotEncoder, <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.RFormula.html?highlight=rformula#pyspark.ml.feature.RFormula" target="_blank">RFormula</a> can do that automatically for you.
# MAGIC
# MAGIC With RFormula, if you have any columns of type String, it treats it as a **categorical feature and string indexes & one hot encodes it** for us. Otherwise, it leaves as it is. Then it combines all of one-hot encoded features and numeric features into a single vector, called **`features`**.
# MAGIC
# MAGIC You can see a detailed example of how to use RFormula <a href="https://spark.apache.org/docs/latest/ml-features.html#rformula" target="_blank">here</a>.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

r_formula = RFormula(formula = "price ~ .", 
                     featuresCol = "features",
                     handleInvalid = "skip",
                     labelCol = "price"
                    )
# handleInvalid: skip to ignore categories that were unseen in the train data

print(r_formula.explainParams())

# COMMAND ----------

# on top of the RFormula, the linear regression model also has to be initiated. Idk why RFormula doesn't do this, but wtv:

lr = LinearRegression(featuresCol = "features", labelCol="price")

# COMMAND ----------

pipeline = Pipeline(stages = [r_formula, lr])

pipeline_model = pipeline.fit(train_df)

pred_df = pipeline_model.transform(test_df)

regression_evaluator = RegressionEvaluator(labelCol = "price")

rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC So ```RFormula``` is a replacement for ```StringIndexer``` and ```OneHotEncoder```.

# COMMAND ----------

# DBTITLE 0,--i18n-c9898a31-90e4-4a6d-87e6-731b95c764bd
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Log Scale
# MAGIC
# MAGIC Our price dependent variable appears to be **log-normally distributed** - and also skewed - , so we are going to try to predict it on the log scale.
# MAGIC
# MAGIC Let's convert our price to be on log scale, and have the linear regression model predict the log price

# COMMAND ----------

dbutils.data.summarize(train_df)

# COMMAND ----------

# I think it's probably an erroneous outlier if the price is higher than 10k. I'll remove those:
train_df.filter(train_df.price > 10000).count()

# COMMAND ----------

train_df = train_df.filter(train_df.price < 10000)

# COMMAND ----------

display(train_df.select("price"))

# COMMAND ----------

# MAGIC %md
# MAGIC The price column is very skewed. And probably log-normal.

# COMMAND ----------

# We can check if the target variable follows a normal distribution or not by plotting a histogram of its values.
# If the histogram looks bell-shaped, then the variable follows a normal distribution. Otherwise, it's not normally distributed. 

# To check whether the dependent variable is log-normally distributed, we can plot a histogram of the log-transformed values of the dependent variable.
# If the histogram looks bell-shaped, then the original dependent variable follows a log-normal distribution. Otherwise, it's not log-normally distributed.



# To perform this task in Spark, we can use the log function from the pyspark.sql.functions module to create a new column in our dataframe with the log-transformed dependent variable.
# Then, we can use the display function to plot a histogram of the log transformed values:
from pyspark.sql.functions import col, log

display(train_df.select(log(col("price"))))


# COMMAND ----------

# MAGIC %md
# MAGIC Wow! it's now normal!

# COMMAND ----------

from pyspark.sql.functions import col, log

log_train_df = train_df.withColumn("logprice", log("price"))
log_test_df = test_df.withColumn("logprice", log("price"))
# This adds the new column log price



# COMMAND ----------

display(log_train_df.limit(5))

# COMMAND ----------

r_formula = RFormula(formula = "logprice ~ . -price", 
                     handleInvalid= "skip",
                     featuresCol= "features",
                     labelCol= "logprice") 

log_lr = LinearRegression(featuresCol= "features", 
                      labelCol= "logprice")



# COMMAND ----------

pipeline = Pipeline(stages=[r_formula, log_lr])
pipeline_model = pipeline.fit(log_train_df)
pred_df = pipeline_model.transform(log_test_df)

# COMMAND ----------

# DBTITLE 0,--i18n-51b5e35f-e527-438a-ab56-2d4d0d389d29
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Exponentiate
# MAGIC
# MAGIC In order to interpret our RMSE, we need to convert our predictions back from logarithmic scale.

# COMMAND ----------

display(pred_df.limit(5))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

r2_evaluator = RegressionEvaluator(predictionCol="exp_prediction", labelCol = "price", metricName = "r2")

# COMMAND ----------

from pyspark.sql.functions import exp

pred_df_exp = pred_df.withColumn("exp_prediction", exp("prediction"))

# COMMAND ----------

display(pred_df_exp.limit(5))

# COMMAND ----------

r2_evaluator.evaluate(pred_df_exp)

# COMMAND ----------

r2_evaluator.setMetricName("rmse").evaluate(pred_df_exp)
