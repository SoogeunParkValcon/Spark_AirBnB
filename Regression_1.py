# Databricks notebook source
# DBTITLE 0,--i18n-62811f6d-e550-4c60-8903-f38d7ed56ca7
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Airbnb Amsterdam: SparkML regression
# MAGIC
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in Amsterdam
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Use the SparkML to build a linear regression model
# MAGIC * Identify the differences between estimators and transformers in Spark ML

# COMMAND ----------

# DBTITLE 0,--i18n-b44be11f-203c-4ea4-bc3e-20e696cabb0e
# MAGIC %md 
# MAGIC ## Load Dataset

# COMMAND ----------

import os
absolute_dir_path = os.path.abspath("./data/")

# COMMAND ----------

absolute_dir_path = "file:" + absolute_dir_path

# COMMAND ----------

# loading the data
airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results") 

# COMMAND ----------

# DBTITLE 0,--i18n-ee10d185-fc70-48b8-8efe-ea2feee28e01
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/301/TrainTestSplit.png)
# MAGIC
# MAGIC **Question**: Why is it necessary to set a seed? What happens if I change my cluster configuration?

# COMMAND ----------

train_df, val_df, test_df= airbnb_df.randomSplit([.7, .15, .15], seed=42)
print(train_df.cache().count())

# COMMAND ----------

val_df.count()

# COMMAND ----------

# DBTITLE 0,--i18n-b70f996a-31a2-4b62-a699-dc6026105465
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's change the # of partitions (to simulate a different cluster configuration), and see if we get the same number of data points in our training set.

# COMMAND ----------

train_repartition_df, val_repartition_df, test_repartition_df = (airbnb_df
                                             .repartition(24)
                                             .randomSplit([.7, .15, .15], seed=42))

print(train_repartition_df.count())

# so these random splits of train and test sets are different from the sets train and test sets above

# COMMAND ----------

display(train_repartition_df.limit(5))


# COMMAND ----------

display(train_df.limit(5))

# COMMAND ----------

# DBTITLE 0,--i18n-5b96c695-717e-4269-84c7-8292ceff9d83
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC We are going to build a very simple model predicting **`price`** just given the number of **`bedrooms`**.
# MAGIC
# MAGIC **Question**: What are some assumptions of the linear regression model?

# COMMAND ----------

train_df.columns

# COMMAND ----------

display(train_df.select("price", "review_scores_rating").summary())
# the summary method tells us these things

# COMMAND ----------

# DBTITLE 0,--i18n-4171a9ae-e928-41e3-9689-c6fcc2b3d57c
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC There does appear to be some outliers in our dataset for the price ($10,000 a night??). Just keep this in mind when we are building our models.
# MAGIC
# MAGIC We will use <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html?highlight=linearregression#pyspark.ml.regression.LinearRegression" target="_blank">LinearRegression</a> to build our first model.
# MAGIC
# MAGIC The cell below will fail because the Linear Regression estimator expects a vector of values as input. We will fix that with VectorAssembler below.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="review_scores_rating", labelCol="price")


# COMMAND ----------

print(lr.explainParams())
# the explainParams() is a nice method that lets us run through the model

# COMMAND ----------

lr_model = lr.fit(train_df)
# the predictor feature should be of type VectorUDT.. instead of double



# COMMAND ----------

# DBTITLE 0,--i18n-f1353d2b-d9b8-4c8c-af18-2abb8f0d0b84
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC What went wrong? Turns out that the Linear Regression **estimator** (**`.fit()`**) expected a column of Vector type as input.
# MAGIC
# MAGIC We can easily get the values from the **`bedrooms`** column into a single vector using <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a>. VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.
# MAGIC
# MAGIC You can see an example of how to use VectorAssembler on the <a href="https://spark.apache.org/docs/latest/ml-features.html#vectorassembler" target="_blank">ML Programming Guide</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC The vector assembler concatenates the values for each predictor and puts it in a vector, which is saved in one column of the DF.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["review_scores_rating"], outputCol="features")
# takes in the column "bedrooms" and makes a new column called "features"

vec_train_df = vec_assembler.transform(train_df)

# COMMAND ----------

display(vec_train_df.limit(5))

# COMMAND ----------

# now we can use the "features" column
lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-ab8f4965-71db-487d-bbb3-329216580be5
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Inspect the Model

# COMMAND ----------

print(lr_model.coefficients)

# COMMAND ----------

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

lr_model.summary.r2

# COMMAND ----------

lr_model.summary.rootMeanSquaredError

# COMMAND ----------

lr_model.summary.r2

# COMMAND ----------

# DBTITLE 0,--i18n-ae6dfaf9-9164-4dcc-a699-31184c4a962e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply Model to Test Set

# COMMAND ----------

vec_test_df = vec_assembler.transform(test_df)
# also transform the test data in hand
# using the same transformer defined previously
# this creates the "features" vector which is the vector assembled

pred_df = lr_model.transform(vec_test_df)

display(pred_df.select("review_scores_rating", "features", "price", "prediction").limit(5))

# COMMAND ----------

# manual calculation of the difference between columns
hi = (pred_df["price"] - pred_df["prediction"])**2

pred_df = pred_df.withColumn("difference", hi)

# COMMAND ----------

display(pred_df.select("", "features", "price" , "prediction", "difference").limit(5))

# COMMAND ----------

display(pred_df.select("price", "prediction").limit(5))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate the Model
# MAGIC
# MAGIC Let's see how our linear regression model with just one variable does. Does it beat our baseline model?

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")


# COMMAND ----------

r2_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "price", metricName = "r2")

print(r2_evaluator.evaluate(pred_df))
