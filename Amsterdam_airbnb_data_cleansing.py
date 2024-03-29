# Databricks notebook source
# DBTITLE 0,--i18n-8c6d3ef3-e44b-4292-a0d3-1aaba0198525
# MAGIC %md 
# MAGIC
# MAGIC # Airbnb Amsterdam Data
# MAGIC
# MAGIC We will be using Spark to do some exploratory data analysis & cleansing of the Aibnb data from Amsterdam.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Explore dataset based on summary statistics
# MAGIC * Identify and remove outliers in a dataset
# MAGIC * Impute missing data 
# MAGIC * Create an imputer pipeline using Spark ML

# COMMAND ----------

# DBTITLE 0,--i18n-1e2c921e-1125-4df3-b914-d74bf7a73ab5
# MAGIC %md 
# MAGIC ## 📌 Requirements
# MAGIC
# MAGIC **Required Databricks Runtime Version:** 
# MAGIC * Please note that in order to run this notebook, you must use one of the following Databricks Runtime(s): **12.2.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading the data - download

# COMMAND ----------

import os
from pathlib import Path
import urllib.request as request


# COMMAND ----------

file_name = "data/listings.csv"
url_name = "https://github.com/SoogeunParkValcon/Datasets/raw/main/listings.csv"

# COMMAND ----------

# method that downloads the data from the url
 
def download_file(file_name, url_name):
  if not os.path.exists(file_name):
      filename, headers = request.urlretrieve(
      url = url_name, 
      filename = file_name
      )
      

# COMMAND ----------

download_file(file_name, url_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Loading the data - ```spark.read.csv```:

# COMMAND ----------

absolute_file_path = os.path.abspath("./data/listings.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC It's really annyoing, but databricks uses this "distributed file system" thing, starting with ```dfbs```. That doesn't work with this. We have to start with ```file:```

# COMMAND ----------

absolute_file_path = "file:" + absolute_file_path

# COMMAND ----------

print(absolute_file_path)

# COMMAND ----------


raw_df = spark.read.csv(absolute_file_path, 
                        header="true", 
                        inferSchema="true", 
                        multiLine="true", 
                        escape='"')

# header = header exists
# inferSchema = let spark determine the schema
# multiLine = strings are allowed to have line breaks
# escape = strings are allowed to have commas

display(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Studying the features:

# COMMAND ----------

print(raw_df.columns)
# let's have a look at the columns

# COMMAND ----------

print(raw_df.count()) # number of rows

print(len(raw_df.columns)) # number of columns

# COMMAND ----------

print(raw_df.shape()) # spark dataframe does not have the object "shape"

# COMMAND ----------

raw_df.cube("neighbourhood_cleansed").count().show()
# this is how I get the frequency table of the features with Spark DF

# COMMAND ----------

display(raw_df.groupby("neighbourhood_cleansed").count().orderBy("count", ascending=False))
# similar way of doing the same thing, with the .groupby method

# COMMAND ----------

display(raw_df.select("neighbourhood_cleansed").distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ```id``` is ofc worth keeping:

# COMMAND ----------

# each row has a unique id
raw_df.select("id").distinct().count()

# COMMAND ----------

raw_df.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC Host response time can also be a fun feature:

# COMMAND ----------

display(raw_df.groupBy("host_response_time").count())

# COMMAND ----------

# MAGIC %md
# MAGIC There are too many ```property_type```'s. I will not include this:

# COMMAND ----------

display(raw_df.groupBy("property_type").count())

# COMMAND ----------

#I wanna know how many values of the feature "accommodates" is NA
display(raw_df.groupBy("accommodates").count())

print(f"This feature has {raw_df.filter(raw_df.accommodates.isNull()).count()} missing values.")


# COMMAND ----------

display(raw_df.groupBy("bathrooms_text").count())

print(f"This feature has {raw_df.filter(raw_df.bathrooms_text.isNull()).count()} missing values.")


# COMMAND ----------

display(raw_df.groupBy("beds").count())

print(f"This feature has {raw_df.filter(raw_df.beds.isNull()).count()} missing values.")


# COMMAND ----------

# MAGIC %md
# MAGIC By studying the features, I've decided on the following:

# COMMAND ----------

columns_to_keep = [
    "id",
    "host_response_time",
    "accommodates",
    "neighbourhood_cleansed",
    "bathrooms_text",
    "beds",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "calculated_host_listings_count",
    "reviews_per_month",
    "price"
]


# COMMAND ----------

base_df = raw_df.select(columns_to_keep) 
# above, we select columns of interest from raw_df and return a new DataFrame base_df

print(base_df.cache().count())
# Caching a DataFrame in memory improves the performance of the queries that are often performed on the DataFrame.
# Calling the count() method of the DataFrame forces the cache operation to occur.

print(len(base_df.columns))
# output the number of columns in the base_df DataFrame

display(base_df)
# output the DataFrame


# COMMAND ----------

from pyspark.sql.functions import col, sum

display(base_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in base_df.columns]))
# Using a list comprehension to iterate over columns in `base_df`, and for each column:
# - apply the `isNull` function to check if that column cell is null, 
# - chain `cast("int")` to convert the boolean value to an integer
# - apply the `sum` function to get the count of null values
# - alias the resulting DataFrame column with the column name

# display the resulting DataFrame


# COMMAND ----------

display(base_df.select([sum((col(c) == "N/A").cast("int")).alias(c) for c in base_df.columns]))
# Using a list comprehension to iterate over columns in `base_df`, and for each column:
# - apply the check if that column cell is equal to "N/A" using the (col(c) == "N/A") expression, 
# - chain `cast("int")` to convert the boolean value to an integer
# - apply the `sum` function to get the count of "N/A" values
# - alias the resulting DataFrame column with the column name

# display the resulting DataFrame


# COMMAND ----------

from pyspark.sql.functions import when

base_df = base_df.withColumn("host_response_time", when(col("host_response_time") == "N/A", None).otherwise(col("host_response_time")))


# COMMAND ----------

display(base_df)
# Displays the first 5 rows of the dataframe 'base_df'


# COMMAND ----------

base_df = base_df.drop("host_response_time")

# COMMAND ----------

# DBTITLE 0,--i18n-a12c5a59-ad1c-4542-8695-d822ec10c4ca
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC  
# MAGIC ## Fixing Data Types
# MAGIC
# MAGIC Take a look at the schema above. You'll notice that the **`price`** field got picked up as string. For our task, we need it to be a numeric (double type) field. 
# MAGIC
# MAGIC Let's fix that.

# COMMAND ----------

from pyspark.sql.functions import translate

fixed_price_df = base_df.withColumn("price", translate(col("price"), "$,", ""))

fixed_price_df = fixed_price_df.withColumn("price", col("price").cast("double"))



# COMMAND ----------

# DBTITLE 0,--i18n-4ad08138-4563-4a93-b038-801832c9bc73
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Summary statistics
# MAGIC
# MAGIC Two options:
# MAGIC * **`describe`**: count, mean, stddev, min, max
# MAGIC * **`summary`**: describe + interquartile range (IQR)
# MAGIC
# MAGIC **Question:** When to use IQR/median over mean? Vice versa?

# COMMAND ----------

display(fixed_price_df.describe())

# COMMAND ----------

display(fixed_price_df.summary())

# COMMAND ----------

# DBTITLE 0,--i18n-bd55efda-86d0-4584-a6fc-ef4f221b2872
# MAGIC %md 
# MAGIC
# MAGIC ### Explore Dataset with Data Profile
# MAGIC
# MAGIC The **Data Profile** feature in Databricks notebooks offers valuable insights and benefits for data analysis and exploration. By leveraging Data Profile, users gain a comprehensive overview of their **dataset's characteristics, statistics, and data quality metrics**. This feature enables data scientists and analysts to understand the data distribution, identify missing values, detect outliers, and explore descriptive statistics efficiently.
# MAGIC
# MAGIC There are two ways of viewing Data Profiler. The first option is the UI.
# MAGIC
# MAGIC - After using `display` function to show a data frame, click **+** icon next to the *Table* in the header. 
# MAGIC - Click **Data Profile**. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC This functionality is also available through the dbutils API in Python, Scala, and R, using the dbutils.data.summarize(df) command. We can also use **`dbutils.data.summarize(df)`** to display Data Profile UI.
# MAGIC
# MAGIC Note that this features will profile the entire data set in the data frame or SQL query results, not just the portion displayed in the table

# COMMAND ----------

dbutils.data.summarize(fixed_price_df) # very nice overview function of the dataframe in hand

# COMMAND ----------

# MAGIC %md 
# MAGIC Does this nice ```dbutils.data.summarize()``` also work on pandas or PySpark pandas dataframe??

# COMMAND ----------

import pyspark.pandas as ps

# from spark dataframe into PySpark pandas dataframe
pyspark_pandas_df = ps.DataFrame(fixed_price_df)
display(pyspark_pandas_df)

# COMMAND ----------

# MAGIC %md
# MAGIC _It indeed works on the_ ```pyspark-pandas-df```

# COMMAND ----------

dbutils.data.summarize(pyspark_pandas_df)

# COMMAND ----------

import pandas as pd

# converting a spark dataframe to pandas dataframe
pandas_df = fixed_price_df.toPandas()

# display the pandas df
display(pandas_df)


# COMMAND ----------

pandas_df["beds"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Nice, it also works on ```pd.dataframe```... interesting.

# COMMAND ----------

dbutils.data.summarize(pandas_df)

# COMMAND ----------

# DBTITLE 0,--i18n-e9860f92-2fbe-4d23-b728-678a7bb4734e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Getting rid of extreme values
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the **`price`** column.

# COMMAND ----------

display(fixed_price_df.select("price").describe())

# COMMAND ----------

# DBTITLE 0,--i18n-4a8fe21b-1dac-4edf-a0a3-204f170b05c9
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC There are some super-expensive listings, but it's up to the SME (Subject Matter Experts) to decide what to do with them. We can certainly filter the "free" Airbnbs though.
# MAGIC
# MAGIC Let's see first how many listings we can find where the *price* is zero.

# COMMAND ----------

fixed_price_df.filter(col("price") == 0).count()
# this is how to filter the dataframe in spark
# we have only one zero-priced listing

# COMMAND ----------

# DBTITLE 0,--i18n-bf195d9b-ea4d-4a3e-8b61-372be8eec327
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now only keep rows with a strictly positive *price*.

# COMMAND ----------

pos_prices_df = fixed_price_df.filter(col("price") > 0)

# COMMAND ----------

# DBTITLE 0,--i18n-dc8600db-ebd1-4110-bfb1-ce555bc95245
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the *minimum_nights* column:

# COMMAND ----------

display(pos_prices_df.select("minimum_nights").describe())

# COMMAND ----------

display(pos_prices_df
        .groupBy("minimum_nights").count()
        .orderBy(col("count").desc(), col("minimum_nights"))
       )

# COMMAND ----------

pos_prices_df.cube("minimum_nights").count().show()

# COMMAND ----------

# DBTITLE 0,--i18n-5aa4dfa8-d9a1-42e2-9060-a5dcc3513a0d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC A minimum stay of one year seems to be a reasonable limit here. Let's filter out those records where the *minimum_nights* is greater than 365.

# COMMAND ----------

min_nights_filtered = pos_prices_df.filter(col("minimum_nights") <= 365)

print(min_nights_filtered.count())

# COMMAND ----------

# DBTITLE 0,--i18n-25a35390-d716-43ad-8f51-7e7690e1c913
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Handling Null Values
# MAGIC
# MAGIC There are a lot of different ways to handle null values. Sometimes, null can actually be a key indicator of the thing you are trying to predict (e.g. if you don't fill in certain portions of a form, probability of it getting approved decreases).
# MAGIC
# MAGIC Some ways to handle nulls:
# MAGIC * Drop any records that contain nulls
# MAGIC * Numeric:
# MAGIC   * Replace them with mean/median/zero/etc.
# MAGIC * Categorical:
# MAGIC   * Replace them with the mode
# MAGIC   * Create a special category for null
# MAGIC * Use techniques like ALS (Alternating Least Squares) which are designed to impute missing values
# MAGIC   
# MAGIC **If you do ANY imputation techniques for categorical/numerical features, you MUST include an additional field specifying that field was imputed.**
# MAGIC
# MAGIC SparkML's Imputer (covered below) does not support imputation for categorical features.

# COMMAND ----------

print(min_nights_filtered.count())
print(min_nights_filtered.na.drop().count()) # number of rows after dropping out the missing values

print(min_nights_filtered.count()) # by doing hte method "drop", that does not change the saved object "min_nights_df"

# COMMAND ----------

# DBTITLE 0,--i18n-83e56fca-ce6d-4e3c-8042-0c1c7b9eaa5a
# MAGIC %md 
# MAGIC
# MAGIC ### Impute: Cast to Double
# MAGIC
# MAGIC Imputing in the context of data means replacing missing values with something intentional, such as replacing nulls with an average/mean value. SparkML's <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html?highlight=imputer#pyspark.ml.feature.Imputer" target="_blank">Imputer </a> requires all fields to be of type double. Let's cast all integer fields to double.

# COMMAND ----------

min_nights_filtered.schema.fields

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

integer_columns = [x.name for x in min_nights_filtered.schema.fields if x.dataType == IntegerType()] 
# this looks at the schema and finds the columns that are integer

doubles_df = min_nights_filtered

# from the dataset, convert the Integer types into Double types.
for c in integer_columns:
    doubles_df = doubles_df.withColumn(c, col(c).cast("double"))


# COMMAND ----------


columns = "\n - ".join(integer_columns)
print(f"Columns converted from Integer to Double:\n - {columns}")

# COMMAND ----------

[col(c).isNull()]

# COMMAND ----------

doubles_df.schema.fields

# COMMAND ----------

display(doubles_df.select([sum((col(c).isNull()).cast("int")).alias(c) for c in doubles_df.columns]))

# COMMAND ----------

# DBTITLE 0,--i18n-69b58107-82ad-4cec-8984-028a5df1b69e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Add a dummy column to denote the presence of null values before imputing (i.e. 1.0 = Yes, 0.0 = No).

# COMMAND ----------

from pyspark.sql.functions import when

impute_cols = [
    "beds", "review_scores_rating", "reviews_per_month"
]

for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))

# COMMAND ----------

display(doubles_df.describe())

# COMMAND ----------

# DBTITLE 0,--i18n-c88f432d-1252-4acc-8c91-4834c00da789
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Transformers and Estimators
# MAGIC
# MAGIC Spark ML standardizes APIs for machine learning algorithms to make it easier to combine multiple algorithms into a single pipeline, or workflow. Let's cover two key concepts introduced by the Spark ML API: **`transformers`** and **`estimators`**.
# MAGIC
# MAGIC **Transformer**: Transforms one DataFrame into another DataFrame. It accepts a DataFrame as input, and returns a new DataFrame with one or more columns appended to it. Transformers do not learn any parameters from your data and simply apply rule-based transformations. It has a **`.transform()`** method.
# MAGIC
# MAGIC **Estimator**: An algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model. It has a **`.fit()`** method because it learns (or "fits") parameters from your DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC Since the ```Imputer``` only accepts numerical (double) as inputs, I will remove the ```bathrooms_text``` column:

# COMMAND ----------

doubles_df = doubles_df.drop("bathrooms_text")

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)

imputer_model = imputer.fit(doubles_df) # "fit" for computing the median
imputed_df = imputer_model.transform(doubles_df) # "transform" to apply the median and produce the imputed values

# COMMAND ----------

print(imputer.explainParams()) # gives us the parameters for the imputation model

# COMMAND ----------

print(imputed_df.count())

print(imputed_df.na.drop().count())

print(raw_df.count())

# COMMAND ----------

# DBTITLE 0,--i18n-4df06e83-27e6-4cc6-b66d-883317b2a7eb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC OK, our data is cleansed now. Let's save this DataFrame to Delta so that we can start building models with it.

# COMMAND ----------

absolute_dir_path = os.path.abspath("./data/")

# COMMAND ----------

absolute_dir_path = "file:" + absolute_dir_path

# COMMAND ----------

imputed_df.write.format("delta").mode("overwrite").save(f"{absolute_dir_path}/imputed_results") # writing the data

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
