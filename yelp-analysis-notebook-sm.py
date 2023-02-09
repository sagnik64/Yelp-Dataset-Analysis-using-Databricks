# Databricks notebook source
#import packages
import pyspark.sql.functions as f
from pyspark.sql.window import Window

# COMMAND ----------

storage_account_name = "sprojectproyelp"
container_name = "targetcontainer"
storage_account_access_key = ""


# COMMAND ----------

# DBTITLE 1,Set ADLS configurations
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net", f"{storage_account_access_key}")

# COMMAND ----------

# DBTITLE 1,List Datasets in ADLS
dbutils.fs.ls(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/")

# COMMAND ----------

# DBTITLE 1,Read yelp datasets in ADLS and convert JSON to parquet for better performance
df_business = spark.read.json(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/yelp_academic_dataset_business.json")
df_business.write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/business.parquet")

df_checkin = spark.read.json(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/yelp_academic_dataset_checkin.json")
df_checkin.write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/checkin.parquet")

df_review = spark.read.json(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/yelp_academic_dataset_review.json")
df_review.write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/review.parquet")

df_tip = spark.read.json(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/yelp_academic_dataset_tip.json")
df_tip.write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/tip.parquet")

df_user = spark.read.json(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/yelp_academic_dataset_user.json")
df_user.write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/user.parquet")

# COMMAND ----------

# DBTITLE 1,Convert JSON to Delta Format
df_business.write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_delta/business.delta")

# COMMAND ----------

# DBTITLE 1,Read parquet file
df_business = spark.read.parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/business.parquet")
df_checkin = spark.read.parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/checkin.parquet")
df_review = spark.read.parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/review.parquet")
df_tip = spark.read.parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/tip.parquet")
df_user = spark.read.parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/json_to_parquet/user.parquet")

# COMMAND ----------

display(df_tip)

# COMMAND ----------

# DBTITLE 1,Total records in each datasets
print("df_business:", df_business.count())
print("df_checkin:", df_checkin.count())
print("df_review:", df_review.count())
print("df_tip:", df_tip.count())
print("df_user:", df_user.count())

# COMMAND ----------

# DBTITLE 1,Extract year and month from date time
df_tip = df_tip.withColumn("tip_year", f.year(f.to_date(f.col("date"))))
df_tip = df_tip.withColumn("tip_month", f.month(f.to_date(f.col("date"))))
display(df_tip)

# COMMAND ----------

# DBTITLE 1,Partition dataset tip by date column
df_tip.write.mode("overwrite").partitionBy("tip_year",
"tip_month").parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/tip_partitioned_by_year_and_month/")

# COMMAND ----------

print("current number of partitions: " + str(df_user.rdd.getNumPartitions()))

df_reduce_part = df_user.coalesce(10)
print("reduced number of partitions after coalesce: " + str(df_reduce_part.rdd.getNumPartitions()))

df_increased_df = df_user.repartition(30)
print("increased number of partitions after repartition: " + str(df_increased_df.rdd.getNumPartitions()))

# COMMAND ----------

df_user.coalesce(10).write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/coalesce/user.parquet")

# COMMAND ----------

df_user.repartition(10).write.mode('overwrite').parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/repartition/user.parquet")

# COMMAND ----------

df_user = spark.read.parquet(f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/repartition/user.parquet")
display(df_user)

# COMMAND ----------

# DBTITLE 1,Creating Temp View for df_user
df_user.createOrReplaceTempView("user");

# COMMAND ----------

# DBTITLE 1,Finding the top 3 users based on their total number of reviews
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   user_id,
# MAGIC   name,
# MAGIC   review_count
# MAGIC FROM
# MAGIC   user
# MAGIC ORDER BY
# MAGIC   review_count DESC
# MAGIC LIMIT
# MAGIC   3;

# COMMAND ----------

# DBTITLE 1,Finding the top 10 users with most number of fans
# MAGIC %sql
# MAGIC SELECT
# MAGIC   user_id, name, fans
# MAGIC FROM
# MAGIC   user
# MAGIC ORDER BY
# MAGIC   fans DESC
# MAGIC LIMIT
# MAGIC  10;

# COMMAND ----------

display(df_business)

# COMMAND ----------

# DBTITLE 1,Analysing top 10 categories by number of reviews
df_business_cat = df_business.groupBy("categories").agg(f.count("review_count").alias("total_review_count"))
df_top_categories = df_business_cat.withColumn("rank", f.row_number().over(Window.orderBy(f.col('total_review_count').desc())))
df_top_categories = df_top_categories.filter(f.col("rank") <= 10)
display(df_top_categories)

# COMMAND ----------

df_business.createOrReplaceTempView("business")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT categories, total_review_count, rn FROM (
# MAGIC     SELECT business_categories.*, 
# MAGIC     ROW_NUMBER() OVER (ORDER BY total_review_count DESC) rn 
# MAGIC     FROM (SELECT categories, count(review_count) AS total_review_count FROM business GROUP BY categories) business_categories
# MAGIC )
# MAGIC WHERE rn <= 10

# COMMAND ----------

# DBTITLE 1,Analysing top businesses which have over 1000 reviews
df_business_reviews = df_business.groupBy("categories").agg(f.count("review_count").alias("total_review_count"))
df_top_business = df_business_reviews.filter(df_business_reviews["total_review_count"] >= 1000).orderBy(f.desc("total_review_count"))
display(df_top_business)

# COMMAND ----------

# DBTITLE 1,Analysing Business Data : Number of restaurants per state
df_num_of_restaurants = df_business.select('state').groupBy('state').count().orderBy(f.desc("count"))
display(df_num_of_restaurants)

# COMMAND ----------

df_business.createOrReplaceTempView("business_restaurant")

# COMMAND ----------

# DBTITLE 1,Analysing top 3 restaurants in each state
# MAGIC %sql
# MAGIC SELECT * FROM (
# MAGIC     SELECT STATE,name,review_count,
# MAGIC     ROW_NUMBER() OVER ( PARTITION BY STATE ORDER BY review_count DESC) rn 
# MAGIC     FROM business_restaurant
# MAGIC )
# MAGIC WHERE rn <= 3

# COMMAND ----------

# DBTITLE 1,Listing the top restaurants in a state by the number of reviews
df_business_Arizona = df_business.filter(df_business['state']=='AZ')
df_Arizona = df_business_Arizona.groupBy("name").agg(f.count("review_count").alias("total_review_count"))
window = Window.orderBy(df_Arizona['total_review_count'].desc())
df_Arizona_best_rest = df_Arizona.select('*', f.rank().over(window).alias('rank')).filter(f.col('rank') <= 10)
display(df_Arizona_best_rest)

# COMMAND ----------

# DBTITLE 1,Numbers of restaurants in Arizona state per city
df_business_Arizona = df_business.filter(df_business['state']=='AZ')
df_business_Arizona = df_business_Arizona.groupBy('city').count().orderBy(f.desc("count"))
display(df_business_Arizona)

# COMMAND ----------

# DBTITLE 1,Select City with highest number of restaurants
window = Window.orderBy(df_business_Arizona['count'].desc())
city_with_max_rest = df_business_Arizona \
                    .select('*', f.rank().over(window).alias('rank')).filter(f.col('rank') <= 1) \
                    .drop('rank')
display(city_with_max_rest)

# COMMAND ----------

# DBTITLE 1,Broadcast Join
from pyspark.sql.functions import broadcast

df_best_restaurants = df_business.join(broadcast(city_with_max_rest),"city", 'inner')

df_best_restaurants = df_best_restaurants.groupBy("name","stars").agg(f.count("review_count").alias("review_count"))

df_best_restaurants = df_best_restaurants.filter(df_best_restaurants["review_count"] >= 10)

df_best_restaurants = df_best_restaurants.filter(df_best_restaurants["stars"] >= 3)

# COMMAND ----------

display(df_best_restaurants)

# COMMAND ----------

# DBTITLE 1,Most rated Italian restaurant in Pheonix
df_business_Pheonix = df_business.filter(df_business.city == 'Phoenix')
df_business_italian = df_business_Pheonix.filter(df_business.categories.contains('Italian'))

# COMMAND ----------

df_best_italian_restaurants = df_business_italian.groupBy("name").agg(f.count("review_count").alias("review_count"))

df_best_italian_restaurants = df_best_italian_restaurants.filter(df_best_italian_restaurants["review_count"] >= 5)

display(df_best_italian_restaurants)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT city FROM business;
