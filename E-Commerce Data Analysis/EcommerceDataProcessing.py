from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofweek, hour,min, max, count, when, sum
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EcommerceDataProcessing") \
    .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .enableHiveSupport() \
    .getOrCreate()

# Load data from HDFS
df = spark.read.csv("hdfs://localhost:9000/user/sunbeam/ecommerce_data/input/2019-Oct.csv", header=True, inferSchema=True)

# Fill missing values
df = df.fillna({'brand': 'unknown', 'category_code': 'unknown'})

# Convert event_time to timestamp and extract time-based features
df = df.withColumn("event_time", to_timestamp(col("event_time")))
df = df.withColumn("year", year(col("event_time"))) \
       .withColumn("month", month(col("event_time"))) \
       .withColumn("day_of_week", dayofweek(col("event_time"))) \
       .withColumn("hour", hour(col("event_time")))

# Session-based features
window_spec = Window.partitionBy("user_session")
df = df.withColumn("session_start", min("event_time").over(window_spec))
df = df.withColumn("session_end", max("event_time").over(window_spec))
df = df.withColumn("session_duration", (col("session_end").cast("long") - col("session_start").cast("long")))
df = df.withColumn("session_event_count", count("*").over(window_spec))

# User engagement metrics
df = df.withColumn("viewed", when(col("event_type") == "view", 1).otherwise(0)) \
       .withColumn("added_to_cart", when(col("event_type") == "cart", 1).otherwise(0)) \
       .withColumn("purchased", when(col("event_type") == "purchase", 1).otherwise(0))

user_engagement = df.groupBy("user_id") \
    .agg(sum("viewed").alias("total_views"),
         sum("added_to_cart").alias("total_add_to_cart"),
         sum("purchased").alias("total_purchases"))

user_engagement = user_engagement.withColumn("view_to_cart_rate", col("total_add_to_cart") / col("total_views")) \
                                 .withColumn("cart_to_purchase_rate", col("total_purchases") / col("total_add_to_cart"))

# Revenue analysis for Average Order Value (AOV)
session_revenue = df.filter(col("event_type") == "purchase") \
    .groupBy("user_session") \
    .agg(sum("price").alias("total_revenue"))

# Write processed data back to HDFS
# df.write.parquet("hdfs://localhost:9000/user/sunbeam/ecommerce_data/input/processed_data.parquet", mode="overwrite")
# user_engagement.write.parquet("hdfs://localhost:9000/user/sunbeam/ecommerce_data/input/user_engagement.parquet", mode="overwrite")
# session_revenue.write.parquet("hdfs://localhost:9000/user/sunbeam/ecommerce_data/input/session_revenue.parquet", mode="overwrite")

jdbc_url = "jdbc:mysql://localhost:3306/ecommerce_db"
properties = {
    "user": "root",
    "password": "manager",
    "driver": "com.mysql.cj.jdbc.Driver"
}

# Write processed data to MySQL
df.write.jdbc(url=jdbc_url, table="processed_data", mode="overwrite", properties=properties)
user_engagement.write.jdbc(url=jdbc_url, table="user_engagement", mode="overwrite", properties=properties)
session_revenue.write.jdbc(url=jdbc_url, table="session_revenue", mode="overwrite", properties=properties)


# Stop Spark session
spark.stop()
