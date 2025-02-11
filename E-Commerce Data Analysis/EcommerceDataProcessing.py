from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofweek, hour, min, max, count, when, sum
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EcommerceDataStreaming") \
    .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.kafka.bootstrap.servers", "localhost:9092") \
    .enableHiveSupport() \
    .getOrCreate()

# Define Schema for the CSV file.
schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("event_time", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_session", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("category_code", StringType(), True)
])

# Set up Kafka Stream as a DataFrame
df_kafka = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "ecommerce_data") \
    .load()

# Decode the binary message (value) to string and parse CSV data
df = df_kafka.selectExpr("CAST(value AS STRING) as csv_data")

# Spark's CSV parsing capabilities to convert CSV data into a DataFrame with the defined schema
df = df.selectExpr("csv_data") \
       .withColumn("csv_data", spark.read.csv(df["csv_data"], header=True, schema=schema))

# Flatten the CSV data into individual columns
df = df.select("csv_data.*")

# Fill missing values
df = df.fillna({'brand': 'unknown', 'category_code': 'unknown'})

# Convert event_time to timestamp and extract time-based features
df = df.withColumn("event_time", to_timestamp(col("event_time"))) \
       .withColumn("year", year(col("event_time"))) \
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

# JDBC configuration for MySQL
jdbc_url = "jdbc:mysql://localhost:3306/ecommerce_db"
properties = {
    "user": "root",
    "password": "manager",
    "driver": "com.mysql.cj.jdbc.Driver"
}

# Writing the processed data back to MySQL as a streaming sink
query_df = df.writeStream \
    .foreachBatch(lambda batch_df, batch_id: batch_df.write.jdbc(url=jdbc_url, table="processed_data", mode="append", properties=properties)) \
    .outputMode("append") \
    .start()

query_user_engagement = user_engagement.writeStream \
    .foreachBatch(lambda batch_df, batch_id: batch_df.write.jdbc(url=jdbc_url, table="user_engagement", mode="append", properties=properties)) \
    .outputMode("complete") \
    .start()

query_session_revenue = session_revenue.writeStream \
    .foreachBatch(lambda batch_df, batch_id: batch_df.write.jdbc(url=jdbc_url, table="session_revenue", mode="append", properties=properties)) \
    .outputMode("complete") \
    .start()

# Wait for the termination of the streams
query_df.awaitTermination()
query_user_engagement.awaitTermination()
query_session_revenue.awaitTermination()

# Stop Spark session
spark.stop()

