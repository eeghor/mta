from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

import re
import os
import json
import builtins

# in pyspark Spark session is readily available as spark
spark = SparkSession.builder.master("local").appName("test session").getOrCreate()

# set a smaller number of executors because this is running locally
spark.conf.set("spark.sql.shuffle.partitions", "4")

# read the data file
df = spark.read.option("inferSchema", "true") \
		.option("header", "true") \
		.csv("data/data.csv.gz")

# spit path into a list
df = df.withColumn('path', split(df.path, r'\s*>\s*'))

df.show(n=3)