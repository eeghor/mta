from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

import re
import os
import json
import builtins
import pandas as pd
from collections import defaultdict


def normalize_dict(d):
	"""
	returns a value-normalized version of dictionary d
	"""
	sum_all_values = builtins.sum(d.values())

	for _ in d:
		d[_] = builtins.round(d[_]/sum_all_values, 6)

	return d

def remove_loops_path(s):

	"""
	input s is a path as a list of channels
	remove transitions from a channel directly to itself, e.g. a > a
	"""
	
	clean_path = []

	for i, c in enumerate(s, 1):

		if i == 1:
			clean_path.append(c)
		else:
			if c != clean_path[-1]:
				clean_path.append(c)


	return clean_path

remove_loops_path_UDF = udf(remove_loops_path, ArrayType(StringType()))

def remove_loops(df):

	df = df.withColumn('path', split(df.path, r'\s*>\s*'))

	df.path = remove_loops_path_UDF(df.path)

	# spit path into a list
	df = df.withColumn('path', concat_ws(' > ', df.path))

	# since now we may have some duplicate paths, we need to remove duplicates and update conversion counts
	df = df.groupBy('path').sum().toDF(*['path', 'total_conversions', 'total_conversion_value', 'total_null'])

	return df

def touch(df, t='first'):

	_touch = defaultdict(int)

	df = remove_loops(df)
	df = df.withColumn('path', split(df.path, r'\s*>\s*'))
	df = df.withColumn('ch_1', df.path.getItem(0) if t == 'first' else df.path.getItem(size(df.path)-1))

	for row in df.select('ch_1', 'total_conversions').groupBy('ch_1').sum().toDF('channel', 'counts').collect():
		_touch[row['channel']] = row['counts']

	return normalize_dict(_touch)

# in pyspark Spark session is readily available as spark
spark = SparkSession.builder.master("local").appName("test session").getOrCreate()

# set a smaller number of executors because this is running locally
spark.conf.set("spark.sql.shuffle.partitions", "4")

# read the data file
df = spark.read.option("inferSchema", "true") \
		.option("header", "true") \
		.csv("data/data.csv.gz")

attribution = defaultdict(lambda: defaultdict(float))

attribution['last_touch'] = touch(df, 'last')
attribution['first_touch'] = touch(df, 'first')

res = pd.DataFrame.from_dict(attribution)

print(res)
