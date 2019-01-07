from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from itertools import chain, tee, combinations

import re
import os
import json
import builtins
import pandas as pd
from collections import defaultdict
from itertools import chain


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

def order_by_exposure_time(s):
	
	clean_path = []

	for i, c in enumerate(s, 1):

		if i == 1:
			clean_path.append(c)
		else:
			if c not in clean_path:
				clean_path.append(c)

	return clean_path

order_by_exposure_time_UDF = udf(order_by_exposure_time, ArrayType(StringType()))
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

keep_unique = udf(lambda s: list(builtins.set(s)), ArrayType(StringType()))
count_unique = udf(lambda s: len(builtins.set(s)), IntegerType())

def linear(df, share='same'):

	_lin = defaultdict(float)

	df = df.withColumn('path', split(df.path, r'\s*>\s*'))
	df = df.withColumn('n', count_unique(df.path))
	df = df.withColumn('s', df.total_conversions/df.n)

	for row in df.select(explode(keep_unique(df.path)).alias('channel'), df.s).groupBy('channel').sum().toDF('channel', 'counts').collect():
		_lin[row['channel']] = row['counts']

	return normalize_dict(_lin)

costs = udf(lambda p, s: {c: i*s/builtins.sum(range(1,len(p) + 1)) for i, c in enumerate(p, 1)}, MapType(StringType(), FloatType()))

def time_decay(df):

	dec_ = defaultdict(float)

	df = df.withColumn('path', split(df.path, r'\s*>\s*'))

	df = df.withColumn('path', order_by_exposure_time_UDF(df.path))

	df = df.withColumn('credits', costs(df.path, df.total_conversions))

	for row in df.select(explode(df.credits)).groupBy('key').sum().toDF('channel', 'counts').collect():
		dec_[row['channel']] = row['counts']

	return normalize_dict(dec_)

pos_creds = udf(lambda channels, n, convs: {c: cr for c, cr in zip(channels, 
						[convs/1.] if n == 1 else [convs/2.]*2 if n == 2 else [0.4*convs] + [0.2*convs]*(n-2) + [0.4*convs])}, MapType(StringType(), FloatType()))

def position_based(df):

	posb = defaultdict(float)

	df = remove_loops(df)

	df = df.withColumn('path', split(df.path, r'\s*>\s*'))

	df = df.withColumn('n', count_unique(df.path))
	df = df.withColumn('cr', pos_creds(df.path, df.n, df.total_conversions))

	for row in df.select(explode(df.cr)).groupBy('key').sum().toDF('channel', 'counts').collect():
		posb[row['channel']] = row['counts']

	return normalize_dict(posb)


def window(path, conversions, nulls):

	aug_path = ['(start)'] + path

	it1, it2 = tee(aug_path)
	next(it2, None)

	c = [f'({it1},{it2})' for it1, it2 in zip(it1, it2)]

	if nulls:
		c.append(f'({path[-1]}, (null))')
	if conversions:
		c.append(f'({path[-1]}, (conversion)')

	return c

window_udf = udf(window, ArrayType(StringType()))

def pair_convs_and_exits(df):

	"""
	return a dictionary that maps each pair of touch points on the path to the number of conversions and
	exits this pair were involved into
	"""

	df = remove_loops(df)

	df = df.withColumn('path', split(df.path, r'\s*>\s*'))

	df = df.withColumn('dicts', window_udf(df.path, df.total_conversions, df.total_null))

	k = defaultdict(lambda: defaultdict(int))

	for row in df.select(explode(df.dicts), df.total_conversions, df.total_null).collect():
		k[row['col']]['conversions'] += row['total_conversions']
		k[row['col']]['nulls'] += row['total_null']

	return k


# in pyspark Spark session is readily available as spark
spark = SparkSession.builder.master("local").appName("test session").getOrCreate()

# set a smaller number of executors because this is running locally
spark.conf.set("spark.sql.shuffle.partitions", "4")

# read the data file
df = spark.read.option("inferSchema", "true") \
		.option("header", "true") \
		.csv("data/data.csv.gz")

c = defaultdict(int)

attribution = defaultdict(lambda: defaultdict(float))

# attribution['last_touch'] = touch(df, 'last')
# attribution['first_touch'] = touch(df, 'first')
# attribution['linear'] = linear(df)
# attribution['time_decay'] = time_decay(df)
# attribution['position_based'] = position_based(df)

# res = pd.DataFrame.from_dict(attribution)

# print(res)

print(pair_convs_and_exits(df))

# print(c)


# df.show(5)
