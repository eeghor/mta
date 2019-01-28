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


class BaselineModel:

	"""
	basic class implementing functionality commonly required by attribution models
	"""

	def __init__(self):

		self.NULL = '(null)'
		self.START = '(start)'
		self.CONV = '(conversion)'

	def load(self, file):

		"""
		load data; this method needs to support ingestion from several possible sources:
			- local hard drive
			- buckets (S3 or GC)
		"""

		self.sep = ','
		self.data = pd.read_csv(file)

		self.required_columns = set('path total_conversions total_conversion_value total_null exposure_times'.split())

		if not (set(self.data.columns) <= self.required_columns):  # note: ok to have extra columns
			raise ValueError(f'some required column names are missing!')

	def normalize_dict(d):
	"""
	returns a value-normalized version of dictionary d
	"""
	sum_all_values = builtins.sum(d.values())

	for _ in d:
		d[_] = builtins.round(d[_]/sum_all_values, 6)

	return d


class Shapley(BaselineModel):

	def __init__(self):
		pass

		
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

def ord_tuple(t, start='(start)'):

	"""
	return tuple t ordered 
	"""

	sort = lambda t: tuple(sorted(list(t)))

	return (t[0],) + sort(t[1:]) if (t[0] == start) and (len(t) > 1) else sort(t)


def outcome_counts(tp_list, convs, nulls, nc=3, count_duplicates=False):

	"""
	calculate the sum of all conversions and exits (nulls) associated with presence
	of various combination of touch points on a path tp_list

	inputs:
	-------

		tp_list: a list of touch points, e.g. [alpha, beta, gamma, alpha, mu, ...]
		convs: total number of conversions for this path
		nulls: total number of nulls for this path
		nc: length of element subsequences for combinations, e.g. 2 for pairs, 1 for singles, etc.
		count_duplicates: if True, count combinations

	output:
	------
		a dictionary mapping combinations to counts of conversions and nulls and probabilities of conversion, 
		e.g. {(alpha, gamma): {'cs': 3, 
								'ns': 6}, ...}
	"""

	dedupl_tp_list = [tp_list[0]]

	if not count_duplicates:
		for _ in tp_list[1:]:
			if _ not in dedupl_tp_list[-1]:
				dedupl_tp_list.append(_)
		tp_list = dedupl_tp_list

	r = defaultdict(lambda: defaultdict(float))

	for n in range(1, nc+1):

		# combinations('ABCD', 2) --> AB AC AD BC BD CD
		for c in combinations(tp_list, n):
			
			t = ord_tuple(c)  # tuple(sorted(list(t)))

			if t != ('(start)',):
				r[t]['cs'] += convs
				r[t]['ns'] += nulls

	return r

def trans_matrix(k):

	"""
	calculate transition matrix which will actually be a dictionary mapping 
	a pair (a, b) to the probability of moving from a to b, e.g. T[(a, b)] = 0.5
	"""

	tr = defaultdict(float)

	outs = defaultdict(int)

	for pair in k:

		outs[pair[0]] += k[pair]['conversions'] + k[pair]['null']

	for pair in k:

		tr[pair] = (k[pair]['conversions'] + k[pair]['null'])/outs[pair[0]]

	return tr


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

# k = pair_convs_and_exits(df)
# t = trans_matrix(k)

# print(t)


# df.show(5)
o = combination_contributions(['(start)', 'alpha', 'gamma', 'beta', 'gamma', 'kappa'], 3, 16, nc=3)
print(o)
