import pandas as pd
from itertools import chain, tee, combinations
from functools import reduce, wraps
from operator import mul
from collections import defaultdict, Counter
import random
import time
import numpy as np
import copy
import json
import os
import sys
import json
import arrow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def show_time(func):

	"""
	timer decorator
	"""

	@wraps(func)
	def wrapper(*args, **kwargs):

		t0 = time.time()

		print(f'running {func.__name__}.. ', end='')
		sys.stdout.flush()

		v = func(*args, **kwargs)

		m, s = divmod(time.time() - t0, 60)

		st = 'elapsed time:'

		if m:
			st += ' ' + f'{m:.0f} min'
		if s:
			st += ' ' + f'{s:.3f} sec'

		print(st)

		return v

	return wrapper

@show_time
def remove_loops(data, sep=' > ', dedupe=True):

		"""
		remove transitions from a channel directly to itself, e.g. a > a
		"""
		data = data.copy()
		cpath = []
		data['path'] = data['path'].apply(lambda _: [ch.strip() for ch in _.split('>')]) 

		for row in data.itertuples():

			clean_path = []

			for i, p in enumerate(row.path, 1):

				if i == 1:
					clean_path.append(p)
				else:
					if p != clean_path[-1]:
						clean_path.append(p)

			cpath.append(sep.join(clean_path))

		data_ = pd.concat([pd.DataFrame({'path': cpath}), 
								data[[c for c in data.columns if c != 'path']]], axis=1)

		if dedupe:
			_ = data_.groupby('path').sum().reset_index()

			data = _.join(data_[['path']].set_index('path'), 
											on='path', how='inner').drop_duplicates(['path'])
			return data

		else:
			return data_
@show_time
def data_pipeline(data, allow_loops=False, sep=' > ', dedupe=True):
	data = data.copy()
	if not allow_loops:
		data = remove_loops(data, sep, dedupe)
	data['path'] = data['path'].apply(lambda _: [ch.strip() for ch in _.split(sep.strip())])
	return data

class MTA():
	@show_time
	def __init__(self, df=pd.DataFrame(), data='data.csv.gz', allow_loops=False, sep=' > '):

		if len(df) > 0:
			self.data = df
		else:
			self.data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data', data))
		self.sep = sep
		self.allow_loops = allow_loops
		self.NULL = '(null)'
		self.START = '(start)'
		self.CONV = '(conversion)'
		self.model_params = {}

		if not (set(self.data.columns) <= set('path total_conversions total_conversion_value total_null'.split())):
			raise ValueError(f'wrong column names in {data}!')
		
		self.data = data_pipeline(self.data, allow_loops, sep, dedupe=True)
		
		# make a sorted list of channel names
		self.channels = sorted(list({ch for ch in chain.from_iterable(self.data['path'])}))
		# add some extra channels
		self.channels_ext = [self.START] + self.channels + [self.CONV, self.NULL]
		# make dictionary mapping a channel name to it's index
		self.c2i = {c: i for i, c in enumerate(self.channels_ext)}
		# and reverse
		self.i2c = {i: c for c, i in self.c2i.items()}

		self.removal_effects = defaultdict(float)
		# touch points by channel
		self.tps_by_channel = {'c1': ['beta', 'iota', 'gamma'], 
								'c2': ['alpha', 'delta', 'kappa', 'mi'],
								'c3': ['epsilon', 'lambda', 'eta', 'theta', 'zeta']}

		self.attribution = defaultdict(lambda: defaultdict(float))

	def __repr__(self):

		return f'{self.__class__.__name__} with {len(self.channels)} channels: {", ".join(self.channels)}'


	def normalize_dict(self, d):
		"""
		returns a value-normalized version of dictionary d
		"""
		sum_all_values = sum(d.values())

		for _ in d:
			d[_] = round(d[_]/sum_all_values, 6)

		return d

	def ordered_tuple(self, t):

		"""
		return tuple t ordered 
		"""

		sort = lambda t: tuple(sorted(list(t)))

		return (t[0],) + sort(t[1:]) if (t[0] == self.START) and (len(t) > 1) else sort(t)

	@show_time
	def shao_input(self):

		"""
		probabilistic model by Shao and Li (supposed to be equivalent to Shapley); explanation in the original paper may seem rather unclear but
		this https://stats.stackexchange.com/questions/255312/multi-channel-attribution-modelling-using-a-simple-probabilistic-model 
		is definitely helpful
		"""

		r = defaultdict(lambda: defaultdict(float))

		# count user conversions and nulls for each visited channel and channel pair

		for row in self.data.itertuples():

			for n in range(1, 3):

				# # combinations('ABCD', 2) --> AB AC AD BC BD CD
				
				for ch in combinations(set(row.path), n):
					
					t = self.ordered_tuple(ch)

					r[t][self.CONV] += row.total_conversions
					r[t][self.NULL] += row.total_null

		for _ in r:
			r[_]['conv_prob'] = r[_][self.CONV]/(r[_][self.CONV] + r[_][self.NULL])

		self.model_params['shao'] = r
		return self

	@show_time
	def shao_output(self, data=None):
		if data is None:
			data = self.data
		else:
			data = data_pipeline(data, allow_loops=self.allow_loops, sep=self.sep, dedupe=False)
		data['shao_attribution'] = np.nan
		# self.C = defaultdict(float)
		i = 0
		for row in data.itertuples():
			row_contribution = {}
			for ch_i in set(row.path):
				pc = 0    # contribution for current path

				other_channels = set(row.path) - {ch_i}

				k = 2*len(other_channels) if other_channels else 1 

				for ch_j in other_channels:

					pc += (self.model_params['shao'][self.ordered_tuple((ch_i, ch_j))]['conv_prob'] - 
													self.model_params['shao'][(ch_i,)]['conv_prob'] - 
													self.model_params['shao'][(ch_j,)]['conv_prob'])

				pc = self.model_params['shao'][(ch_i,)]['conv_prob']  + pc/k
				# self.C[ch_i] += row.total_conversions*pc
				row_contribution[ch_i] = max([0,pc]) # store contribution for each channel on each row
			# must normalize contributions for each row so they add up to 1
			try:
				row_contribution = self.normalize_dict(row_contribution)
			except ZeroDivisionError:
				row_contribution = {}
			data['shao_attribution'].iloc[i] = json.dumps(row_contribution) # output row-level attribution
			i += 1

		data['shao_attribution'] = data['shao_attribution'].apply(json.loads)

		# if normalize:
		# 	self.C = self.normalize_dict(self.C)
		# if data==self.data:
		# 	self.
		# self.attribution['shao'] = self.C

		return data

if __name__ == '__main__':

	mta = MTA(data='data.csv.gz', allow_loops=False)

	(mta
		# .linear(share='proportional') 
		# .time_decay(count_direction='right') 
		# .shapley() 
		.shao_input() 
		.shao_output() 
		# .first_touch() 
		# .position_based() 
		# .last_touch() 
		# .markov(sim=False) 
		# .logistic_regression() 
		# .additive_hazard() 
		.show())
	