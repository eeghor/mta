import pandas as pd
from itertools import chain, tee
from functools import reduce
from operator import mul
from collections import defaultdict, Counter

class MCA:

	def __init__(self, data='data/data.csv'):

		self.data = pd.read_csv(data)

		assert set(self.data.columns) <= set('path total_conversions total_conversion_value total_null'.split()), \
						print(f'wrong column names in {data}!')

		self.total_conversions = self.data['total_conversions'].sum()

		# split into a list and attach start labels
		self.data['path'] = self.data['path'].str.split('>').apply(lambda _: ['<start>'] + [w.strip() for w in _]) 

		self.channels = list({ch for ch in chain.from_iterable(self.data['path'])} - {'<start>'})

		print(self.data.head())
		print(f'channels: {len(self.channels)}')
		print(f'conversions: {self.total_conversions}')

		self.trans_probs = defaultdict(float)

	def heuristic_models(self, normalized=True):

		"""
		calculate channel contributions assuming the last and first touch attribution
		"""

		self.first_touch = []
		self.last_touch = []

		for c in self.channels:

			self.first_touch.append((c, self.data.loc[self.data['path'].apply(lambda _: _[1] == c), 'total_conversions'].sum()))
			self.last_touch.append((c, self.data.loc[self.data['path'].apply(lambda _: _[-1] == c), 'total_conversions'].sum()))

		# rank from high to low
		self.first_touch = sorted(self.first_touch, key=lambda x: x[1], reverse=True)
		self.last_touch = sorted(self.last_touch, key=lambda x: x[1], reverse=True)

		return self

	def pairs(self, lst):

		it1, it2 = tee(lst)
		next(it2, None)

		return zip(it1, it2)

	def count_pairs(self):

		# ignore loops

		trans = defaultdict(int)

		for ch_list, convs, nulls in zip(self.data['path'], 
											self.data['total_conversions'], 
												self.data['total_null']):

			for t in self.pairs(ch_list):

				if t[0] != t[1]:
					trans[t] += (convs + nulls)

			if nulls:
				trans[(ch_list[-1], '<null>')] += nulls
			if convs:
				trans[(ch_list[-1], '<conversion>')] += convs

		return trans


	def trans_matrix(self):

		ways_from = defaultdict(int)

		pair_counts = self.count_pairs()

		for pair in pair_counts:
			ways_from[pair[0]] += pair_counts[pair]

		for pair in pair_counts:
			self.trans_probs[pair] = pair_counts[pair]/ways_from[pair[0]]

		return self

	def prob_conversion(self):

		conv_probs_by_path = []

		for ch_list, convs, nulls in zip(self.data['path'], 
											self.data['total_conversions'], 
													self.data['total_null']):

			if convs:
				conv_probs_by_path.append(reduce(mul, [self.trans_probs[pair] for pair in self.pairs(ch_list + ['<conversion>'])]))

		return sum(conv_probs_by_path)


if __name__ == '__main__':

	mca = MCA().heuristic_models()

	print('channels:', mca.channels)

	print('first touch:')
	print(mca.first_touch)

	print('last touch:')
	print(mca.last_touch)

	mca.trans_matrix()

	print({p: mca.trans_probs[p] for p in mca.trans_probs if p[0] == 'gamma'})

	print('prob_conversion = ', mca.prob_conversion())

