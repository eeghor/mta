import pandas as pd
from itertools import chain, tee
from functools import reduce
from operator import mul
from collections import defaultdict, Counter
import random
import time
import numpy as np
import copy

class MCA:

	def __init__(self, data='data/data.csv'):

		self.data = pd.read_csv(data)

		assert set(self.data.columns) <= set('path total_conversions total_conversion_value total_null'.split()), \
						print(f'wrong column names in {data}!')

		# split journey into a list of visited channels
		self.data['path'] = self.data['path'].str.split('>').apply(lambda _: [ch.strip() for ch in _]) 
		# make a sorted list of channel names
		self.channels = sorted(list({ch for ch in chain.from_iterable(self.data['path'])}))
		# add some extra channels
		self.channels_ext = ['<start>'] + self.channels + ['<conversion>', '<null>']
		# make dictionary mapping a channel name to it's index
		self.channels_to_idxs = {c: i for i, c in enumerate(self.channels_ext)}

		print(f'rows: {len(self.data):,}')
		print(f'channels: {len(self.channels)}')
		print(f'conversions: {self.data["total_conversions"].sum():,}')
		print(f'exits: {self.data["total_null"].sum():,}')

		self.m = np.zeros(shape=(len(self.channels_ext), len(self.channels_ext)))

		self.trans_probs = defaultdict(float)

		random.seed(1)

	def heuristic_models(self):

		"""
		calculate channel contributions assuming the last and first touch attribution
		"""

		self.first_touch = []
		self.last_touch = []

		for c in self.channels_ext:

			self.first_touch.append((c, self.data.loc[self.data['path'].apply(lambda _: _[0] == c), 'total_conversions'].sum()))
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

			for t in self.pairs(['<start>'] + ch_list):

				if t[0] != t[1]:
					trans[t] += (convs + nulls)

			if nulls:
				trans[(ch_list[-1], '<null>')] += nulls
			if convs:
				trans[(ch_list[-1], '<conversion>')] += convs

		return trans


	def trans_matrix(self):

		print('estimating transition matrix...', end='')

		ways_from = defaultdict(int)

		pair_counts = self.count_pairs()

		for pair in pair_counts:

			ways_from[pair[0]] += pair_counts[pair]

		for pair in pair_counts:

			outs = ways_from.get(pair[0], 0)
			self.trans_probs[pair] = pair_counts[pair]/outs if outs else 0

		for p in self.trans_probs:

			idx_channel_from = self.channels_to_idxs[p[0]]
			idx_channel_to = self.channels_to_idxs[p[1]]

			self.m[idx_channel_from][idx_channel_to] = self.trans_probs[p]

		# check if sums of rows are equal to one
		for ch in self.channels:
			assert np.abs(np.sum(self.m[self.channels_to_idxs[ch]]) - 1) < 1e-6, print(f'row value for channel {ch} don\'t sum up to one!')

		print('ok')

		return self

	def simulate_path(self, n=1e6, drop_state=None):

		conv_or_null = defaultdict(int)
		channel_idxs = list(self.channels_to_idxs.values())

		null_idx = self.channels_to_idxs['<null>']

		m = copy.copy(self.m)

		if drop_state:

			drop_idx = self.channels_to_idxs[drop_state]
			# no exit from this state, i.e. it becomes <null>
			m[drop_idx] = 0

		else:

			drop_idx = null_idx

		for _ in range(n):

			init_idx = self.channels_to_idxs['<start>']
			final_state = None

			while not final_state:

				next_idx = np.random.choice(channel_idxs, p=m[init_idx], replace=False)

				if next_idx == self.channels_to_idxs['<conversion>']:
					conv_or_null['<conversion>'] += 1
					final_state = True
				elif next_idx in {null_idx, drop_idx}:
					conv_or_null['<null>'] += 1
					final_state = True
				else:
					init_idx = next_idx

		return conv_or_null

	def removal_effects(self):

		print('calculating removal effects...')

		conversions_by_channel = defaultdict()
		removal_effects = defaultdict(float)

		print('no removals...', end='')
		conversions_by_channel['no_removals'] = self.simulate_path()
		print('ok')

		for i, ch in enumerate(self.channels, 1):
			print(f'{i}/{len(self.channels)}: removed channel {ch}...', end='')
			conversions_by_channel[ch] = self.simulate_path(drop_state=ch)
			print('ok')
			removal_effects[ch] = (conversions_by_channel['no_removals']['<conversion>'] - conversions_by_channel[ch]['<conversion>'])/conversions_by_channel['no_removals']['<conversion>']

		for ch in self.channels:
			print(f'{ch}: {removal_effects[ch]}')

		return self


if __name__ == '__main__':

	mca = MCA() \
			.heuristic_models() \
			.trans_matrix() \
			.removal_effects()