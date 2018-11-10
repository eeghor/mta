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

		self.total_conversions = self.data['total_conversions'].sum()

		# split into a list and attach start labels
		self.data['path'] = self.data['path'].str.split('>').apply(lambda _: ['<start>'] + [w.strip() for w in _]) 

		self.channels = ['<start>'] + sorted(list({ch for ch in chain.from_iterable(self.data['path'])} - {'<start>'})) + ['<conversion>', '<null>']
		self.ch_index = {c: i for i, c in enumerate(self.channels)}

		print(f'channels ({len(self.channels)}): {self.ch_index}')
		print(f'conversions: {self.total_conversions}')

		self.m = np.zeros(shape=(len(self.channels), len(self.channels)))

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
			outs = ways_from.get(pair[0], 0)
			self.trans_probs[pair] = pair_counts[pair]/outs if outs else 0

		for p in self.trans_probs:
			self.m[self.ch_index[p[0]]][self.ch_index[p[1]]] = self.trans_probs[p]

		return self

	def prob_conv_by_starting_channel(self, start_channel=None, drop_channel=None):

		conv_probs_by_path = []

		for ch_list, convs, nulls in zip(self.data['path'], 
											self.data['total_conversions'], 
													self.data['total_null']):

			if convs and (ch_list[1] == start_channel) and (drop_channel not in ch_list):

				conv_probs_by_path.append(reduce(mul, [self.trans_probs[pair] 
								for pair in self.pairs(ch_list + ['<conversion>']) if pair[0] != pair[1]]))

		return sum(conv_probs_by_path)

	def simulate_path(self, n, drop_state=None):

		conv_or_null = defaultdict(int)
		channel_idxs = list(self.ch_index.values())

		null_idx = self.ch_index['<null>']

		m = copy.copy(self.m)

		if drop_state:

			drop_idx = self.ch_index[drop_state]
			# no exit from this state, i.e. it becomes <null>
			m[drop_idx] = 0

		else:

			drop_idx = null_idx

		for _ in range(n):

			init_idx = self.ch_index['<start>']
			final_state = None

			while not final_state:

				next_idx = np.random.choice(channel_idxs, p=m[init_idx])

				if next_idx == self.ch_index['<conversion>']:
					conv_or_null['<conversion>'] += 1
					final_state = True
				elif next_idx in {null_idx, drop_idx}:
					conv_or_null['<null>'] += 1
					final_state = True
				else:
					init_idx = next_idx

		return conv_or_null



	def removal_effects(self):

		overall_prob = []
		prob_ch = defaultdict(list)
		prob_ch_av = defaultdict(float)

		for _ in range(10000):

			if (_%1000) and (_>0) == 0:
				print(_)

			rch = random.randint(0, len(self.channels) - 1)
			overall_prob.append(self.prob_conv_by_starting_channel(start_channel=self.channels[rch]))

			for ch in self.channels:
				prob_ch[ch].append(self.prob_conv_by_starting_channel(start_channel=self.channels[rch], drop_channel=ch))

		op = reduce(lambda x,y: x+y, overall_prob)/len(overall_prob)

		for ch in self.channels:
			prob_ch_av[ch] = reduce(lambda x,y: x+y, prob_ch[ch])/len(prob_ch[ch])

		print('op=', op)

		for ch in prob_ch_av:
			prob_ch_av[ch] = (op - prob_ch_av[ch])/op

		print(prob_ch_av)

		reffs = defaultdict(float)

		for ch in prob_ch_av:
			reffs[ch] = prob_ch_av[ch]/sum(list(prob_ch_av.values()))

		print(reffs)

		


if __name__ == '__main__':

	mca = MCA().heuristic_models()

	print('channels:', mca.channels)

	print('first touch:')
	print(mca.first_touch)

	print('last touch:')
	print(mca.last_touch)

	mca.trans_matrix()

	conversions_by_channel = defaultdict()
	
	t0 = time.time()
	print(mca.simulate_path(n=1000000, drop_state='eta'))
	print('elapsed time: {:0.1f} sec'.format(time.time() - t0))

	# t0 = time.time()
	# mca.removal_effects()

	# print(time.time() - t0)

