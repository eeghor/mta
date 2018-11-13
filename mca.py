import pandas as pd
from itertools import chain, tee, combinations
from functools import reduce
from operator import mul
from collections import defaultdict
import random
import time
import numpy as np
import copy

def show_time(func):

	"""
	timer decorator
	"""

	def wrapper(*args, **kwargs):

		t0 = time.time()
		v = func(*args, **kwargs)
		print('elapsed time: {:.0f} min {:.0f} sec'.format(*divmod(time.time() - t0, 60)))

		return v

	return wrapper

class MCA:

	def __init__(self, data='data/data.csv'):

		self.data = pd.read_csv(data)

		if not (set(self.data.columns) <= set('path total_conversions total_conversion_value total_null'.split())):
			raise ValueError(f'wrong column names in {data}!')

		# split journey into a list of visited channels
		self.data['path'] = self.data['path'].str.split('>').apply(lambda _: [ch.strip() for ch in _]) 
		# make a sorted list of channel names
		self.channels = sorted(list({ch for ch in chain.from_iterable(self.data['path'])}))
		# add some extra channels
		self.channels_ext = ['<start>'] + self.channels + ['<conversion>', '<null>']
		# make dictionary mapping a channel name to it's index
		self.channels_to_idxs = {c: i for i, c in enumerate(self.channels_ext)}

		self.m = np.zeros(shape=(len(self.channels_ext), len(self.channels_ext)))
		self.removal_effects = defaultdict(float)
		self.trans_probs = defaultdict(float)

		self.C = defaultdict(float)

	def show_data(self):

		print(f'rows: {len(self.data):,}')
		print(f'channels: {len(self.channels)}')
		print(f'conversions: {self.data["total_conversions"].sum():,}')
		print(f'exits: {self.data["total_null"].sum():,}')
		print('sample:')
		print(self.data.head())

		return self

	def normalize_dict(self, d):
		"""
		returns a value-normalized version of dictionary d
		"""
		sum_all_values = sum(d.values())

		for _ in d:
			d[_] = round(d[_]/sum_all_values, 6)

		return d

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

			trans[(ch_list[-1], '<null>')] += nulls
			trans[(ch_list[-1], '<conversion>')] += convs

		return trans

	@show_time
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
			assert np.abs(np.sum(self.m[self.channels_to_idxs[ch]]) - 1) < 1e-6, print(f'row values for channel {ch} don\'t sum up to one!')

		print('ok')

		return self

	@show_time
	def simulate_path(self, n=int(1e6), drop_state=None):

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

	def calculate_removal_effects(self, normalize=True):

		print('calculating removal effects...')

		cvs = defaultdict()  # conversions by channel

		print('no removals...')
		cvs['no_removals'] = self.simulate_path()

		for i, ch in enumerate(self.channels, 1):

			print(f'{i}/{len(self.channels)}: removed channel {ch}...')

			cvs[ch] = self.simulate_path(drop_state=ch)
			self.removal_effects[ch] = round((cvs['no_removals']['<conversion>'] - 
												cvs[ch]['<conversion>'])/cvs['no_removals']['<conversion>'], 6)

		if normalize:
			self.removal_effects = self.normalize_dict(self.removal_effects)

		return self

	def removal_probs(self, drop=None):

		p_full = 0

		print('drop=', drop)

		d_ = self.data[~self.data['path'].apply(lambda x: drop in x) & (self.data['total_conversions'] > 0)]

		p_paths = []

		for ch_list, convs in zip(d_['path'], d_['total_conversions']):
			
			p_path = 1

			for t in self.pairs(['<start>'] + ch_list + ['<conversion>']):

				if t[0] != t[1]:
					p_path *= self.m[self.channels_to_idxs[t[0]]][self.channels_to_idxs[t[1]]]

			p_paths.append(p_path)

			if p_path > 1:
				print('p_path=', p_path)

		p_full = sum(p_paths)
		print(p_full)



		return p_full


	def markov(self, normalize=True):

		self.trans_matrix()
		# self.calculate_removal_effects(normalize=normalize)
		fl = self.removal_probs()

		rp = defaultdict(float)

		for ch in self.channels:
			rp[ch] = (fl - self.removal_probs(drop=ch))/fl

		rp = self.normalize_dict(rp)

		print(rp)


		return self

	def shao(self, normalize=True):

		"""
		probabilistic model by Shao and Li (supposed to be equivalent to Shapley)
		"""

		n_channel = defaultdict(lambda: defaultdict(float))

		# count user conversions and nulls for each visited channel and channel pair

		for ch_list, convs, nulls in zip(self.data['path'], 
											self.data['total_conversions'], 
												self.data['total_null']):

			for ch in ch_list:

				n_channel[ch]['<conversion>'] += convs
				n_channel[ch]['<null>'] += nulls

			path_pairs = set()

			for ctup in combinations(ch_list, 2):

				if ctup[0] != ctup[1]:
					ctup_ = tuple(sorted(ctup))

					if ctup_ not in path_pairs:

						path_pairs.add(ctup_)

						n_channel[ctup_]['<conversion>'] += convs
						n_channel[ctup_]['<null>'] += nulls

		# now calculate conditional probabilities of conversion given exposure to channel or channel pair

		for _ in n_channel:

			n_channel[_]['conv_prob'] = n_channel[_]['<conversion>']/(n_channel[_]['<conversion>'] + n_channel[_]['<null>'])


		# calculate channel contributions

		u = set()

		for ch in self.channels:

			for another_ch in set(self.channels) - {ch}:

				t = tuple(sorted((another_ch, ch)))

				if t not in u:
					u.add(t)

					self.C[ch] += ((n_channel[t]['conv_prob'] - n_channel[ch]['conv_prob'] - n_channel[another_ch]['conv_prob']))

			self.C[ch] = self.C[ch]/(2*(len(self.channels) - 1)) + n_channel[ch]['conv_prob']

		if normalize:
			self.C = self.normalize_dict(self.C)

		return self

	def get_generated_conversions(self, max_subset_size=3):

		self.cc = defaultdict(lambda: defaultdict(float))

		for ch_list, convs, nulls in zip(self.data['path'], 
											self.data['total_conversions'], 
												self.data['total_null']):

			# only look at journeys with conversions
			for n in range(1, max_subset_size + 1):

				for tup in combinations(set(ch_list), n):

					tup_ = tuple(sorted(tup))

					self.cc[tup_]['<conversion>'] += convs
					self.cc[tup_]['<null>'] += nulls

		return self

	def v(self, coalition):
		
		"""
		total number of conversions generated by all subsets of the coalition;
		coalition is a tuple of channels
		"""

		s = len(coalition)

		total_convs = 0

		for n in range(1, s+1):
			for tup in combinations(coalition, n):
				tup_ = tuple(sorted(tup))
				total_convs += self.cc[tup_]['<conversion>']

		return total_convs

	def w(self, s, n):
		
		return np.math.factorial(s)*(np.math.factorial(n - s -1))/np.math.factorial(n)


	def shapley(self, max_coalition_size=2, normalize=True):

		"""
		Shapley model; channels are players, the characteristic function maps a coalition A to the 
		the total number of conversions generated by all the subsets of the coalition

		see https://medium.com/data-from-the-trenches/marketing-attribution-e7fa7ae9e919
		"""

		self.get_generated_conversions(max_subset_size=3)

		self.phi = defaultdict(float)

		for ch in self.channels:
			# all subsets of channels that do NOT include ch
			for n in range(1, max_coalition_size + 1):
				for tup in combinations(set(self.channels) - {ch}, n):
					self.phi[ch] += (self.v(tup + (ch,)) - self.v(tup))*self.w(len(tup), len(self.channels))

		if normalize:
			self.phi = self.normalize_dict(self.phi)

		return self

if __name__ == '__main__':

	mca = MCA()

	mca.markov()

	# # print('Markov:\n', mca.removal_effects)

	# # mca.get_generated_conversions(max_subset_size=3)

	mca.shapley()

	print('Shapley:\n',mca.phi)

	mca.shao()

	print('Shao:\n',mca.C)