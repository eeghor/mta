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

class MTA:

	def __init__(self, data='data.csv.gz', allow_loops=False, add_timepoints=True, sep=' > '):

		self.data = pd.read_csv(os.path.join('data', data))
		self.sep = sep
		self.NULL = '(null)'
		self.START = '(start)'
		self.CONV = '(conversion)'

		if not (set(self.data.columns) <= set('path total_conversions total_conversion_value total_null exposure_times'.split())):
			raise ValueError(f'wrong column names in {data}!')
		
		if add_timepoints:
			self.add_exposure_times(1)

		if not allow_loops:
			self.remove_loops()

		# we'll work with lists in path and exposure_times from now on
		self.data[['path', 'exposure_times']] = self.data[['path', 'exposure_times']].applymap(lambda _: [ch.strip() for ch in _.split(self.sep.strip())])
		
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

	def add_exposure_times(self, dt=None):

		"""
		generate synthetic exposure times; if dt is specified, the exposures will be dt=1 sec away from one another, otherwise
		we'll generate time spans randomly

		- the times are of the form 2018-11-26T03:54:26.532091+00:00
		"""

		if 'exposure_times' in self.data.columns:
			return self

		ts = []    # this will be a list of time instant lists one per path 

		if dt:

			_t0 = arrow.utcnow()

			self.data['path'].str.split('>') \
				.apply(lambda _: [ch.strip() for ch in _]) \
				.apply(lambda lst: ts.append(self.sep.join([r.format('YYYY-MM-DD HH:mm:ss') 
									for r in arrow.Arrow.range('second', _t0, _t0.shift(seconds=+(len(lst) - 1)))])))

		self.data['exposure_times'] = ts

		return self

	@show_time
	def remove_loops(self):

		"""
		remove transitions from a channel directly to itself, e.g. a > a
		"""

		cpath = []
		cexposure = []

		self.data[['path', 'exposure_times']] = self.data[['path', 'exposure_times']].applymap(lambda _: [ch.strip() for ch in _.split('>')]) 

		for row in self.data.itertuples():

			clean_path = []
			clean_exposure_times = []

			for i, p in enumerate(row.path, 1):

				if i == 1:
					clean_path.append(p)
					clean_exposure_times.append(row.exposure_times[i-1])
				else:
					if p != clean_path[-1]:
						clean_path.append(p)
						clean_exposure_times.append(row.exposure_times[i-1])

			cpath.append(self.sep.join(clean_path))
			cexposure.append(self.sep.join(clean_exposure_times))

		self.data_ = pd.concat([pd.DataFrame({'path': cpath}), 
								self.data[[c for c in self.data.columns if c not in 'path exposure_times'.split()]],
								pd.DataFrame({'exposure_times': cexposure})], axis=1)

		_ = self.data_[[c for c in self.data.columns if c != 'exposure_times']].groupby('path').sum().reset_index()

		self.data = _.join(self.data_[['path', 'exposure_times']].set_index('path'), 
											on='path', how='inner').drop_duplicates(['path'])

		return self

	def normalize_dict(self, d):
		"""
		returns a value-normalized version of dictionary d
		"""
		sum_all_values = sum(d.values())

		for _ in d:
			d[_] = round(d[_]/sum_all_values, 6)

		return d

	@show_time
	def linear(self, share='same', normalize=True):

		"""
		either give exactly the same share of conversions to each visited channel (option share=same) or
		distribute the shares proportionally, i.e. if a channel 1 appears 2 times on the path and channel 2 once
		then channel 1 will receive double credit

		note: to obtain the same result as ChannelAttbribution produces for the test data set, you need to

			- select share=proportional
			- allow loops - use the data set as is without any modifications
		"""

		if share not in 'same proportional'.split():
			raise ValueError('share parameter must be either *same* or *proportional*!')

		self.linear = defaultdict(float)

		for row in self.data.itertuples():

			if row.total_conversions:

				if share == 'same':

					n = len(set(row.path))    # number of unique channels visited during the journey
					s = row.total_conversions/n    # each channel is getting an equal share of conversions

					for c in set(row.path):
						self.linear[c] += s

				elif share == 'proportional':

					c_counts = Counter(row.path)  # count how many times channels appear on this path
					tot_appearances = sum(c_counts.values())

					c_shares = defaultdict(float)

					for c in c_counts:

						c_shares[c] = c_counts[c]/tot_appearances

					for c in set(row.path):

						self.linear[c] += row.total_conversions*c_shares[c]

		if normalize:
			self.linear = self.normalize_dict(self.linear)

		self.attribution['linear'] = self.linear

		return self

	@show_time
	def time_decay(self, count_direction='left', normalize=True):

		"""
		time decay - the closer to conversion was exposure to a channel, the more credit this channel gets

		this can work differently depending how you get timing sorted. 

		example: a > b > c > b > a > c > (conversion)
	
		we can count timing backwards: c the latest, then a, then b (lowest credit) and done. Or we could count left to right, i.e.
		a first (lowest credit), then b, then c. 

		"""

		self.time_decay = defaultdict(float)

		if count_direction not in 'left right'.split():
			raise ValueError('argument count_direction must be *left* or *right*!')

		for row in self.data.itertuples():

			if row.total_conversions:

				channels_by_exp_time = []

				_ = row.path if count_direction == 'left' else row.path[::-1]

				for c in _:
					if c not in channels_by_exp_time:
						channels_by_exp_time.append(c)

				if count_direction == 'right':
					channels_by_exp_time = channels_by_exp_time[::-1]

				# first channel gets 1, second 2, etc.

				score_unit = 1./sum(range(1, len(channels_by_exp_time) + 1))

				for i, c in enumerate(channels_by_exp_time, 1):
					self.time_decay[c] += i*score_unit*row.total_conversions

		if normalize:
			self.time_decay = self.normalize_dict(self.time_decay)

		self.attribution['time_decay'] = self.time_decay

		return self

	@show_time
	def first_touch(self, normalize=True):

		first_touch = defaultdict(int)

		for c in self.channels:

			# total conversions for all paths where the first channel was c
			first_touch[c] = self.data.loc[self.data['path'].apply(lambda _: _[0] == c), 'total_conversions'].sum()

		if normalize:
			first_touch = self.normalize_dict(first_touch)

		self.attribution['first_touch'] = first_touch

		return self

	@show_time
	def last_touch(self, normalize=True):

		last_touch = defaultdict(int)

		for c in self.channels:

			# total conversions for all paths where the last channel was c
			last_touch[c] = self.data.loc[self.data['path'].apply(lambda _: _[-1] == c), 'total_conversions'].sum()

		if normalize:
			last_touch = self.normalize_dict(last_touch)

		self.attribution['last_touch'] = last_touch

		return self

	def pairs(self, lst):

		it1, it2 = tee(lst)
		next(it2, None)

		return zip(it1, it2)

	def count_pairs(self):

		"""
		count how many times channel pairs appear on all recorded customer journey paths
		"""

		c = defaultdict(int)

		for row in self.data.itertuples():

			for ch_pair in self.pairs([self.START] + row.path):
				c[ch_pair] += (row.total_conversions + row.total_null)

			c[(row.path[-1], self.NULL)] += row.total_null
			c[(row.path[-1], self.CONV)] += row.total_conversions

		return c

	def ordered_tuple(self, t):

		"""
		return tuple t ordered 
		"""

		if not isinstance(t, tuple):
			raise TypeError(f'provided value {t} is not tuple!')

		if all([len(t) == 1, t[0] in '(start) (null) (conversion)'.split()]):
			raise Exception(f'wrong transition {t}!')

		if (len(t) > 1) and (t[-1] == self.START): 
			raise Exception(f'wrong transition {t}!')

		if (len(t) > 1) and (t[0] == self.START):
			return (t[0],) + tuple(sorted(list(t[1:])))

		if (len(t) > 1) and (t[-1] in '(null) (conversion)'.split()):
			return tuple(sorted(list(t[:-1]))) + (t[-1],)

		return tuple(sorted(list(t)))

	def trans_matrix(self):

		"""
		calculate transition matrix which will actually be a dictionary mapping 
		a pair (a, b) to the probability of moving from a to b, e.g. T[(a, b)] = 0.5
		"""

		tr = defaultdict(float)

		outs = defaultdict(int)

		# here pairs are unordered
		pair_counts = self.count_pairs()

		for pair in pair_counts:

			outs[pair[0]] += pair_counts[pair]

		for pair in pair_counts:

			tr[pair] = pair_counts[pair]/outs[pair[0]]

		return tr

	@show_time
	def simulate_path(self, trans_mat, drop_channel=None, n=int(1e6)):

		"""
		generate n random user journeys and see where these users end up - converted or not;
		drop_channel is a channel to exclude from journeys if specified
		"""

		outcome_counts = defaultdict(int)

		idx0 = self.c2i[self.START]
		null_idx = self.c2i[self.NULL]
		conv_idx = self.c2i[self.CONV]

		drop_idx = self.c2i[drop_channel] if drop_channel else null_idx

		for _ in range(n):

			stop_flag = None

			while not stop_flag:

				probs = [trans_mat.get((self.i2c[idx0], self.i2c[i]), 0) for i in range(len(self.channels_ext))]

				# index of the channel where user goes next
				idx1 = np.random.choice([self.c2i[c] for c in self.channels_ext], p=probs, replace=False)

				if idx1 == conv_idx:
					outcome_counts[self.CONV] += 1
					stop_flag = True
				elif idx1 in {null_idx, drop_idx}:
					outcome_counts[self.NULL] += 1
					stop_flag = True
				else:
					idx0 = idx1

		return outcome_counts

	def prob_convert(self, trans_mat, drop=None):

		_d = self.data[self.data['path'].apply(lambda x: drop not in x) & (self.data['total_conversions'] > 0)]

		p = 0

		for row in _d.itertuples():

			pr_this_path = []

			for t in self.pairs([self.START] + row.path + [self.CONV]):

				pr_this_path.append(trans_mat.get(t, 0))

			p += reduce(mul, pr_this_path)

		return p

	@show_time
	def markov(self, sim=False, normalize=True):

		markov = defaultdict(float)

		# calculate the transition matrix
		tr = self.trans_matrix()

		if not sim:
			
			p_conv = self.prob_convert(trans_mat=tr)

			for c in self.channels:
				markov[c] = (p_conv - self.prob_convert(trans_mat=tr, drop=c))/p_conv
		else:

			outcomes = defaultdict(lambda: defaultdict(float))
			# get conversion counts when all chennels are in place
			outcomes['full'] = self.simulate_path(trans_mat=tr, drop_channel=None)

			for c in self.channels:

				outcomes[c] = self.simulate_path(trans_mat=tr, drop_channel=c)
				# removal effect for channel c
				markov[c] = (outcomes['full'][self.CONV] - outcomes[c][self.CONV])/outcomes['full'][self.CONV]

		if normalize:
			markov = self.normalize_dict(markov)

		self.attribution['markov'] = markov


		return self

	@show_time
	def shao(self, normalize=True):

		"""
		probabilistic model by Shao and Li (supposed to be equivalent to Shapley); explanation in the original paper may seem rather unclear but
		this https://stats.stackexchange.com/questions/255312/multi-channel-attribution-modelling-using-a-simple-probabilistic-model 
		is definitely helpful
		"""

		r = defaultdict(lambda: defaultdict(float))

		# count user conversions and nulls for each visited channel and channel pair

		for row in self.data.itertuples():

			for n in range(1, 3):

				for ch in combinations(set(row.path), n):
					
					t = self.ordered_tuple(ch)

					r[t][self.CONV] += row.total_conversions
					r[t][self.NULL] += row.total_null

		for _ in r:
			r[_]['conv_prob'] = r[_][self.CONV]/(r[_][self.CONV] + r[_][self.NULL])

		# calculate channel contributions

		self.C = defaultdict(float)

		for row in self.data.itertuples():

			for ch_i in set(row.path):

				if row.total_conversions:

					pc = 0    # contribution for current path

					other_channels = set(row.path) - {ch_i}

					k = 2*len(other_channels) if other_channels else 1 

					for ch_j in other_channels:

						pc += (r[self.ordered_tuple((ch_i, ch_j))]['conv_prob'] - 
														r[(ch_i,)]['conv_prob'] - 
														r[(ch_j,)]['conv_prob'])

					pc = r[(ch_i,)]['conv_prob']  + pc/k

					self.C[ch_i] += row.total_conversions*pc


		if normalize:
			self.C = self.normalize_dict(self.C)

		self.attribution['shao'] = self.C

		return self

	def get_generated_conversions(self, max_subset_size=3):

		self.cc = defaultdict(lambda: defaultdict(float))

		for ch_list, convs, nulls in zip(self.data['path'], 
											self.data['total_conversions'], 
												self.data['total_null']):

			# only look at journeys with conversions
			for n in range(1, max_subset_size + 1):

				for tup in combinations(set(ch_list), n):

					tup_ = self.ordered_tuple(tup)

					self.cc[tup_][self.CONV] += convs
					self.cc[tup_][self.NULL] += nulls

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
				tup_ = self.ordered_tuple(tup)
				total_convs += self.cc[tup_][self.CONV]

		return total_convs

	def w(self, s, n):
		
		return np.math.factorial(s)*(np.math.factorial(n - s -1))/np.math.factorial(n)

	@show_time
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

		self.attribution['shapley'] = self.phi

		return self

	@show_time
	def logistic_regression(self, test_size=0.25, ps=0.5, pc=0.5, normalize=True, n=2000):

		"""
		test_size is the proportion of data set to use as the test set
		ps is the proportion of rows to sample
		pc is the proportion of columns to sample
		n is the number of iterations
		"""

		lr = defaultdict(float)

		# expand the original data set into a feature matrix
		lists_ = []
		flags_ = []

		for i, row in enumerate(self.data.itertuples()):

			for _ in range(row.total_conversions):
				lists_.append({c: 1 for c in row.path})
				flags_.append(1)

			for _ in range(row.total_null):
				lists_.append({c: 1 for c in row.path})
				flags_.append(0)

		data_ = pd.concat([pd.DataFrame(lists_).fillna(0), 
								pd.DataFrame({'conv': flags_})], axis=1).sample(frac=1.0)

		for _ in range(1, n + 1):

			dd = pd.concat([data_[data_['conv'] == 1].sample(frac=0.5), 
							data_[data_['conv'] == 0].sample(frac=0.5)])

			# randomly sample features (channels) and rows (journeys)
			dd = dd.sample(frac=ps)

			dd = pd.concat([dd.drop('conv', axis=1).sample(frac=pc, axis=1), dd['conv']], axis=1)
	
			# split into training/test set
			X_train, X_test, y_train, y_test = train_test_split(dd.drop('conv', axis=1), dd['conv'], 
														test_size=test_size, random_state=36, stratify=dd['conv'])

			# fit logistic regression classifier
			clf = LogisticRegression(random_state=32, solver='lbfgs', fit_intercept=False).fit(X_train, y_train)
	
			# predict conversion labels for the test set
			yh = clf.predict(X_test)

			minc = min(clf.coef_[0])
			maxc = max(clf.coef_[0])

			for c, coef in zip(X_train.columns, [k*k for k in clf.coef_[0]]):
				lr[c] += coef/n

		if normalize:
			lr = self.normalize_dict(lr)

		self.attribution['linreg'] = lr

		return self

	def show(self):

		"""
		show simulation results
		"""

		res = pd.DataFrame.from_dict(mta.attribution)

		print(res)

	def rois(self, attrib, spend, cv):

		"""
		calculate ROIs as suggested in paper
		Geyik et al (2014) - Multi-Touch Attribution Based Budget Allocation in Online Advertising

		attrib is a dictionary of attributions per touch point
		spend is a dictionary of spent dollars per channel 
		cv is the conversion value (in dollars)
		"""	

		roi = defaultdict(float)

		for c in self.tps_by_channel:
			roi[c] = sum([attrib[tp] for tp in self.tps_by_channel[c]])*cv/spend[c]

		return roi

	def pi(self, path, exposure_times, conv_flag, beta_by_channel, omega_by_channel):

		"""

		calculate contribution of channel i to conversion of journey (user) u - (p_i^u) in the paper

		 - path is a list of states that includes (start) but EXCLUDES (null) or (conversion)
		 - exposure_times is list of exposure times
		
		"""

		p = {c: 0 for c in path}    # contributions by channel

		# all contributions are zero if no conversion
		if not conv_flag:
			return p

		dts = [(arrow.get(exposure_times[-1]) - arrow.get(t)).seconds for t in exposure_times]

		_ = defaultdict(float)

		for c, dt in zip(path, dts):
			_[c] += beta_by_channel[c]*omega_by_channel[c]*np.exp(-omega_by_channel[c]*dt)

		for c in _:
			p[c] = _[c]/sum(_.values())

		return p

	def update_coefs(self, beta, omega):

		"""
		return updated beta and omega
		"""

		delta = 1e-3

		beta_num = defaultdict(float)
		beta_den = defaultdict(float)
		omega_den = defaultdict(float)

		for u, row in enumerate(self.data.itertuples()):

			p = self.pi(row.path, row.exposure_times, row.total_conversions, beta, omega)

			r = copy.deepcopy(row.path)

			dts = [(arrow.get(row.exposure_times[-1]) - arrow.get(t)).seconds for t in row.exposure_times]

			while r:

				# pick channels starting from the last one
				c = r.pop()
				dt = dts.pop()

				beta_den[c] += (1.0 - np.exp(-omega[c]*dt))
				omega_den[c] += (p[c]*dt + beta[c]*dt*np.exp(-omega[c]*dt))

				beta_num[c] += p[c]
	
		# now that we gone through every user, update coefficients for every channel

		beta0 = copy.deepcopy(beta)
		omega0 = copy.deepcopy(omega)

		df = []

		for c in self.channels:
			
			beta_num[c] = (beta_num[c] > 1e-6)*beta_num[c]
			beta_den[c] = (beta_den[c] > 1e-6)*beta_den[c]
			omega_den[c] = max(omega_den[c], 1e-6)

			if beta_den[c]:
				beta[c] = beta_num[c]/beta_den[c]

			omega[c] = beta_num[c]/omega_den[c]

			df.append(abs(beta[c] - beta0[c]) < delta)
			df.append(abs(omega[c] - omega0[c]) < delta)

		return (beta, omega, sum(df))

	@show_time
	def additive_hazard(self, epochs=20, normalize=True):

		"""
		additive hazard model as in Multi-Touch Attribution in On-line Advertising with Survival Theory
		"""

		beta = {c: random.uniform(0.001,1) for c in self.channels}
		omega = {c: random.uniform(0.001,1) for c in self.channels}

		for _ in range(epochs):

			beta, omega, h = self.update_coefs(beta, omega)

			if h == 2*len(self.channels):
				print(f'converged after {_ + 1} iterations')
				break

		# time window: take the max time instant across all journeys that converged

		additive_hazard = defaultdict(float)

		for u, row in enumerate(self.data.itertuples()):

			p = self.pi(row.path, row.exposure_times, row.total_conversions, beta, omega)

			for c in p:
				additive_hazard[c] += p[c]

		if normalize:
			additive_hazard = self.normalize_dict(additive_hazard)

		self.attribution['add_haz'] = additive_hazard

		return self

if __name__ == '__main__':

	mta = MTA(allow_loops=False)

	mta.linear(share='proportional') \
			.time_decay(count_direction='right') \
			.shapley() \
			.shao() \
			.first_touch() \
			.last_touch() \
			.markov(sim=False) \
			.logistic_regression() \
			.additive_hazard() \
			.show()
	