import pandas as pd
from itertools import chain
from collections import defaultdict, Counter

class MCA:

	def __init__(self, data='data/data.csv'):

		self.data = pd.read_csv(data)

		assert set(self.data.columns) <= set('path total_conversions total_conversion_value total_null'.split()), \
						print(f'wrong column names in {data}!')

		self.total_conversions = self.data['total_conversions'].sum()

		self.channels = {ch for ch in chain.from_iterable(self.data['path'].str.split('>').apply(lambda _: [w.strip() for w in _]))}

		print(f'channels: {len(self.channels)}')
		print(f'conversions: {self.total_conversions}')

	def heuristic_models(self, normalized=True):

		"""
		calculate channel contributions assuming the last and first touch attribution
		"""

		self.first_touch = []
		self.last_touch = []

		for c in self.channels:

			self.first_touch.append((c, self.data.loc[self.data['path'].str.strip().str.startswith(c), 'total_conversions'].sum()))
			self.last_touch.append((c, self.data.loc[self.data['path'].str.strip().str.endswith(c), 'total_conversions'].sum()))

		# rank from high to low
		self.first_touch = sorted(self.first_touch, key=lambda x: x[1], reverse=True)
		self.last_touch = sorted(self.last_touch, key=lambda x: x[1], reverse=True)

		# divide by total conversions if needed
		if normalized:
			self.first_touch

		return self

	def markov_model(self):

		


if __name__ == '__main__':

	mca = MCA().heuristic_models()

	print('first touch:')
	print(mca.first_touch)

	print('last touch:')
	print(mca.last_touch)

