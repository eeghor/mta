from setuptools import setup
import os

setup(name='mta',
      version='0.0.1',
      description='Multi-Touch Attrobution',
      classifiers=[
      	'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6'
      ],
      url='https://github.com/eeghor/mta',
      author='Igor Korostil',
      author_email='eeghor@gmail.com',
      license='MIT',
      packages=['mta'],
      install_requires=['unidecode'],
      python_requires='>=3.6',
      package_data={'mta': ['data/*.csv']},
      keywords='attribution marketing')