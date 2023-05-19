
from setuptools import setup
from setuptools import find_packages

long_description = '''
NetworkAnalysis is a package that facilitates the analysis of omics data using networks.
and is distributed under the MIT license.
'''

setup(name='NetworkAnalysis',
      version='0.0.1',
      description='Python framework for graph handling and graph representation learning',
      long_description=long_description,
      author='Maarten Larmuseau & Pieter-Paul Strybol',
      author_email='pieterpaul.strybol@ugent.be, maarten.larmuseau@ugent.be',
      url='https://github.com/pstrybol/NetworkAnalysis',
      license='MIT',
      install_requires=[
			'scikit-learn',
			'matplotlib',
			'lifelines',
			'networkx',
			'umap',
			'statsmodels',
			'seaborn',
			'joblib',
			'gseapy',
			'pytest',
			'coverage',
			'coverage_badge',
            'gensim',
					],

      classifiers=[
         "Development Status :: 3 - Alpha",
      ],
packages=find_packages())
