#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.StringSimilarityErrorModule import StringSimilarityErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.SemanticErrorModule import SemanticErrorModule
from activedetect.error_detectors.DistributionErrorModule import DistributionErrorModule
from activedetect.error_detectors.CharSimilarityErrorModule import CharSimilarityErrorModule

from Benchmark import Benchmark

class LABenchmark(Benchmark):

	def getDataset(self):
		c = CSVLoader()
		return c.loadFile('datasets/ipums.la.99')

	def getQuantitativeConfig(self):
		return [{'thresh': 10}]

	def getADConfig(self):
		d_detect = DistributionErrorModule
		q_detect = QuantitativeErrorModule
		s_detect = SemanticErrorModule
		str_detect = StringSimilarityErrorModule
		char_detect = CharSimilarityErrorModule

		config = [{'thresh':20}, 
				  {'thresh': 10}, 
				  {'thresh': 10, 'corpus': 'corpora/text8'}, 
				  {'thresh': 10},
				  {'thresh': 10}]

		return ([d_detect,q_detect, s_detect, str_detect, char_detect], config)

	def _groundTruth(self, dataset):
		return set([i for i,l in enumerate(dataset) if '000000' in l[48] or \
													   '501' in l[55]])



