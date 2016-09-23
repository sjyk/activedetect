#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.StringSimilarityErrorModule import StringSimilarityErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.SemanticErrorModule import SemanticErrorModule
from activedetect.error_detectors.DistributionErrorModule import DistributionErrorModule
from activedetect.error_detectors.CharSimilarityErrorModule import CharSimilarityErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

from Benchmark import Benchmark

class NFLBenchmark(Benchmark):

	def getDataset(self):
		c = CSVLoader()
		return c.loadFile('datasets/nflplaybyplay2015.csv')

	def getQuantitativeConfig(self):
		return [{'thresh': 10}]

	def getADConfig(self):
		q_detect = QuantitativeErrorModule
		s_detect = SemanticErrorModule
		str_detect = StringSimilarityErrorModule
		char_detect = CharSimilarityErrorModule
		punc_detect = PuncErrorModule

		config = [{'thresh': 10}, 
				  {'thresh': 10, 'corpus': 'corpora/text8'}, 
				  {'thresh': 10},
				  {'thresh': 10},
				  {}]

		return ([q_detect, s_detect, str_detect, char_detect, punc_detect], config)

	def _groundTruth(self, dataset):
		return set([i for i,l in enumerate(dataset) if 'Timeout' in l[19] or \
													   'END' in l[19] or \
													   self._na_count(l) > 16 or \
													   'Kickoff' in l[28] or \
													   'Extra Point' in l[28] or \
													   'Two-Minute' in l[19]])

	def _na_count(self, l):
		return len([c for c in l if 'NA' in c])


