#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.StringSimilarityErrorModule import StringSimilarityErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.SemanticErrorModule import SemanticErrorModule
from activedetect.error_detectors.CharSimilarityErrorModule import CharSimilarityErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

from Benchmark import Benchmark

class USCensusBenchmark(Benchmark):

	def getDataset(self):
		c = CSVLoader()
		return c.loadFile('datasets/adult.data')

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
		return set([i for i,l in enumerate(dataset) if ' ?' in l or ' 99999' in l])



