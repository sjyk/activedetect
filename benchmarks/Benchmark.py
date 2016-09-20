#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from  activedetect.error_detectors.ErrorDetector import ErrorDetector
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
import numpy as np
import pickle

"""
This is an abstract class that defines a benchmark
"""
class Benchmark():

	def getDataset(self):
		raise NotImplemented("Loads the data from some source")

	def getQuantitativeConfig(self):
		raise NotImplemented("Returns config for quantitative detection")

	def getADConfig(self):
		raise NotImplemented("Returns detectors, config for AD")

	def _groundTruth(self, dataset):
		raise NotImplemented("Ground Truth Not Implemented")

	def _quantitative(self, dataset):
		config = self.getQuantitativeConfig()
		
		e = ErrorDetector(dataset, modules=[QuantitativeErrorModule], config=config)
		e.fit()

		return set([error['cell'][0] for error in e])

	def _ad(self, dataset):
		config = self.getADConfig()
		
		e = ErrorDetector(dataset, modules=config[0], config=config[1])
		e.fit()

		return set([error['cell'][0] for error in e])

	def pr(self, gt, s2):
		tp = len(gt.intersection(s2))+0.0
		fp = len([fp for fp in s2 if fp not in gt])
		fn = len([fn for fn in gt if fn not in s2])
		return (tp/(tp + fp), tp/(tp+fn))

	def saveResults(self):
		dataset = self.getDataset()
		gt = self._groundTruth(dataset)
		quantitative = self._quantitative(dataset)
		ad = self._ad(dataset)
		print self.pr(gt, ad), self.pr(gt, quantitative)





