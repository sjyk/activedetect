#!/usr/bin/env python

from loaders.csv_loader import CSVLoader
from error_detectors.ErrorDetector import ErrorDetector
from model_based.SafeSetFilter import SafeSetFilter
from model_based.HardFilter import HardFilter
from error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from error_detectors.StringSimilarityErrorModule import StringSimilarityErrorModule
import numpy as np

#get dataset from https://www.dropbox.com/s/71pal0jo7qioxcd/nflplaybyplay2015.csv?dl=0

"""
Model Free
"""
"""
c = CSVLoader()
loadedData = c.loadFile('datasets/nflplaybyplay2015.csv')

e = ErrorDetector(loadedData)

e.fit()

for error in e:
	print error
"""

