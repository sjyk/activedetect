#!/usr/bin/env python

from loaders.csv_loader import CSVLoader
from error_detectors.EnsembleErrorDetector import EnsembleErrorDetector
from model_based.SafeSetFilter import SafeSetFilter
import numpy as np

"""
c = CSVLoader()
loadedData = c.loadFile('datasets/nflplaybyplay2015.csv')
t = LoLTypeInference()
print t.getDataTypes(loadedData[2])
"""

"""
c = CSVLoader()
loadedData = c.loadFile('datasets/adult.data')

e = EnsembleErrorDetector(loadedData)

e.fit()

for error in e:
	print error
"""

c = CSVLoader()
loadedData = c.loadFile('datasets/adult.data')

e = EnsembleErrorDetector(loadedData)

import pickle
m = pickle.load(open('datasets/adult-rl-misp.p','rb'))

unlabeleddataset = [d[0:len(d)-1] for d in loadedData]
labels = np.array([int('<' in d[-1]) for d in loadedData])

h = SafeSetFilter(e, m, unlabeleddataset, labels)

h.fit()

for error in h:
	print error



