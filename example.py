#!/usr/bin/env python

from loaders.csv_loader import CSVLoader
from error_detectors.ErrorDetector import ErrorDetector
from model_based.SafeSetFilter import SafeSetFilter
from model_based.HardFilter import HardFilter
import numpy as np

"""
Model Free
"""
"""
c = CSVLoader()
loadedData = c.loadFile('datasets/adult.data')

e = ErrorDetector(loadedData)

e.fit()

for error in e:
	print error

"""

"""
Model Based 1
"""

c = CSVLoader()
loadedData = c.loadFile('datasets/adult.data')

e = ErrorDetector(loadedData)

import pickle
m = pickle.load(open('datasets/adult-rl-misp.p','rb'))

h = HardFilter(e, m)

h.fit()

for error in h:
	print error

exit()


"""
Model Based
"""

c = CSVLoader()
loadedData = c.loadFile('datasets/adult.data')

e = ErrorDetector(loadedData)

import pickle
m = pickle.load(open('datasets/adult-rl-misp.p','rb'))

unlabeleddataset = [d[0:len(d)-1] for d in loadedData]
labels = np.array([int('<' in d[-1]) for d in loadedData])

h = SafeSetFilter(e, m, unlabeleddataset, labels)

h.fit()

for error in h:
	print error



