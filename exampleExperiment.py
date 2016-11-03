#!/usr/bin/env python
from activedetect.loaders.csv_loader import CSVLoader
from sklearn.ensemble import RandomForestClassifier
from activedetect.experiments.Experiment import Experiment

"""
Example Experiment Script
"""

#Loads the first 100 lines of the dataset
#loaded data is a list of lists [ [r1], [r2],...,[r100]]
c = CSVLoader()
loadedData = c.loadFile('datasets/adult.data')[0:100]

#all but the last column are features
features = [l[0:-1] for l in loadedData]

#last column is a label, turn into a float
labels = [1.0*(l[-1]==' <=50K') for l in loadedData]

#run the experiment, results are stored in uscensus.log
e = Experiment(features, labels, RandomForestClassifier(), "uscensus")
e.runAllAccuracy()

