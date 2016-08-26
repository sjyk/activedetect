#!/usr/bin/env python


from loaders.csv_loader import CSVLoader
from loaders.type_inference import LoLTypeInference

c = CSVLoader()
loadedData = c.loadFile('datasets/adult.data')
t = LoLTypeInference()



print t.getDataTypes(loadedData[2])
