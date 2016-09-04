"""
Given a dataset returns the cells that 
are possibly erroneous.
"""
import numpy as np
from loaders.type_inference import LoLTypeInference
from StringSimilarityErrorModule import StringSimilarityErrorModule
from QuantitativeErrorModule import QuantitativeErrorModule
from SemanticErrorModule import SemanticErrorModule
from DistributionErrorModule import DistributionErrorModule

class ErrorDetector:

	def __init__(self, 
				 dataset, 
				 cols = None,
				 modules = [],
				 config =  []):	

		if len(config) != len(modules):
			raise ValueError("Config must be the same length as the modules list")

		if len(modules) == 0:
			self.default__init__(dataset, cols)
			return

		self.cols = cols

		self.dataset = dataset

		self.types = LoLTypeInference().getDataTypes(dataset)

		self.modules = [d(**config[i]) for i, d in enumerate(modules)] 

		self.all_errors = [[[] for x in range(len(self.types))] for y in range(len(dataset))] 
		
		self.error_list = []

		self.iterator = None


	"""
	default detectors and config
	"""
	def default__init__(self, dataset, cols):
		d_detect = DistributionErrorModule
		q_detect = QuantitativeErrorModule
		s_detect = SemanticErrorModule
		str_detect = StringSimilarityErrorModule
		config = [{'thresh':20}, 
				  {'thresh': 10}, 
				  {'thresh': 10, 'corpus': 'corpora/text8'}, 
				  {'thresh': 10}]

		return self.__init__(dataset, cols, [d_detect,q_detect, s_detect, str_detect], config)



	"""
	Based on the data type this applies one of the detectors
	"""
	def __predictCol(self, col):
		col_type = self.types[col]

		error_dict = {}
		for d in self.modules:
			if col_type in d.availTypes():
				vals = [r[col] for r in self.dataset]
				error_dict[d.desc()] = d.getRecordSet(d.predict(vals)[0], self.dataset, col)

		return error_dict

	"""
	The main user facing method to get all of the errors
	"""
	def fit(self, logging=True):
		if logging:
			print "[Fitting Model]..."

		for i,t in enumerate(self.types):

			if  not self.cols == None and i not in self.cols:
				continue 

			output = self.__predictCol(i)
			
			if logging:
				print "[Fitting Model], column: ", i, t

			for k in output:
				for index in output[k][1]:
					self.all_errors[index][i].append(k)
					self.error_list.append((index, i))

		self.iterator = self.error_list.__iter__()
		
		return self.all_errors

	
	#iterator interface to the error detector
	def __iter__(self):
		return self

	#gets the next error
	def next(self):
		v = self.iterator.next()
		return {'cell': v, 
				'error_types':self.all_errors[v[0]][v[1]],
				'cell_value': self.dataset[v[0]][v[1]],
				'record_value': self.dataset[v[0]]}










