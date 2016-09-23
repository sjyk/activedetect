"""
Given a dataset returns the cells that 
are possibly erroneous.
"""
import numpy as np
from activedetect.loaders.type_inference import LoLTypeInference
from StringSimilarityErrorModule import StringSimilarityErrorModule
from QuantitativeErrorModule import QuantitativeErrorModule
from SemanticErrorModule import SemanticErrorModule
from CharSimilarityErrorModule import CharSimilarityErrorModule
from PuncErrorModule import PuncErrorModule

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

		self.modules = modules
		self.config = config

		self.cols = cols

		self.dataset = dataset

		self.types = LoLTypeInference().getDataTypes(dataset)

		self.modules = [d(**config[i]) for i, d in enumerate(modules)] 

		self.all_errors = [[[] for x in range(len(self.types))] for y in range(len(dataset))] 
		
		self.error_list = []

		self.iterator = None

		self.logger = None


	"""
	Adds a logger to the error detector
	"""
	def addLogger(self, logger):
		self.logger = logger

	"""
	default detectors and config
	"""
	def default__init__(self, dataset, cols):
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

		return self.__init__(dataset, cols, [q_detect, s_detect, str_detect, char_detect, punc_detect], config)



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
	def fit(self):
		
		#log schema, config, file type
		if self.logger != None:
			self.logger.logSchema(self.types)
			self.logger.logFileType(len(self.dataset), len(self.types))
			self.logger.logConfig(self.modules, self.config)

		for i,t in enumerate(self.types):

			if  not self.cols == None and i not in self.cols:
				continue 

			output = self.__predictCol(i)

			for k in output:
				for index in output[k][1]:
					self.all_errors[index][i].append(k)
					self.error_list.append((index, i))
					
					if self.logger != None:
						self.logger.logError(self.errorToDict((index, i)))

		self.iterator = self.error_list.__iter__()
		
		return self.all_errors

	
	#iterator interface to the error detector
	def __iter__(self):
		return self

	def errorToDict(self, v):
		return {'cell': v, 
				'error_types':self.all_errors[v[0]][v[1]],
				'cell_value': self.dataset[v[0]][v[1]],
				'record_value': self.dataset[v[0]]}

	#gets the next error
	def next(self):
		v = self.iterator.next()
		return self.errorToDict(v)










