"""
Given a dataset returns the cells that 
are possibly erroneous.
"""
import numpy as np
from DistributionErrorDetector import DistributionErrorDetector
from loaders.type_inference import LoLTypeInference
from StringSimilarityErrorDetector import StringSimilarityErrorDetector
from QuantitativeErrorDetector import QuantitativeErrorDetector
from SemanticErrorDetector import SemanticErrorDetector

class EnsembleErrorDetector:

	def __init__(self, 
				 dataset, 
				 cat_thresh=10, 
				 num_thresh=10, 
				 dist_thresh=20,
				 string_thresh=10,
				 corpus='corpora/text8'):		

		self.types = LoLTypeInference().getDataTypes(dataset)
		self.cat_thresh = cat_thresh
		self.num_thresh = num_thresh
		self.string_thresh = string_thresh
		self.dist_thresh = dist_thresh
		self.dataset = dataset

		self.d_detect = DistributionErrorDetector(self.dist_thresh)
		self.q_detect = QuantitativeErrorDetector(self.num_thresh)
		self.s_detect = SemanticErrorDetector(corpus, self.num_thresh)
		self.str_detect = None

		self.all_errors = [[[] for x in range(len(self.types))] for y in range(len(dataset))] 
		self.error_list = []

		self.iterator = None


	"""
	Based on the data type this applies one of the detectors
	"""
	def __predictCol(self, col):
		col_type = self.types[col]

		if col_type == 'numerical':
			vals = [r[col] for r in self.dataset]
			
			return {'quantitative': self.q_detect.getRecordSet(self.q_detect.predict(vals)[0], 
		    								  self.dataset, 
		    								  col),

					'distribution': self.d_detect.getRecordSet(self.d_detect.predict(vals)[0], 
		    								  self.dataset, 
		    								  col)}

		elif col_type == 'string':

			vals = [r[col] for r in self.dataset]

			self.str_detect = StringSimilarityErrorDetector(vals)

			return {'string': self.str_detect.getRecordSet(self.str_detect.predict(vals)[0], 
		    								  self.dataset, 
		    								  col),
			
					'distribution': self.d_detect.getRecordSet(self.d_detect.predict(vals)[0], 
		    								  self.dataset, 
		    								  col)}

		elif col_type == 'categorical':

			vals = [r[col] for r in self.dataset]

			return  {'semantic': self.s_detect.getRecordSet(self.s_detect.predict(set(vals))[0], 
		    								  self.dataset, 
		    								  col),
			
					'distribution': self.d_detect.getRecordSet(self.d_detect.predict(vals)[0], 
		    								  self.dataset, 
		    								  col)}
		else:
			#we aren't handling special cases yet
			return None


	"""
	The main user facing method to get all of the errors
	"""
	def fit(self, logging=True):
		if logging:
			print "[Fitting Model]..."

		for i,t in enumerate(self.types):
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










