"""
Detects Semantic Errors in A Dataset Using 
Word2Vec
"""
from gensim.models.word2vec import Word2Vec
import numpy as np
import os.path

class PatternErrorFinder(object):

	def __init__(self, thresh=3.5):

		self.thresh = thresh

	"""
	Turns an error set into a set of records
	"""
	def getRecordSet(self, dataset):

		p = len(dataset[0])
		self.model = Word2Vec(dataset, size=100, window=p, min_count=1, workers=4, hs=1, negative=0)
		vals = self.model.score(dataset)

		mean = np.mean(vals)
		std = np.std(vals)

		erecords = []
		indices = []

		for i,v in enumerate(vals):
			if mean-v > self.thresh*std:
				erecords.append(dataset[i])
				indices.append(i)

		return erecords, indices

	"""
	"""
	def desc(self):
		return "A value was found with an abnormal word2vec similarity score thresh= " + str(self.thresh)






		
