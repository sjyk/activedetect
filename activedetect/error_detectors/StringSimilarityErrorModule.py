"""
Detects Semantic Errors in A Dataset Using 
Word2Vec
"""
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import numpy as np
import re
from ErrorModule import ErrorModule

class StringSimilarityErrorModule(ErrorModule):


	#takes in a list of words
	def __init__(self, thresh=3.5):		
		self.thresh = thresh


	"""
	Returns the subset of a records that are potentially
	erroneous

	use dictionary instead

	O(N^2): N distinct values
	"""
	def predict(self, strings):
		
		self.model = word2vec.Word2Vec(strings,hs=1)

		error = []
		incorpus = []

		incorpus = strings
		error = []
		
		#for each val in the column
		string_scores = []
		scoredict = {}

		for s in strings:

			cleaned_string = s.lower().strip()

			if len(cleaned_string) == 0:
				error.append(s)
				incorpus.remove(s)
			else:
				score = np.squeeze(self.model.score([cleaned_string]))/len(cleaned_string)
				string_scores.append(score)
				scoredict[s] = score

		mad = np.std(string_scores)
		median = np.mean(string_scores)

		for s in scoredict:
			if np.abs(scoredict[s] - median) > self.thresh*mad:
				error.append(s)
				incorpus.remove(s)

		return error, incorpus
		


	""" Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
	def mad(self, arr):
		arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
		med = np.median(arr)
		return np.median(np.abs(arr - med))


	"""
	Turns an error set into a set of records
	"""
	def getRecordSet(self, errors, dataset, col):

		indices = []
		erecords = []

		for i,d in enumerate(dataset):

			val = d[col]

			if val in set(errors):
				indices.append(i)
				erecords.append(d)
				
		return erecords, indices

	def desc(self):
		return "A string was found that was not well predicted by a sequential module thresh= " + str(self.thresh)


	def availTypes(self):
		return ['string']




		
