"""
Detects Semantic Errors in A Dataset Using 
Word2Vec
"""
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import numpy as np
import re

class SemanticErrorDetector:

	def __init__(self, corpus='corpora/text8', thresh=3.5):
		sentences = word2vec.Text8Corpus(corpus)
		
		self.model = word2vec.Word2Vec(sentences)

		self.thresh = thresh


	"""
	Returns the subset of a domain that is potentially
	erroneous

	use dictionary instead

	O(N^2): N distinct values
	"""
	def predict(self, domain):
		error = []
		incorpus = []
		
		#for each val in the domain
		for d in domain:

			#tokenizes categorical attrs
			tokens = re.findall(r"[\w']+", d)

			#if no tokens error
			if len(tokens) == 0:
				error.append(d)
			else:

				#iterate through tokens add to corpus
				match = False
				for t in tokens:
					if t in self.model:
						incorpus.append(t)
						match = True

				#if no matches error
				if not match:
					error.append(d)



		#build similarity graph, take in-degree
		aggsim = {}
		vals = []
		for i in incorpus:
			agg = 0
			for j in incorpus:
				agg = agg + self.model.similarity(i,j)

			aggsim[i] = agg
			vals.append(agg)



		#take MAD to filter corpus
		mad = self.mad(vals)
		median = np.median(vals)

		for a in aggsim:
			if np.abs(aggsim[a] - median) > self.thresh*mad:
				error.append(a)
				incorpus.remove(a)

		#return error and incorpus
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
			tokens = set(re.findall(r"[\w']+", val))

			match = False

			for e in errors:

				if e in tokens or \
				   (len(tokens) == 0 and e in val):
					match = True
					break

			if match:
				indices.append(i)
				erecords.append(d)

		return erecords, indices






		
