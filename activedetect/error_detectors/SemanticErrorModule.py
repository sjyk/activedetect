"""
Detects Semantic Errors in A Dataset Using 
Word2Vec
"""
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import word2vec
import numpy as np
import re
from ErrorModule import ErrorModule
import os.path

class SemanticErrorModule(ErrorModule):

	def __init__(self, corpus='corpora/text8', thresh=3.5, fail_thresh=0.8):
		
		#compiles the corpus first time
		if os.path.isfile(corpus+'-pretrained.bin'):
			self.model = Word2Vec.load(corpus+'-pretrained.bin')
		else:
			sentences = word2vec.LineSentence(corpus)
			self.model = word2vec.Word2Vec(sentences)
			self.model.save(corpus+'-pretrained.bin')


		self.model.init_sims(replace=True)

		self.thresh = thresh

		self.fail_thresh = fail_thresh


	"""
	Returns the subset of a domain that is potentially
	erroneous

	use dictionary instead

	O(N^2): N distinct values
	"""
	def predict(self, vals):
		
		domain = set(vals)

		error = set()
		incorpus = set()
		modeled_corpus = set()
		
		#for each val in the domain
		for d in domain:

			#tokenizes categorical attrs
			tokens = [t.strip().lower() for t in re.findall(r"[\w']+", d)]

			#if no tokens error
			if len(tokens) == 0:
				error.add(d)
			else:

				#iterate through tokens add to corpus
				match = False
				for t in tokens:

					if t not in STOPWORDS and t in self.model:
						incorpus.add(t)
						modeled_corpus.add(t)
						match = True

				#if no matches add it back to incorpus
				if not match:
					incorpus.add(d)

		#build similarity graph, take in-degree

		#if len(modeled_corpus) > 0:
		#	print self.model.doesnt_match(modeled_corpus)

		aggsim = {}
		vals = []
		for i in modeled_corpus:
			
			if i in self.model:
				agg = 0

				for j in modeled_corpus:

					if j in self.model:
						#print i, j
						agg = agg + self.model.similarity(i,j)

				aggsim[i] = agg
				vals.append(agg)

		#take MAD to filter corpus
		mad = self.mad(vals)
		median = np.median(vals)

		for a in aggsim:

			#if a == 'b':
			#	print aggsim[a], median, self.thresh*mad

			if np.abs(median - aggsim[a]) > self.thresh*mad:
				error.add(a)
				incorpus.remove(a)
				#print a, median, aggsim[a], mad, self.thresh

		#return error and incorpus
		return list(error), list(incorpus)
		


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
			tokens = [t.strip().lower() for t in re.findall(r"[\w']+", val)]

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

	"""
	"""
	def desc(self):
		return "A value was found with an abnormal word2vec similarity score thresh= " + str(self.thresh)


	def availTypes(self):
		return ['categorical']






		
