"""
Detects Quantitative Errors in a dataset using
standard deviations
"""
import numpy as np

class QuantitativeErrorDetector:

	def __init__(self, thresh=3.5):
		self.thresh = thresh


	"""
	Returns the subset of a domain that is potentially
	erroneous
	"""
	def predict(self, vals):
		
		vals = filter( lambda x : not np.isinf(x) ,[self.tryParse(v) for v in list(vals)])

		std = np.std(vals)
		mean = np.mean(vals)
		
		#make a copy
		incorpus = []
		error = []
		incorpus.extend(vals)

		for a in vals:
			if np.abs(a - mean) > self.thresh*std:
				error.append(a)
				incorpus.remove(a)

		return error, incorpus


	def tryParse(self, num):
		try:
			return float(num)
		except ValueError:
			return np.inf


	"""
	Turns an error set into a set of records
	"""
	def getRecordSet(self, errors, dataset, col):

		indices = []
		erecords = []

		for i,d in enumerate(dataset):
			val = self.tryParse(d[col])

			match = False

			for e in errors:

				if e == val:
					match = True

			#match or can't parse
			if match or np.isinf(val):
				indices.append(i)
				erecords.append(d)

		return erecords, indices