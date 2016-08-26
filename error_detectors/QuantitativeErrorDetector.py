"""
Detects Quantitative Errors in a dataset using
Median Absolute Deviation
"""
import numpy as np

class QuantitativeErrorDetector:

	"""
	Returns the subset of a domain that is potentially
	erroneous
	"""
	def predict(self, vals, thresh=10):
		
		vals = [float(v) for v in list(vals)]

		mad = self.mad(vals)
		median = np.median(vals)
		
		#make a copy
		incorpus = []
		error = []
		incorpus.extend(vals)

		for a in vals:
			if np.abs(a - median) > thresh*mad:
				error.append(a)
				incorpus.remove(a)

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
			val = float(d[col])

			match = False

			for e in errors:

				if e == val:
					match = True

			if match:
				indices.append(i)
				erecords.append(d)

		return erecords, indices