"""
Detects Distribution Errors in a discrete distribution
dataset using Median Absolute Deviation
"""
import numpy as np

class DistributionErrorDetector:

	def __init__(self, thresh=3.5, fail_thresh=2):
		self.thresh = thresh
		self.fail_thresh = fail_thresh

	"""
	Returns the subset of a domain that is potentially
	erroneous
	"""
	def predict(self, vals):

		dist = {}

		for v in vals:
			if v not in dist:
				dist[v] = 0

			dist[v] = dist[v] + 1

		
		valsv = set(vals)
		valsd = [dist[v] for v in valsv]


		mad = np.std(valsd)

		median = np.mean(valsd)
		
		#make a copy
		incorpus = []
		error = []
		incorpus.extend(valsv)


		#fail if less than the fail threshold, or if
		#mad not informative
		if len(valsd) <= self.fail_thresh or mad < 1.0:
			return error, incorpus

		for a in valsv:
			
			if np.abs(dist[a] - median) > self.thresh*mad:
				#print a, dist[a], mad, median
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
			val = d[col]

			match = False

			for e in errors:

				if e == val:
					match = True

			if match:
				indices.append(i)
				erecords.append(d)

		return erecords, indices