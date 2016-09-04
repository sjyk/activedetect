"""
Detects Distribution Errors in a discrete distribution
dataset using standard deviations
"""
import numpy as np
from ErrorModule import ErrorModule

class DistributionErrorModule(ErrorModule):

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

	"""
	Returns a description
	"""
	def desc(self):
		return "A distinct value was found with a count of greater than > " + str(self.thresh) + " stds from the mean count"


	def availTypes(self):
		return ['numerical', 'categorical', 'string']