#!/usr/bin/env python
import csv
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import unicodedata

#basic error handling
def tryParse(X):

	vals = []

	if X.shape == (1,1):
		try:
			vals.append(float(X.tolist()[0][0]))
		except ValueError:
			vals.append(0)

		return vals


	for x in np.squeeze(X.T):
		try:
			vals.append(float(x))
		except ValueError:
			vals.append(0)

	return vals

def tryParseList(Y):

	return tryParse(np.array(Y))

#converts the labeled dataset into features and labels
def featurize(features_dataset, types):
	feature_list = []

	for i,t in enumerate(types):
		col = [f[i] for f in features_dataset]

		if t == "string" or t == "categorical" or t =="address":
			vectorizer = CountVectorizer(min_df=1, token_pattern='\S+')
			feature_list.append(vectorizer.fit_transform(col))
		else:
			vectorizer = FunctionTransformer(tryParse)
			feature_list.append(scipy.sparse.csr_matrix(vectorizer.fit_transform(col)).T)

	features = scipy.sparse.hstack(feature_list).tocsr()
	return features
