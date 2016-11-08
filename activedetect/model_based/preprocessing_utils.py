#!/usr/bin/env python
import csv
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
	transform_list = []

	for i,t in enumerate(types):
		col = [f[i] for f in features_dataset]

		if t == "string" or t == "categorical" or t =="address":
			vectorizer = CountVectorizer(min_df=1, token_pattern='\S+')
			vectorizer.fit(col)
			feature_list.append(vectorizer.transform(col))
			###print 
			transform_list.append(vectorizer)
		else:
			vectorizer = FunctionTransformer(tryParse)
			vectorizer.fit(col)
			feature_list.append(scipy.sparse.csr_matrix(vectorizer.transform(col)).T)
			transform_list.append(vectorizer)

	features = scipy.sparse.hstack(feature_list).tocsr()
	return features, transform_list

#converts the labeled dataset into features and labels
def featurizeFromList(features_dataset, types, tlist):
	feature_list = []
	transform_list = []

	for i,t in enumerate(types):
		col = [f[i] for f in features_dataset]

		if t == "string" or t == "categorical" or t =="address":
			vectorizer = tlist[i]
			feature_list.append(vectorizer.transform(col))
		else:
			vectorizer = tlist[i]
			#print scipy.sparse.csr_matrix(vectorizer.transform(col)).T
			feature_list.append(scipy.sparse.csr_matrix(vectorizer.transform(col)).T)

	features = scipy.sparse.hstack(feature_list).tocsr()
	return features

def get_acc_scores(ytrue, ypred, yscores=None):
	
	if yscores == None:
		yscores = ypred
	
	return [accuracy_score(ytrue, ypred), f1_score(ytrue, ypred), roc_auc_score(ytrue, yscores, 'weighted')]




