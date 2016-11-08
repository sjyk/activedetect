"""
This class defines the composition of a classifier 
and a error detector
"""
import numpy as np
from scipy import stats
from collections import Counter
import copy

from activedetect.model_based.preprocessing_utils import *


class CleanClassifier(object):

    avail_train = ['impute_mean', 'impute_median', 'impute_mode', \
                    'discard', 'default']

    avail_test = ['impute_mean', 'impute_median', 'impute_mode', \
                  'default']

    def __init__(self,
                 model, 
                 detector, 
                 training_features,
                 training_labels,
                 feature_types,
                 train_action, 
                 test_action):

        self.model = model
        self.detector = detector
        self.train_action = train_action
        self.test_action = test_action
        self.training_features = training_features
        self.training_labels = training_labels
        self.clean_training_data = [v for v in training_features if not self.detector(v)[0]]
        self.default_pred = self.most_common(training_labels)
        self.types = feature_types


    def tryParse(self, num):
        try:
            return float(num)
        except ValueError:
            return np.inf 

    def most_common(self, lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    def gatherStatistics(self):

        self.stats = {}

        for i,t in enumerate(self.types):

            if t == 'numerical':

                cleanvals = [self.tryParse(v[i]) for v in self.clean_training_data \
                             if not np.isinf(self.tryParse(v[i]))]

                self.stats[i] = {'mean': np.mean(cleanvals), 
                                 'median': np.median(cleanvals), 
                                 'mode': stats.mode(cleanvals)[0][0]}

            elif t == 'categorical' or t == 'string':

                self.stats[i] = {'mean': None, 
                                 'median': None, 
                                 'mode': self.most_common([v[i] for v in self.clean_training_data])}

        #print self.stats


    def fit(self):
        self.gatherStatistics()

        training_features_copy = copy.copy(self.training_features)
        training_labels_copy = copy.copy(self.training_labels)

        indices_to_delete = set()

        if self.types == None:
            raise ValueError("Please run gatherStatistics() first")
        
        for i,v in enumerate(training_features_copy):
            error, col = self.detector(v)

            if error and col == -1:

                if self.train_action == 'default':
                    training_labels_copy[i] = self.default_pred
                else:
                    indices_to_delete.add(i)

            elif error and self.types[col] == 'numerical':

                if self.train_action == 'impute_mode':
                    training_features_copy[i][col] = str(self.stats[col]['mode'])
                elif self.train_action == 'impute_mean':
                    training_features_copy[i][col] = str(self.stats[col]['mean'])
                elif self.train_action == 'impute_median':
                    training_features_copy[i][col] = str(self.stats[col]['median'])
                elif self.train_action == 'default':
                    training_labels_copy[i] = self.default_pred
                else:
                    indices_to_delete.add(i)

            elif error and self.types[col] == 'categorical':

                if self.train_action == 'impute_mode':
                    training_features_copy[i][col] = str(self.stats[col]['mode'])
                elif self.train_action == 'impute_mean':
                    training_features_copy[i][col] = ''
                elif self.train_action == 'impute_median':
                    training_features_copy[i][col] = str(self.stats[col]['mode'])
                elif self.train_action == 'default':
                    training_labels_copy[i] = self.default_pred
                else:
                    indices_to_delete.add(i)

            elif error:

                if self.train_action == 'impute_mode':
                    training_features_copy[i][col] = ''
                elif self.train_action == 'impute_mean':
                    training_features_copy[i][col] = ''
                elif self.train_action == 'impute_median':
                    training_features_copy[i][col] = ''
                elif self.train_action == 'default':
                    training_labels_copy[i] = self.default_pred
                else:
                    indices_to_delete.add(i)


        training_features_copy =  [t for i, t in enumerate(training_features_copy) if i not in indices_to_delete]
        training_labels_copy =  [t for i, t in enumerate(training_labels_copy) if i not in indices_to_delete]

        X, transforms = featurize(training_features_copy, self.types)
        
        self.transforms = transforms

        y = np.array(training_labels_copy)

        return self.model.fit(X,y)


    def predict(self, test_features):
        test_features_copy = copy.copy(test_features)

        predictions = {}

        for i,v in enumerate(test_features_copy):

            error, col = self.detector(v)

            if error and col == -1:

                if self.test_action == 'default':
                    predictions[i] = self.default_pred

            elif error and self.types[col] == 'numerical':

                if self.test_action == 'impute_mode':
                    test_features_copy[i][col] = str(self.stats[col]['mode'])
                elif self.test_action == 'impute_mean':
                    test_features_copy[i][col] = str(self.stats[col]['mean'])
                elif self.test_action == 'impute_median':
                    test_features_copy[i][col] = str(self.stats[col]['median'])
                else:
                    predictions[i] = self.default_pred

            elif error and self.types[col] == 'categorical':

                if self.test_action == 'impute_mode':
                    test_features_copy[i][col] = str(self.stats[col]['mode'])
                elif self.test_action  == 'impute_mean':
                    test_features_copy[i][col] = ''
                elif self.test_action  == 'impute_median':
                    test_features_copy[i][col] = str(self.stats[col]['mode'])
                else:
                    predictions[i] = self.default_pred

            elif error:

                if self.test_action  == 'impute_mode':
                    test_features_copy[i][col] = ''
                elif self.test_action  == 'impute_mean':
                    test_features_copy[i][col] = ''
                elif self.test_action  == 'impute_median':
                    test_features_copy[i][col] = ''
                else:
                    predictions[i] = self.default_pred

        X = featurizeFromList(test_features_copy, self.types, self.transforms)
        predictions_nom = self.model.predict(X)

        try:
            scores = self.model.predict_proba(X)[:,1]
        except:
            scores = predictions_nom


        for k in predictions:
            predictions_nom[k] = predictions[k]

            if predictions[k] == 1:
                scores[k] = 1.0
            else:
                scores[k] = 0

        return predictions_nom, scores






