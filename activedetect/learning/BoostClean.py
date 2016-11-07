"""
This class implements the main boosting loop
"""
import copy
from activedetect.error_detectors.ErrorDetector import ErrorDetector
from EvaluateCleaning import EvaluateCleaning
from CleanClassifier import CleanClassifier
from sklearn.metrics import accuracy_score, f1_score
from activedetect.model_based.preprocessing_utils import *
import numpy as np

class BoostClean(object):

    def __init__(self,
                 modules,
                 config,
                 base_model,
                 features,
                 labels,
                 logging,
                 #optimization flags
                 materialize=True,
                 dfnmemo=True):

        self.modules = modules
        self.config = config

        self.modules.append("None")
        self.config.append({})

        self.base_model = base_model
        self.features = features
        self.labels = labels

        self.ensemble = []

        self.weights = None

        self.logging = logging

        self.materialize = materialize

        self.dfn_cache = {}

        self.dfnmemo = dfnmemo


    def runRound(self, avail_modules, avail_config, selected, materialized_cache):

        trial = {}

        for i, module in enumerate(avail_modules):

            #print avail_modules

            if module == "None":

                dfn = lambda row: (False, -1)

                if self.materialize and ((i, 'impute_mean', 'impute_mean') in materialized_cache):
                    trial[(i, 'impute_mean', 'impute_mean')] = materialized_cache[(i, 'impute_mean', 'impute_mean')]
                else:
                    clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                    cleanClassifier, ypred, ytrue = clf.run(dfn, 'impute_mean', 'impute_mean')
                    materialized_cache[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue)
                    trial[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue)

                #if the weights are none initialize
                if self.weights == None:
                    self.weights = np.ones((len(ypred),1))/len(ypred)
            
            elif not (len(avail_modules)-1, 'impute_mean', 'impute_mean') in selected:
                mlist = [module]
                clist = [avail_config[i]]

                if self.dfnmemo and (i in self.dfn_cache):
                    dfn = self.dfn_cache[i]
                else:
                    detector = ErrorDetector(self.features,modules=mlist, config=clist)
                    detector.addLogger(self.logging)
                    detector.fit()
                    dfn = detector.getDetectorFunction()
                    self.dfn_cache[i] = dfn

                for tr in CleanClassifier.avail_train:
                    for te in CleanClassifier.avail_test:

                        if (i, tr, te) in selected:
                            continue

                        if self.materialize and ((i, tr, te) in materialized_cache):
                            trial[(i, tr, te)] = materialized_cache[(i, tr, te)]
                            continue

                        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                        cleanClassifier, ypred, ytrue = clf.run(dfn, tr, te)
                        materialized_cache[(i, tr, te)] = (cleanClassifier, ypred, ytrue)
                        trial[(i, tr, te)] = (cleanClassifier, ypred, ytrue)

                        #if the weights are none initialize
                        if self.weights == None:
                            self.weights = np.ones((len(ypred),1))/len(ypred)
        
        argmax = None
        maxv = 0

        for k in trial:

            arg = (trial[k], k, avail_modules)
            cur = accuracy_score(trial[k][2], trial[k][1], sample_weight=np.asarray(self.weights).reshape(-1))

            if maxv < cur:
                argmax = arg
                maxv = cur

        argmax, maxv = self.refitMax(argmax)

        return maxv, argmax, materialized_cache


    def refitMax(self, argmax):
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        result = clf.run(argmax[0][0].detector, argmax[0][0].train_action, argmax[0][0].test_action)
        cur = accuracy_score(result[2], result[1], sample_weight=np.asarray(self.weights).reshape(-1))

        return (result, argmax[1], argmax[2]), cur


    def calculateStep(self, ypred, yactual):
        eps = np.sum(self.weights[np.where( np.array(ypred) != np.array(yactual))])
        return 0.5*np.log((1-eps)/eps)

    def updateWeights(self, ypred, yactual, alpha):

        for i in range(0, len(ypred)):
            if ypred[i] == yactual[i]:
                self.weights[i] = self.weights[i]*np.exp(-alpha)
            else:
                self.weights[i] = self.weights[i]*np.exp(alpha)

        self.weights = self.weights / np.sum(self.weights)

    def run(self, j=1):

        modules = copy.copy(self.modules)
        config  = copy.copy(self.config)
        selected = set()
        cache = {}
        
        for roundNo in range(0,j):
            acc, argmax, cache = self.runRound(modules, config, selected, cache)

            alpha = self.calculateStep(argmax[0][1], argmax[0][2])

            self.ensemble.append((argmax[0][0], alpha))

            self.updateWeights(argmax[0][1], argmax[0][2], alpha)

            selected.add(argmax[1])

            self.logging.logResult(["cleaner_boostclean", roundNo, str(self.modules[argmax[1][0]])])

            self.logging.logResult(["acc_boostclean", roundNo, self.evaluateEnsembleAccuracy(roundNo)])

        return self.ensemble


    def predict(self, X):

        base = np.asarray(np.zeros((len(X),1))).reshape(-1)

        for e in self.ensemble:
            base[:] = base[:] + e[1]*np.array([y*2-1 for y in e[0].predict(X)])

        return (base >= 0)


    def evaluateEnsembleAccuracy(self, roundNo):
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        X = clf.test_features
        ypred = self.predict(X)
        ytrue = np.array(clf.test_labels)
        return acc_score(ytrue, ypred, roundNo)




            


