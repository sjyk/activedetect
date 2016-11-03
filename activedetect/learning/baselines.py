"""
This class implements the no cleaning-baseline
"""
import copy
from activedetect.error_detectors.ErrorDetector import ErrorDetector
from EvaluateCleaning import EvaluateCleaning
from CleanClassifier import CleanClassifier
from sklearn.metrics import accuracy_score, f1_score
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
import numpy as np

class NoClean(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        dfn = lambda row: (False, -1)
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        cleanClassifier, ypred, ytrue = clf.run(dfn, 'impute_mean', 'impute_mean')
        self.logging.logResult(["acc_noclean", accuracy_score(ytrue, ypred)])


class ICClean(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [PuncErrorModule]
        clist = [{}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist, use_word2vec=False)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        cleanClassifier, ypred, ytrue = clf.run(dfn, 'impute_mean', 'impute_mean')
        self.logging.logResult(["acc_icclean", accuracy_score(ytrue, ypred)])


class QClean(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [QuantitativeErrorModule]
        clist = [{'thresh': 10}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist, use_word2vec=False)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        cleanClassifier, ypred, ytrue = clf.run(dfn, 'impute_mean', 'impute_mean')
        self.logging.logResult(["acc_qclean", accuracy_score(ytrue, ypred)])


class BestSingle(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [QuantitativeErrorModule, PuncErrorModule]
        clist = [{'thresh': 10}, {}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        
        v, i = self.runRound(mlist, clist, set())

        self.logging.logResult(["acc_bs", v])

    def runRound(self, avail_modules, avail_config, selected):

        trial = {}

        for i, module in enumerate(avail_modules):

            #print avail_modules

            if module == "None":

                dfn = lambda row: (False, -1)
                clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                cleanClassifier, ypred, ytrue = clf.run(dfn, 'impute_mean', 'impute_mean')
                trial[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue)
            
            else:
                mlist = [module]
                clist = [avail_config[i]]

                detector = ErrorDetector(self.features,modules=mlist, config=clist)
                detector.fit()
                dfn = detector.getDetectorFunction()

                for tr in CleanClassifier.avail_train:
                    for te in CleanClassifier.avail_test:

                        if (i, tr, te) in selected:
                            continue

                        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                        cleanClassifier, ypred, ytrue = clf.run(dfn, tr, te)
                        trial[(i, tr, te)] = (cleanClassifier, ypred, ytrue)

        
        argmax = None
        maxv = 0

        for k in trial:

            arg = (trial[k], k, avail_modules)
            cur = accuracy_score(trial[k][2], trial[k][1])

            if maxv < cur:
                argmax = arg
                maxv = cur

        self.logging.logResult(["cleaner_bs", roundNo, str(self.modules[argmax[1][0]])])

        return maxv, argmax


class WorstSingle(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [QuantitativeErrorModule, PuncErrorModule]
        clist = [{'thresh': 10}, {}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        
        v, i = self.runRound(mlist, clist, set())

        self.logging.logResult(["acc_ws", v])

    def runRound(self, avail_modules, avail_config, selected):

        trial = {}

        for i, module in enumerate(avail_modules):

            #print avail_modules

            if module == "None":

                dfn = lambda row: (False, -1)
                clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                cleanClassifier, ypred, ytrue = clf.run(dfn, 'impute_mean', 'impute_mean')
                trial[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue)
            
            else:
                mlist = [module]
                clist = [avail_config[i]]

                detector = ErrorDetector(self.features,modules=mlist, config=clist)
                detector.fit()
                dfn = detector.getDetectorFunction()

                for tr in CleanClassifier.avail_train:
                    for te in CleanClassifier.avail_test:

                        if (i, tr, te) in selected:
                            continue

                        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                        cleanClassifier, ypred, ytrue = clf.run(dfn, tr, te)
                        trial[(i, tr, te)] = (cleanClassifier, ypred, ytrue)

        
        argmin = None
        minv = 1

        for k in trial:

            arg = (trial[k], k, avail_modules)
            cur = accuracy_score(trial[k][2], trial[k][1])

            if minv >= cur:
                argmin = arg
                minv = cur

        self.logging.logResult(["cleaner_ws", roundNo, str(self.modules[argmin[1][0]])])

        return minv, argmin




            


