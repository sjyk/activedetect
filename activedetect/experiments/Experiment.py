"""
This class defines the main experiment routines
"""
from activedetect.reporting.CSVLogging import CSVLogging
from activedetect.learning.baselines import *
from activedetect.learning.BoostClean import BoostClean
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

import datetime

class Experiment(object):

    def __init__(self,
                 features,
                 labels,
                 model,
                 experiment_name):
        """
        * features a list of lists
        * a list of binary attributes 1/0
        """

        self.features = features
        self.labels = labels
        logger = CSVLogging(experiment_name+".log")
        self.logger = logger
        self.model = model


    def runAllAccuracy(self):
        q_detect = QuantitativeErrorModule
        punc_detect = PuncErrorModule
        config = [{'thresh': 10},  {}]

        start = datetime.datetime.now()

        b = BoostClean(modules=[q_detect, punc_detect],
               config=config,
               base_model=self.model,
               features=self.features,
               labels=self.labels,
               logging=self.logger)

        b.run(j=5)

        self.logger.logResult(["time_boostclean", str(datetime.datetime.now()-start)])

        start = datetime.datetime.now()

        b = NoClean( base_model=self.model,
               features=self.features,
               labels=self.labels,
               logging=self.logger)

        b.run()

        self.logger.logResult(["time_noclean", str(datetime.datetime.now()-start)])

        start = datetime.datetime.now()

        b = ICClean( base_model=self.model,
               features=self.features,
               labels=self.labels,
               logging=self.logger)

        b.run()

        self.logger.logResult(["time_icclean", str(datetime.datetime.now()-start)])

        start = datetime.datetime.now()

        b = QClean( base_model=self.model,
               features=self.features,
               labels=self.labels,
               logging=self.logger)

        b.run()

        self.logger.logResult(["time_qclean", str(datetime.datetime.now()-start)])

        start = datetime.datetime.now()

        b = BestSingle( base_model=self.model,
               features=self.features,
               labels=self.labels,
               logging=self.logger)

        b.run()

        self.logger.logResult(["time_bs", str(datetime.datetime.now()-start)])

        start = datetime.datetime.now()

        b = WorstSingle( base_model=self.model,
               features=self.features,
               labels=self.labels,
               logging=self.logger)

        b.run()

        self.logger.logResult(["time_ws", str(datetime.datetime.now()-start)])


