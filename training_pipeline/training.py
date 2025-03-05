'''Luigi feature engineering and model training pipeline.'''

import pickle
import luigi
from luigi.format import Nop

import configuration as config
from training_pipeline.functions import data_cleaning
from training_pipeline.functions import data_encoding
from training_pipeline.functions import survival_modeling
from training_pipeline.functions import kld_scoring
from training_pipeline.functions import classifier
from training_pipeline.functions import regressor

class DataCleaning(luigi.Task):

    def output(self):
        return luigi.LocalTarget(config.DATA_CLEANING_RESULT, format=Nop)

    def run(self):
        data=data_cleaning.run()
        
        with self.output().open('w') as output_file:
            pickle.dump(data, output_file)


class DataEncoding(luigi.Task):

    def requires(self):
        return DataCleaning()

    def output(self):
        return luigi.LocalTarget(config.DATA_ENCODING_RESULT, format=Nop)

    def run(self):
        data=data_encoding.run()

        with self.output().open('w') as output_file:
            pickle.dump(data, output_file)


class SurvivalFeatures(luigi.Task):

    def requires(self):
        return DataEncoding()
    
    def output(self):
        return luigi.LocalTarget(config.SURVIVAL_FEATURES_RESULT, format=Nop)

    def run(self):
        data=survival_modeling.run()

        with self.output().open('w') as output_file:
            pickle.dump(data, output_file)


class KLDFeatures(luigi.Task):

    def requires(self):
        return SurvivalFeatures()
    
    def output(self):
        return luigi.LocalTarget(config.KLD_FEATURES_RESULT, format=Nop)

    def run(self):
        data=kld_scoring.run()

        with self.output().open('w') as output_file:
            pickle.dump(data, output_file)


class ClassifierTraining(luigi.Task):

    def requires(self):
        return KLDFeatures()
    
    def output(self):
        return luigi.LocalTarget(config.EFS_FEATURE_RESULT, format=Nop)

    def run(self):
        data=classifier.run()

        with self.output().open('w') as output_file:
            pickle.dump(data, output_file)


class RegressorTraining(luigi.Task):

    def requires(self):
        return ClassifierTraining()
    
    def output(self):
        return luigi.LocalTarget(config.PREDICTIONS, format=Nop)
    
    def run(self):
        data=regressor.run()

        with self.output().open('w') as output_file:
            pickle.dump(data, output_file)


def run():
    '''Main function to run Luigi training pipeline.'''

    luigi.build(
        [
            DataCleaning(),
            DataEncoding(),
            SurvivalFeatures(),
            KLDFeatures(),
            ClassifierTraining(),
            RegressorTraining()
        ],
        local_scheduler=True
    )