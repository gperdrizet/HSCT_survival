'''Luigi feature engineering and model training pipeline.'''

import pickle
import luigi
from luigi.format import Nop

import training_pipeline.configuration as config
from training_pipeline.functions import data_cleaning
from training_pipeline.functions import data_encoding
from training_pipeline.functions import survival_modeling

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
        return luigi.LocalTarget(config.SURVIVAL_FEATURES_RESULTS, format=Nop)

    def run(self):
        data=survival_modeling.run()

        with self.output().open('w') as output_file:
            pickle.dump(data, output_file)


# class KLDFeatures(luigi.Task):

#     def requires(self):
#         return SurvivalFeatures()
    
#     def output(self):
#         return luigi.LocalTarget(config.TFIDF_LUT, format = Nop)

#     def run(self):
#         tfidf_lut = data_funcs.make_tfidf_lut()

#         with self.output().open('w') as output_file:
#             dump(tfidf_lut, output_file, protocol = 5)


# class ClassifierTraining(luigi.Task):

#     def requires(self):
#         return KLDFeatures()
    
#     def output(self):
#         return luigi.LocalTarget(config.TFIDF_SCORE_ADDED)

#     def run(self):
#         data = data_funcs.add_tfidf_score()

#         with self.output().open('w') as output_file:
#             json.dump(data, output_file)


# class RegressorTraining(luigi.Task):

#     def requires(self):
#         return ClassifierTraining()
    
#     def output(self):
#         return luigi.LocalTarget(config.TFIDF_SCORE_KLD_KDE, format = Nop)
    
#     def run(self):
#         kl_kde = data_funcs.tfidf_score_kld_kde()

#         with self.output().open('w') as output_file:
#             dump(kl_kde, output_file, protocol = 5)

def run():
    '''Main function to run Luigi training pipeline.'''

    luigi.build(
        [
            DataCleaning(),
            DataEncoding(),
            SurvivalFeatures(),
            # KLDFeatures(),
            # ClassifierTraining(),
            # RegressorTraining()
        ],
        local_scheduler=True
    )