'''Data preprocessing and inference pipeline. Handles data cleaning and encoding
and feature engineering for inference.'''

import time
import pandas as pd

import configuration as config
from inference_pipeline.functions import cleaning
from inference_pipeline.functions import encoding
from inference_pipeline.functions import feature_engineering
from inference_pipeline.functions import prediction

INPUT_DATA_FILE='./pipeline_assets/data/train.csv'
PREDICTION_OUTPUT_FILE='./submissions/predictions/two_stage_ensemble_submission.csv'

# Models and other assets for each pipeline step
models='./pipeline_assets/models'
DATA_CLEANING_ASSETS=f'{models}/01-data_cleaning.pkl'
DATA_ENCODING_ASSETS=f'{models}/02-data_encoding.pkl'
SURVIVAL_MODEL_ASSETS=f'{models}/03-survival.pkl'
KLD_MODEL_ASSETS=f'{models}/04-kullback-leibler_divergence.pkl'
CLASSIFIER_MODEL_FILE=f'{models}/05-EFS_classifier.pkl'
REGRESSOR_ASSETS_FILE=f'{models}/06-regressor.pkl'


def run(sample_fraction:float=None):
    '''Main function to run inference pipeline.'''

    print()
    print('########################################################')
    print('# HSCT Survival inference pipeline #####################')
    print('########################################################')
    print()

    # Read the data
    data_df=pd.read_csv(config.INPUT_DATA_FILE)

    # Save the id column for later
    ids=data_df['ID']

    # Drop unnecessary columns
    data_df.drop(['efs', 'efs_time', 'ID'], axis=1, inplace=True, errors='ignore')

    # Take a sample for rapid development and testing, if desired.
    if sample_fraction != None:
        data_df=data_df.sample(frac=sample_fraction)

    print(f'Loaded data from: {INPUT_DATA_FILE}')
    print(f'Data shape: {data_df.shape}')
    print(f'nan count: {data_df.isnull().sum().sum()}')
    print()

    # Translate feature level values and replace string NAN
    # placeholders with actual np.nan.
    print('Starting data cleaning.')
    start_time=time.time()

    data_df=cleaning.run(
        data_df=data_df,
        data_cleaning_assets=DATA_CLEANING_ASSETS
    )

    dt=time.time()-start_time
    print(f'Data cleaning complete, run time: {dt:.0f} seconds')
    print(f'nan count: {data_df.isnull().sum().sum()}')
    print()

    # Encodes features with SciKit-learn continuous target encoder, then
    # applies a power transform with standardization using the Yeo-Johnson
    # method. Missing values filled with KNN imputation.
    print('Starting feature encoding.')
    start_time=time.time()

    data_df=encoding.run(
        data_df=data_df,
        data_cleaning_assets=DATA_CLEANING_ASSETS,
        data_encoding_assets=DATA_ENCODING_ASSETS
    )

    dt=time.time()-start_time
    print(f'Feature encoding complete, run time: {dt:.0f} seconds')
    print(f'nan count: {data_df.isnull().sum().sum()}')
    print()

    # Does feature engineering. Adds survival model features and their
    # corresponding Kullback-Leibler divergence scores. Also adds
    # learned EFS probability
    print(f'Starting feature engineering.')
    start_time=time.time()

    data_df=feature_engineering.run(
        data_df=data_df,
        survival_model_assets=SURVIVAL_MODEL_ASSETS,
        kld_model_assets=KLD_MODEL_ASSETS,
        classifier_model_file=CLASSIFIER_MODEL_FILE
    )

    dt=time.time()-start_time
    print(f'Feature engineering complete, run time: {dt:.0f} seconds')
    print(f'nan count: {data_df.isnull().sum().sum()}')
    print()
    print('Inference dataset:')
    print(data_df.head().transpose())
    print()

    # Does the actual inference run
    print(f'Starting inference.')
    start_time=time.time()

    predictions=prediction.run(
        data_df=data_df,
        assets_file=REGRESSOR_ASSETS_FILE
    )

    predictions_df=pd.DataFrame.from_dict({'ID': ids, 'prediction': predictions})

    dt=time.time()-start_time
    print(f'Inference complete, run time: {dt:.0f} seconds')
    print()
    print('Predictions:')
    print(predictions_df.head(20))

    predictions_df.to_csv(PREDICTION_OUTPUT_FILE, index=False)