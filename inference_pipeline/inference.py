'''Data preprocessing and inference pipeline. Handles data cleaning and encoding
and feature engineering for inference.'''

import time
import pandas as pd
from inference_pipeline import cleaning
from inference_pipeline import encoding
from inference_pipeline import feature_engineering
from inference_pipeline import prediction

####################################################
# Asset file paths #################################
####################################################

# Data
INPUT_DATA_FILE='./data/raw/train.csv'
FEATURE_TYPES_DICT_FILE='./data/processed/01.1-feature_type_dict.pkl'
FEATURE_LEVEL_DICTS_FILE='./data/processed/01.1-feature_value_translation_dicts.pkl'
NAN_DICTS_FILE='./data/processed/01.1-nan_placeholders_list.pkl'
COXPH_FEATURES_FILE='./data/processed/02.1-coxPH_significant_features.pkl'
WAFT_FEATURES_FILE='./data/processed/02.2-weibullAFT_significant_features.pkl'

# Models
TARGET_ENCODER_FILE='./models/01.2-continuous_target_encoder.pkl'
POWER_TRANSFORMER_FILE='./models/01.2-continuous_target_power_transformer.pkl'
KNN_IMPUTER_FILE='./models/01.2-numerical_imputer.pkl'
COXPH_MODEL_FILE='./models/02.1-coxPH_model.pkl'
WAFT_MODEL_FILE='./models/02.2-weibullAFT_model.pkl'
KLD_MODELS_FILE='./models/02.3-kld_models.pkl'
EFS_MODEL_FILE='./models/02.4-EFS_classifier_model.pkl'
MODEL_FILE='./models/03.3-XGBoost_engineered_features_tuned.pkl'


def run(sample_fraction:float=None):
    '''Main function to run inference pipeline.'''

    print()
    print('########################################################')
    print('# HSCT Survival inference pipeline #####################')
    print('########################################################')
    print()

    # Read the data
    data_df=pd.read_csv(INPUT_DATA_FILE)

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
        feature_level_dicts_file=FEATURE_LEVEL_DICTS_FILE,
        nan_dicts_file=NAN_DICTS_FILE
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
        feature_types_dict_file=FEATURE_TYPES_DICT_FILE,
        target_encoder_file=TARGET_ENCODER_FILE,
        power_transformer_file=POWER_TRANSFORMER_FILE,
        knn_imputer_file=KNN_IMPUTER_FILE
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
        coxph_features_file=COXPH_FEATURES_FILE,
        waft_features_file=WAFT_FEATURES_FILE,
        coxph_model_file=COXPH_MODEL_FILE,
        waft_model_file=WAFT_MODEL_FILE,
        kld_models_file=KLD_MODELS_FILE,
        efs_model_file=EFS_MODEL_FILE
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
        model_file=MODEL_FILE
    )

    predictions_df=pd.DataFrame.from_dict({'ID': ids, 'prediction': predictions})

    dt=time.time()-start_time
    print(f'Inference complete, run time: {dt:.0f} seconds')
    print()
    print('Predictions:')
    print(predictions_df.head(20))
