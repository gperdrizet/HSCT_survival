'''Data preprocessing and inference pipeline. Handles data cleaning and encoding
and feature engineering for inference. Makes predictions.'''

import time
import pickle
import multiprocessing as mp
from typing import Callable

import numpy as np
import pandas as pd

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


def clean_data(
        data_df:pd.DataFrame,
        data_cleaning_assets:str
) -> pd.DataFrame:

    '''Main function to do feature cleaning.'''

    with open(data_cleaning_assets, 'rb') as input_file:
        assets=pickle.load(input_file)

    for feature, translation_dict in assets['feature_levels'].items():
        data_df[feature]=data_df[feature].replace(translation_dict)

    data_df.replace(assets['nan_placeholders'], inplace=True)

    return data_df


def encode_data(
        data_df:pd.DataFrame,
        data_cleaning_assets:str,
        data_encoding_assets:str,
) -> pd.DataFrame:

    '''Main function to do data encoding.'''


    #######################################################
    # ASSET LOADING #######################################
    #######################################################

    # Load feature type definitions
    with open(data_cleaning_assets, 'rb') as input_file:
        assets=pickle.load(input_file)

    feature_types_dict=assets['feature_types']

    # Load encoder and transformer models
    with open(data_encoding_assets, 'rb') as input_file:
        assets=pickle.load(input_file)

    target_encoder=assets['target_encoder']
    knn_imputer=assets['knn_imputer']


    #######################################################
    # FEATURE ENCODING ####################################
    #######################################################

    # Get categorical features
    categorical_df=data_df[feature_types_dict['Nominal'] + feature_types_dict['Ordinal']]

    # Encode the nominal & ordinal features
    encoded_categorical_features=target_encoder.transform(categorical_df)

    # Rebuild the dataframe
    encoded_categorical_features_df=pd.DataFrame(
        encoded_categorical_features,
        columns=feature_types_dict['Nominal'] + feature_types_dict['Ordinal']
    )

    #######################################################
    # DATA CLEANING #######################################
    #######################################################

    # Clean NANs in the interval features
    imputed_interval_df=impute_numerical_features(
        df=data_df,
        features=feature_types_dict['Interval'],
        knn_imputer=knn_imputer
    )

    # Join the data back together
    data_df=pd.concat([encoded_categorical_features_df, imputed_interval_df], axis=1)

    return data_df


def impute_numerical_features(
        df:pd.DataFrame,
        features:list,
        knn_imputer:Callable
) -> pd.DataFrame:

    '''Takes a set of numerical features, fills NAN with KNN imputation, returns clean features
    as Pandas dataframe.'''

    # Select all of the numeric columns for input into imputation
    numerical_df=df.select_dtypes(include='number').copy()

    # Impute missing values
    imputed_data=knn_imputer.transform(numerical_df)

    # Re-build dataframe
    imputed_df=pd.DataFrame(
        imputed_data,
        columns=numerical_df.columns
    )

    # Select only the target features
    imputed_df=imputed_df[features].copy()

    # Fix the index
    imputed_df.set_index(df.index, inplace=True)

    # Set the types
    imputed_df=imputed_df.astype('float64').copy()

    return imputed_df


def engineer_features(
        data_df:pd.DataFrame,
        survival_model_assets:str,
        kld_model_assets:str,
        classifier_model_file:str
) -> pd.DataFrame:

    '''Main function to run feature engineering operations.'''

    #######################################################
    # ASSET LOADING #######################################
    #######################################################

    with open(survival_model_assets, 'rb') as input_file:
        assets=pickle.load(input_file)

    coxph_features=assets['coxph_features']
    waft_features=assets['weibullaft_features']
    coxph_model=assets['coxph_model']
    waft_model=assets['weibullaft_model']

    with open(kld_model_assets, 'rb') as input_file:
        kld_models=pickle.load(input_file)

    with open(classifier_model_file, 'rb') as input_file:
        classifier_model=pickle.load(input_file)

    #######################################################
    # FEATURE ENGINEERING #################################
    #######################################################

    data_df=cox_ph(
        data_df=data_df,
        coxph_features=coxph_features,
        coxph_model=coxph_model
    )

    print(f' CoxPH features added, nan count: {data_df.isnull().sum().sum()}')

    data_df=weibull_aft(
        data_df=data_df,
        waft_features=waft_features,
        waft_model=waft_model
    )

    print(f' Weibul AFT features added, nan count: {data_df.isnull().sum().sum()}')

    data_df=kullback_leibler_score(
        data_df=data_df,
        kld_models=kld_models
    )

    print(f' Kullback-Leibler divergence scores added, nan count: {data_df.isnull().sum().sum()}')

    data_df=learned_efs(
        data_df=data_df,
        classifier_model=classifier_model
    )

    print(f' Learned EFS probability added, nan count: {data_df.isnull().sum().sum()}')
    print()

    return data_df


def cox_ph(
        data_df:pd.DataFrame,
        coxph_features:list,
        coxph_model:Callable
) -> pd.DataFrame:

    '''Adds Cox PH features.'''

    survival_functions=coxph_model.predict_survival_function(data_df[coxph_features])
    partial_hazards=coxph_model.predict_partial_hazard(data_df[coxph_features])
    data_df['coxph_survival']=survival_functions.iloc[-1]
    data_df['coxph_partial_hazard']=partial_hazards

    return data_df


def weibull_aft(
        data_df:pd.DataFrame,
        waft_features:list,
        waft_model:Callable
) -> pd.DataFrame:

    '''Adds Weibull AFT features.'''

    survival_functions=waft_model.predict_survival_function(data_df[waft_features])
    expectations=waft_model.predict_expectation(data_df[waft_features])
    data_df['weibullaft_survival']=survival_functions.iloc[-1]
    data_df['weibullaft_expectation']=expectations

    return data_df


def kullback_leibler_score(
        data_df:pd.DataFrame,
        kld_models:dict
) -> pd.DataFrame:

    '''Adds Kullback-Leibler divergence scores for Cox PH and Weibull AFT features'''

    for feature, kernel_density_estimate in kld_models.items():

        data=np.array(data_df[feature])
        workers=mp.cpu_count() - 4

        with mp.Pool(workers) as p:
            kld_score=np.concatenate(p.map(kernel_density_estimate, np.array_split(data, workers)))

        data_df[f'{feature}_kld']=kld_score

    return data_df


def learned_efs(
        data_df:pd.DataFrame,
        classifier_model:Callable
) -> pd.DataFrame:

    '''Adds learned EFS probability feature.'''

    data_df['learned_efs']=classifier_model.predict_proba(data_df)[:,1]

    return data_df


def predict(data_df:pd.DataFrame, assets_file:str) -> list:
    '''Main inference function.'''

    # Load model
    with open(assets_file, 'rb') as input_file:
        assets=pickle.load(input_file)

    # Unpack the assets
    scaler=assets['scaler']
    model=assets['model']

    # Scale the data
    data_df=scaler.transform(data_df)

    # Make predictions
    predictions=model.predict(data_df)

    return predictions


if __name__ == '__main__':

    print()
    print('########################################################')
    print('# HSCT Survival inference pipeline #####################')
    print('########################################################')
    print()

    # Read the data
    data_df=pd.read_csv(INPUT_DATA_FILE)

    # Save the id column for later
    ids=data_df['ID']

    # Drop unnecessary columns, if present
    data_df.drop(['efs', 'efs_time', 'ID'], axis=1, inplace=True, errors='ignore')

    print(f'Loaded data from: {INPUT_DATA_FILE}')
    print(f'Data shape: {data_df.shape}')
    print(f'nan count: {data_df.isnull().sum().sum()}')
    print()

    # Translate feature level values and replace string NAN
    # placeholders with actual np.nan.
    print('Starting data cleaning.')
    start_time=time.time()

    data_df=clean_data(
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

    data_df=encode_data(
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

    data_df=engineer_features(
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

    predictions=predict(
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