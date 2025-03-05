'''Encodes data for inference.'''

import pickle
from typing import Callable
import pandas as pd

def run(
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
