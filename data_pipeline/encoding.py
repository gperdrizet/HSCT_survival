'''Encodes data for inference.'''

import pickle
import pandas as pd
from typing import Callable

def run(
        data_df:pd.DataFrame,
        feature_types_dict_file:str,
        target_encoder_file:str,
        power_transformer_file:str,
        knn_imputer_file: str
) -> pd.DataFrame:

    '''Main function to do data encoding.'''

    #######################################################
    # Asset loading #######################################
    #######################################################

    # Load the feature data type definitions
    with open(feature_types_dict_file, 'rb') as input_file:
        feature_types_dict=pickle.load(input_file)

    # Load the target encoder
    with open(target_encoder_file, 'rb') as input_file:
        target_encoder=pickle.load(input_file)

    # Load the power transformer
    with open(power_transformer_file, 'rb') as input_file:
        power_transformer=pickle.load(input_file)

    # Load the KNN imputer
    with open(knn_imputer_file, 'rb') as input_file:
        knn_imputer=pickle.load(input_file)

    # Get categorical features
    categorical_df=data_df[feature_types_dict['Nominal'] + feature_types_dict['Ordinal']]

    # Encode the nominal & ordinal features
    encoded_categorical_features=target_encoder.transform(categorical_df)

    # Rebuild the dataframe
    encoded_categorical_features_df=pd.DataFrame(
        encoded_categorical_features,
        columns=feature_types_dict['Nominal'] + feature_types_dict['Ordinal']
    )

    # Clean NANs in the interval features
    imputed_interval_df=impute_numerical_features(
        df=data_df,
        features=feature_types_dict['Interval'],
        knn_imputer=knn_imputer
    )

    # Join the data back together
    data_df=pd.concat([encoded_categorical_features_df, imputed_interval_df], axis=1)

    # Power transform the features
    transformed_data=power_transformer.transform(data_df)

    # Rebuild the dataframe
    data_df=pd.DataFrame(
        transformed_data,
        columns=data_df.columns
    )

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
