'''Functions to encode and transform data for feature engineering.'''

import pickle
from typing import Callable

import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import KNNImputer

from .. import configuration as config

def run() -> dict:
    '''Main function to do data encoding.'''

    # Load data
    with open(config.DATA_CLEANING_RESULT, 'rb') as input_file:
        data_df=pickle.load(input_file)

    # Save the labels
    labels_df=data_df[['efs', 'efs_time']].copy()

    # Drop labels and IDs
    features_df=data_df.drop(['efs', 'efs_time', 'ID'], axis=1).copy()

    # Load feature type definitions
    with open(config.DATA_CLEANING_ASSETS, 'rb') as input_file:
        assets=pickle.load(input_file)

    feature_types_dict=assets['feature_types']

    # Get categorical features
    categorical_df=features_df[feature_types_dict['Nominal'] + feature_types_dict['Ordinal']]
    
    # Encode the nominal & ordinal features
    target_encoder=TargetEncoder(target_type='continuous')
    encoded_categorical_features=target_encoder.fit_transform(categorical_df, labels_df['efs_time'])

    # Rebuild the dataframes
    encoded_categorical_features_df=pd.DataFrame(
        encoded_categorical_features,
        columns=feature_types_dict['Nominal'] + feature_types_dict['Ordinal']
    )

    # Clean NANs in the interval features with imputation
    interval_df=impute_numerical_features(
        df=data_df,
        features=feature_types_dict['Interval'],
        knn_imputer=KNNImputer(n_neighbors=5, weights='distance')
    )

    # Join the data back together
    features_df=pd.concat([encoded_categorical_features_df, interval_df], axis=1)

    # Assemble dataset dictionary
    data={
        'features': features_df,
        'labels': labels_df
    }

    return data


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
    imputed_data=knn_imputer.fit_transform(numerical_df)

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