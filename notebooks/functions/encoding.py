'''Collection of functions implementing various encoding schemes.'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer


def one_hot_nan_encoded(
        data_df:pd.DataFrame,
        features:list,
        one_hot_drop:str='first'
) -> pd.DataFrame:
    
    '''One hot encodes features using a 'missing' level for any NANs.
    Returns dataframe containing encoded features only.'''

    print(f'\nOne-hot encoding input data: {data_df.shape}')

    # Isolate target features
    data_df=data_df[features].copy()
    print(f'Feature data: {data_df.shape}')

    # Translate NAN to 'missing'
    data_df.replace({np.nan: 'missing'}, inplace=True)

    # Encode the features
    encoder=OneHotEncoder(drop=one_hot_drop, min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False)
    encoded_feature_data=encoder.fit_transform(data_df)

    # Rebuild the dataframe
    encoded_features_df=pd.DataFrame(
        encoded_feature_data,
        columns=encoder.get_feature_names_out()
    )

    # Fix the index
    encoded_features_df.set_index(data_df.index, inplace=True)

    # Set the types
    encoded_features_df=encoded_features_df.astype('int32').copy()
    print(f'One-hot encoded feature data: {encoded_features_df.shape}')

    return encoded_features_df


def one_hot_encode_nan_imputed(
        data_df:pd.DataFrame,
        features:list,
        one_hot_drop:str='first',
        knn_neighbors:int=5
) -> pd.DataFrame:
    
    '''Takes dataframe and list of features. Handles NAN by first label 
    encoding, then KNN imputing missing values and finally one-hot encodes. 
    Returns dataframe containing encoded features only.'''

    print(f'\nOne-hot encoding input data: {data_df.shape}')

    # Isolate target features
    data_df=data_df[features].copy()
    print(f'Feature data: {data_df.shape}')

    # Ordinal encode features, preserving NANs
    encoder=OrdinalEncoder()
    encoded_feature_data=encoder.fit_transform(data_df)

    # Impute missing values in ordinal encoded features
    imputer=KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
    imputed_feature_data=imputer.fit_transform(encoded_feature_data)

    # Round to the nearest int
    imputed_feature_data=np.rint(imputed_feature_data)

    # Re-build dataframe
    imputed_features_df=pd.DataFrame(
        imputed_feature_data, 
        columns=data_df.columns
    )

    # One hot encode the imputed features
    encoder=OneHotEncoder(drop=one_hot_drop, min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False)
    encoded_feature_data=encoder.fit_transform(imputed_features_df)

    # Re-build dataframe
    encoded_features_df=pd.DataFrame(
        encoded_feature_data, 
        columns=encoder.get_feature_names_out()
    )

    # Fix index
    encoded_features_df.set_index(data_df.index, inplace=True)

    # Set dtype
    encoded_features_df=encoded_features_df.astype('int32').copy()
    print(f'On-hot encoded, imputed feature data: {encoded_features_df.shape}')

    return encoded_features_df


def ordinal_encode_nan_encoded(data_df:pd.DataFrame, features:list) -> pd.DataFrame:
    '''Ordinal encodes features using a 'missing' level for any NANs.
    Returns dataframe containing encoded features only.'''

    print(f'\nOrdinal encoding input data: {data_df.shape}')

    # Isolate target features
    data_df=data_df[features].copy()
    print(f'Feature data: {data_df.shape}')

    # Translate NAN to 'missing'
    data_df.replace({np.nan: 'missing'}, inplace=True)

    # Make sure everything is string
    data_df=data_df.astype(str)

    # Encode the features
    encoder=OrdinalEncoder()
    encoded_feature_data=encoder.fit_transform(data_df)

    # Rebuild the dataframe
    encoded_features_df=pd.DataFrame(
        encoded_feature_data,
        columns=encoder.get_feature_names_out()
    )

    # Fix the index
    encoded_features_df.set_index(data_df.index, inplace=True)

    # Set the types
    encoded_features_df=encoded_features_df.astype('int32').copy()
    print(f'Ordinal encoded feature data: {encoded_features_df.shape}')

    return encoded_features_df


def ordinal_encode_nan_imputed(data_df:pd.DataFrame, features:list, knn_neighbors:int=5) -> pd.DataFrame:

    print(f'\nOrdinal encoding input data: {data_df.shape}')

    # Isolate target features
    data_df=data_df[features].copy()
    print(f'Feature data: {data_df.shape}')

    # Make sure everything is string
    data_df=data_df.astype(str)

    # Encode the features
    encoder=OrdinalEncoder()
    encoded_feature_data=encoder.fit_transform(data_df)
    print(f'Ordinal encoded feature data: {encoded_feature_data.shape}')

    # Impute missing values in label encoded features
    imputer=KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
    imputed_feature_data=imputer.fit_transform(encoded_feature_data)

    # Round to nearest int
    imputed_feature_data=np.rint(imputed_feature_data)

    # Re-build dataframe
    imputed_features_df=pd.DataFrame(
        imputed_feature_data, 
        columns=data_df.columns
    )

    # Fix the index
    imputed_features_df.set_index(data_df.index, inplace=True)

    # Set the types
    imputed_features_df=imputed_features_df.astype('int32').copy()
    print(f'Imputed, ordinal encoded feature data: {imputed_features_df.shape}')

    return imputed_features_df


def impute_numerical_features(data_df: pd.DataFrame, features:str, knn_neighbors:int=5) -> pd.DataFrame:
    '''Takes a set of numerical features, fills NAN with KNN imputation, returns clean features
    as Pandas dataframe.'''

    # Select all of the numeric columns for input into imputation
    data_df=data_df.select_dtypes(include='number').copy()
    print(f'\nImputation input data: {data_df.shape}')

    # Impute missing values
    imputer=KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
    imputed_feature_data=imputer.fit_transform(data_df)

    # Re-build dataframes
    imputed_features_df=pd.DataFrame(
        imputed_feature_data, 
        columns=data_df.columns
    )

    # Select only the target features
    imputed_features_df=imputed_features_df[features].copy()

    # Fix the index
    imputed_features_df.set_index(data_df.index, inplace=True)

    # Set the types
    imputed_features_df=imputed_features_df.astype('float64').copy()
    print(f'Imputed numerical data: {imputed_features_df.shape}')

    return imputed_features_df