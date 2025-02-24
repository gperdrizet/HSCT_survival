'''Collection of functions implementing various encoding schemes.'''

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer


def one_hot_nan_encoded(
        training_df:pd.DataFrame,
        testing_df:pd.DataFrame,
        features:list,
        models_path:str,
        one_hot_drop:str='first'
) -> pd.DataFrame:
    
    '''One hot encodes features using a 'missing' level for any NANs.
    Returns dataframe containing encoded features only.'''

    print(f'\nOne-hot encoding input data: {training_df.shape}')

    # Isolate target features
    training_df=training_df[features].copy()
    testing_df=testing_df[features].copy()
    print(f'Feature data: {training_df.shape}')

    # Translate NAN to 'missing'
    training_df.replace({np.nan: 'missing'}, inplace=True)
    testing_df.replace({np.nan: 'missing'}, inplace=True)

    # Encode the features
    encoder=OneHotEncoder(
        drop=one_hot_drop,
        min_frequency=5,
        handle_unknown='infrequent_if_exist',
        sparse_output=False
    )

    encoder.fit(training_df)
    encoded_training_data=encoder.transform(training_df)
    encoded_testing_data=encoder.transform(testing_df)

    # Save the encoder
    with open(f'{models_path}/01.2-one_hot_encoder_nan_encoded.pkl', 'wb') as output_file:
        pickle.dump(encoder, output_file)

    # Rebuild the dataframes
    encoded_training_df=pd.DataFrame(
        encoded_training_data,
        columns=encoder.get_feature_names_out()
    )

    encoded_testing_df=pd.DataFrame(
        encoded_testing_data,
        columns=encoder.get_feature_names_out()
    )

    # Fix the index
    encoded_training_df.set_index(training_df.index, inplace=True)
    encoded_testing_df.set_index(testing_df.index, inplace=True)

    # Set the types
    encoded_training_df=encoded_training_df.astype('int32').copy()
    encoded_testing_df=encoded_testing_df.astype('int32').copy()
    print(f'One-hot encoded feature data: {encoded_testing_df.shape}')

    return encoded_training_df, encoded_testing_df


def one_hot_encode_nan_imputed(
        training_df:pd.DataFrame,
        testing_df:pd.DataFrame,
        features:list,
        models_path:str,
        one_hot_drop:str='first',
        knn_neighbors:int=5
) -> pd.DataFrame:
    
    '''Takes dataframe and list of features. Handles NAN by first label 
    encoding, then KNN imputing missing values and finally one-hot encodes. 
    Returns dataframe containing encoded features only.'''

    print(f'\nOne-hot encoding input data: {training_df.shape}')

    # Isolate target features
    training_df=training_df[features].copy()
    testing_df=testing_df[features].copy()
    print(f'Feature data: {training_df.shape}')

    # Ordinal encode features, preserving NANs
    encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(training_df)
    encoded_training_data=encoder.transform(training_df)
    encoded_testing_data=encoder.transform(testing_df)

    # Save the encoder
    with open(f'{models_path}/01.2-ordinal_encoder_nan_imputed.pkl', 'wb') as output_file:
        pickle.dump(encoder, output_file)

    # Impute missing values in ordinal encoded features
    imputer=KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
    imputer.fit(encoded_training_data)
    imputed_training_data=imputer.transform(encoded_training_data)
    imputed_testing_data=imputer.transform(encoded_testing_data)

    # Save the imputer
    with open(f'{models_path}/01.2-one_hot_imputer_nan_imputed.pkl', 'wb') as output_file:
        pickle.dump(imputer, output_file)

    # Round to the nearest int
    imputed_training_data=np.rint(imputed_training_data)
    imputed_testing_data=np.rint(imputed_testing_data)

    # Re-build dataframes
    imputed_training_df=pd.DataFrame(
        imputed_training_data, 
        columns=training_df.columns
    )

    imputed_testing_df=pd.DataFrame(
        imputed_testing_data, 
        columns=testing_df.columns
    )

    # One hot encode the imputed features
    encoder=OneHotEncoder(
        drop=one_hot_drop,
        min_frequency=5,
        handle_unknown='infrequent_if_exist',
        sparse_output=False
    )

    encoder.fit(imputed_training_df)
    encoded_training_data=encoder.transform(imputed_training_df)
    encoded_testing_data=encoder.transform(imputed_testing_df)

    # Save the encoder
    with open(f'{models_path}/01.2-one_hot_encoder_nan_imputed.pkl', 'wb') as output_file:
        pickle.dump(encoder, output_file)

    # Re-build dataframes
    encoded_training_df=pd.DataFrame(
        encoded_training_data, 
        columns=encoder.get_feature_names_out()
    )

    encoded_testing_df=pd.DataFrame(
        encoded_testing_data, 
        columns=encoder.get_feature_names_out()
    )

    # Fix index
    encoded_training_df.set_index(training_df.index, inplace=True)
    encoded_testing_df.set_index(testing_df.index, inplace=True)

    # Set dtype
    encoded_training_df=encoded_training_df.astype('int32').copy()
    encoded_testing_df=encoded_testing_df.astype('int32').copy()
    print(f'On-hot encoded, imputed feature data: {encoded_training_df.shape}')

    return encoded_training_df, encoded_testing_df


def ordinal_encode_nan_encoded(
        training_df:pd.DataFrame,
        testing_df:pd.DataFrame,
        features:list,
        models_path:str
) -> pd.DataFrame:
    
    '''Ordinal encodes features using a 'missing' level for any NANs.
    Returns dataframe containing encoded features only.'''

    print(f'\nOrdinal encoding input data: {training_df.shape}')

    # Isolate target features
    training_df=training_df[features].copy()
    testing_df=testing_df[features].copy()
    print(f'Feature data: {training_df.shape}')

    # Translate NAN to 'missing'
    training_df.replace({np.nan: 'missing'}, inplace=True)
    testing_df.replace({np.nan: 'missing'}, inplace=True)
    
    # Make sure everything is string
    training_df=training_df.astype(str)
    testing_df=testing_df.astype(str)

    # Encode the features
    encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(training_df)
    encoded_training_data=encoder.transform(training_df)
    encoded_testing_data=encoder.transform(testing_df)

    # Save the encoder
    with open(f'{models_path}/01.2-ordinal_encoder_nan_encoded.pkl', 'wb') as output_file:
        pickle.dump(encoder, output_file)

    # Rebuild the dataframes
    encoded_training_df=pd.DataFrame(
        encoded_training_data,
        columns=encoder.get_feature_names_out()
    )

    encoded_testing_df=pd.DataFrame(
        encoded_testing_data,
        columns=encoder.get_feature_names_out()
    )

    # Fix the index
    encoded_training_df.set_index(training_df.index, inplace=True)
    encoded_testing_df.set_index(testing_df.index, inplace=True)

    # Set the types
    encoded_training_df=encoded_training_df.astype('int32').copy()
    encoded_testing_df=encoded_testing_df.astype('int32').copy()
    print(f'Ordinal encoded feature data: {encoded_training_df.shape}')

    return encoded_training_df, encoded_testing_df


def ordinal_encode_nan_imputed(
        training_df:pd.DataFrame,
        testing_df:pd.DataFrame,
        features:list,
        models_path:str,
        knn_neighbors:int=5
) -> pd.DataFrame:

    print(f'\nOrdinal encoding input data: {training_df.shape}')

    # Isolate target features
    training_df=training_df[features].copy()
    testing_df=testing_df[features].copy()
    print(f'Feature data: {training_df.shape}')

    # Make sure everything is string
    training_df=training_df.astype(str)
    testing_df=testing_df.astype(str)

    # Encode the features
    encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(training_df)
    encoded_training_data=encoder.transform(training_df)
    encoded_testing_data=encoder.transform(testing_df)

    # Save the encoder
    with open(f'{models_path}/01.2-ordinal_encoder_nan_imputed.pkl', 'wb') as output_file:
        pickle.dump(encoder, output_file)

    print(f'Ordinal encoded feature data: {encoded_training_data.shape}')

    # Impute missing values in ordinal encoded features
    imputer=KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
    imputer.fit(encoded_training_data)
    imputed_training_data=imputer.transform(encoded_training_data)
    imputed_testing_data=imputer.transform(encoded_testing_data)

    # Save the imputer
    with open(f'{models_path}/01.2-ordinal_imputer_nan_imputed.pkl', 'wb') as output_file:
        pickle.dump(imputer, output_file)

    # Round to nearest int
    imputed_training_data=np.rint(imputed_training_data)
    imputed_testing_data=np.rint(imputed_testing_data)

    # Re-build dataframes
    imputed_training_df=pd.DataFrame(
        imputed_training_data, 
        columns=training_df.columns
    )

    imputed_testing_df=pd.DataFrame(
        imputed_testing_data, 
        columns=testing_df.columns
    )

    # Fix the index
    imputed_training_df.set_index(training_df.index, inplace=True)
    imputed_testing_df.set_index(testing_df.index, inplace=True)

    # Set the types
    imputed_training_df=imputed_training_df.astype('int32').copy()
    imputed_testing_df=imputed_testing_df.astype('int32').copy()
    print(f'Imputed, ordinal encoded feature data: {imputed_training_df.shape}')

    return imputed_training_df, imputed_testing_df


def impute_numerical_features(
        training_df: pd.DataFrame,
        testing_df: pd.DataFrame,
        features:str,
        models_path:str,
        knn_neighbors:int=5
) -> pd.DataFrame:
    
    '''Takes a set of numerical features, fills NAN with KNN imputation, returns clean features
    as Pandas dataframe.'''

    # Select all of the numeric columns for input into imputation
    training_df=training_df.select_dtypes(include='number').copy()
    testing_df=testing_df.select_dtypes(include='number').copy()
    print(f'\nImputation input data: {training_df.shape}')

    # Impute missing values
    imputer=KNNImputer(n_neighbors=knn_neighbors, weights='uniform')
    imputer.fit(training_df)
    imputed_training_data=imputer.transform(training_df)
    imputed_testing_data=imputer.transform(testing_df)

    # Save the imputer
    with open(f'{models_path}/01.2-numerical_imputer.pkl', 'wb') as output_file:
        pickle.dump(imputer, output_file)

    # Re-build dataframes
    imputed_training_df=pd.DataFrame(
        imputed_training_data, 
        columns=training_df.columns
    )

    # Re-build dataframes
    imputed_testing_df=pd.DataFrame(
        imputed_testing_data, 
        columns=testing_df.columns
    )

    # Select only the target features
    imputed_training_df=imputed_training_df[features].copy()
    imputed_testing_df=imputed_testing_df[features].copy()

    # Fix the index
    imputed_training_df.set_index(training_df.index, inplace=True)
    imputed_testing_df.set_index(testing_df.index, inplace=True)

    # Set the types
    imputed_training_df=imputed_training_df.astype('float64').copy()
    imputed_testing_df=imputed_testing_df.astype('float64').copy()
    print(f'Imputed numerical data: {imputed_training_df.shape}')

    return imputed_training_df, imputed_testing_df