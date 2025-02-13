import os
import pickle
import pandas as pd

# Figure out if we are in a Kaggle notebook or not based on the current path and set file paths accordingly
if os.getcwd() == '/kaggle/working':
    print('Running on Kaggle')
    data_dictionary_file='/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'
    model_file='/kaggle/input/hsct-survival-uncensored-regression/03.4-uncensored_linear_regression.pkl'
    knn_imputer_file='/kaggle/input/hsct-survival-uncensored-regression/02-KNN_imputer.pkl'
    one_hot_encoder_file='/kaggle/input/hsct-survival-uncensored-regression/02-one_hot_encoder.pkl'
    testing_data_file='/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
    submission_file='submission.csv'

else:
    data_dictionary_file='../../data/raw/data_dictionary.csv'
    model_file='../../models/03.4-uncensored_linear_regression.pkl'
    knn_imputer_file='../../models/02-KNN_imputer.pkl'
    one_hot_encoder_file='../../models/02-one_hot_encoder.pkl'
    testing_data_file='../../data/raw/test.csv'
    submission_file='../predictions/polynomial_regression_submission.csv'


if __name__=='__main__':

    ###################################
    # LOAD ASSETS #####################
    ###################################

    # Load the model
    with open(model_file, 'rb') as input_file:
        model=pickle.load(input_file)

    # Load the one-hot encoder
    with open(one_hot_encoder_file, 'rb') as input_file:
        encoder=pickle.load(input_file)

    # Load the KNN imputer
    with open(knn_imputer_file, 'rb') as input_file:
        imputer=pickle.load(input_file)

    # Load testing data and column definitions
    testing_data=pd.read_csv(testing_data_file)
    data_dictionary=pd.read_csv(data_dictionary_file)

    ###################################
    # PROCESS TESTING DATA ############
    ###################################

    # Save the ID and drop
    testing_ids=testing_data['ID']
    testing_data.drop('ID', axis=1, inplace=True)
    print(f'Testing features: {testing_data.shape}')

    # Get lists of categorical and numerical column names
    categorical_feature_names=data_dictionary['variable'][data_dictionary['type'] == 'Categorical']
    numerical_feature_names=data_dictionary['variable'][data_dictionary['type'] == 'Numerical']

    # Remove the feature column from the column names lists
    categorical_feature_names=categorical_feature_names[categorical_feature_names != 'efs']
    numerical_feature_names=numerical_feature_names[numerical_feature_names != 'efs_time']

    # Split the testing dataframe
    testing_categorical_df=testing_data[categorical_feature_names].copy()
    testing_numerical_df=testing_data[numerical_feature_names].copy()

    print(f'Testing numerical features: {testing_numerical_df.shape}')
    print(f'Testing categorical features: {testing_categorical_df.shape}')

    # Replace NAN with 'Missing' string
    testing_categorical_df.fillna('Missing', inplace=True)
    print(f'Testing categorical features: {testing_categorical_df.shape}')

    # Fill missing data
    testing_numerical_data=imputer.transform(testing_numerical_df)

    # Re-build dataframe
    testing_numerical_df=pd.DataFrame(testing_numerical_data, columns=testing_numerical_df.columns)
    print(f'Testing numerical features: {testing_numerical_df.shape}')

    # Encode the features
    testing_categorical_data=encoder.transform(testing_categorical_df)

    # Rebuild the dataframe
    feature_names=encoder.get_feature_names_out()
    testing_categorical_df=pd.DataFrame(testing_categorical_data, columns=feature_names)
    print(f'Testing categorical features: {testing_categorical_df.shape}')

    # Recombine numerical and categorical features
    testing_features_df=pd.concat(
        [
            testing_numerical_df.reset_index(drop=True), 
            testing_categorical_df.reset_index(drop=True)
        ],
        axis=1
    )

    print(f'Testing features: {testing_features_df.shape}')

    predicted_efs_time=model.predict(testing_features_df)
    predictions_df=pd.DataFrame.from_dict({'ID': testing_ids, 'prediction': predicted_efs_time.flatten()})
    predictions_df.describe()
    predictions_df.to_csv(submission_file, index=False)
