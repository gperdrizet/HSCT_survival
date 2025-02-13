import os
import pickle
import pandas as pd
import xgboost as xgb

pd.set_option('display.max_rows', 500)

# Figure out if we are in a Kaggle notebook or not based on the current path and set file paths accordingly
if os.getcwd() == '/kaggle/working':

    print('Running on Kaggle')

    # Data files
    testing_data_file='/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
    submission_file='submission.csv'

    # Feature info files
    feature_types_dict_file='/kaggle/input/d/gperdrizet/hsct-survival-xgboost-regression-model-assets/01.1-feature_type_dict.pkl'
    feature_value_translation_dicts_file='/kaggle/input/d/gperdrizet/hsct-survival-xgboost-regression-model-assets/01.1-feature_value_translation_dicts.pkl'
    nan_placeholders_dict_file='/kaggle/input/d/gperdrizet/hsct-survival-xgboost-regression-model-assets/01.1-nan_placeholders_list.pkl'

    # Model files
    model_file='/kaggle/input/d/gperdrizet/hsct-survival-xgboost-regression-model-assets/05.2-XGBoost_tuned.pkl'
    knn_imputer_numerical_features_file='/kaggle/input/d/gperdrizet/hsct-survival-xgboost-regression-model-assets/02.1-KNN_imputer_numerical_features.pkl'
    knn_imputer_categorical_features_file='/kaggle/input/d/gperdrizet/hsct-survival-xgboost-regression-model-assets/02.1-KNN_imputer_categorical_features.pkl'
    one_hot_encoder_nan_imputed_file='/kaggle/input/d/gperdrizet/hsct-survival-xgboost-regression-model-assets/02.1-multicollinear_one_hot_encoder_nan_imputed.pkl'

else:
    # Data files
    testing_data_file='../../data/raw/test.csv'
    submission_file='../predictions/xgboost_submission.csv'

    # Feature info files
    feature_types_dict_file='../../data/processed/01.1-feature_type_dict.pkl'
    feature_value_translation_dicts_file='../../data/processed/01.1-feature_value_translation_dicts.pkl'
    nan_placeholders_dict_file='../../data/processed/01.1-nan_placeholders_list.pkl'

    # Model files
    model_file='../../models/05.2-XGBoost_tuned.pkl'
    knn_imputer_numerical_features_file='../../models/02.1-KNN_imputer_numerical_features.pkl'
    knn_imputer_categorical_features_file='../../models/02.1-KNN_imputer_categorical_features.pkl'
    one_hot_encoder_nan_imputed_file='../../models/02.1-multicollinear_one_hot_encoder_nan_imputed.pkl'


if __name__=='__main__':

    ###################################
    # LOAD ASSETS #####################
    ###################################

    # Missing data placeholder strings
    with open(nan_placeholders_dict_file, 'rb') as input_file:
        nan_placeholders_dict=pickle.load(input_file)

    print(f'Missing data place-holder strings: {list(nan_placeholders_dict.keys())}\n')

    # Feature variable type classifications
    with open(feature_types_dict_file, 'rb') as input_file:
        feature_types_dict=pickle.load(input_file)

    print('Feature types:\n')
    for feature_type, features in feature_types_dict.items():
        print(f'{feature_type}\n{features}\n')

    # Feature value translation dictionaries for ordinal categorical features
    with open(feature_value_translation_dicts_file, 'rb') as input_file:
        feature_value_translation_dicts=pickle.load(input_file)

    print('Ordinal categorical feature value translations:')

    for feature, translation_dict in feature_value_translation_dicts.items():
        print(f'\n{feature}:')
        for old_value, new_value in translation_dict.items():
            print(f'  {old_value} -> {new_value}')

    print()

    # KNN imputer model for label encoded categorical features
    with open(knn_imputer_categorical_features_file, 'rb') as input_file:
        categorical_knn_imputer=pickle.load(input_file)

    # KNN imputer model for numerical features
    with open(knn_imputer_numerical_features_file, 'rb') as input_file:
        numerical_knn_imputer=pickle.load(input_file)

    # One-hot encoder for NAN imputed data
    with open(one_hot_encoder_nan_imputed_file, 'rb') as input_file:
        one_hot_encoder=pickle.load(input_file)

    # Load the model
    with open(model_file, 'rb') as input_file:
        model=pickle.load(input_file)

    # Load the dataset
    data_df=pd.read_csv(testing_data_file)
        

    ###################################
    # Prep. data ######################
    ###################################

    # Set the ID column as the index
    data_df.set_index('ID', drop=True, inplace=True)

    # Translate categorical ordinal to numerical ordinal
    for feature, translation_dict in feature_value_translation_dicts.items():
        data_df.replace(translation_dict, inplace=True)

    # Translate missing value string to np.nan
    data_df.replace(nan_placeholders_dict, inplace=True)
    print(f'{data_df.info()}\n')
    print(f'Data: {data_df.shape}')

    # Split the data into numerical/ordinal and true categorical
    feature_types_dict['True numerical features'].remove('efs_time')
    feature_types_dict['True categorical features'].remove('efs')
    numerical_features_df=data_df[feature_types_dict['Ordinal features']+feature_types_dict['True numerical features']].copy()
    categorical_features_df=data_df[feature_types_dict['True categorical features']].copy()
    print(f'Numerical data: {numerical_features_df.shape}')
    print(f'Categorical data: {categorical_features_df.shape}')

    # Label encode categorical features, preserving nans
    translation_dicts={}

    for feature in categorical_features_df.columns:

        feature_level_counts=categorical_features_df[feature].value_counts()
        translation_dict={}

        for i, level in enumerate(feature_level_counts.index):
            translation_dict[level]=str(i)

        categorical_features_df[feature]=categorical_features_df[feature].replace(translation_dict)
        translation_dicts[feature]=translation_dict

    # Impute missing values categorical features
    imputed_categorical_features=categorical_knn_imputer.transform(categorical_features_df)

    # Re-build dataframe
    categorical_features_df=pd.DataFrame(
        imputed_categorical_features, 
        columns=categorical_features_df.columns
    )

    categorical_features_df.set_index(data_df.index, inplace=True)

    # Round to nearest int
    categorical_features_df=categorical_features_df.map(lambda x: int(round(x, 0)) if isinstance(x, (int, float)) else x)
    print(f'Imputed categorical data: {categorical_features_df.shape}')

    # Get categories back
    for feature in categorical_features_df.columns:
        translation_dict={int(value): key for key, value in translation_dicts[feature].items()}
        categorical_features_df[feature]=categorical_features_df[feature].replace(translation_dict)

    # Encode the categorical features
    encoded_imputed_categorical_features=one_hot_encoder.transform(categorical_features_df)

    # Re-build dataframe
    categorical_features_df=pd.DataFrame(
        encoded_imputed_categorical_features, 
        columns=one_hot_encoder.get_feature_names_out()
    )

    categorical_features_df.set_index(data_df.index, inplace=True)
    print(f'Encoded imputed categorical data: {categorical_features_df.shape}')

    # Impute missing values in the numerical features
    imputed_numerical_features=numerical_knn_imputer.transform(numerical_features_df)

    # Re-build dataframes
    numerical_features_df=pd.DataFrame(
        imputed_numerical_features, 
        columns=numerical_features_df.columns
    )

    numerical_features_df.set_index(data_df.index, inplace=True)
    print(f'Imputed numerical data: {numerical_features_df.shape}')

    # Set the types
    categorical_features_df=categorical_features_df.astype('int32').copy()
    numerical_features_df=numerical_features_df.astype('float64').copy()
    numerical_features_df['year_hct']=numerical_features_df['year_hct'].astype('int32').copy()

    # Join categorical and numerical data
    data_df=pd.concat([numerical_features_df, categorical_features_df], axis=1)
    print(f'Re-combined data: {data_df.shape}\n')

    # Fix the efs column name
    data_df.rename(columns={'efs_1': 'efs'}, inplace=True)

    # Clean up column names
    data_df.columns=data_df.columns.str.replace('[\\[\\]<]', '', regex=True)

    print(f'{data_df.head().transpose()}\n')


    ###################################
    # Predict and submit ##############
    ###################################

    # Make predictions
    data=xgb.DMatrix(data_df)
    predictions=model.predict(data)

    # Assemble submission
    submission_df=pd.DataFrame.from_dict({'ID': data_df.index, 'prediction': predictions})

    # Save
    submission_df.to_csv(submission_file, index=False)