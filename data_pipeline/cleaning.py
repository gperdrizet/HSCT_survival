'''Function to clean up features and translate levels for encoding.'''

import pickle
import pandas as pd

def run(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Main function to do feature cleaning.'''

    feature_value_translation_dicts_file='../data/processed/01.1-feature_value_translation_dicts.pkl'
    nan_placeholders_dict_file='../data/processed/01.1-nan_placeholders_list.pkl'

    with open(feature_value_translation_dicts_file, 'rd') as input_file:
        feature_value_translation_dicts=pickle.load(input_file)

    for feature, translation_dict in feature_value_translation_dicts.items():
        data_df[feature]=data_df[feature].replace(translation_dict)

    
    with open(nan_placeholders_dict_file, 'rb') as input_file:
        missing_values=pickle.load(input_file)

    data_df.replace(missing_values, inplace=True)

    return data_df