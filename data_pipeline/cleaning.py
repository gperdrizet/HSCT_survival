'''Function to clean up features and translate levels for encoding.'''

import pickle
import pandas as pd

def run(
        data_df:pd.DataFrame,
        feature_level_dicts:str,
        nan_dicts:str
) -> pd.DataFrame:
    '''Main function to do feature cleaning.'''

    with open(feature_level_dicts, 'rb') as input_file:
        feature_value_translation_dicts=pickle.load(input_file)

    for feature, translation_dict in feature_value_translation_dicts.items():
        data_df[feature]=data_df[feature].replace(translation_dict)

    with open(nan_dicts, 'rb') as input_file:
        missing_values=pickle.load(input_file)

    data_df.replace(missing_values, inplace=True)

    return data_df
