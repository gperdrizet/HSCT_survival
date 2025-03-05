'''Function to clean up features and translate levels for encoding.'''

import pickle
import pandas as pd

def run(
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
