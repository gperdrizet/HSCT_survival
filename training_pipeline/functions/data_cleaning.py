'''Data cleaning functions.'''

import pickle
import pandas as pd
from .. import configuration as config

def run() -> pd.DataFrame:
    '''Main function to run data cleaning.'''

    data_df=pd.read_csv(config.INPUT_DATA_FILE)

    with open(config.DATA_CLEANING_ASSETS, 'rb') as input_file:
        assets=pickle.load(input_file)

    for feature, translation_dict in assets['feature_levels'].items():
        data_df[feature]=data_df[feature].replace(translation_dict)

    data_df.replace(assets['nan_placeholders'], inplace=True)

    return data_df