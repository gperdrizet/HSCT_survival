'''Data data processing pipeline. Handles data cleaning and encoding
and feature engineering for inference.'''
import pandas as pd
from data_pipeline import cleaning
# from data_pipeline import encoding

INPUT_DATA='./data/raw/train.csv'
FEATURE_LEVEL_DICTS='./data/processed/01.1-feature_value_translation_dicts.pkl'
NAN_DICTS='./data/processed/01.1-nan_placeholders_list.pkl'

def run():
    '''Main function to run data pipeline.'''

    data_df=pd.read_csv(INPUT_DATA)

    # Translate feature level values and replace string NAN
    # placeholders with actual np.nan
    data_df=cleaning.run(
        data_df=data_df,
        feature_level_dicts=FEATURE_LEVEL_DICTS,
        nan_dicts=NAN_DICTS
    )

    print(data_df.head().transpose())

    return data_df
