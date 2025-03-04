'''Data data processing pipeline. Handles data cleaning and encoding
and feature engineering for inference.'''

from data_pipeline import cleaning
from data_pipeline import encoding

def run(data_df):
    '''Main function to run data pipeline.'''

    data_df=cleaning.run(data_df)
    data_df=encoding.run(data_df)

    return data_df
