'''Data data processing pipeline. Handles data cleaning and encoding
and feature engineering for inference.'''

from cleaning import clean
from encoding import encode

def run(data_df):
    '''Main function to run data pipeline.'''

    data_df=clean(data_df)
    data_df=encode(data_df)
    
    return data_df
