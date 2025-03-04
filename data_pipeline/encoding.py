'''Encodes data for inference.'''

import pandas as pd

def run(data_df:pd.DataFrame) -> pd.DataFrame:
    '''Main function to do data encoding'''

    # Feature data type definition file
    feature_types_dict_file='../data/processed/01.1-feature_type_dict.pkl'
    
    return