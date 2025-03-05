'''Runs EFS time inference using tuned XGBoost regression model.'''

import pickle
import pandas as pd
from xgboost import DMatrix


def run(data_df:pd.DataFrame, model_file:str) -> pd.DataFrame:
    '''Main inference function.'''

    #######################################################
    # ASSET LOADING #######################################
    #######################################################

    with open(model_file, 'rb') as input_file:
        model=pickle.load(input_file)

    #######################################################
    # PREDICTION ##########################################
    #######################################################

    dfeatures=DMatrix(data_df)
    predictions=model.predict(dfeatures)

    return predictions