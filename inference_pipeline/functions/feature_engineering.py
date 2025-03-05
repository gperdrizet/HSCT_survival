'''Does feature engineering for inference, adds the following:

    1. Cox PH surivival probability at study end
    2. Cox PH hazard ratio
    3. Weibull AFT survival probability at study end
    4. Weibull AFT expectation value
    5. Kullback-Leibler divergence score of above values
    6. Learned EFS probability
'''

import pickle
from typing import Callable
import multiprocessing as mp

import numpy as np
import pandas as pd
from xgboost import DMatrix


def run(
        data_df:pd.DataFrame,
        coxph_features_file:str,
        waft_features_file:str,
        coxph_model_file:str,
        waft_model_file:str,
        kld_models_file:str,
        efs_model_file:str
) -> pd.DataFrame:

    '''Main function to run feature engineering operations.'''

    #######################################################
    # ASSET LOADING #######################################
    #######################################################

    with open(coxph_features_file, 'rb') as input_file:
        coxph_features=pickle.load(input_file)

    with open(waft_features_file, 'rb') as input_file:
        waft_features=pickle.load(input_file)

    with open(coxph_model_file, 'rb') as input_file:
        coxph_model=pickle.load(input_file)

    with open(waft_model_file, 'rb') as input_file:
        waft_model=pickle.load(input_file)

    with open(kld_models_file, 'rb') as input_file:
        kld_models=pickle.load(input_file)

    with open(efs_model_file, 'rb') as input_file:
        efs_model=pickle.load(input_file)

    #######################################################
    # FEATURE ENGINEERING #################################
    #######################################################

    data_df=cox_ph(
        data_df=data_df,
        coxph_features=coxph_features,
        coxph_model=coxph_model
    )

    print(f' CoxPH features added, nan count: {data_df.isnull().sum().sum()}')

    data_df=weibull_aft(
        data_df=data_df,
        waft_features=waft_features,
        waft_model=waft_model
    )

    print(f' Weibul AFT features added, nan count: {data_df.isnull().sum().sum()}')

    data_df=kullback_leibler_score(
        data_df=data_df,
        kld_models=kld_models
    )

    print(f' Kullback-Leibler divergence scores added, nan count: {data_df.isnull().sum().sum()}')

    data_df=learned_efs(
        data_df=data_df,
        efs_model=efs_model
    )

    print(f' Learned EFS probability added, nan count: {data_df.isnull().sum().sum()}')
    print()

    return data_df


def cox_ph(
        data_df:pd.DataFrame,
        coxph_features:list,
        coxph_model:Callable
) -> pd.DataFrame:

    '''Adds Cox PH features.'''

    survival_functions=coxph_model.predict_survival_function(data_df[coxph_features])
    partial_hazards=coxph_model.predict_partial_hazard(data_df[coxph_features])
    data_df['CoxPH survival']=survival_functions.iloc[-1]
    data_df['CoxPH partial hazard']=partial_hazards

    return data_df


def weibull_aft(
        data_df:pd.DataFrame,
        waft_features:list,
        waft_model:Callable
) -> pd.DataFrame:

    '''Adds Weibull AFT features.'''

    survival_functions=waft_model.predict_survival_function(data_df[waft_features])
    expectations=waft_model.predict_expectation(data_df[waft_features])
    data_df['WeibullAFT survival']=survival_functions.iloc[-1]
    data_df['WeibullAFT expectation']=expectations

    return data_df


def kullback_leibler_score(
        data_df:pd.DataFrame,
        kld_models:dict
) -> pd.DataFrame:

    '''Adds Kullback-Leibler divergence scores for Cox PH and Weibull AFT features'''

    for feature, kernel_density_estimate in kld_models.items():

        data=np.array(data_df[feature])
        workers=mp.cpu_count() - 4

        with mp.Pool(workers) as p:
            kld_score=np.concatenate(p.map(kernel_density_estimate, np.array_split(data, workers)))

        data_df[f'{feature} KLD']=kld_score

    return data_df


def learned_efs(
        data_df:pd.DataFrame,
        efs_model:Callable
) -> pd.DataFrame:

    '''Adds learned EFS probability feature.'''

    dfeatures=DMatrix(data_df)

    data_df['learned_efs']=efs_model.predict(dfeatures)

    return data_df
