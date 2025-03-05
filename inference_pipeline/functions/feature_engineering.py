'''Does feature engineering for inference, adds the following:

    1. Cox PH surivival probability at study end
    2. Cox PH hazard ratio
    3. Weibull AFT survival probability at study end
    4. Weibull AFT expectation value
    5. Kullback-Leibler divergence score of above values
    6. Learned EFS probability
'''

import pickle
import multiprocessing as mp
from typing import Callable

import numpy as np
import pandas as pd


def run(
        data_df:pd.DataFrame,
        survival_model_assets:str,
        kld_model_assets:str,
        classifier_model_file:str
) -> pd.DataFrame:

    '''Main function to run feature engineering operations.'''

    #######################################################
    # ASSET LOADING #######################################
    #######################################################

    with open(survival_model_assets, 'rb') as input_file:
        assets=pickle.load(input_file)

    coxph_features=assets['coxph_features']
    waft_features=assets['weibullaft_features']
    coxph_model=assets['coxph_model']
    waft_model=assets['weibullaft_model']

    with open(kld_model_assets, 'rb') as input_file:
        kld_models=pickle.load(input_file)

    with open(classifier_model_file, 'rb') as input_file:
        classifier_model=pickle.load(input_file)

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
        classifier_model=classifier_model
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
    data_df['coxph_survival']=survival_functions.iloc[-1]
    data_df['coxph_partial_hazard']=partial_hazards

    return data_df


def weibull_aft(
        data_df:pd.DataFrame,
        waft_features:list,
        waft_model:Callable
) -> pd.DataFrame:

    '''Adds Weibull AFT features.'''

    survival_functions=waft_model.predict_survival_function(data_df[waft_features])
    expectations=waft_model.predict_expectation(data_df[waft_features])
    data_df['weibullaft_survival']=survival_functions.iloc[-1]
    data_df['weibullaft_expectation']=expectations

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

        data_df[f'{feature}_kld']=kld_score

    return data_df


def learned_efs(
        data_df:pd.DataFrame,
        classifier_model:Callable
) -> pd.DataFrame:

    '''Adds learned EFS probability feature.'''

    data_df['learned_efs']=classifier_model.predict_proba(data_df)[:,1]

    return data_df
