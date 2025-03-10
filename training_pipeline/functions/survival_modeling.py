'''Functions to run survival modeling with the Lifelines package.'''

import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter

import configuration as config


def run() -> dict:
    '''Main function to run survival modeling.'''

    # Load data
    with open(config.DATA_ENCODING_RESULT, 'rb') as input_file:
        data=pickle.load(input_file)

    assets={}

    data, assets=coxph_model(data, assets)
    data, assets=waft_model(data, assets)

    with open(config.SURVIVAL_MODEL_ASSETS, 'wb') as output_file:
        pickle.dump(assets, output_file)

    return data


def coxph_model(data:dict, assets:dict) -> Tuple[dict, dict]:
    '''Runs Cox Proportional Hazards survival modeling.'''

    # Combine features and labels
    training_df=pd.concat([data['features'], data['labels']], axis=1)

    # Fit the model
    cph_model=CoxPHFitter()
    cph_model.fit(training_df, duration_col='efs_time', event_col='efs')

    # Select features by p-value
    feature_pvals=cph_model.summary['p']
    significant_features_df=training_df[feature_pvals[feature_pvals < 0.05].index].copy()

    # Save significant features
    assets['coxph_features']=list(significant_features_df.columns)

    # Refit the model with only the significant features
    significant_features_df['efs']=training_df['efs']
    significant_features_df['efs_time']=training_df['efs_time']
    cph_model=CoxPHFitter()
    cph_model.fit(significant_features_df, duration_col='efs_time', event_col='efs')

    # Save the model
    assets['coxph_model']=cph_model

    # Forecast participant survival
    survival_functions=cph_model.predict_survival_function(significant_features_df)
    partial_hazards=cph_model.predict_partial_hazard(significant_features_df)
    data['features']['coxph_survival']=survival_functions.iloc[-1]
    data['features']['coxph_partial_hazard']=np.log(partial_hazards)

    return data, assets


def waft_model(data:dict, assets:dict) -> Tuple[dict, dict]:
    '''Weibull Accelerated Failure Time survival modeling.'''

    # Combine features and labels
    training_df=pd.concat([data['features'], data['labels']], axis=1)

    # Fit the model
    waft_model=WeibullAFTFitter()
    waft_model.fit(training_df, duration_col='efs_time', event_col='efs')
    
    # Select features by p-value
    feature_pvals=waft_model.summary['p']
    feature_pvals=feature_pvals.droplevel(0)
    feature_pvals.drop('Intercept', axis=0, inplace=True)
    significant_features_df=training_df[feature_pvals[feature_pvals < 0.05].index].copy()

    # Save the significant features list
    assets['weibullaft_features']=list(significant_features_df.columns)

    # Refit the model with only the significant features
    significant_features_df['efs']=training_df['efs']
    significant_features_df['efs_time']=training_df['efs_time']
    waft_model=WeibullAFTFitter()
    waft_model.fit(significant_features_df, duration_col='efs_time', event_col='efs')

    # Save the model
    assets['weibullaft_model']=waft_model

    # Forecast participant survival
    survival_functions=waft_model.predict_survival_function(significant_features_df)
    expectations=waft_model.predict_expectation(significant_features_df)
    data['features']['weibullaft_survival']=survival_functions.iloc[-1]
    data['features']['weibullaft_expectation']=np.log(expectations)

    return data, assets