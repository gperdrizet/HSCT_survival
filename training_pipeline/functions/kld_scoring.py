'''Estimates Kullback-Leibler divergence distributions between EFS zero and one
participants for survival features, adds KLD score features for each.'''

import pickle
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .. import configuration as config


def run() -> dict:
    '''Main function to run KLD estimation and feature addition.'''

    # Load data
    with open(config.SURVIVAL_FEATURES_RESULT, 'rb') as input_file:
        data=pickle.load(input_file)

    # Combine features and labels
    training_df=pd.concat([data['features'], data['labels']], axis=1)

    # Split the data into EFS zero and EFS one
    efs_one=training_df[training_df['efs'] == 0]
    efs_zero=training_df[training_df['efs'] == 1]

    # Get kernel density estimates for both for each survival feature
    features=['coxph_survival','coxph_partial_hazard','weibullaft_survival','weibullaft_expectation']
    efs_one_kdes={}
    efs_zero_kdes={}

    for feature in features:
        efs_one_kdes[feature]=feature_kde(efs_one[feature])
        efs_zero_kdes[feature]=feature_kde(efs_zero[feature])

    # Get Kullback-Leibler divergences for each feature
    klds={}

    for feature in features:
        klds[feature]=kld(efs_zero_kdes[feature], efs_one_kdes[feature])

    # Get a kernel density estimate for the Kullback-Leibler divergence
    # values for each feature
    kld_kdes={}

    for feature in features:
        kld_kdes[feature]=kld_kde(klds[feature])

    # Score the features in the original data and add the result as new features
    for feature in features:

        data=np.array(training_df[feature])
        workers=mp.cpu_count() - 10

        with mp.Pool(workers) as p:
            kld_score=np.concatenate(p.map(kld_kdes[feature], np.array_split(data, workers)))

        data['features'][f'{feature}_kld']=kld_score

    return data


def feature_kde(data: pd.Series) -> np.array:
    '''Makes kernel density estimate on feature. Returns kernel density
    values across data range with padding.'''

    data=np.array(data)

    data_min=min(data)
    data_max=max(data)
    data_range=data_max - data_min
    padding=data_range * 0.05
    x=np.linspace(data_min - padding, data_max + padding)

    kde=gaussian_kde(
        data, 
        bw_method='silverman'
    )

    return np.array(kde(x))


def kld(efs_zero_data:list, efs_one_data:list) -> list:
    '''Calculates Kullback-Leibler divergence of two lists.'''

    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        kld_values=efs_one_data * np.log2(efs_one_data/efs_zero_data)

    return kld_values


def kld_kde(data:np.array) -> gaussian_kde:
    '''Gets kernel density estimate.'''

    # Construct new padded x range
    data_min=min(data)
    data_max=max(data)
    data_range=data_max - data_min
    padding=data_range * 0.05
    x=np.linspace(data_min - padding, data_max + padding)

    # Shift the kld values so that they are non-negative
    data=data + abs(min(data))

    # Scale the values so when we convert to integer we get good
    # resolution, e.g. we don't want to collapse 2.1, 2.2, 2.3 etc.,
    # to 2. Instead, 2100.0, 2200.0, 2300.0 become 2100, 2200, 2300 etc.
    data=data * 10000

    # Convert to integer
    data_counts=data.astype(int)

    # Now, construct a list where each value of x appears a number of times
    # equal to it's KLD 'count'
    data_scores=[]

    for i, _ in enumerate(data_counts):
        data_scores.extend([x[i]] * data_counts[i])

    kde=gaussian_kde(
        data_scores, 
        bw_method='silverman'
    )

    return kde