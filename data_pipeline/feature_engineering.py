'''Does feature engineering for inference, adds the following:

    1. Cox PH surivival probability at study end
    2. Cox PH hazard ratio
    3. Weibull AFT survival probability at study end
    4. Weibull AFT expectation value
    5. Kullback-Leibler divergence score of above values
    6. Learned EFS probability
'''

import pandas as pd


def run(
        data_df: pd.DataFrame
) -> pd.DataFrame:

    '''Main function to run feature engineering operations.'''

    data_df=cox_ph(data_df)
    data_df=weibull_aft(data_df)
    data_df=kullback_leibler_score(data_df)
    data_df=learned_efs(data_df)

    return data_df


def cox_ph(
        data_df:pd.DataFrame
) -> pd.DataFrame:

    '''Adds Cox PH features.'''

    return data_df


def weibull_aft(data_df:pd.DataFrame) -> pd.DataFrame:

    '''Adds Weibull AFT features.'''

    return data_df


def kullback_leibler_score(
        data_df:pd.DataFrame
) -> pd.DataFrame:

    '''Adds Kullback-Leibler divergence scores for Cox PH
    and Weibull AFT features'''

    return data_df


def learned_efs(
        data_df:pd.DataFrame
) -> pd.DataFrame:

    '''Adds learned EFS probability feature.'''

    return data_df
