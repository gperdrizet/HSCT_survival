'''Helper functions for experimentation notebooks.'''

import pandas as pd
import pandas.api.types
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from lifelines.utils import concordance_index


def cross_val(model, features: pd.DataFrame, labels: pd.Series) -> list[float]:
    '''Reusable helper function to run cross-validation on a model. Takes model,
    Pandas data frame of features and Pandas data series of labels. Returns 
    list of cross-validation fold accuracy scores as percents.'''

    # Define the cross-validation strategy
    cross_validation=KFold(n_splits=7, shuffle=True, random_state=315)

    # Run the cross-validation, collecting the scores
    scores=cross_val_score(
        model,
        features,
        labels,
        cv=cross_validation,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )

    # Print mean and standard deviation of the scores
    print(f'Cross validation RMSE {-scores.mean():.2f} +/- {scores.std():.2f}')

    # Return the scores
    return scores



class ParticipantVisibleError(Exception):
    pass


def competition_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    '''Scoring function adapted from competition. Removed deletion of 'row_id_column_name',
    Otherwise, unchanged. See here: https://www.kaggle.com/code/metric/eefs-concordance-index'''
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'

    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
        
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)

    metric_list = []

    for race in merged_df_race_dict.keys():

        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]

        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        
        metric_list.append(c_index_race)
        
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))