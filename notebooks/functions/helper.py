'''Helper functions for experimentation notebooks.'''

import itertools
import pandas as pd
import pandas.api.types
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error
from lifelines.utils import concordance_index
from typing import Tuple


def cross_val(model, features: pd.DataFrame, labels: pd.Series, folds: int=10) -> list[float]:
    '''Reusable helper function to run cross-validation on a model. Takes model,
    Pandas data frame of features and Pandas data series of labels. Returns 
    list of cross-validation fold RMSE scores.'''

    # Define the cross-validation strategy
    cross_validation=KFold(n_splits=folds, shuffle=True, random_state=315)

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
    return -np.array(scores)


def xgb_cross_val(
    xgb_params: dict,
    features_df: pd.DataFrame,
    labels: pd.Series,
    folds: int=10,
    boosting_rounds: int=500,
    early_stopping_rounds: int=10
):
    
    '''Cross-validates an XGBoost model.'''

    # Cross-validation splitter
    k_fold=KFold(n_splits=folds, shuffle=True, random_state=42)

    label_type=labels.name
    training_df=pd.concat([features_df, labels], axis=1)

    # Collector for scores
    scores=[]

    # Loop on cross-validation folds
    for _, (training_idx, validation_idx) in enumerate(k_fold.split(training_df)):

        # Get the features for this fold
        training_features=training_df.iloc[training_idx].drop(label_type, axis=1)
        validation_features=training_df.iloc[validation_idx].drop(label_type, axis=1)

        # Get the labels
        training_labels=training_df.iloc[training_idx][label_type]
        validation_labels=training_df.iloc[validation_idx][label_type]

        # Convert to DMaxtrix for XGBoost training
        dtraining=xgb.DMatrix(training_features, label=training_labels)
        dvalidation=xgb.DMatrix(validation_features, label=validation_labels)

        # Train the model
        naive_model=xgb.train(
            xgb_params,
            dtraining,
            num_boost_round=boosting_rounds,
            evals=[(dtraining, 'training'), (dvalidation, 'validation')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=0
        )

        # Get validation RMSE for this fold
        predictions=naive_model.predict(dvalidation)
        scores.append(root_mean_squared_error(validation_labels, predictions))

    return scores


def search_space_samples(**search_space):
    '''Takes a dictionary of hyperparameters, where key is string name
    and value is a list of values. Returns individual dictionaries 
    containing the cartesian product of all hyperparameter values'''
    
    parameters=search_space.keys()

    for values in itertools.product(*search_space.values()):
        yield dict(zip(parameters, values))


def xgb_hyperparameter_search(
        search_space: dict,
        features_df: pd.DataFrame,
        labels: pd.Series
) -> Tuple:
    
    '''Does hyperparameter grid search on XGBoost mode. Takes dictionary of
    hyperparameter lists to search.Runs cross-validation on each combination 
    of hyperparameters. Returns dictionary where key is mean RMSE of 
    cross-validation folds and value is hyperparameter dictionary and
    tuned model trained on full training dataset.'''

    # Make search space combinations
    samples=search_space_samples(**search_space)

    results={
        'RMSE mean': [],
        'RMSE standard deviation': [],
        'Hyperparameters': []
    }

    # Loop on hyperparameter samples
    for hyperparameters in samples:

        # Cross-validate with the hyperparameters
        scores=xgb_cross_val(
            hyperparameters,
            features_df,
            labels,
            boosting_rounds=1000,
            early_stopping_rounds=100
        )

        results['RMSE mean'].append(np.array(scores).mean())
        results['RMSE standard deviation'].append(np.array(scores).std())
        results['Hyperparameters'].append(hyperparameters)

    # Get winning hyperparameter set
    results_df=pd.DataFrame.from_dict(results)
    results_df.sort_values('RMSE mean', inplace=True, ascending=False)
    results_df.reset_index(inplace=True, drop=True)

    winning_hyperparameters=dict(results_df.iloc[-1]['Hyperparameters'])
    print(f'Winning hyperparameters: {winning_hyperparameters}')

    # Train with winning hyperparameters on complete training set
    dtraining=xgb.DMatrix(features_df, label=labels)

    tuned_model=xgb.train(
        winning_hyperparameters,
        dtraining,
        num_boost_round=1000,
        evals=[(dtraining, 'training')],
        early_stopping_rounds=10,
        verbose_eval=0
    )

    return results_df, tuned_model


def score_predictions(
    model_description: str,
    predictions: list,
    labels_df: pd.DataFrame,
    race_group: list,
    results: dict,
    label_type: str='efs_time'
) -> dict:

    '''Takes predictions, labels and results dictionary. Calculates 
    RMSE, concordance index and stratified concordance index. Returns
    updated results dictionary.'''

    # Add model description to results
    results['Model'].append(model_description)

    # Save the RMSE for later
    results['RMSE'].append(root_mean_squared_error(labels_df[label_type], predictions))

    # Save the concordance index for later
    results['C-index'].append(
        concordance_index(
            labels_df[label_type],
            -predictions,
            labels_df['efs']
        )
    )

    # Get and save stratified concordance index for later
    results_df=pd.DataFrame({'ID': labels_df.index, 'prediction': predictions})
    results_df['race_group']=race_group
    results_df['efs_time']=labels_df[label_type]
    results_df['efs']=labels_df['efs']
    solution=results_df.drop(['ID', 'prediction'], axis=1)
    submission=results_df.drop(['race_group','efs_time','efs'], axis=1)
    results['Stratified C-index'].append(competition_score(solution, submission))

    return results


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