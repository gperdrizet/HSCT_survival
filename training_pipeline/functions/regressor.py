'''Trains voting ensemble of bagging regressors.'''

import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.linear_model import SGDRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import configuration as config

def run() -> list:
    '''Main function to do model training.'''

    # Load data
    with open(config.EFS_FEATURE_RESULT, 'rb') as input_file:
        data=pickle.load(input_file)

    # Extract features and labels
    features_df=data['features']
    labels_df=data['labels']

    feature_scaler=StandardScaler()
    features_df=feature_scaler.fit_transform(features_df)

    labels_df['efs_time']=np.log(labels_df['efs_time'])


    print(f"EFS times: {labels_df['efs_time']}")

    # Define models
    models={
        'SGD':SGDRegressor(penalty='elasticnet', max_iter=10000),
        'CatBoost':CatBoostRegressor(thread_count=4, verbose=0),
        'XGBoost':XGBRegressor(n_jobs=4)
    }

    # Set hyperparameters
    model_hyperparameters={
        'SGD': {'alpha': 0.0002, 'l1_ratio': 0.3, 'learning_rate': 'adaptive', 'loss': 'squared_error'},
        'CatBoost': {'depth': 5, 'model_size_reg': 0.001, 'n_estimators': 200},
        'XGBoost': {'max_depth': 5, 'n_estimators': 50, 'subsample': 1}

    }

    # Train bagging regressors
    for model_name, model in models.items():

        print(f'Calibrating {model_name}')
        hyperparameters=model_hyperparameters[model_name]
        model.set_params(**hyperparameters)

        bagging_model=BaggingRegressor(
            estimator=model,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False, 
            n_jobs=-1
        )

        bagging_model.fit(features_df, labels_df['efs_time'])
        models[model_name]=bagging_model

        bagging_models=list(zip(list(models.keys()), list(models.values())))

        # Train the voting ensemble
        ensemble_model=VotingRegressor(
            estimators=bagging_models,
            n_jobs=-1
        )

        ensemble_model.fit(features_df, labels_df['efs_time'])

        # Make predictions
        predictions=ensemble_model.predict(features_df)
        print(f'Predictions: {predictions}')

        # Save the voting ensemble model and scaler
        assets={
            'scaler':feature_scaler,
            'model':ensemble_model
        }

        with open(config.REGRESSOR_MODEL_ASSETS, 'wb') as output_file:
            pickle.dump(assets, output_file)

        return predictions