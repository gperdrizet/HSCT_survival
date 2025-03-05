'''Trains classifier and then adds learned EFS probability to data for
regression model training.'''

import pickle

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import configuration as config


def run() -> dict:
    '''Main function to do training and feature addition.'''

    # Load data
    with open(config.KLD_FEATURES_RESULT, 'rb') as input_file:
        data=pickle.load(input_file)

    # Extract features and labels
    features_df=data['features']
    labels_df=data['labels']

    # Define models
    models={
        'AdaBoost':AdaBoostClassifier(),
        'CatBoost':CatBoostClassifier(thread_count=4, verbose=0),
        'XGBoost':XGBClassifier(n_jobs=4)
    }

    # Set hyperparameters
    model_hyperparameters={
        'AdaBoost': {'learning_rate': 1, 'n_estimators': 200},
        'CatBoost': {'depth': 5, 'model_size_reg': 0.001, 'n_estimators': 200},
        'XGBoost': {'max_depth': 5, 'n_estimators': 50, 'subsample': 1}

    }

    # Calibrate the models with cross-validation
    for model_name, model in models.items():

        print(f'Calibrating {model_name}')
        hyperparameters=model_hyperparameters[model_name]
        model.set_params(**hyperparameters)

        calibrated_model=CalibratedClassifierCV(
            model,
            cv=3,
            n_jobs=-1)

        calibrated_model.fit(features_df, labels_df['efs'])
        models[model_name]=calibrated_model

    calibrated_classifiers=list(zip(list(models.keys()), list(models.values())))

    # Train the voting ensemble
    ensemble_model=VotingClassifier(
        estimators=calibrated_classifiers,
        voting='soft',
        n_jobs=-1
    )

    ensemble_model.fit(features_df, labels_df['efs'])

    # Make predictions
    predictions=ensemble_model.predict_proba(features_df)

    # Add feature back to original data
    data['features']['learned_efs']=predictions[:,1]

    # Save the voting ensemble model
    with open(config.CLASSIFIER_MODEL_ASSETS, 'wb') as output_file:
        pickle.dump(ensemble_model, output_file)

    return data