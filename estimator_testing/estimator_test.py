'''Runs test of Scikit-learn classifiers.'''

import os
import time
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

os.environ['OMP_NUM_THREADS']='2'

def run(input_file:str, output_file:str, models:dict, hyperparameters:dict, task:str, cv_splits:int=4):
    '''Main function to run test of estimators.'''

    print()
    print('#########################################')
    print('###### SciKit-learn estimator test ######')
    print('#########################################')
    print()

    with open(input_file, 'rb') as data_file:
        data_dict=pickle.load(data_file)

    feature_scaler=StandardScaler()
    training_features=feature_scaler.fit_transform(data_dict['Training features'])
    testing_features=feature_scaler.transform(data_dict['Testing features'])

    if task == 'classification':
        training_labels=data_dict['Training labels']['efs']
    
    elif task == 'regression':
        training_labels=np.log(data_dict['Training labels']['efs_time'])

    print('Models:')
    for name in models.keys():
        print(f' {name}')

    print()

    if Path(output_file).is_file():
        print(f'Have old results:')
        with open(output_file, 'rb') as results_file:
            old_results=pickle.load(results_file)

        for name in old_results.keys():
            print(f" {name}: {old_results[name]['Best hyperparameters']}")

    print()

    splitter=ShuffleSplit(n_splits=cv_splits, test_size=0.3)
    results={}

    for name, model in models.items():

        start_time=time.time()
        optimize=True
        results[name]={}
        results[name]['Hyperparameters']=hyperparameters[name]

        if name in list(old_results.keys()):
            if results[name]['Hyperparameters']==old_results[name]['Hyperparameters']:
                print(f"{name} already optimized: {old_results[name]['Best hyperparameters']}")
                winning_parameters=old_results[name]['Best hyperparameters']
                optimize=False

            else:
                print(f'{name}: hyperparameter space updated, re-running optimization')

        else:
            print(f'\nOptimizing {name}')

        if optimize is True:

            optimization=GridSearchCV(
                model,
                hyperparameters[name],
                cv=splitter
            )

            optimization_result=optimization.fit(
                training_features,
                training_labels
            )

            winning_parameters=optimization_result.best_params_
            results[name]['Best hyperparameters']=winning_parameters

            model.set_params(**winning_parameters)

            if task == 'classification':
                scoring='accuracy'
                model=CalibratedClassifierCV(model)
            
            elif task == 'regression':
                scoring='neg_mean_squared_error'

            model.fit(training_features, training_labels)
            results[name]['Model']=model

            scores=cross_val_score(
                model,
                training_features,
                training_labels,
                cv=splitter,
                scoring=scoring,
                n_jobs=-1
            )

            score_mean=np.mean(scores)
            score_std=np.std(scores)

            testing_predictions=model.predict(testing_features)

            results[name]['Cross validation scores']=scores
            results[name]['Testing predictions']=testing_predictions

            with open(output_file, 'wb') as output:
                pickle.dump(results, output)

            runtime=(time.time()-start_time)/60

            print(f'{name} {scoring}: {score_mean*100:.1f}+/{score_std*100:.1f}%, runtime: {runtime:.0f} minutes')

    return
