'''Runs test of Scikit-learn classifiers.'''

import time
import pickle

import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

import classifier_testing.hyperparameters as hyperparams

def run(input_file:str, output_file:str):
    '''Main function to run test of classifiers.'''

    print()
    print('#########################################')
    print('###### SciKit-learn classifier test #####')
    print('#########################################')
    print()

    with open(input_file, 'rb') as data_file:
        data_dict=pickle.load(data_file)

    classifiers={
        'Nearest Neighbors':KNeighborsClassifier(),
        'Radius Neighbors':RadiusNeighborsClassifier(),
        'Linear SVM':SVC(kernel='linear', max_iter=5000),
        'RBF SVM':SVC(kernel='rbf', max_iter=5000),
        'Polynomial SVM':SVC(kernel='poly', max_iter=5000),
        'Gaussian Process':GaussianProcessClassifier(),
        'Decision Tree':DecisionTreeClassifier(),
        'Random Forest':RandomForestClassifier(),
        'Neural Net':MLPClassifier(max_iter=1000),
        'AdaBoost':AdaBoostClassifier(),
        'Naive Bayes':GaussianNB(),
        'QDA':QuadraticDiscriminantAnalysis(),
        'SGD classifier': SGDClassifier(),
        'XGBoost':XGBClassifier(),
        'CatBoost':CatBoostClassifier()
    }

    print('Classifiers')
    for name in classifiers.keys():
        print(f' {name}')

    print()

    feature_scaler=StandardScaler()
    training_features=feature_scaler.fit_transform(data_dict['Training features'])
    testing_features=feature_scaler.transform(data_dict['Testing features'])

    training_labels=data_dict['Training labels']['efs']

    splitter=ShuffleSplit(n_splits=16, test_size=0.3)
    results={}

    for name, classifier in classifiers.items():

        start_time=time.time()
        results[name]={}
        results[name]['Hyperparameter distributions']=hyperparams.distributions[name]

        with open(output_file, 'rb') as results_file:
            old_results=pickle.load(results_file)

        if name in old_results.keys():
            if hyperparams.distributions[name]==old_results[name]['Hyperarameter ditributions']:
                print(f'{name}: skipped')
                continue

        optimization=RandomizedSearchCV(
            classifier,
            hyperparams.distributions[name],
            cv=splitter,
            n_iter=16
        )

        optimization_result=optimization.fit(
            training_features,
            training_labels
        )

        winning_parameters=optimization_result.best_params_
        results[name]['Best parameters']=winning_parameters

        classifier.set_params(**winning_parameters)

        scores=cross_val_score(
            classifier,
            training_features,
            training_labels,
            cv=splitter,
            scoring='accuracy',
            n_jobs=-1
        )

        score_mean=np.mean(scores)
        score_std=np.std(scores)

        classifier.fit(training_features, training_labels)
        testing_predictions=classifier.predict(testing_features)

        results[name]['Cross validation scores']=scores
        results[name]['Testing predictions']=testing_predictions

        with open(output_file, 'wb') as output:
            pickle.dump(results, output)

        runtime=(time.time()-start_time)/60

        print(f'{name} accuracy: {score_mean*100:.1f}+/{score_std*100:.1f}%, runtime: {runtime:.0f} minutes')

    return
