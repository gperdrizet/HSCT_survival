'''Runs test of Scikit-learn classifiers.'''

import time
import pickle

import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def run(input_file:str, output_file:str):
    '''Main function to run test of classifiers.'''

    print()
    print('#########################################')
    print('###### SciKit-learn classifier test #####')
    print('#########################################')
    print()

    with open(input_file, 'rb') as data_file:
        data_dict=pickle.load(data_file)

    names=[
        'Nearest Neighbors',
        'Linear SVM',
        'RBF SVM',
        'Polynomial SVM',
        'Gaussian Process',
        'Decision Tree',
        'Random Forest',
        'Neural Net',
        'AdaBoost',
        'Naive Bayes',
        'QDA',
        'XGBoost'
    ]

    print('Classifiers')
    for name in names:
        print(f' {name}')

    classifiers=[
        KNeighborsClassifier(16),
        SVC(kernel='linear', max_iter=5000),
        SVC(kernel='rbf', max_iter=5000),
        SVC(kernel='poly', max_iter=5000),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        xgb.XGBClassifier()
    ]

    print()

    scalar=StandardScaler()
    splitter=ShuffleSplit(n_splits=16, test_size=0.3)
    results={}

    for name, classifier in zip(names, classifiers):

        start_time=time.time()
        pipeline=Pipeline([('Scaler', scalar), ('Classifier', classifier)])

        scores=cross_val_score(
            pipeline,
            data_dict['Training features'],
            data_dict['Training labels']['efs'],
            cv=splitter,
            scoring='accuracy',
            n_jobs=-1
        )

        score_mean=np.mean(scores)
        score_std=np.std(scores)

        classifier.fit(data_dict['Training features'], data_dict['Training labels']['efs'])
        testing_predictions=classifier.predict(data_dict['Testing features'])

        results[name]={}
        results[name]['Cross validation scores']=scores
        results[name]['Testing predictions']=testing_predictions

        with open(output_file, 'wb') as output:
            pickle.dump(results, output)

        runtime=(time.time()-start_time)/60

        print(f'{name} accuracy: {score_mean*100:.1f}+/{score_std*100:.1f}%, runtime: {runtime:.0f} minutes')

    return
