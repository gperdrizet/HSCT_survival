'''Estimator definitions and hyperparameter distributions
for GridSearchCV with SciKit-learn classifiers.'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

models={
    'Nearest Neighbors':KNeighborsClassifier(n_jobs=1),
    'Linear SVM':SVC(kernel='linear', class_weight='balanced', max_iter=1000),
    'RBF SVM':SVC(kernel='rbf', class_weight='balanced', max_iter=1000),
    'Polynomial SVM':SVC(kernel='poly', class_weight='balanced', max_iter=1000),
    #'Gaussian Process':GaussianProcessClassifier(),
    'Decision Tree':DecisionTreeClassifier(class_weight='balanced'),
    'Random Forest':RandomForestClassifier(class_weight='balanced', n_jobs=1),
    'Neural Net':MLPClassifier(max_iter=1000),
    'AdaBoost':AdaBoostClassifier(),
    'Naive Bayes':GaussianNB(),
    'QDA':QuadraticDiscriminantAnalysis(),
    'SGD': SGDClassifier(penalty='elasticnet'),
    'XGBoost':XGBClassifier(n_jobs=1),
    'CatBoost':CatBoostClassifier(thread_count=1)
}

hyperparameters={
    'Nearest Neighbors':{
        'n_neighbors': [2,4,8],
        'weights': ['uniform','distance'],
        'leaf_size': [15,30,60],
        'p': [1,2,3]
    },
    'Linear SVM':{
        'C': [0.5,1.0,2],
        'decision_function_shape': ['ovo','ovr']
    },
    'RBF SVM':{
        'C': [0.5,1.0,2],
        'decision_function_shape': ['ovo','ovr'],
        'gamma': ['scale','auto'],
    },
    'Polynomial SVM':{
        'C': [0.5,1.0,2],
        'decision_function_shape': ['ovo','ovr'],
        'gamma': ['scale','auto'],
        'degree': [2,3,4],
        'coef0': [0.0,0.001,0.01],
    },
    'Gaussian Process':{
        'n_restarts_optimizer': [0,1,2],
        'max_iter_predict': [50,100,200]
    },
    'Decision Tree':{
        'criterion': ['gini','entropy','log_loss'],
        'splitter': ['best','random'],
        'max_depth': [5,10,20],
        'max_features': [0.5,0.75,1]
    },
    'Random Forest':{
        'n_estimators': [50,100,200],
        'criterion': ['gini','entropy','log_loss'],
        'max_features': [0.25,0.5,0.75,1],
        'ccp_alpha': [0.0,0.01,0.1]
    },
    'Neural Net':{
        'hidden_layer_sizes': [16,32,64,128,256],
        'alpha': [0.000025,0.00005,0.0001,0.0002],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    },
    'AdaBoost':{
        'n_estimators': [25, 50, 100],
        'learning_rate': [0.25, 0.5, 1]
    },
    'Naive Bayes':{
        'var_smoothing': [10**-9, 10**-8]
    },
    'QDA':{'reg_param': [0.0,0.001,0.01]},
    'SGD classifier':{
        'alpha': [0.00005,0.0001,0.0002],
        'l1_ratio': [0.075,0.15,0.30],
        'learning_rate': ['optimal', 'invscaling', 'adaptive']
    },
    'XGBoost':{
        'n_estimators': [50,100,200],
        'max_depth': [5,10,20],
        'subsample': [0.5,0.75,1]
    },
    'CatBoost':{
        'n_estimators': [50,100,200],
        'depth': [5,10,20],
        'model_size_reg':[0.001,0.01,0.1]
    }
}