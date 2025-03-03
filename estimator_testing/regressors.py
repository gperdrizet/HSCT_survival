'''Estimator definitions and hyperparameter distributions
for GridSearchCV with SciKit-learn classifiers.'''

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

models={
    'Nearest Neighbors':KNeighborsRegressor(n_jobs=1),
    'Linear SVM':SVR(kernel='linear'),
    'RBF SVM':SVR(kernel='rbf'),
    'Polynomial SVM':SVR(kernel='poly'),
    # 'Gaussian Process':GaussianProcessRegressor(),
    'Decision Tree':DecisionTreeRegressor(),
    'Random Forest':RandomForestRegressor(n_jobs=1),
    'Neural Net':MLPRegressor(max_iter=1000),
    'AdaBoost':AdaBoostRegressor(),
    'SGD': SGDRegressor(penalty='elasticnet'),
    'XGBoost':XGBRegressor(n_jobs=1),
    'CatBoost':CatBoostRegressor(thread_count=1)
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
        'epsilon': [0.05,1,2],
        'max_iter': [10000]
    },
    'RBF SVM':{
        'C': [0.5,1.0,2],
        'epsilon': [0.05,1,2],
        'gamma': ['scale','auto'],
        'max_iter': [10000]
    },
    'Polynomial SVM':{
        'C': [0.5,1.0,2],
        'epsilon': [0.05,1,2],
        'gamma': ['scale','auto'],
        'degree': [2,3],
        'coef0': [0.0,0.001,0.01],
        'max_iter': [10000]
    },
    'Gaussian Process':{
        'n_restarts_optimizer': [0,1,2]
    },
    'Decision Tree':{
        'criterion': ['squared_error','friedman_mse','absolute_error'],
        'splitter': ['best','random'],
        'max_depth': [5,10,20],
        'max_features': [0.5,0.75,1]
    },
    'Random Forest':{
        'n_estimators': [50,100,200],
        'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
        'max_depth': [5,10,20],
        'max_features': [0.5,0.75,1],
        'ccp_alpha': [0.0,0.01,0.1]
    },
    'Neural Net':{
        'hidden_layer_sizes': [16,32,64,128,256],
        'alpha': [0.000025,0.00005,0.0001,0.0002],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    },
    'AdaBoost':{
        'n_estimators': [25, 50, 100],
        'learning_rate': [0.25, 0.5, 1],
        'loss': ['linear', 'square', 'exponential']
    },
    'SGD':{
        'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
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