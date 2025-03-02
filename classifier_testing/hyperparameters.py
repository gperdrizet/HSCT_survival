'''Hyperparameter distributions for SciKit-learn RandomSearchCV.'''

distributions={
    'Nearest Neighbors':{
        'n_neighbors': [2,4,8],
        'weights': ['uniform','distance'],
        'leaf_size': [15,30,60],
        'p': [1,2,3,4]
    },
    'Radius Neighbors':{
        'radius':[10],
        'weights': ['uniform','distance'],
        'leaf_size': [15,30,60],
        'p': [1,2,3,4]
    },
    'Linear SVM':{
        'kernel': ['linear'],
        'C': [0.001,0.1,1.0],
        'class_weight': ['balanced'],
        'decision_function_shape': ['ovo','ovr']
    },
    'RBF SVM':{
        'kernel': ['rbf'],
        'gamma': ['scale','auto'],
        'C': [0.001, 0.1, 1.0],
        'class_weight': ['balanced'],
        'decision_function_shape': ['ovo','ovr']
    },
    'Polynomial SVM':{
        'kernel': ['poly'],
        'gamma': ['scale','auto'],
        'degree': [2,3,4],
        'coef0': [0.0,0.001,0.01],
        'C': [0.001,0.1,1.0],
        'class_weight': ['balanced'],
        'decision_function_shape': ['ovo','ovr']
    },
    'Gaussian Process':{
        'n_restarts_optimizer': [0,1,2],
        'max_iter_predict': [50,100,200]
    },
    'Decision Tree':{
        'criterion': ['gini','entropy','log_loss'],
        'splitter': ['best','random'],
        'max_depth': [5,10,15,20],
        'class_weight': ['balanced'],
        'max_features': [0.25,0.5,0.75,1]
    },
    'Random Forest':{
        'n_estimators': [50,100,200],
        'criterion': ['gini','entropy','log_loss'],
        'class_weight': ['balanced'],
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
    # 'Naive Bayes':{},
    'QDA':{'reg_param': [0.0,0.001,0.01]},
    'SGD classifier':{
        'alpha': [0.00005,0.0001,0.0002],
        'l1_ratio': [0.075,0.15,0.30],
        'learning_rate': ['optimal', 'invscaling', 'adaptive']
    },
    # 'XGBoost':,
    # 'CatBoost':
}