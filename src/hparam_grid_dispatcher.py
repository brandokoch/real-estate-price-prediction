hparam_grids={
    'rf':{
        'n_estimators':[120,300,500,800,1200],
        'max_depth':[15,25,30,None],
        'min_samples_split':[2,5,10,15,100],
        'min_samples_leaf':[2,4,5,10],
        'max_features':[None, 0.5]
    },
    'xgb':{
        'eta':[0.01, 0.015, 0.025, 0.05, 0.1],
        'gamma':[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'max_depth':[3,5,7,9,12,15,17,25],
        'min_child_weight':[1, 3, 5, 7],
        'subsample':[0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0],
        'lambda':[0.01,0.1,1.0],
        'alpha':[0, 0.1, 0.5, 1.0]
    },
    'lin_reg':{
        'fit_intercept':[False, True],
        'normalize':[True,False]
    }
}