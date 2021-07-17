import argparse
import os
import joblib
import time
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

import preprocessor_dispatcher
import model_dispatcher
import config


def train(model, metric):

    df=pd.read_pickle(config.TRAINING_DATA_FOLDS)
    fold_count=df.kfold.nunique()
    metric_method=getattr(metrics, metric)
    dst_dir=os.path.join(config.MODEL_OUTPUT,model)
    os.makedirs(dst_dir, exist_ok=True)

    scores=np.array([])

    #
    # Evaluate performance with k-fold cross-validation
    #
    for fold in range(fold_count):
        # Choose data folds
        train_df=df[df.kfold!=fold].reset_index(drop=True)
        valid_df=df[df.kfold==fold].reset_index(drop=True)

        # Seperate features from labels
        X_train_df=train_df.drop(['price','kfold'],axis=1)
        X_valid_df=valid_df.drop(['price','kfold'], axis=1)
        y_train_df=train_df.price
        y_valid_df=valid_df.price

        # Run the data through the preprocessor
        preprocess_pipeline=preprocessor_dispatcher.preprocessors[config.PREPROCESSOR]

        X_train=preprocess_pipeline.fit_transform(X_train_df)
        X_valid=preprocess_pipeline.transform(X_valid_df)
        y_train=y_train_df.to_numpy(dtype='float64')
        y_valid=y_valid_df.to_numpy(dtype='float64')

        # Model Initialization and Training
        reg=model_dispatcher.models[model]
        reg.fit(X_train,y_train)
        
        # Model Eval
        preds=reg.predict(X_valid)
        score=metric_method(y_valid,preds)
        scores=np.append(scores,score)

        # Save Model
        joblib.dump(reg, os.path.join(dst_dir, f"{model}_{fold}.bin"))
        print(f'Model = {model}, Fold = {fold+1}/{fold_count}, {metric} = {score:.2f}')

    print('-'*100)
    print('SUMMARY:')
    print(f"Model = {model}, {metric} Mean = {scores.mean():.3f}, {metric} Std = {scores.std():.3f}")
    print('-'*100)

    #
    # Train Final model on all folds
    #
    X_df=df.drop(['price','kfold'], axis=1)
    y_df=df.price

    preprocess_pipeline=preprocessor_dispatcher.preprocessors[config.PREPROCESSOR]

    X=preprocess_pipeline.fit_transform(X_df)
    y=y_df.to_numpy(dtype='float64')

    reg=model_dispatcher.models[model]
    reg.fit(X,y)

    joblib.dump(preprocess_pipeline, os.path.join(dst_dir, f"{config.PREPROCESSOR}.bin"))
    joblib.dump(reg, os.path.join(dst_dir, f"{model}.bin"))




    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default='rf_best')
    parser.add_argument("--metric",
                        type=str,
                        choices=['r2_score','mean_absolute_error','mean_squared_error','mean_absolute_percentage_error'],
                        default='r2_score')

    args=parser.parse_args()
    train(args.model, args.metric)