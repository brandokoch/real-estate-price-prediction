import argparse
import os
import joblib

import pandas as pd
from sklearn.model_selection import GridSearchCV

import preprocessor_dispatcher
import config
import model_dispatcher
import hparam_grid_dispatcher


def train(model, scoring):

    df = pd.read_pickle(config.TRAINING_DATA_FOLDS)
    dst_dir = os.path.join(config.MODEL_OUTPUT, model + "_hparam_search")
    os.makedirs(dst_dir, exist_ok=True)

    X_df = df.drop(["price", "kfold"], axis=1)
    y_df = df.price

    # Run the data through the preprocessor
    preprocess_pipeline = preprocessor_dispatcher.preprocessors[config.PREPROCESSOR]

    X = preprocess_pipeline.fit_transform(X_df)
    y = y_df.to_numpy(dtype="float64")

    # Model and Hparam Grid Initialization
    hparam_grid = hparam_grid_dispatcher.hparam_grids[model]
    reg = model_dispatcher.models[model]

    # Hparam Search
    reg_grid = GridSearchCV(
        reg, hparam_grid, cv=5, n_jobs=10, scoring=scoring, verbose=10
    )
    reg_grid.fit(X, y)

    # Log results
    print("-" * 100)
    print("SUMMARY:")
    print(f"Best {scoring} score = {reg_grid.best_score_}")
    print(f"Best hparams = {reg_grid.best_params_}")
    print("-" * 100)

    with open(os.path.join(dst_dir, "logs.txt"), "w") as f:
        f.write(f"Best {scoring} score = {reg_grid.best_score_} \n")
        f.write(f"Best hparams = {reg_grid.best_params_} \n")

    # Save Best Model
    joblib.dump(reg_grid.best_estimator_, os.path.join(dst_dir, f"{model}_best.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lin_reg",
        help="Model name with the hparam grid available in hparam_grid_dispatcher.py",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["r2"],
        default="r2"
    )

    args = parser.parse_args()
    train(args.model, args.scoring)