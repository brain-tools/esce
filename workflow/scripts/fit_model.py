"""
fit_model.py
====================================

"""
    
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid


class BaseModel(ABC):
    """
    
    Base model class for each model.
    
    
    """

    # alternative for switching later
    scale_features: bool
    scale_targets: bool

    def __init__(self, model_generator: Callable[..., Any], model_name: str):
        """Initialize class using a model that is initialized later."""
        self.model_generator = model_generator
        self.model_name = model_name

    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):  # type: ignore
        """
        
        Provide a score for the model performance on the data.

        Args:
            x: features
            y: targets
            idx: train, validation, test indices
            **kwargs: hyperparmeters
        
        """

        # generate model based on hyperparams
        model = self.model_generator(**kwargs)

        # scale features (x) if necessary
        x_scaler = StandardScaler() if self.scale_features else None
        x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
        if x_scaler:
            x_train_scaled = x_scaler.fit_transform(x_train)
            x_val_scaled = x_scaler.transform(x_val)
            x_test_scaled = x_scaler.transform(x_test)
        else:
            x_train_scaled = x_train
            x_val_scaled = x_val
            x_test_scaled = x_test

        # scale targets (y) if necessary
        y_scaler = StandardScaler() if self.scale_targets else None
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        if y_scaler:
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = y_train

        model.fit(x_train_scaled, y_train_scaled)

        y_hat_train_scaled = model.predict(x_train_scaled)
        y_hat_val_scaled = model.predict(x_val_scaled)
        y_hat_test_scaled = model.predict(x_test_scaled)

        # revert the scaling depending on the metrics
        if y_scaler:
            y_hat_train = y_scaler.inverse_transform(
                y_hat_train_scaled.reshape(-1, 1)
            ).flatten()
            y_hat_val = y_scaler.inverse_transform(
                y_hat_val_scaled.reshape(-1, 1)
            ).flatten()
            y_hat_test = y_scaler.inverse_transform(
                y_hat_test_scaled.reshape(-1, 1)
            ).flatten()
        else:
            y_hat_train = y_hat_train_scaled
            y_hat_val = y_hat_val_scaled
            y_hat_test = y_hat_test_scaled

        return self.compute_metrics(
            y_hat_train, y_hat_val, y_hat_test, y_train, y_val, y_test
        )

    @abstractmethod
    def compute_metrics(
        self,
        y_hat_train: np.ndarray,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        pass


class ClassifierModel(BaseModel):
    """Base class for classifier models."""

    scale_features = True
    scale_targets = False

    def compute_metrics(
        self,
        y_hat_train,
        y_hat_val,
        y_hat_test,
        y_train,
        y_val,
        y_test,
    ):
        return {
            "acc_train": accuracy_score(y_train, y_hat_train),
            "acc_val": accuracy_score(y_val, y_hat_val),
            "acc_test": accuracy_score(y_test, y_hat_test),
            "f1_train": f1_score(y_train, y_hat_train, average="weighted"),
            "f1_val": f1_score(y_val, y_hat_val, average="weighted"),
            "f1_test": f1_score(y_test, y_hat_test, average="weighted"),
        }


class RegressionModel(BaseModel):
    """Base class for regression models."""

    scale_features = True
    scale_targets = True

    def compute_metrics(
        self,
        y_hat_train,
        y_hat_val,
        y_hat_test,
        y_train,
        y_val,
        y_test,
    ):
        return {
            "r2_train": r2_score(y_train, y_hat_train),
            "r2_val": r2_score(y_val, y_hat_val),
            "r2_test": r2_score(y_test, y_hat_test),
            "mae_train": mean_absolute_error(y_train, y_hat_train),
            "mae_val": mean_absolute_error(y_val, y_hat_val),
            "mae_test": mean_absolute_error(y_test, y_hat_test),
            "mse_train": mean_squared_error(y_train, y_hat_train),
            "mse_val": mean_squared_error(y_val, y_hat_val),
            "mse_test": mean_squared_error(y_test, y_hat_test),
        }


MODELS = {
    "majority-classifier": ClassifierModel(
        lambda **args: DummyClassifier(strategy="most_frequent", **args),
        "majority classifier",
    ),
    "ridge-cls": ClassifierModel(
        lambda **args: RidgeClassifier(**args), "ridge classifier"
    ),
    "ridge-reg": RegressionModel(lambda **args: Ridge(**args), "ridge regressor"),
}


def get_existing_scores(scores_path_list):
    """
    In the case where data was precalculated due to extra minor adjustment, we extract the existing scores if the files exist
    """
    df_list = []
    for filename in scores_path_list:
        if os.stat(filename).st_size > 0:
            df_list.append(pd.read_csv(filename, index_col=False))

    if df_list:
        return pd.concat(
            df_list,
            axis=0,
            ignore_index=True,
        )
    else:
        return pd.DataFrame()


def fit(
    features_path,
    targets_path,
    split_path,
    scores_path,
    model_name,
    grid_path,
    existing_scores_path_list,
):
    split = json.load(open(split_path, "r"))
    if "error" in split:
        Path(scores_path).touch() # if error occurs, create file for snakemake to run smoothly
        return

    x = np.load(features_path)
    y = np.load(targets_path)

    # making sure the split files are correct, meaning only fully available data should be included
    assert np.isfinite(x[split["idx_train"]]).all()
    assert np.isfinite(y[split["idx_train"]]).all()

    grid = yaml.safe_load(open(grid_path, "r"))
    model = MODELS[model_name]

    df_existing_scores = get_existing_scores(existing_scores_path_list)

    scores = []
    for params in ParameterGrid(grid[model_name]): # for each hyperparam combination

        # extracting only scores corresponding to the hyperparam combinations
        df_existing_scores_filtered = lambda: df_existing_scores.loc[
            (df_existing_scores[list(params)] == pd.Series(params)).all(axis=1)
        ]
        if not df_existing_scores.empty and not df_existing_scores_filtered().empty:
            score = dict(df_existing_scores_filtered().iloc[0])
            # print("retrieved score", score)
        else: # if the scores don't exist, calculate them manually
            score = model.score(
                x,
                y,
                idx_train=split["idx_train"],
                idx_val=split["idx_val"],
                idx_test=split["idx_test"],
                **params
            )
            score.update(params)
            score.update({"n": split["samplesize"], "s": split["seed"]})
            # print("computed score", score)

        scores.append(score)

    pd.DataFrame(scores).to_csv(scores_path, index=None)


assert snakemake.wildcards.model in MODELS, "model not found"
fit(
    snakemake.input.features,
    snakemake.input.targets,
    snakemake.input.split,
    snakemake.output.scores,
    snakemake.wildcards.model,
    snakemake.input.grid,
    snakemake.params.existing_scores,
)
