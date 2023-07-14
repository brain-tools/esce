"""
base_models.py
====================================



"""


from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """Base model class for each model."""

    scale_features: bool
    scale_targets: bool

    def __init__(self, model_generator: Callable[..., Any], model_name: str):
        """Initialize class using a model that is initialized later."""
        self.model_generator = model_generator
        self.model_name = model_name

    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):  # type: ignore
        """Provide a score for the model performance on the data."""

        # generate model based on the given hyperparameters in kwargs
        model = self.model_generator(**kwargs)

        # scale features if scale_features is True
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

        # scale targets if scale_targets is True
        y_scaler = StandardScaler() if self.scale_targets else None
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        if y_scaler:
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = y_train

        # fit model and predict
        model.fit(x_train_scaled, y_train_scaled)
        y_hat_train_scaled = model.predict(x_train_scaled)
        y_hat_val_scaled = model.predict(x_val_scaled)
        y_hat_test_scaled = model.predict(x_test_scaled)

        # scale y_hat back to original scale (so that MAE has unit of original target variable)
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

        # compute metrics (definded in subclasses)
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
        """Compute metrics for the model performance on the data."""
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
