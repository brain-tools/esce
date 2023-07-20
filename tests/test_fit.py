import json

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification

from workflow.scripts.fit_model import fit


def test_fit(tmpdir):
    # Create a toy binary classification problem
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )

    # Split the data into train, validation and test sets
    split = {
        "idx_train": list(range(0, 60)),
        "idx_val": list(range(60, 80)),
        "idx_test": list(range(80, 100)),
        "samplesize": 100,
        "seed": 42,
    }

    # Save the features, targets, and split to temporary files
    features_path = str(tmpdir.join("features.npy"))
    targets_path = str(tmpdir.join("targets.npy"))
    split_path = str(tmpdir.join("split.json"))

    with h5py.File(features_path, "w") as f:
        f.create_dataset("data", data=X)
        f.create_dataset("mask", data=np.isfinite(X).all(axis=1))

    with h5py.File(targets_path, "w") as f:
        f.create_dataset("data", data=y)
        f.create_dataset("mask", data=np.isfinite(y))

    with open(split_path, "w") as f:
        json.dump(split, f)

    # Define the model and grid parameters
    model_name = "ridge-cls"
    grid_path = tmpdir.join("grid.yaml")
    hp = {"alpha": [0.1, 1.0, 10.0]}
    grid = {model_name: hp}

    # Run the fit function
    scores_path = tmpdir.join("scores.csv")
    existing_scores_path_list = []
    fit(
        features_path,
        targets_path,
        split_path,
        scores_path,
        model_name,
        grid,
        existing_scores_path_list,
    )

    # Load the scores and check if they are correct
    scores = pd.read_csv(scores_path)

    assert "acc_train" in scores.columns
    assert "acc_val" in scores.columns
    assert "acc_test" in scores.columns
    assert "n" in scores.columns
    assert "s" in scores.columns

    assert scores["n"][0] == split["samplesize"]
    assert scores["s"][0] == split["seed"]


def test_fit_with_existing_scores(tmpdir):
    # Create a toy binary classification problem
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )

    # Split the data into train, validation and test sets
    split = {
        "idx_train": list(range(0, 60)),
        "idx_val": list(range(60, 80)),
        "idx_test": list(range(80, 100)),
        "samplesize": 100,
        "seed": 42,
    }

    # Save the features, targets, and split to temporary files
    features_path = str(tmpdir.join("features.npy"))
    targets_path = str(tmpdir.join("targets.npy"))
    split_path = str(tmpdir.join("split.json"))

    with h5py.File(features_path, "w") as f:
        f.create_dataset("data", data=X)
        f.create_dataset("mask", data=np.isfinite(X).all(axis=1))

    with h5py.File(targets_path, "w") as f:
        f.create_dataset("data", data=y)
        f.create_dataset("mask", data=np.isfinite(y))

    with open(split_path, "w") as f:
        json.dump(split, f)

    # Define the model and grid parameters
    model_name = "ridge-cls"
    grid_path = tmpdir.join("grid.yaml")
    hp = {"alpha": [0.1, 1.0, 10.0]}
    grid = {model_name: hp}

    # Prepare an existing scores file
    EXISTING_ACC = -1
    existing_scores_path = str(tmpdir.join("existing_scores.csv"))
    df_existing_scores = pd.DataFrame(
        [
            {
                "alpha": 0.1,
                "acc_train": EXISTING_ACC,
                "acc_val": EXISTING_ACC,
                "acc_test": EXISTING_ACC,
                "n": 100,
                "s": 42,
            }
        ]
    )
    df_existing_scores.to_csv(existing_scores_path, index=False)

    # Run the fit function
    scores_path = str(tmpdir.join("scores.csv"))
    existing_scores_path_list = [existing_scores_path]
    fit(
        features_path,
        targets_path,
        split_path,
        scores_path,
        model_name,
        grid,
        existing_scores_path_list,
    )

    # Load the scores and check if they are correct
    scores = pd.read_csv(scores_path)

    assert "acc_train" in scores.columns
    assert "acc_val" in scores.columns
    assert "acc_test" in scores.columns
    assert "n" in scores.columns
    assert "s" in scores.columns

    assert scores["n"][0] == split["samplesize"]
    assert scores["s"][0] == split["seed"]

    assert EXISTING_ACC in scores["acc_train"].values

    assert scores["alpha"][0] == df_existing_scores["alpha"][0]
    assert scores["acc_train"][0] == df_existing_scores["acc_train"][0]
    assert scores["acc_val"][0] == df_existing_scores["acc_val"][0]
    assert scores["acc_test"][0] == df_existing_scores["acc_test"][0]
