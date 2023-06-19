from workflow.scripts.fit_model import fit ,ClassifierModel, RegressionModel, get_existing_scores
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error


def test_fit():
    features_path = "tests/data/fit_model-fit_features.npy"
    targets_path = "tests/data/fit_model-fit_targets.npy"
    split_path = "tests/data/fit_model-fit_split.json"
    scores_path = "tests/data/fit_model-fit_scores.csv"
    model_name = "majority-classifier" # "ridge-cls", "ridge-reg"
    grid_path = "config/grids/default.yaml"
    existing_scores_path_list = ["tests/data/fit_model-fit_existing_scores1.csv", "tests/data/fit_model-fit_existing_scores2.csv"]

    # Sample data
    split_data = {
        "idx_train": [0, 1, 2],
        "idx_val": [3, 4, 5],
        "idx_test": [6, 7, 8],
        "samplesize": 9,
        "seed": 123,
    }
    Path(split_path).touch()
    with open(split_path, "w") as f:
        json.dump(split_data, f)

    x = np.random.rand(9, 10)
    y = np.random.rand(9)
    np.save(features_path, x)
    np.save(targets_path, y)
    existing_scores_data = {
        "param1": [1, 2, 3],
        "param2": ["a", "b", "c"],
        "score": [0.1, 0.2, 0.3],
        "n": [100, 200, 300],
        "s": [1, 2, 3]

    }
    existing_scores_df = pd.DataFrame(existing_scores_data)
    for path in existing_scores_path_list:
        existing_scores_df.to_csv(path, index=None)

    fit(
        features_path,
        targets_path,
        split_path,
        str(scores_path),
        model_name,
        grid_path,
        existing_scores_path_list,
    )

    assert os.path.isfile(str(scores_path)), 'scores file doesn\'t exist'
    scores_df = pd.read_csv(str(scores_path))

    assert len(scores_df) == 1  # No. of parameter combinations
    assert "param1" in scores_df.columns
    assert "param2" in scores_df.columns
    assert "score" in scores_df.columns
    assert "n" in scores_df.columns
    assert "s" in scores_df.columns
    assert scores_df["n"].unique() == [100], '8'
    assert scores_df["s"].unique() == [1], '0'

    # Remove temporary files
    for file in [features_path, targets_path, split_path, scores_path] + existing_scores_path_list:
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)


class TestClassifierModel(ClassifierModel):

    # to avoid PytestCollectionWarning
    __test__ = False

    def compute_metrics(
        self,
        y_hat_train,
        y_hat_val,
        y_hat_test,
        y_train,
        y_val,
        y_test,
    ):
    
        metrics = {
            "acc_train": accuracy_score(y_train, y_hat_train),
            "acc_val": accuracy_score(y_val, y_hat_val),
            "acc_test": accuracy_score(y_test, y_hat_test),
            "f1_train": f1_score(y_train, y_hat_train, average="weighted"),
            "f1_val": f1_score(y_val, y_hat_val, average="weighted"),
            "f1_test": f1_score(y_test, y_hat_test, average="weighted"),
        }
        return metrics

def test_DummyClassifierModel():
    num_samples = 100
    num_features = 10
    x = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, 2, size=num_samples)

    model = TestClassifierModel(lambda **kwargs: DummyClassifier(strategy="most_frequent", **kwargs), "majority classifier")

    idx_train = np.arange(80)
    idx_val = np.arange(80, 90)
    idx_test = np.arange(90, 100)

    metrics = model.score(x, y, idx_train, idx_val, idx_test)

    assert "acc_train" in metrics
    assert "acc_val" in metrics
    assert "acc_test" in metrics
    assert "f1_train" in metrics
    assert "f1_val" in metrics
    assert "f1_test" in metrics

    del model

def test_RidgeClassifierModel():
    num_samples = 100
    num_features = 10
    x = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, 2, size=num_samples)

    model = TestClassifierModel(lambda **kwargs: RidgeClassifier(**kwargs), "ridge classifier")

    idx_train = np.arange(80)
    idx_val = np.arange(80, 90)
    idx_test = np.arange(90, 100)

    metrics = model.score(x, y, idx_train, idx_val, idx_test)

    assert "acc_train" in metrics
    assert "acc_val" in metrics
    assert "acc_test" in metrics
    assert "f1_train" in metrics
    assert "f1_val" in metrics
    assert "f1_test" in metrics

    del model

class TestRegressionModel(RegressionModel):

    # To avoid PytestCollectionWarning
    __test__ = False

    def compute_metrics(
        self,
        y_hat_train,
        y_hat_val,
        y_hat_test,
        y_train,
        y_val,
        y_test,
    ):
    
        metrics = {
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
        return metrics

def test_RegressionModel():
    num_samples = 100
    num_features = 10
    x = np.random.randn(num_samples, num_features)
    y = np.random.randn(num_samples)

    model = TestRegressionModel(lambda **kwargs: Ridge(**kwargs), "ridge regressor")

    idx_train = np.arange(80)
    idx_val = np.arange(80, 90)
    idx_test = np.arange(90, 100)

    metrics = model.score(x, y, idx_train, idx_val, idx_test)

    assert "r2_train" in metrics
    assert "r2_val" in metrics
    assert "r2_test" in metrics
    assert "mae_train" in metrics
    assert "mae_val" in metrics
    assert "mae_test" in metrics
    assert "mse_train" in metrics
    assert "mse_val" in metrics
    assert "mse_test" in metrics

    del model


def test_get_existing_scores():
    
    ###
    # Test case 1: existing score files with values
    ###
    # Create sample existing scores files
    existing_scores_paths = ["tests/data/fit_model-existing_scores1.csv", "tests/data/fit_model-existing_scores2.csv"]
    existing_scores1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    existing_scores1.to_csv(existing_scores_paths[0], index=None)
    existing_scores2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})
    existing_scores2.to_csv(existing_scores_paths[1], index=None)

    result = get_existing_scores(existing_scores_paths)

    expected = pd.DataFrame({"col1": [1, 2, 5, 6], "col2": [3, 4, 7, 8]})

    # Assert result is as expected
    pd.testing.assert_frame_equal(result, expected)

    ###
    # Test case 2: there is no existing score files
    ###
    result = get_existing_scores([])
    expected = pd.DataFrame()
    pd.testing.assert_frame_equal(result, expected)

    # Remove temporary files
    for file in existing_scores_paths:
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)
