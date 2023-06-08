# from workflow.scripts.fit_model import BaseModel, ClassifierModel, RegressionModel, get_existing_scores, fit

# import os
# import json
# import yaml
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import Any, Callable, Dict
# from abc import ABC, abstractmethod
# from sklearn.model_selection import ParameterGrid
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
# from tempfile import NamedTemporaryFile

# class TestModel(BaseModel):
#     def compute_metrics(
#         self,
#         y_hat_train: np.ndarray,
#         y_hat_val: np.ndarray,
#         y_hat_test: np.ndarray,
#         y_train: np.ndarray,
#         y_val: np.ndarray,
#         y_test: np.ndarray,
#     ) -> Dict[str, float]:
#         metrics = {}
#         metrics["train_score"] = np.mean(y_hat_train)
#         metrics["val_score"] = np.mean(y_hat_val)
#         metrics["test_score"] = np.mean(y_hat_test)
#         return metrics

# def test_BaseModel():
#     # Sample dataset
#     num_samples = 100
#     num_features = 10
#     x = np.random.randn(num_samples, num_features)
#     y = np.random.randn(num_samples)

#     model = TestModel(lambda **kwargs: None, "test_model")

#     idx_train = np.arange(80)
#     idx_val = np.arange(80, 90)
#     idx_test = np.arange(90, 100)

#     metrics = model.score(x, y, idx_train, idx_val, idx_test)

#     assert "train_score" in metrics
#     assert "val_score" in metrics
#     assert "test_score" in metrics

#     del model

# class TestClassifierModel(ClassifierModel):
#     def compute_metrics(
#         self,
#         y_hat_train,
#         y_hat_val,
#         y_hat_test,
#         y_train,
#         y_val,
#         y_test,
#     ):
    
#         metrics = {
#             "acc_train": accuracy_score(y_train, y_hat_train),
#             "acc_val": accuracy_score(y_val, y_hat_val),
#             "acc_test": accuracy_score(y_test, y_hat_test),
#             "f1_train": f1_score(y_train, y_hat_train, average="weighted"),
#             "f1_val": f1_score(y_val, y_hat_val, average="weighted"),
#             "f1_test": f1_score(y_test, y_hat_test, average="weighted"),
#         }
#         return metrics

# def test_ClassifierModel():
#     # Sample dataset
#     num_samples = 100
#     num_features = 10
#     x = np.random.randn(num_samples, num_features)
#     y = np.random.randint(0, 2, size=num_samples)

#     model = TestClassifierModel(lambda **kwargs: None, "test_model")

#     # Define the train, val, and test indices
#     idx_train = np.arange(80)
#     idx_val = np.arange(80, 90)
#     idx_test = np.arange(90, 100)

#     metrics = model.score(x, y, idx_train, idx_val, idx_test)

#     assert "acc_train" in metrics
#     assert "acc_val" in metrics
#     assert "acc_test" in metrics
#     assert "f1_train" in metrics
#     assert "f1_val" in metrics
#     assert "f1_test" in metrics

#     del model


# class TestRegressionModel(RegressionModel):
#     def compute_metrics(
#         self,
#         y_hat_train,
#         y_hat_val,
#         y_hat_test,
#         y_train,
#         y_val,
#         y_test,
#     ):
    
#         metrics = {
#             "r2_train": r2_score(y_train, y_hat_train),
#             "r2_val": r2_score(y_val, y_hat_val),
#             "r2_test": r2_score(y_test, y_hat_test),
#             "mae_train": mean_absolute_error(y_train, y_hat_train),
#             "mae_val": mean_absolute_error(y_val, y_hat_val),
#             "mae_test": mean_absolute_error(y_test, y_hat_test),
#             "mse_train": mean_squared_error(y_train, y_hat_train),
#             "mse_val": mean_squared_error(y_val, y_hat_val),
#             "mse_test": mean_squared_error(y_test, y_hat_test),
#         }
#         return metrics

# def test_RegressionModel():
#     # Sample dataset
#     num_samples = 100
#     num_features = 10
#     x = np.random.randn(num_samples, num_features)
#     y = np.random.randn(num_samples)

#     model = TestRegressionModel(lambda **kwargs: None, "test_model")

#     idx_train = np.arange(80)
#     idx_val = np.arange(80, 90)
#     idx_test = np.arange(90, 100)

#     metrics = model.score(x, y, idx_train, idx_val, idx_test)

#     assert "r2_train" in metrics
#     assert "r2_val" in metrics
#     assert "r2_test" in metrics
#     assert "mae_train" in metrics
#     assert "mae_val" in metrics
#     assert "mae_test" in metrics
#     assert "mse_train" in metrics
#     assert "mse_val" in metrics
#     assert "mse_test" in metrics

#     del model


# def test_get_existing_scores():
#     # Sample data
#     with NamedTemporaryFile(delete=False) as temp_file1, NamedTemporaryFile(delete=False) as temp_file2:
#         temp_file1.write(b"col1,col2\n1,2\n3,4\n")
#         temp_file2.write(b"col1,col2\n5,6\n7,8\n")
#         temp_file1.flush()
#         temp_file2.flush()

#         path1 = Path(temp_file1.name)
#         path2 = Path(temp_file2.name)
#         result = get_existing_scores([path1, path2])

#         # Verifying
#         expected = pd.DataFrame({"col1": [1, 3, 5, 7], "col2": [2, 4, 6, 8]})
#         pd.testing.assert_frame_equal(result, expected)

#         # Remove temporary files
#         os.remove(temp_file1.name)
#         os.remove(temp_file2.name)

#     # Call with an empty list of file paths
#     result = get_existing_scores([])
#     expected = pd.DataFrame()
#     pd.testing.assert_frame_equal(result, expected)




# def test_fit():
#     with NamedTemporaryFile(delete=False) as features_file, NamedTemporaryFile(delete=False) as targets_file, \
#          NamedTemporaryFile(delete=False) as split_file, NamedTemporaryFile(delete=False) as scores_file, \
#          NamedTemporaryFile(delete=False) as grid_file:
#         # Random features and targets
#         x, y = make_classification(n_samples=100, n_features=5, random_state=42)

#         # Save to temporary files
#         np.save(features_file, x)
#         np.save(targets_file, y)
#         features_path = Path(features_file.name)
#         targets_path = Path(targets_file.name)

#         split_dict = {
#             "idx_train": np.arange(70),
#             "idx_val": np.arange(70, 85),
#             "idx_test": np.arange(85, 100),
#             "samplesize": len(y),
#             "seed": 42
#         }

#         # Save to a temporary file
#         json.dump(split_dict, split_file)
#         split_path = Path(split_file.name)

#         grid_dict = {
#             "LogisticRegression": [
#                 {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]}
#             ]
#         }

#         yaml.safe_dump(grid_dict, grid_file)
#         grid_path = Path(grid_file.name)

#         fit(
#             features_path,
#             targets_path,
#             split_path,
#             scores_file.name,
#             "LogisticRegression",
#             grid_path,
#             []
#         )

#         scores = pd.read_csv(scores_file)

#         assert len(scores) == 6
#         assert all(score["model_name"] == "LogisticRegression" for _, score in scores.iterrows())
#         assert all(score["n"] == len(y) for _, score in scores.iterrows())
#         assert all(score["s"] == 42 for _, score in scores.iterrows())
#         assert all(score["C"] in [0.1, 1.0, 10.0] for _, score in scores.iterrows())
#         assert all(score["penalty"] in ["l1", "l2"] for _, score in scores.iterrows())
#         assert all("acc_train" in score.keys() for _, score in scores.iterrows())

#         # Remove temporary files
#         Path(features_file.name).unlink()
#         Path(targets_file.name).unlink()
#         Path(split_file.name).unlink()
#         Path(scores_file.name).unlink()
#         Path(grid_file.name).unlink()
