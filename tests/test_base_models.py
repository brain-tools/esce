import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from workflow.scripts.fit_model import ClassifierModel, RegressionModel


def test_classifier_model():
    # Create a toy binary classification problem
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )

    # Scale features and targets (for the sake of the example)
    X = X * 100

    # Split the data into train, validation and test sets
    idx_train, idx_val, idx_test = (
        list(range(0, 60)),
        list(range(60, 80)),
        list(range(80, 100)),
    )

    # Solve the problem manually (including feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[idx_train])
    X_val_scaled = scaler.transform(X[idx_val])
    X_test_scaled = scaler.transform(X[idx_test])

    model_manual = LogisticRegression(random_state=42)
    model_manual.fit(X_train_scaled, y[idx_train])

    y_hat_train_manual = model_manual.predict(X_train_scaled)
    y_hat_val_manual = model_manual.predict(X_val_scaled)
    y_hat_test_manual = model_manual.predict(X_test_scaled)

    acc_train_manual = accuracy_score(y[idx_train], y_hat_train_manual)
    acc_val_manual = accuracy_score(y[idx_val], y_hat_val_manual)
    acc_test_manual = accuracy_score(y[idx_test], y_hat_test_manual)

    # Solve the problem using the ClassifierModel class
    model_class = ClassifierModel(LogisticRegression, "logistic_regression")
    metrics = model_class.score(X, y, idx_train, idx_val, idx_test, random_state=42)

    # Compare the metrics
    assert pytest.approx(acc_train_manual) == metrics["acc_train"]
    assert pytest.approx(acc_val_manual) == metrics["acc_val"]
    assert pytest.approx(acc_test_manual) == metrics["acc_test"]


def test_regression_model():
    # Create a toy regression problem
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)

    # Scale features and targets (for the sake of the example)
    X = X * 100
    y = y * 50

    # Split the data into train, validation and test sets
    idx_train, idx_val, idx_test = (
        list(range(0, 60)),
        list(range(60, 80)),
        list(range(80, 100)),
    )

    # Solve the problem manually (including feature scaling for features and targets)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X[idx_train])
    X_val_scaled = x_scaler.transform(X[idx_val])
    X_test_scaled = x_scaler.transform(X[idx_test])

    y_train_scaled = y_scaler.fit_transform(y[idx_train].reshape(-1, 1)).flatten()

    model_manual = Ridge(random_state=42)
    model_manual.fit(X_train_scaled, y_train_scaled)

    y_hat_train_scaled_manual = model_manual.predict(X_train_scaled)
    y_hat_val_scaled_manual = model_manual.predict(X_val_scaled)
    y_hat_test_scaled_manual = model_manual.predict(X_test_scaled)

    # Scale predictions back to original scale
    y_hat_train_manual = y_scaler.inverse_transform(
        y_hat_train_scaled_manual.reshape(-1, 1)
    ).flatten()
    y_hat_val_manual = y_scaler.inverse_transform(
        y_hat_val_scaled_manual.reshape(-1, 1)
    ).flatten()
    y_hat_test_manual = y_scaler.inverse_transform(
        y_hat_test_scaled_manual.reshape(-1, 1)
    ).flatten()

    r2_train_manual = r2_score(y[idx_train], y_hat_train_manual)
    r2_val_manual = r2_score(y[idx_val], y_hat_val_manual)
    r2_test_manual = r2_score(y[idx_test], y_hat_test_manual)

    # Solve the problem using the RegressionModel class
    model_class = RegressionModel(Ridge, "ridge_regression")
    metrics = model_class.score(X, y, idx_train, idx_val, idx_test, random_state=42)

    # Compare the metrics
    assert pytest.approx(r2_train_manual) == metrics["r2_train"]
    assert pytest.approx(r2_val_manual) == metrics["r2_val"]
    assert pytest.approx(r2_test_manual) == metrics["r2_test"]
