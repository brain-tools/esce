"""
test_extrapolate.py
===================

This module contains unit tests for the extrapolate function and related utilities
in the extrapolate module. It tests various scenarios including normal operation,
edge cases, and error handling.

Test Summary:
1. test_extrapolate: Tests the main extrapolate function with sample data.
2. test_extrapolate_empty_input: Tests handling of empty input files.
3. test_extrapolate_insufficient_data: Tests behavior with insufficient data for curve fitting.
4. test_extrapolate_no_r2_test: Tests fallback to accuracy metric when R² is not available.
5. test_power_law_model: Tests the power law model function.
6. test_fit_curve: Tests the curve fitting function with known data.
7. test_fit_curve_failure: Tests handling of curve fitting failures.
8. test_fit_curve_exception: Tests exception handling in curve fitting.
9. test_extrapolate_bootstrap_failure: Tests handling of frequent bootstrap failures.
10. test_extrapolate_file_not_found: Tests extrapolate function with a non-existent input file.
11. test_extrapolate_unreadable_file: Tests extrapolate function with an unreadable input file.
12. test_extrapolate_insufficient_data_bootstrap: Tests extrapolate function with insufficient data, focusing on bootstrap results.
13. test_extrapolate_output_completeness: Tests that the extrapolate function output contains all required fields.
14. test_extrapolate_reproducibility: Tests that the extrapolate function produces consistent results with a fixed random seed.
15. test_extrapolate_bootstrap_file_contents: Tests that the extrapolate function writes bootstrap results to a file and verify the contents of the bootstrap file.

The tests use pytest fixtures and mocking to create controlled test environments.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import os
from unittest.mock import mock_open, patch

from workflow.scripts.extrapolate import extrapolate, MIN_DOF, fit_curve, power_law_model

@pytest.fixture
def stats_df():
    """
    Create a DataFrame with known values for testing.
    
    Returns:
        pd.DataFrame: A sample DataFrame containing test data.
    """
    np.random.seed(42)  # Set seed for reproducibility
    n_samples = 50
    stats = {
        "n": np.repeat([10, 20, 30, 40, 50], n_samples // 5),
        "s": np.tile(range(1, 11), 5),  # 10 seeds for each sample size
        "r2_test": np.random.uniform(0.7, 0.95, n_samples),
        "acc_test": np.random.uniform(0.75, 0.98, n_samples),
    }
    return pd.DataFrame(stats)

def test_extrapolate(tmpdir, stats_df):
    """
    Test the extrapolate function with a sample DataFrame.
    
    Args:
        tmpdir: Pytest fixture for creating a temporary directory.
        stats_df: Pytest fixture providing a sample DataFrame.
    """
    # Prepare input and output paths
    stats_path = Path(tmpdir) / "stats.csv"
    extra_path = Path(tmpdir) / "extra.json"
    bootstrap_path = Path(tmpdir) / "bootstrap.json"
    
    # Save the DataFrame to a temporary CSV file
    stats_df.to_csv(stats_path, index=False)
    
    # Run the extrapolate function
    extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    # Load and validate the extrapolation results
    with open(extra_path) as f:
        result = json.load(f)
    
    # Basic structure checks
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "n_seeds" in result, "Result should contain 'n_seeds'"
    assert "metric" in result, "Result should contain 'metric'"
    assert "x" in result, "Result should contain 'x' (sample sizes)"
    assert "y_mean" in result, "Result should contain 'y_mean'"
    assert "p_mean" in result, "Result should contain 'p_mean' (fitted parameters)"
    
    # Data consistency checks
    assert result["n_seeds"] == len(stats_df["s"].unique()), "Incorrect number of seeds"
    assert result["metric"] in ["r2_test", "acc_test"], "Unexpected metric used"
    assert len(result["x"]) == len(set(stats_df["n"])), "Mismatch in number of sample sizes"
    
    # Fitted curve checks
    assert len(result["p_mean"]) == 3, "Power law fit should have 3 parameters"
    assert result["dof"] >= MIN_DOF, f"Degrees of freedom should be at least {MIN_DOF}"
    
    # Bootstrap checks
    assert "p_bootstrap_mean" in result, "Bootstrap results missing"
    assert len(result["p_bootstrap_mean"]) == 3, "Bootstrap mean should have 3 values"
    
    # Load and validate bootstrap results
    with open(bootstrap_path) as f:
        bootstrap_data = json.load(f)
    
    assert isinstance(bootstrap_data, list), "Bootstrap data should be a list"
    assert len(bootstrap_data) > 0, "Bootstrap data should not be empty"
    
    print("All checks passed successfully.")

def test_extrapolate_empty_input(tmpdir):
    """
    Test extrapolate function with an empty input file.
    
    Args:
        tmpdir: Pytest fixture for creating a temporary directory.
    """
    stats_path = Path(tmpdir) / "empty_stats.csv"
    extra_path = Path(tmpdir) / "empty_extra.json"
    bootstrap_path = Path(tmpdir) / "empty_bootstrap.json"
    
    # Create an empty file
    stats_path.touch()
    
    extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    assert extra_path.exists(), "Extra file should be created even with empty input"
    assert bootstrap_path.exists(), "Bootstrap file should be created even with empty input"
    assert extra_path.stat().st_size == 0, "Extra file should be empty"
    assert bootstrap_path.stat().st_size == 0, "Bootstrap file should be empty"

def test_extrapolate_insufficient_data(tmpdir):
    """
    Test extrapolate function with insufficient data for curve fitting.
    
    Args:
        tmpdir: Pytest fixture for creating a temporary directory.
    """
    stats_df = pd.DataFrame({
        "n": [10, 20],
        "s": [1, 1],
        "r2_test": [0.8, 0.85],
        "acc_test": [0.9, 0.92]
    })
    
    stats_path = Path(tmpdir) / "insufficient_stats.csv"
    extra_path = Path(tmpdir) / "insufficient_extra.json"
    bootstrap_path = Path(tmpdir) / "insufficient_bootstrap.json"
    
    stats_df.to_csv(stats_path, index=False)
    
    extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    with open(extra_path) as f:
        result = json.load(f)
    
    assert np.isnan(result["p_mean"]).all(), "p_mean should be NaN for insufficient data"
    assert np.isnan(result["r2"]), "R² should be NaN for insufficient data"

def test_extrapolate_no_r2_test(tmpdir):
    """
    Test extrapolate function when r2_test is not present in the input.
    
    Args:
        tmpdir: Pytest fixture for creating a temporary directory.
    """
    stats_df = pd.DataFrame({
        "n": [10, 20, 30],
        "s": [1, 1, 1],
        "acc_test": [0.9, 0.92, 0.94]
    })
    
    stats_path = Path(tmpdir) / "no_r2_stats.csv"
    extra_path = Path(tmpdir) / "no_r2_extra.json"
    bootstrap_path = Path(tmpdir) / "no_r2_bootstrap.json"
    
    stats_df.to_csv(stats_path, index=False)
    
    extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    with open(extra_path) as f:
        result = json.load(f)
    
    assert result["metric"] == "acc_test", "Metric should be acc_test when r2_test is not present"

def test_power_law_model():
    """Test the power law model function."""
    x = np.array([1, 2, 3, 4, 5])
    a, b, c = 2, 0.5, 1
    expected = 2 * x**(-0.5) + 1
    np.testing.assert_allclose(power_law_model(x, a, b, c), expected, rtol=1e-7)

def test_fit_curve():
    """Test the fit_curve function with known data."""
    x = np.array([10, 20, 30, 40, 50])
    y = np.array([0.9, 0.85, 0.82, 0.8, 0.79])
    y_e = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    
    result = fit_curve(x, y, y_e)
    
    assert "p_mean" in result, "Result should contain fitted parameters"
    assert "r2" in result, "Result should contain R² score"
    assert "chi2" in result, "Result should contain chi-squared statistic"
    assert "mu" in result, "Result should contain mean of normalized residuals"
    assert "sigma" in result, "Result should contain std of normalized residuals"
    
    assert len(result["p_mean"]) == 3, "Fitted parameters should have 3 values"
    assert not np.isnan(result["r2"]), "R² score should not be NaN"
    assert not np.isnan(result["chi2"]), "Chi-squared should not be NaN"

def test_fit_curve_failure():
    """Test the fit_curve function when curve fitting fails."""
    x = np.array([10, 20])
    y = np.array([0.9, 0.85])
    y_e = np.array([0.01, 0.01])
    
    result = fit_curve(x, y, y_e)
    
    assert np.isnan(result["p_mean"]).all(), "p_mean should be NaN when fitting fails"
    assert np.isnan(result["r2"]), "R² should be NaN when fitting fails"

@patch('workflow.scripts.extrapolate.scipy.optimize.curve_fit')
def test_fit_curve_exception(mock_curve_fit):
    """
    Test the fit_curve function when an exception is raised.
    
    Args:
        mock_curve_fit: Mocked curve_fit function.
    """
    mock_curve_fit.side_effect = RuntimeError("Curve fitting failed")
    
    x = np.array([10, 20, 30, 40, 50])
    y = np.array([0.9, 0.85, 0.82, 0.8, 0.79])
    y_e = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    
    result = fit_curve(x, y, y_e)
    
    assert np.isnan(result["p_mean"]).all(), "p_mean should be NaN when an exception is raised"
    assert np.isnan(result["r2"]), "R² should be NaN when an exception is raised"

def test_extrapolate_bootstrap_failure(tmpdir, stats_df):
    """
    Test extrapolate function when bootstrap fails frequently.
    
    Args:
        tmpdir: Pytest fixture for creating a temporary directory.
        stats_df: Pytest fixture providing a sample DataFrame.
    """
    stats_path = Path(tmpdir) / "stats.csv"
    extra_path = Path(tmpdir) / "extra.json"
    bootstrap_path = Path(tmpdir) / "bootstrap.json"
    
    stats_df.to_csv(stats_path, index=False)
    
    with patch('workflow.scripts.extrapolate.fit_curve') as mock_fit_curve:
        # Make fit_curve fail 95% of the time during bootstrap
        mock_fit_curve.side_effect = lambda *args, **kwargs: (
            {"p_mean": np.array([np.nan, np.nan, np.nan])} if np.random.random() < 0.95
            else {"p_mean": np.array([1.0, 1.0, 1.0])}
        )
        
        extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    with open(extra_path) as f:
        result = json.load(f)
    
    assert all(np.isnan(v) for v in result["p_bootstrap_mean"]), "Bootstrap mean should be NaN when bootstrap fails frequently"
    assert all(np.isnan(v) for v in result["p_bootstrap_std"]), "Bootstrap std should be NaN when bootstrap fails frequently"

def test_extrapolate_file_not_found(tmpdir):
    """Test extrapolate function with a non-existent input file."""
    non_existent_path = str(Path(tmpdir) / "non_existent.csv")
    extra_path = str(Path(tmpdir) / "extra.json")
    bootstrap_path = str(Path(tmpdir) / "bootstrap.json")
    
    with pytest.raises(FileNotFoundError):
        extrapolate(non_existent_path, extra_path, bootstrap_path, repeats=100)

def test_extrapolate_unreadable_file(tmpdir):
    """Test extrapolate function with an unreadable input file."""
    stats_path = str(Path(tmpdir) / "unreadable.csv")
    extra_path = str(Path(tmpdir) / "extra.json")
    bootstrap_path = str(Path(tmpdir) / "bootstrap.json")
    
    # Create a file with no read permissions
    with open(stats_path, 'w') as f:
        f.write("dummy data")
    os.chmod(stats_path, 0o000)
    
    with pytest.raises(PermissionError):
        extrapolate(stats_path, extra_path, bootstrap_path, repeats=100)

def test_extrapolate_insufficient_data_bootstrap(tmpdir):
    """Test extrapolate function with insufficient data, focusing on bootstrap results."""
    stats_df = pd.DataFrame({
        "n": [10, 20],
        "s": [1, 1],
        "r2_test": [0.8, 0.85],
        "acc_test": [0.9, 0.92]
    })
    
    stats_path = Path(tmpdir) / "insufficient_stats.csv"
    extra_path = Path(tmpdir) / "insufficient_extra.json"
    bootstrap_path = Path(tmpdir) / "insufficient_bootstrap.json"
    
    stats_df.to_csv(stats_path, index=False)
    
    extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    with open(extra_path) as f:
        result = json.load(f)
    
    assert np.isnan(result["p_bootstrap_mean"]).all(), "Bootstrap mean should be NaN for insufficient data"
    assert np.isnan(result["p_bootstrap_std"]).all(), "Bootstrap std should be NaN for insufficient data"
    
    # Check that the bootstrap file contains an empty list
    with open(bootstrap_path, 'r') as f:
        bootstrap_content = json.load(f)
    assert bootstrap_content == [], "Bootstrap file should contain an empty list for insufficient data"

def test_extrapolate_output_completeness(tmpdir, stats_df):
    """Test that the extrapolate function output contains all required fields."""
    stats_path = Path(tmpdir) / "stats.csv"
    extra_path = Path(tmpdir) / "extra.json"
    bootstrap_path = Path(tmpdir) / "bootstrap.json"
    
    stats_df.to_csv(stats_path, index=False)
    
    extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    with open(extra_path) as f:
        result = json.load(f)
    
    required_fields = [
        "n_seeds", "metric", "x", "y_mean", "y_std", "y_sem", "mask", "dof",
        "p_mean", "r2", "chi2", "mu", "sigma",
        "p_bootstrap_mean", "p_bootstrap_std", "p_bootstrap_975", "p_bootstrap_025",
        "y_bootstrap_mean", "y_bootstrap_std", "y_bootstrap_975", "y_bootstrap_025"
    ]
    
    for field in required_fields:
        assert field in result, f"Required field '{field}' is missing from the output"

def test_extrapolate_reproducibility(tmpdir, stats_df):
    """Test that the extrapolate function produces consistent results with a fixed random seed."""
    stats_path = Path(tmpdir) / "stats.csv"
    extra_path1 = Path(tmpdir) / "extra1.json"
    bootstrap_path1 = Path(tmpdir) / "bootstrap1.json"
    extra_path2 = Path(tmpdir) / "extra2.json"
    bootstrap_path2 = Path(tmpdir) / "bootstrap2.json"
    
    stats_df.to_csv(stats_path, index=False)
    
    extrapolate(str(stats_path), str(extra_path1), str(bootstrap_path1), repeats=100)
    extrapolate(str(stats_path), str(extra_path2), str(bootstrap_path2), repeats=100)
    
    with open(extra_path1) as f1, open(extra_path2) as f2:
        result1 = json.load(f1)
        result2 = json.load(f2)
    
    assert result1 == result2, "Results should be identical with the same random seed"

    with open(bootstrap_path1) as f1, open(bootstrap_path2) as f2:
        bootstrap1 = json.load(f1)
        bootstrap2 = json.load(f2)
    
    assert bootstrap1 == bootstrap2, "Bootstrap results should be identical with the same random seed"

def test_extrapolate_bootstrap_file_contents(tmpdir, stats_df):
    """
    Test that the extrapolate function writes bootstrap results to a file
    and verify the contents of the bootstrap file.
    """
    stats_path = Path(tmpdir) / "stats.csv"
    extra_path = Path(tmpdir) / "extra.json"
    bootstrap_path = Path(tmpdir) / "bootstrap.json"
    
    stats_df.to_csv(stats_path, index=False)
    
    # Run extrapolate function
    extrapolate(str(stats_path), str(extra_path), str(bootstrap_path), repeats=100)
    
    # Check if bootstrap file exists
    assert bootstrap_path.exists(), "Bootstrap file should be created"
    
    # Read and check contents of bootstrap file
    with open(bootstrap_path, 'r') as f:
        bootstrap_data = json.load(f)
    
    # Verify structure and content of bootstrap data
    assert isinstance(bootstrap_data, list), "Bootstrap data should be a list"
    assert len(bootstrap_data) > 0, "Bootstrap data should not be empty"
    assert len(bootstrap_data) <= 100, "Bootstrap data should not exceed the number of repeats"
    
    # Check structure of each bootstrap result
    for result in bootstrap_data:
        assert isinstance(result, list), "Each bootstrap result should be a list"
        assert len(result) == 3, "Each bootstrap result should have 3 parameters"
        assert all(isinstance(param, (int, float)) for param in result), "Parameters should be numeric"

    # Verify that extra file contains bootstrap statistics
    with open(extra_path, 'r') as f:
        extra_data = json.load(f)
    
    bootstrap_keys = [
        "p_bootstrap_mean", "p_bootstrap_std", "p_bootstrap_975", "p_bootstrap_025",
        "y_bootstrap_mean", "y_bootstrap_std", "y_bootstrap_975", "y_bootstrap_025"
    ]
    for key in bootstrap_keys:
        assert key in extra_data, f"{key} should be present in extra data"
        assert isinstance(extra_data[key], list), f"{key} should be a list"
        assert len(extra_data[key]) > 0, f"{key} should not be empty"

    print("Bootstrap file contents verified successfully.")