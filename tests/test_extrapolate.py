import json

import pandas as pd
import pytest

from workflow.scripts.extrapolate import extrapolate


@pytest.fixture()
def stats_df():
    # Create a DataFrame with known values
    stats = {
        "n": [10, 20, 30, 40, 50],
        "s": [1, 2, 3, 4, 5],
        "r2_test": [0.9, 0.85, 0.8, 0.75, 0.7],
        "acc_test": [0.95, 0.9, 0.85, 0.8, 0.75],
    }
    return pd.DataFrame(stats)


def test_extrapolate(tmpdir, stats_df):
    # Save the DataFrame to a temporary file
    stats_path = str(tmpdir.join("stats.csv"))
    stats_df.to_csv(stats_path, index=False)

    # Define the paths for the extra and bootstrap files
    extra_path = str(tmpdir.join("extra.json"))
    bootstrap_path = str(tmpdir.join("bootstrap.json"))

    # Run the extrapolate function
    extrapolate(stats_path, extra_path, bootstrap_path, repeats=100)

    # Load the result and check if it is correct
    with open(extra_path) as f:
        result = json.load(f)

    assert isinstance(result, dict)  # The result should be a dictionary
    # Add more assertions here to check the values in the result
