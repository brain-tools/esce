from workflow.scripts.extrapolate import NpEncoder, extrapolate
import os
import json
import numpy as np
import pandas as pd


def test_NpEncoder():
    data = {
        "int_value_16": np.int16(0),
        "int_value_64": np.int64(1),
        "float_value_32": np.float32(1.23),
        "float_value_64": np.float64(4.56),
        "array_value": np.array([1, 2, 3, 4, 5]),
        "str_value": "test NpEncoder",
        "bool_value": True,
        "none_value": None,
    }

    encoded_data = json.dumps(data, cls=NpEncoder)

    # Decode the JSON string back into an object
    decoded_data = json.loads(encoded_data)

    # Verify conversions
    assert isinstance(decoded_data["int_value_16"], int), "int16 conversion failed"
    assert isinstance(decoded_data["int_value_64"], int), "int64 conversion failed"
    assert isinstance(
        decoded_data["float_value_32"], float
    ), "float32 conversion failed"
    assert isinstance(
        decoded_data["float_value_64"], float
    ), "float65 conversion failed"
    assert isinstance(decoded_data["array_value"], list), "array conversion failed"
    assert isinstance(decoded_data["str_value"], str), "string conversion failed"
    assert isinstance(decoded_data["bool_value"], bool), "boolean conversion failed"
    assert decoded_data["none_value"] is None, "none value conversion failed"


def test_extrapolate():
    stats_path = "tests/data/extrapolate_stats.csv"
    extra_path = "tests/data/extrapolate_extra.json"
    bootstrap_path = "tests/data/extrapolate_bootstrap.json"

    ###
    # Test Case 1
    ###
    pd.DataFrame(
        {
            "n": [10, 20, 30, 40, 50],
            "s": [1, 1, 1, 2, 2],
            "r2_test": [0.8, 0.9, 0.85, 0.7, 0.75],
        }
    ).to_csv(stats_path, index=False)

    extrapolate(stats_path, str(extra_path), str(bootstrap_path), repeats=100)

    assert os.path.exists(extra_path), "extra file doesn't exist"
    assert os.path.exists(bootstrap_path), "bootstrap file doesn't exist"

    with open(extra_path, "r") as f:
        extra_data = json.load(f)
    with open(bootstrap_path, "r") as f:
        bootstrap_data = json.load(f)

    # Verify the keys in the extra_data dictionary
    expected_keys = [
        "n_seeds",
        "metric",
        "x",
        "y_mean",
        "y_std",
        "y_sem",
        "mask",
        "dof",
        "p_mean",
        "r2",
        "chi2",
        "mu",
        "sigma",
        "p_bootstrap_mean",
        "p_bootstrap_std",
        "p_bootstrap_975",
        "p_bootstrap_025",
        "y_bootstrap_mean",
        "y_bootstrap_std",
        "y_bootstrap_975",
        "y_bootstrap_025",
    ]
    assert all(key in extra_data for key in expected_keys)

    assert extra_data["metric"] == "r2_test"
    assert np.array_equal(extra_data["x"], np.array([10, 20, 30, 40, 50]))
    assert np.array_equal(extra_data["y_mean"], np.array([0.8, 0.9, 0.85, 0.7, 0.75]))
    assert np.array_equal(
        extra_data["y_std"],
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )
    assert np.array_equal(
        extra_data["y_sem"],
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )
    assert np.array_equal(
        extra_data["mask"], np.array([False, False, False, False, False])
    )
    assert extra_data["dof"] == -3
    assert bootstrap_data is None

    for file in [stats_path, extra_path, bootstrap_path]:
        if os.path.exists(file) and os.path.isfile(file):
            os.remove(file)

    ###
    # Test Case 2
    ###
    df = pd.DataFrame(
        {
            "r2_train": [
                0.0002916496,
                1.0,
                0.9999999999,
                0.1744241596,
                0.5312638174,
                0.3054214328,
                0.1020143936,
                0.0840176553,
                0.1503364807,
            ],
            "r2_val": [
                0.0002916496,
                1.0,
                0.9999999999,
                0.1744241596,
                0.5312638174,
                0.3054214328,
                0.1020143936,
                0.0840176553,
                0.1503364807,
            ],
            "r2_test": [
                -0.0403154031,
                0.0,
                0.0,
                -0.0611194339,
                -1.578495288,
                0.1028331212,
                -0.2548612821,
                -0.1367956568,
                -0.3537043287,
            ],
            "mae_train": [
                0.4799299959,
                1.389e-06,
                3.8034e-06,
                0.4101482715,
                0.2801894748,
                0.3843861514,
                0.4376957458,
                0.4732234159,
                0.4230906211,
            ],
            "mae_val": [
                0.5000184842,
                0.5836114127,
                1.1377520108,
                0.3940803717,
                0.2895893372,
                0.4370887027,
                0.4843529896,
                0.4750198631,
                0.5143306905,
            ],
            "mae_test": [
                0.5000803109,
                1.3755731388,
                0.508924909,
                0.4746311127,
                0.6244257608,
                0.4615771291,
                0.5173919943,
                0.5136989859,
                0.5357145016,
            ],
            "mse_train": [
                0.2399300041,
                0.0,
                0.0,
                0.1878185037,
                0.1124966838,
                0.1719081954,
                0.208532213,
                0.227977828,
                0.1973107506,
            ],
            "mse_val": [
                0.2600237279,
                0.3797041599,
                1.4623375421,
                0.1664442254,
                0.1175665421,
                0.2109894189,
                0.2592186171,
                0.2299687257,
                0.2925520739,
            ],
            "mse_test": [
                0.2600788508,
                1.9659927355,
                0.3415243679,
                0.2546686641,
                0.6188388691,
                0.2153200509,
                0.294108113,
                0.2664364821,
                0.317274452,
            ],
            "alpha": [100000.0, 1e-05, 1e-05, 100.0, 1.0, 10.0, 100.0, 100.0, 100.0],
            "n": [10, 10, 10, 20, 20, 20, 30, 30, 30],
            "s": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )
    df.to_csv(stats_path, index=False)

    extrapolate(stats_path, str(extra_path), str(bootstrap_path), repeats=100)

    with open(extra_path, "r") as f:
        extra_data = json.load(f)
    with open(bootstrap_path, "r") as f:
        bootstrap_data = json.load(f)

    assert extra_data["metric"] == "r2_test"
    assert np.array_equal(extra_data["x"], np.array([10, 20, 30]), equal_nan=True)
    assert np.allclose(
        extra_data["y_mean"],
        np.array([-0.013438467691572267, -0.5122605335838996, -0.248453755907222]),
        equal_nan=True,
    )
    assert np.allclose(
        extra_data["y_std"],
        np.array([0.02327610881767601, 0.9270180816941317, 0.1085962028735548]),
        equal_nan=True,
    )
    assert np.allclose(
        extra_data["y_sem"],
        np.array([0.013438467691572267, 0.5352141390097575, 0.06269804696201808]),
        equal_nan=True,
    )
    assert np.array_equal(extra_data["mask"], np.array([False, False, False]))
    assert extra_data["dof"] == -3
    assert bootstrap_data is None

    for file in [stats_path, extra_path, bootstrap_path]:
        if os.path.exists(file) and os.path.isfile(file):
            os.remove(file)
