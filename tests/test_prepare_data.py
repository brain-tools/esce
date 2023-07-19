import h5py
import numpy as np
import pandas as pd
import pytest

from workflow.scripts.prepare_data import prepare_data

TEST_CASES = [
    ("tsv", "targets"),
    ("tsv", "features"),
    ("tsv", "covariates"),
    ("csv", "targets"),
    ("csv", "features"),
    ("csv", "covariates"),
    ("npy", "targets"),
    ("npy", "features"),
    ("npy", "covariates"),
]


@pytest.mark.parametrize(
    (
        "in_file_type",
        "features_targets_covariates",
    ),
    TEST_CASES,
)
def test_prepare_data_custom_datasets(
    tmpdir,
    in_file_type,
    features_targets_covariates,
):
    in_path = str(tmpdir.join("data." + in_file_type))
    out_path = str(tmpdir.join("data.h5"))

    dummy_data = (
        np.random.rand(10, 1)
        if features_targets_covariates == "targets"
        else np.random.rand(10, 2)
    )
    dummy_data = pd.DataFrame(dummy_data)

    if in_path.endswith(".csv"):
        dummy_data.to_csv(in_path, index=False)
    elif in_path.endswith(".tsv"):
        dummy_data.to_csv(in_path, sep="\t", index=False)
    elif in_path.endswith(".npy"):
        np.save(in_path, dummy_data.values)

    custom_datasets = {
        "pytest": {
            "targets": {"normal": in_path},
            "features": {"normal": in_path},
            "covariates": {"normal": in_path},
        }
    }

    prepare_data(
        out_path, "pytest", features_targets_covariates, "normal", custom_datasets
    )

    # Load the data from the output file and check its dimensions
    with h5py.File(out_path, "r") as f:
        output_data = f["data"][:]
        output_mask = f["mask"][:]

    expected_shape = 1 if features_targets_covariates == "targets" else 2

    assert output_data.ndim == expected_shape
    assert output_mask.ndim == 1
