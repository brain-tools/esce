import h5py
import pytest

from workflow.scripts.prepare_data import prepare_data


@pytest.mark.slow()
def test_mnist_features(tmpdir):
    out_path = str(tmpdir / "test.h5")
    dataset = "mnist"
    features_targets_covariates = "features"
    variant = "pixel"
    custom_datasets = {}

    prepare_data(
        out_path, dataset, features_targets_covariates, variant, custom_datasets
    )

    # Load the data from the output file and check dimensions
    with h5py.File(out_path, "r") as f:
        output_data = f["data"][:]

    assert output_data.shape == (70000, 784)


@pytest.mark.slow()
def test_mnist_targets(tmpdir):
    out_path = str(tmpdir / "test.h5")
    dataset = "mnist"
    features_targets_covariates = "targets"
    variant = "ten-digits"
    custom_datasets = {}

    prepare_data(
        out_path, dataset, features_targets_covariates, variant, custom_datasets
    )

    # Load the data from the output file and check dimensions
    with h5py.File(out_path, "r") as f:
        output_data = f["data"][:]

    assert output_data.shape == (70000,)
