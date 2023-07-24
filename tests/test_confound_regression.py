
import h5py
import numpy as np
import pytest

from workflow.scripts.confound_regression import confound_regression


@pytest.mark.parametrize(
    ("data", "confounds", "expected"),
    [
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.zeros((5, 1)),
        ),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]), np.zeros((2, 2))),
    ],
)
def test_confound_regression(tmpdir, data, confounds, expected):
    out_path = str(tmpdir / "corrected.h5")
    data_path = str(tmpdir / "data.h5")
    confound_path = str(tmpdir / "confounds.h5")

    with h5py.File(data_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("mask", data=np.ones(len(data)))

    with h5py.File(confound_path, "w") as f:
        f.create_dataset("data", data=confounds)
        f.create_dataset("mask", data=np.ones(len(confounds)))

    # Run the function
    confound_regression(data_path, confound_path, out_path)

    # Load the corrected data
    with h5py.File(out_path, "r") as f:
        corrected_data = f["data"][:]

    # Check output shape
    assert corrected_data.shape == expected.shape

    # Check if the corrected data is as expected
    assert np.allclose(corrected_data, expected)
