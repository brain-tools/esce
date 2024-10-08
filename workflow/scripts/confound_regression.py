import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple
from numpy.typing import NDArray


def load_h5_data(file_path: str) -> Tuple[NDArray, NDArray]:
    """
    Load data and mask from an H5 file.

    Args:
        file_path: Path to the H5 file.

    Returns:
        Tuple containing the data array and mask array.
    """
    with h5py.File(file_path, "r") as f:
        data = f["data"][:]
        mask = f["mask"][:]
    return data, mask


def confound_regression(data_path: str, confounds_path: str, out_path: str) -> None:
    """
    Perform confound regression on input data.

    A confound is a variable that is correlated with both the dependent variable and the independent variable.
    For instance, when predicting the disease status of a person based on their brain structure, both variables may be dependent on age.
    In such a case, a machine learning model may learn to predict the disease status based on age-related changes in brain structure, instead of the disease status itself.

    We can eliminate the effect of confounding variables by confound regression, i.e. regressing out the confounding variables from the data.
    This function reads a data file, runs linear confound correction, then saves the new corrected data file.

    The confound data can be multivariate, i.e. have multiple columns. Note that in such a case, weighing the confound data is not possible.
    If you want a certain weighing, you need to create a new confound data file with weighted univariate confound data (i.e. a linear combination of the columns).

    Args:
        data_path: Path to the pre-confound corrected data.
        confounds_path: Path to the confounds.
        out_path: Path to save the newly corrected data.
    """
    # Load data and confounds
    data_raw, data_mask = load_h5_data(data_path)
    confounds, confounds_mask = load_h5_data(confounds_path)

    # Ensure data and confounds are 2D
    if len(data_raw.shape) == 1:
        data_raw = data_raw.reshape(-1, 1)
    if len(confounds.shape) == 1:
        confounds = confounds.reshape(-1, 1)

    # Check if data and confounds have the same number of samples
    assert len(data_raw) == len(confounds), f"Mismatch in data samples: data n={len(data_raw)}, confounds n={len(confounds)}"

    # Create a combined mask for valid data points
    xy_mask = np.logical_and(data_mask, confounds_mask)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(confounds[xy_mask], data_raw[xy_mask])

    # Apply confound regression
    data_corrected = np.empty(data_raw.shape)
    for i in range(data_raw.shape[0]):
        if xy_mask[i]:
            data_corrected[i] = data_raw[i] - model.predict(confounds[i].reshape(1, -1))
        else:
            data_corrected[i][:] = np.nan

    # Save the corrected data
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data_corrected)
        f.create_dataset("mask", data=xy_mask)


if __name__ == "__main__":
    confound_regression(
        snakemake.input.features, snakemake.input.confounds, snakemake.output.features
    )
