import h5py
import numpy as np
from sklearn.linear_model import LinearRegression


def confound_regression(data_path: str, confounds_path: str, out_path: str):
    """

    A confound is a variable that is correlated with both the dependent variable and the independent variable.
    For instance, when predicting the disease status of a person based on their brain structure, both variables may be dependent on age.
    In such a case, a machine learning model may learn to predict the disease status based on age-related changes in brain structure, instead of the disease status itself.

    We can eliminate the effect of confounding variables by confound regression, i.e. regressing out the confounding variables from the data.
    This function reads a data file, runs linear confound correction, then save the new corrected data file.

    The confound data can be mulitvariate, i.e. have multiple columns. Note that in such a case, weighing the confound data is not possible.
    If you want to a certain weighing, you need to create a new confound data file with weighted univariate confound data (i.e. a linear combination of the columns).

    Args:
        data_path: path to the pre-confound corrected data
        confounds_path: path to the counfounds
        out_path: path to save the newly corrected data

    """
    with h5py.File(data_path, "r") as f:
        data_raw = f["data"][:]
        data_mask = f["mask"][:]
    with h5py.File(confounds_path, "r") as f:
        confounds = f["data"][:]
        confounds_mask = f["mask"][:]

    if len(data_raw.shape) == 1:
        data_raw = data_raw.reshape(-1, 1)
    if len(confounds.shape) == 1:
        confounds = confounds.reshape(-1, 1)
    assert len(data_raw) == len(confounds)

    xy_mask = np.logical_and(data_mask, confounds_mask)

    model = LinearRegression()
    model.fit(confounds[xy_mask], data_raw[xy_mask])
    # to deal with nan values in confounds we loop over the data
    data_corrected = np.empty(data_raw.shape)
    for i in range(data_raw.shape[0]):
        if xy_mask[i]:
            data_corrected[i] = data_raw[i] - model.predict(confounds[i].reshape(1, -1))
        else:
            data_corrected[i][:] = np.nan

    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data_corrected)
        f.create_dataset("mask", data=xy_mask)

if __name__ == "__main__":
    confound_regression(
        snakemake.input.features, snakemake.input.confounds, snakemake.output.features
    )
