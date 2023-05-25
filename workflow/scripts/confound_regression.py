"""
confound_regression.py
====================================

"""
import numpy as np
from sklearn.linear_model import LinearRegression



def confound_regression(data_path: str, confounds_path: str, out_path: str):
    """
    A confounder is a variable whose presence affects the variables being studied 
    so that the results do not reflect the actual relationship. 
    
    Since the saturating prediction performance for certain "confounder" 
    may cause the model to mostly rely on confounding variables to derive its prediction,
    there are various ways to exclude or control confounding variables.
    
    When experimental designs are premature, impractical, or impossible, 
    researchers must rely on statistical methods (e.g. regression models) to eliminate potentially confounding effects. 
    
    This function reads data, run linear confound correction, then save the new corrected dataset.
    
    Args:
        data_path: path to the pre-confound corrected data
        confounds_path: path to the counfounds
        out_path: path to save the newly corrected data

    """
    data_raw = np.load(data_path)
    confounds = np.load(confounds_path)

    if len(data_raw.shape) == 1:
        data_raw = data_raw.reshape(-1, 1)
    if len(confounds.shape) == 1:
        confounds = confounds.reshape(-1, 1)
    assert len(data_raw) == len(confounds)

    #? what is the purpose of this step?
    x_mask = np.all(np.isfinite(data_raw), 1)
    y_mask = np.all(np.isfinite(confounds), 1)
    xy_mask = np.logical_and(x_mask, y_mask)

    model = LinearRegression()
    model.fit(confounds[xy_mask], data_raw[xy_mask])
    data_predicted = model.predict(confounds)
    data_corrected = data_raw - data_predicted

    np.save(out_path, data_corrected)


confound_regression(
    snakemake.input.features, snakemake.input.confounds, snakemake.output.features
)
