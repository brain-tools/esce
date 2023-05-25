"""
extrapolate.py
====================================

"""
    
import json

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import r2_score
from pathlib import Path
import os


MIN_DOF = 2


class NpEncoder(json.JSONEncoder):
    """
    Transform numpy arrays into datatypes that JSON module can handle

    """
    def default(self, obj):
        """
        Turn datatypes into something that JSON modules can handle
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def fit_curve(x, y, y_e):
    """
    Fitting the learning curve. Input a set of sample sizes (x) the metrics (y), and standard error of the metrics over the seed.

    Args:
        x: a set of sample sizes
        y: the metrics
        y_e: standard error

    Returns:
        result: a dictionary of stats 
    """
    result = {
        "p_mean": np.nan, # parameter of the learning curve fit
        "r2": np.nan, # R^2, the goodness-of-fit statistics  between the empirical data & the curve fitted
        "chi2": np.nan, # chisq, the X^2 that's part of the goodness-of-fit statistics 
        "mu": np.nan, # the mean residuals between the empirical data and the curve we fitted
        "sigma": np.nan, # the standard deviation residuals between the empirical data and the curve we fitted
    }

    # degrees of freedom: a measure of how much data we have to fill up the curve, the higher the dof is, the more data we have the better it fits
    # the lower the dof is, the worse the curve is
    dof = len(x) - 3
    if dof < MIN_DOF:
        return result

    try:
        p_mean, _ = scipy.optimize.curve_fit(
            lambda t, a, b, c: a * t ** (-b) + c,
            x,
            y,
            sigma=y_e,
            maxfev=5000, # The maximum number of calls to the function
            p0=(-1, 0.01, 0.7), # Initial guess for the parameters (length N)
            bounds=((-np.inf, 0, 0), (0, 1, 1)), # Lower and upper bounds on parameters
        )
        result["p_mean"] = p_mean
        result["r2"] = r2_score(
            y_mean, p_mean[0] * x ** (-p_mean[1]) + p_mean[2]
        )
        # sem: standard error of the mean
        result["chi2"] = (
            sum(
                (y_mean - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2])) ** 2
                / y_sem ** 2
            )
            / result["dof"]
        )
        result["mu"], result["sigma"] = np.mean(
            (y_mean - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2]))
            / y_sem
        ), np.std(
            (y_mean - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2]))
            / y_sem
        )
        return result
    except:
        return result


def extrapolate(stats_path: str, extra_path: str, bootstrap_path: str, repeats: int):
    """
    
    The gain in prediction accuracy that is enabled by an increase in sample size can be modeled and extrapolated. 
    This allowed us to, in essence, forecast the prediction performance that we would likely reach at sample sizes orders of magnitude larger than the datasets of today. 

    Args:
        stats_path: path to the stats
        extra_path: if stats size is 0, or doesn't exist, create an extra file 
        bootstrap_path: same as extra_path
    """
    
    if os.stat(stats_path).st_size == 0: # if the result is empty, could be caued by insufficient sample size
        Path(extra_path).touch() # for the snakemake workflow to know the process is finished
        Path(bootstrap_path).touch()
        return

    df = pd.read_csv(stats_path, index_col=False)

    ###
    # parametrics caculation based on the assumption that they are under Gaussian distribution
    ###
    metric = "r2_test" if "r2_test" in df.columns else "acc_test" # decide which metrics based on classification or regression
    result = {"n_seeds": len(df["s"].unique())}

    # extract relevant information for each sample size
    x, y_mean, y_std, y_sem, mask = [], [], [], [], []
    for n in sorted(df["n"].unique()):
        x.append(n) # collect unique sample sizes under x
        y_mean.append(df[df["n"] == n][metric].mean())
        y_std.append(df[df["n"] == n][metric].std())
        y_sem.append(df[df["n"] == n][metric].sem())
        # check the accuracy is at least 1 above standard error to exclude datapoints from extrapolating when the models performs poorly
        # discard sample sizes that are perform too poorly 
        mask.append(bool((y_mean[-1] - y_sem[-1]) > 0))
    x = np.asarray(x)
    y_mean = np.asarray(y_mean)
    y_std = np.asarray(y_std)
    y_sem = np.asarray(y_sem)

    result.update(
        {
            "metric": metric,
            "x": x,
            "y_mean": y_mean,
            "y_std": y_std,
            "y_sem": y_sem,
            "mask": mask,
            "dof": sum(mask) - 3,
        }
    )
    result.update(fit_curve(x[mask], y_mean[mask], y_sem[mask]))

    ###
    # Bootstrapping: calculating under the assumption of non-Gaussian distribution 
    #                estimates confidence level with non-parametrics under the assumption of non-Gaussian
    ###
    # parameter & empirical scores
    p_bootstrap = []
    y_bootstrap = [[] for _ in x] # x: fully available sample sizes
    # 
    for _ in range(repeats):
        y_bs_sample_mean, y_bs_sample_std, y_bs_sample_sem = [], [], []
        for i, n in enumerate(x): # for each sample size
            y_bs_sample = df[df["n"] == n][metric].sample(frac=1, replace=True) # subsample with replacement
            y_bs_sample_mean.append(y_bs_sample.mean())
            y_bs_sample_sem.append(y_bs_sample.sem())
            y_bootstrap[i].append(y_bs_sample.mean())
        p_ = fit_curve(
            x[mask],
            np.asarray(y_bs_sample_mean)[mask],
            np.asarray(y_bs_sample_sem)[mask],
        )["p_mean"]
        if np.isfinite(p_).all():
            p_bootstrap.append(p_) # deriving repeats number of curves 

    if len(p_bootstrap) > 0.9 * repeats: # check > 90% valid fits, else skip
        result.update(
            {
                "p_bootstrap_mean": np.mean(p_bootstrap, 0),
                "p_bootstrap_std": np.std(p_bootstrap, 0),
                "p_bootstrap_975": np.percentile(p_bootstrap, 97.5, axis=0), 
                "p_bootstrap_025": np.percentile(p_bootstrap, 2.5, axis=0), # 95% confidence interval with the 2.5 cut off points
            }
        )
    else: # if not enough data, fill in with nan
        result.update(
            {
                "p_bootstrap_mean": np.nan,
                "p_bootstrap_std": np.nan,
                "p_bootstrap_975": np.nan,
                "p_bootstrap_025": np.nan,
            }
        )
        p_bootstrap = None

    result.update(
        {
            "y_bootstrap_mean": np.mean(y_bootstrap, 1),
            "y_bootstrap_std": np.std(y_bootstrap, 1),
            "y_bootstrap_975": np.percentile(y_bootstrap, 97.5, axis=1),
            "y_bootstrap_025": np.percentile(y_bootstrap, 2.5, axis=1),
        }
    )

    with open(extra_path, "w") as f:
        json.dump(result, f, cls=NpEncoder, indent=0)

    with open(bootstrap_path, "w") as f:
        json.dump(p_bootstrap, f, cls=NpEncoder, indent=0)


extrapolate(
    snakemake.input.scores,
    snakemake.output.stats,
    snakemake.output.bootstraps,
    snakemake.params.bootstrap_repetitions,
)
