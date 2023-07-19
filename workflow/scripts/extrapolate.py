import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import r2_score

MIN_DOF = 2


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def fit_curve(x, y, y_e):
    result = {
        "p_mean": np.nan,
        "r2": np.nan,
        "chi2": np.nan,
        "mu": np.nan,
        "sigma": np.nan,
    }

    dof = len(x) - 3
    if dof < MIN_DOF:
        print(f"Warning: dof = {dof} < {MIN_DOF}")
        return result

    try:
        p_mean, _ = scipy.optimize.curve_fit(
            lambda t, a, b, c: a * t ** (-b) + c,
            x,
            y,
            # sigma=y_e,
            maxfev=5000,
            p0=(-1, 0.1, 0.5),
            bounds=((-np.inf, 0, -np.inf), (0, np.inf, np.inf)),
        )
        result["p_mean"] = p_mean
        result["r2"] = r2_score(y, p_mean[0] * x ** (-p_mean[1]) + p_mean[2])
        result["chi2"] = (
            sum((y - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2])) ** 2 / y_e**2) / dof
        )
        result["mu"], result["sigma"] = np.mean(
            (y - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2])) / y_e
        ), np.std((y - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2])) / y_e)
        return result
    except Exception as e:
        print(e)
        return result


def extrapolate(
    stats_path: str,
    extra_path: str,
    bootstrap_path: str,
    repeats: int,
):
    if os.stat(stats_path).st_size == 0:
        Path(extra_path).touch()
        Path(bootstrap_path).touch()
        return

    df = pd.read_csv(stats_path, index_col=False)

    metric = "r2_test" if "r2_test" in df.columns else "acc_test"
    result = {"n_seeds": len(df["s"].unique())}

    x, y_mean, y_std, y_sem, mask = [], [], [], [], []
    for n in sorted(df["n"].unique()):
        x.append(n)
        y_mean.append(df[df["n"] == n][metric].mean())
        y_std.append(df[df["n"] == n][metric].std())
        y_sem.append(df[df["n"] == n][metric].sem())
        mask.append(bool((y_mean[-1] - y_sem[-1]) > 0))

    x = np.asarray(x)
    y_mean = np.asarray(y_mean)
    y_std = np.asarray(y_std)
    y_sem = np.asarray(y_sem)
    mask = np.asarray(mask)

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

    p_bootstrap = []
    y_bootstrap = [[] for _ in x]
    for _ in range(repeats):
        y_bs_sample_mean, y_bs_sample_std, y_bs_sample_sem = [], [], []
        for i, n in enumerate(x):
            y_bs_sample = df[df["n"] == n][metric].sample(frac=1, replace=True)
            y_bs_sample_mean.append(y_bs_sample.mean())
            y_bs_sample_sem.append(y_bs_sample.sem())
            y_bootstrap[i].append(y_bs_sample.mean())
        p_ = fit_curve(
            x[mask],
            np.asarray(y_bs_sample_mean)[mask],
            np.asarray(y_bs_sample_sem)[mask],
        )["p_mean"]
        if np.isfinite(p_).all():
            p_bootstrap.append(p_)

    print(f"p_bootstrap: {len(p_bootstrap)} out of {repeats}")
    if len(p_bootstrap) > 0.9 * repeats:
        result.update(
            {
                "p_bootstrap_mean": np.mean(p_bootstrap, 0),
                "p_bootstrap_std": np.std(p_bootstrap, 0),
                "p_bootstrap_975": np.percentile(p_bootstrap, 97.5, axis=0),
                "p_bootstrap_025": np.percentile(p_bootstrap, 2.5, axis=0),
            }
        )
    else:
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

if __name__ == "__main__":
    extrapolate(
        snakemake.input.scores,
        snakemake.output.stats,
        snakemake.output.bootstraps,
        snakemake.params.bootstrap_repetitions,
    )
