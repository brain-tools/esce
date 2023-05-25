"""
prepare_data.py
====================================

"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


predefined_datasets = {
    ("mnist", "pixel"): lambda: fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )[0],
    ("mnist", "ten-digits"): lambda: fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )[1].astype(int),
    ("mnist", "odd-even"): lambda: (
        fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)[1].astype(
            int
        )
        % 2
    ).astype(int),
}


def prepare_data(
    out_path: str,
    dataset: str,
    features_targets_covariates: str,
    variant: str,
    custom_datasets: dict,
):
    """
    Prepare features, targets, and covariates in the data.

    Args:
        out_path: storage path for output data
        dataset: dataset name for extracting from predifined datasets
        features_targets_covariates: the string decides which the function prepares
        variant: a special data type that can contain any kind of data
        custom_datasets: path for custom datasets 
    """
    # read in predefined datasets or prepare empty files
    if (dataset, variant) in predefined_datasets:
        data = predefined_datasets[(dataset, variant)]()
    elif features_targets_covariates == "covariates" and variant in [ # if predefined datasets don't exist, use dummy empty files
        "none",
        "balanced",
    ]:
        data = []
    # turn datas into numpy arrays
    else:
        in_path = Path(custom_datasets[dataset][features_targets_covariates][variant])
        if in_path.suffix == ".csv":
            data = pd.read_csv(in_path).values
        if in_path.suffix == ".tsv":
            data = pd.read_csv(in_path, delimiter="\t").values
        if in_path.suffix == ".npy":
            data = np.load(in_path)

    if features_targets_covariates == "targets":
        data = data.reshape(-1)

    np.save(out_path, data)


prepare_data(
    snakemake.output.npy,
    snakemake.wildcards.dataset,
    snakemake.wildcards.features_or_targets
    if hasattr(snakemake.wildcards, "features_or_targets")
    else "covariates",
    snakemake.wildcards.name,
    snakemake.params.custom_datasets,
)
