"""
aggregate.py
====================================

Aggregates the scores for each hyperparameter combination, and selects the best
hyperparameter combination, grouped by sample size and seed.

"""


import os
from pathlib import Path

import pandas as pd


def aggregate(
    score_path_list: str,
    stats_path: str,
):
    """

    For each score file in score_path_list,
    identify best performing hyperparameter combination on validation set
    and collect corresponding metrics.

    save resulting table to new csv file stats_path

    Args:
        score_path_list: the path for the input score files
        stats_path: the path for the output stats csv file. which stores the best
            performing combinationbased on the the average coefficient of determination
            or accuracy on validation set

    """
    df_list = []
    for filename in score_path_list:
        # ignore empty files (insufficient samples in dataset)
        if os.stat(filename).st_size > 0:
            df_list.append(pd.read_csv(filename, index_col=False))

    # create empty token file for snakemake if empty (insufficient samples in dataset)
    if not df_list:
        Path(stats_path).touch()
        return

    df = pd.concat(
        df_list,
        axis=0,
        ignore_index=True,
    )

    # R^2 is the average coefficient of determination
    # R^2 or accuracy due to classification/regression models
    metric = "r2_val" if "r2_val" in df.columns else "acc_val"
    # n: trainin sample size ; s: random seed -- groupby
    #  --> (hyperparameters + seed) combination
    idx_best = df.groupby(["n", "s"])[metric].idxmax()
    # best (hyperparameters + seed) combination
    df.loc[idx_best].to_csv(stats_path, index=False)


if __name__ == "__main__":
    aggregate(snakemake.input.scores, snakemake.output.scores)
