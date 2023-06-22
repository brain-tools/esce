import os
from pathlib import Path

import pandas as pd


def aggregate(
    score_path_list: str,
    stats_path: str,
):
    """
    for each score file in score_path_list,
    identify best performing hyperparameter combination on validation set
    and collect corresponding metrics.

    save resulting table to new csv file stats_path
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

    metric = "r2_val" if "r2_val" in df.columns else "acc_val"
    idx_best = df.groupby(["n", "s"])[metric].idxmax()
    df.loc[idx_best].to_csv(stats_path, index=False)

if __name__ == "__main__":
    aggregate(snakemake.input.scores, snakemake.output.scores)
