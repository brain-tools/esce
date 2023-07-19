import pandas as pd

from workflow.scripts.aggregate import aggregate


def test_aggregate(tmpdir):
    # Create temporary score files
    scores1 = {"n": [100, 200], "s": [42, 43], "acc_val": [0.8, 0.9]}
    scores2 = {"n": [100, 200], "s": [42, 43], "acc_val": [0.7, 0.85]}

    df1 = pd.DataFrame(scores1)
    df2 = pd.DataFrame(scores2)

    scores_path1 = str(tmpdir.join("scores1.csv"))
    scores_path2 = str(tmpdir.join("scores2.csv"))

    df1.to_csv(scores_path1, index=False)
    df2.to_csv(scores_path2, index=False)

    score_path_list = [scores_path1, scores_path2]

    # Path for the output stats file
    stats_path = str(tmpdir.join("stats.csv"))

    # Run the aggregate function
    aggregate(score_path_list, stats_path)

    # Load the stats and check if they are correct
    stats = pd.read_csv(stats_path)

    assert "n" in stats.columns
    assert "s" in stats.columns
    assert "acc_val" in stats.columns

    assert len(stats) == 2  # There should be two rows in the stats file
    assert (
        stats.loc[stats["n"] == 100]["acc_val"].values[0] == 0.8
    )  # Check the max 'acc_val' for n=100
    assert (
        stats.loc[stats["n"] == 200]["acc_val"].values[0] == 0.9
    )  # Check the max 'acc_val' for n=200
