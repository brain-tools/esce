import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import yaml
import json
import re

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
# this is a hack to make yaml load the floats correctly


def process_results(available_results):
    """Read available results and return them as a pandas dataframe."""
    df = pd.DataFrame(available_results, columns=["full_path"])
    df[
        [
            "dataset",
            "model",
            "features",
            "features_cni",
            "target",
            "targets_cni",
            "matching",
            "grid",
        ]
    ] = (
        df["full_path"]
        .str.replace("results/", "")
        .str.replace("statistics/", "")
        .str.replace(".stats.json", "")
        .str.replace("/", "_")
        .str.split("_", expand=True)
    )
    df["cni"] = df[["features_cni", "targets_cni", "matching"]].agg("-".join, axis=1)
    return df


def plot(
    stats_file_list, output_filename, color_variable, linestyle_variable, title, max_x=6
):
    """Plot the results of a model.

    Args:
        stats_file_list: list of paths to the results files
        output_filename: path to save the plot
        color_variable: variable to use for color
        linestyle_variable: variable to use for linestyle
        title: title of the plot
        max_x: maximum sample size in powers of 10
    """
    df = process_results(stats_file_list)

    data = []
    for _, row in df.iterrows():
        with open(row.full_path) as f:
            score = yaml.load(f, Loader=loader)
        if not score:
            print(row.full_path, "is empty - skipping")
            continue
            
        df_ = pd.DataFrame(
            {"n": score["x"], "y": score["y_mean"], "y_std": score["y_std"]}
        )
        df_["model"] = row.model
        df_["features"] = row.features
        df_["target"] = row.target
        df_["cni"] = row.cni
        data.append(df_)

    # skip if no data
    if len(data) > 0:
        data = pd.concat(data, axis=0, ignore_index=True)
        data = data.sort_values(color_variable)
    else:
        Path(output_filename).touch()
        return

    fig = px.line(
        data,
        x="n",
        y="y",
        error_y="y_std",
        color=color_variable,
        line_dash=linestyle_variable,
        log_x=True,
        template="simple_white",
    )
    fig.update_layout(
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "xanchor": "center",
            "y": -0.125,
            "x": 0.5,
        },
        title_text=textwrap.fill(title, 90).replace("\n", "<br>"),
        title_x=0.5,
        font={"size": 10},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )

    cnt = 0
    for i, (_, row) in enumerate(df.iterrows()):
        with open(row.full_path.replace("stats.json", "bootstrap.json")) as f:
            p = yaml.load(f, Loader=loader)

        if not p:
            continue

        for p_ in p:
            x_exp = np.logspace(np.log10(128), np.log10(max_x))
            y_exp = p_[0] * np.power(x_exp, -p_[1]) + p_[2]
            fig.add_trace(
                go.Scatter(
                    x=x_exp,
                    y=y_exp,
                    line={
                        "color": fig.data[cnt].line.color,
                        "dash": fig.data[cnt].line.dash,
                    },
                    opacity=2 / len(p),
                    showlegend=False,
                )
            )
        cnt +=1
    fig.update_yaxes(rangemode="nonnegative")

    fig.write_image(output_filename)


if __name__ == "__main__":
    plot(
        stats_file_list=snakemake.params.stats,
        output_filename=snakemake.output.plot,
        color_variable=snakemake.params.color_variable,
        linestyle_variable=snakemake.params.linestyle_variable,
        title=snakemake.params.title,
        max_x=snakemake.params.max_x,)
