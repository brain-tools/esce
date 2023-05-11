import yaml, glob, os, textwrap
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import textwrap
import numpy as np
from pathlib import Path

def process_results(available_results):
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


def plot(stats_file_list, output_filename, color_variable, linestyle_variable, title, max_x=6):
    df = process_results(stats_file_list)

    data = []
    for _, row in df.iterrows():
        if not os.path.exists(row.full_path):
            print(row.full_path, "does not exist - skipping")
            continue
        with open(row.full_path) as f:
            score = yaml.safe_load(f)
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

    print(data)
    # skip if no data
    if len(data) > 0:
        data = pd.concat(data, axis=0, ignore_index=True)
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
        legend=dict(orientation="h", yanchor="top", xanchor="center", y=-0.125, x=0.5),
        title_text=textwrap.fill(title, 90).replace("\n", "<br>"),
        title_x=0.5,
        font={"size": 10},
        margin=dict(l=20, r=20, t=40, b=20),
    )

    for i, (_, row) in enumerate(df.iterrows()):
        with open(row.full_path.replace("stats.json", "bootstrap.json")) as f:
            p = yaml.safe_load(f)
        if not p:
            continue
        for p_ in p:
            x_exp = np.logspace(np.log10(128), np.log10(max_x))
            y_exp = p_[0] * np.power(x_exp, -p_[1]) + p_[2]
            fig.add_trace(
                go.Scatter(
                    x=x_exp,
                    y=y_exp,
                    line=dict(color=fig.data[i].line.color, dash=fig.data[i].line.dash),
                    opacity=2 / len(p),
                    showlegend=False,
                )
            )
    fig.update_yaxes(rangemode="nonnegative")

    fig.write_image(output_filename)


plot(
    stats_file_list=snakemake.params.stats,
    output_filename=snakemake.output.plot,
    color_variable=snakemake.params.color_variable,
    linestyle_variable=snakemake.params.linestyle_variable,
    title=snakemake.params.title,
    max_x=snakemake.params.max_x,)
