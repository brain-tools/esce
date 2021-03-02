import seaborn as sns
from typing import Dict
from matplotlib import pylab
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from esce.models import MODEL_NAMES, MODELS
from esce.grid import GRID
import ast
from itertools import chain
from pathlib import Path

pylab.rc("font", family="serif", serif="Times")
pylab.rc("xtick", labelsize=8)
pylab.rc("ytick", labelsize=8)
pylab.rc("axes", labelsize=8)
sns.set_theme()


def hp_plot(
    root: Path,
    title: str,
    df: pd.DataFrame,
    grid: Dict[str, Dict[str, np.ndarray]],
    show: bool = False,
    target: str = "acc_test",
) -> None:
    model_names = df["model"].unique()
    plots_per_model = {}
    for model_name in model_names:
        params = grid[model_name].keys()
        plots_per_model[model_name] = len(params)

    # Plots hyperparameters per model
    for model_name in model_names:
        df_ = df[df["model"] == model_name]
        if plots_per_model[model_name] == 0:
            continue

        df_new = df_.copy()
        df_new["params"] = df_["params"].apply(lambda x: ast.literal_eval(x))

        names = df_.n.unique()
        palette = sns.color_palette("mako_r", n_colors=len(np.unique(names)))
        legend = [mpatches.Patch(color=i, label=j) for i, j in zip(palette, names)]

        fig, ax = pylab.subplots(
            1, plots_per_model[model_name], dpi=200, sharey="row", squeeze=False
        )

        for i, param in enumerate(grid[model_name].keys()):
            ax_ = ax[0, i]
            df_new[param] = df_new["params"].apply(lambda x: x[param])
            idx = df_new.groupby(["model", "n", "s", param])[target].idxmax()
            df_tmp = df_new.loc[idx]

            ax1 = sns.lineplot(
                x=param,
                y=target,
                data=df_tmp,
                hue="n",
                legend=False,
                ax=ax_,
                palette=palette,
            )
            ax1.set_xscale("log", base=2)
            ax1.set_ylabel("Accuracy")

        fig.subplots_adjust(bottom=0.2)
        pylab.plt.figtext(0.5, 0.9, title, fontsize=8, ha="center")

        pylab.plt.suptitle(MODEL_NAMES[model_name])
        pylab.figlegend(
            handles=legend,
            ncol=len(names),
            fontsize=8,
            loc="lower center",
            frameon=False,
        )
        fig.savefig(root / f"hp_new_{model_name}_{title}.png")
        if show:
            pylab.plt.show()

        idx = df_.groupby(["model", "n", "s"])[target].idxmax()
        df_ = df_.loc[idx]
        fig, ax = pylab.subplots(
            1, plots_per_model[model_name], dpi=200, sharey="row", squeeze=False
        )
        for i, param_name in enumerate(grid[model_name].keys()):
            ax_ = ax[0, i]

            # Create small table, infer type back
            tmp = df_[["n", target]].copy()
            tmp[param_name] = df_["params"].apply(
                lambda x: ast.literal_eval(x)[param_name]
            )
            tmp.infer_objects()

            # Convert data to log if float
            if tmp[param_name].dtype in [np.float64, np.float32]:
                log_param = np.log2(tmp[param_name])
                param_name = "log " + param_name
                tmp[param_name] = log_param

            ax1 = sns.scatterplot(
                x=param_name,
                y=target,
                hue="n",
                data=tmp,
                ax=ax_,
                ci="sd",
                palette=palette,
                legend=False,
            )
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel(param_name)

        fig.subplots_adjust(bottom=0.2)
        pylab.plt.figtext(0.5, 0.9, title, fontsize=8, ha="center")

        pylab.plt.suptitle(MODEL_NAMES[model_name])
        pylab.figlegend(
            handles=legend,
            ncol=len(names),
            fontsize=8,
            loc="lower center",
            frameon=False,
        )
        fig.savefig(root / f"hp_{model_name}_{title}.png")
        if show:
            pylab.plt.show()

    pylab.plt.close("all")


def sc_plot(root: Path, title: str, df: pd.DataFrame, show: bool = False) -> None:
    fig, ax = pylab.subplots(1, 1, dpi=200)
    models = list(df["model"].unique())
    ticks = df["n"].unique()

    names = [MODEL_NAMES[i] for i in models]
    palette = [MODEL_COLORS[i] for i in models]
    palette = sns.xkcd_palette(palette)
    legend = [mpatches.Patch(color=i, label=j) for i, j in zip(palette, names)]

    ax = sns.lineplot(
        x="n",
        y="acc_test",
        hue="model",
        data=df,
        ci="sd",
        err_style="bars",
        palette=palette,
        ax=ax,
        legend=False,
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Sample Size")
    # ax.set_xticks(ticks=list(np.arange(len(ticks))), labels=list(df.n.unique()))
    # ax.set_xlim(0, np.max(ticks))

    ax.set_xscale("log")
    ax.set_xticks(ticks)

    # ax.get_xaxis().set_ticks(list(np.arange(len(ticks))))
    # ax.get_xaxis().set_ticklabels(ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_tick_params(which="minor", size=0)
    ax.get_xaxis().set_tick_params(which="minor", width=0)

    fig.subplots_adjust(bottom=0.2)

    pylab.plt.suptitle("Sample scores")
    pylab.plt.title(title, y=1.0, fontsize=8)
    pylab.figlegend(
        handles=legend, ncol=2, fontsize=8, loc="lower center", frameon=False
    )
    fig.savefig(root / f"sc_plot_{title}.png")
    if show:
        pylab.plt.show()


MODEL_COLORS = {
    "lda": "purple",
    "logit": "blue",
    "forest": "green",
    "ols": "pink",
    "lasso": "brown",
    "ridge": "red",
    "svm-linear": "light blue",
    "svm-rbf": "teal",
    "svm-sigmoid": "orange",
    "svm-polynomial": "light green",
}
