"""
plot_hps.py
====================================

"""
import yaml, glob, os, textwrap
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go



def plot(
    stats_filename, output_filename, grid_filename, hyperparameter_scales, model_name, title
):
    """
    Extract hyperparameter and their performance for the model, 
    then scale and plot the figure.

    Args:
        stats_filename: stats file name for stats retrieval
        output_filename: output plot image name that is stored by the function
        grid_filename: grid file name for grid retreival 
        hyperparameter_scales: scale of hyperparameters
        model_name: name provided to extract hyperparameters from each model 
        title: title of the figure 

    """

    scores = pd.read_csv(stats_filename)
    grid = yaml.safe_load(open(grid_filename, "r"))[model_name]
    hp_names = list(grid.keys())
    metric = "r2_val" if "r2_val" in scores.columns else "acc_val"

    df = scores.melt(
        id_vars=["n", metric], value_vars=hp_names, var_name="hyperparameter"
    )
    fig = px.scatter(
        df, x="n", y="value", color=metric, facet_col="hyperparameter", log_x=True
    )
    fig.update_yaxes(matches=None, showticklabels=True)

    fig.update_layout(
        plot_bgcolor="white",
        title_text=textwrap.fill(title, 90).replace("\n", "<br>"),
        title_x=0.5,
        font={"size": 10},
        margin=dict(l=20, r=20, t=80, b=20),
    )

    for i, v in enumerate(hp_names):
        if v in hyperparameter_scales:
            fig.add_hline(y=min(grid[v]), line_dash="dot", col=i + 1)
            fig.add_hline(y=max(grid[v]), line_dash="dot", col=i + 1)
            if hyperparameter_scales[v] == "log":
                fig.update_yaxes(type="log", col=i + 1)

    fig.write_image(output_filename)


plot(
    stats_filename=snakemake.input.scores,
    output_filename=snakemake.output.plot,
    grid_filename=snakemake.params.grid,
    hyperparameter_scales=snakemake.params.hyperparameter_scales,
    model_name=snakemake.wildcards.model,
    title=snakemake.params.title,
)
