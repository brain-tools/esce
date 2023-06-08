from workflow.scripts.plot import process_results, plot
from workflow.scripts.extrapolate import NpEncoder


import os
import json
from pathlib import Path
import yaml
import textwrap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tempfile import NamedTemporaryFile

def test_process_results():

    available_results = [
        "results/statistics/dataset1_model1_features1_featuresCNI1_target1_targetsCNI1_matching1_grid1.stats.json",
        "results/statistics/dataset2_model2_features2_featuresCNI2_target2_targetsCNI2_matching2_grid2.stats.json",
    ]

    expected_output = pd.DataFrame(
        {
            "full_path": [
                "results/statistics/dataset1_model1_features1_featuresCNI1_target1_targetsCNI1_matching1_grid1.stats.json",
                "results/statistics/dataset2_model2_features2_featuresCNI2_target2_targetsCNI2_matching2_grid2.stats.json",
            ],
            "dataset": ["dataset1", "dataset2"],
            "model": ["model1", "model2"],
            "features": ["features1", "features2"],
            "features_cni": ["featuresCNI1", "featuresCNI2"],
            "target": ["target1", "target2"],
            "targets_cni": ["targetsCNI1", "targetsCNI2"],
            "matching": ["matching1", "matching2"],
            "grid": ["grid1", "grid2"],
            "cni": ["featuresCNI1-targetsCNI1-matching1", "featuresCNI2-targetsCNI2-matching2"],
        }
    )

    assert process_results(available_results).equals(expected_output), 'result dataframe processed incorrectly'
    

def test_plot():

    stats_file_list = [
        "tests/data/example-features-a_example-covariates-of-no-interest_example-targets_none_none_default.stats.json",
        "tests/data/example-features-b_example-covariates-of-no-interest_example-targets_none_none_default.stats.json"
    ]
    
    bootstrap_file_list = [
        "tests/data/example-features-a_example-covariates-of-no-interest_example-targets_none_none_default.bootstrap.json",
        "tests/data/example-features-b_example-covariates-of-no-interest_example-targets_none_none_default.bootstrap.json"
    ]

    stats = [{
        "x": [
            10,
            20,
            30
        ],
        "y_mean": [
            -0.013438467691572267,
            -0.5122605335838996,
            -0.248453755907222
        ],
        "y_std": [
            0.02327610881767601,
            0.9270180816941317,
            0.1085962028735548
        ]
    },{
        "x": [
            10,
            20,
            30
        ],
        "y_mean": [
            0.001029732130729,
            -0.24388659377273078,
            -0.9252934714314738
        ],
        "y_std": [
            0.0017835483686087852,
            0.21380769781674483,
            1.2665951095048933
        ]
    }
    ]

    Path(stats_file_list[0]).touch()
    Path(stats_file_list[1]).touch()
    with open(stats_file_list[0], "w") as f:
        json.dump(stats[0], f, cls=NpEncoder, indent=0)
    with open(stats_file_list[1], "w") as f:
        json.dump(stats[1], f, cls=NpEncoder, indent=0)

    Path(bootstrap_file_list[0]).touch()
    Path(bootstrap_file_list[1]).touch()
    with open(bootstrap_file_list[0], "w") as f:
        json.dump({}, f, cls=NpEncoder, indent=0)
    with open(bootstrap_file_list[1], "w") as f:
        json.dump({}, f, cls=NpEncoder, indent=0)

    with open(bootstrap_file_list[0]) as f:
        p = yaml.safe_load(f)
        print(p, type(p), not p)

    output_filename = "tests/plots/plot_output.png"
    color_variable = "features"
    linestyle_variable = "model"
    title = "example features vs. example targets"
    max_x = 100000

    plot(
        stats_file_list,
        output_filename,
        color_variable,
        linestyle_variable,
        title,
        max_x,
    )

    assert os.path.exists(output_filename), 'plot function from plot.py failed'


