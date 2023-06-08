from workflow.scripts.plot_hps import plot

import os
import yaml
import pandas as pd
import plotly.graph_objects as go

def test_plot():
    ###
    # Test case 1: there is r2_value
    ###
    stats_filename = "tests/plots/plot_hps_stats.csv"
    output_filename = "tests/plots/plot_hps_output(r2).png"
    grid_filename = "tests/plots/plot_hps_grid.yaml"
    hyperparameter_scales = {
        "alpha": "log"
    }

    scores = {'r2_train': {'0': 0.0002916496, '1': 1.0, '2': 0.9999999999, '3': 0.1744241596, '4': 0.5312638174, '5': 0.3054214328, '6': 0.1020143936, '7': 0.0840176553, '8': 0.1503364807}, 'r2_val': {'0': -0.0400949115, '1': 0.0, '2': 0.0, '3': -0.0402764086, '4': 0.5101394081, '5': 0.1208774213, '6': -0.0368744683, '7': 0.0188001035, '8': -0.2482221821}, 'r2_test': {'0': -0.0403154031, '1': 0.0, '2': 0.0, '3': -0.0611194339, '4': -1.578495288, '5': 0.1028331212, '6': -0.2548612821, '7': -0.1367956568, '8': -0.3537043287}, 'mae_train': {'0': 0.4799299959, '1': 1.389e-06, '2': 3.8034e-06, '3': 0.4101482715, '4': 0.2801894748, '5': 0.3843861514, '6': 0.4376957458, '7': 0.4732234159, '8': 0.4230906211}, 'mae_val': {'0': 0.5000184842, '1': 0.5836114127, '2': 1.1377520108, '3': 0.3940803717, '4': 0.2895893372, '5': 0.4370887027, '6': 0.4843529896, '7': 0.4750198631, '8': 0.5143306905}, 'mae_test': {'0': 0.5000803109, '1': 1.3755731388, '2': 0.508924909, '3': 0.4746311127, '4': 0.6244257608, '5': 0.4615771291, '6': 0.5173919943, '7': 0.5136989859, '8': 0.5357145016}, 'mse_train': {'0': 0.2399300041, '1': 0.0, '2': 0.0, '3': 0.1878185037, '4': 0.1124966838, '5': 0.1719081954, '6': 0.208532213, '7': 0.227977828, '8': 0.1973107506}, 'mse_val': {'0': 0.2600237279, '1': 0.3797041599, '2': 1.4623375421, '3': 0.1664442254, '4': 0.1175665421, '5': 0.2109894189, '6': 0.2592186171, '7': 0.2299687257, '8': 0.2925520739}, 'mse_test': {'0': 0.2600788508, '1': 1.9659927355, '2': 0.3415243679, '3': 0.2546686641, '4': 0.6188388691, '5': 0.2153200509, '6': 0.294108113, '7': 0.2664364821, '8': 0.317274452}, 'alpha': {'0': 100000.0, '1': 1e-05, '2': 1e-05, '3': 100.0, '4': 1.0, '5': 10.0, '6': 100.0, '7': 100.0, '8': 100.0}, 'n': {'0': 10, '1': 10, '2': 10, '3': 20, '4': 20, '5': 20, '6': 30, '7': 30, '8': 30}, 's': {'0': 0, '1': 1, '2': 2, '3': 0, '4': 1, '5': 2, '6': 0, '7': 1, '8': 2}}
    scores = pd.DataFrame.from_dict(scores)

    scores.to_csv(stats_filename, index=False)

    grid = {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
    with open(grid_filename, "w") as f:
        yaml.dump({"model_name": grid}, f)

    plot(
        str(stats_filename),
        str(output_filename),
        str(grid_filename),
        hyperparameter_scales,
        model_name = "model_name",
        title = "Performance by Hyperparameters"
    )

    # Verify the existence of the output file
    assert os.path.exists(output_filename), 'plot hyperparams(r2)\'s plot file doesn\'t exist'

    for file in [stats_filename]:
        os.remove(file)

    ###
    # Test case 2: there is no r2_value
    ###
    output_filename = "tests/plots/plot_hps_output(acc).png"
    scores = scores.rename(columns= {'r2_val': 'acc_val'})

    scores.to_csv(stats_filename, index=False)

    plot(
        str(stats_filename),
        str(output_filename),
        str(grid_filename),
        hyperparameter_scales,
        model_name = "model_name",
        title = "Performance by Hyperparameters"
    )

    assert os.path.exists(output_filename), 'plot hyperparams(acc)\'s plot file doesn\'t exist'

    for file in [stats_filename, grid_filename, output_filename]
        os.remove(file)
