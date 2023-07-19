import os

import yaml

from scripts.fit_model import MODELS, RegressionModel
from scripts.prepare_data import predefined_datasets as PREDEFINED_DATASETS


def validate_details(config):
    errors = []

    # check that each experiment has the expected values
    for exp_name, exp in config["experiments"].items():

        # check that dataset exists
        if (
            exp["dataset"] not in PREDEFINED_DATASETS
            and exp["dataset"] not in config["custom_datasets"]
        ):
            errors.append(
                f"experiment {exp_name}['dataset'] should be a string from: {PREDEFINED_DATASETS.keys()} or {config['custom_datasets'].keys()}"
            )
        else:
            # check that all features, targets, covariates exist
            dataset = (
                PREDEFINED_DATASETS[exp["dataset"]]
                if exp["dataset"] in PREDEFINED_DATASETS
                else config["custom_datasets"][exp["dataset"]]
            )
            for feature in exp["features"]:
                if feature not in dataset["features"]:
                    errors.append(
                        f"experiment {exp_name} feature {feature} must exist in either predefined  or custom datasets"
                    )
            for feature_cni in exp["features_cni"]:
                if feature_cni != "none" and feature_cni not in dataset["covariates"]:
                    errors.append(
                        f"experiment {exp_name} feature_cni {feature_cni} must exist in either predefined  or custom datasets"
                    )
            for target in exp["targets"]:
                if target not in dataset["targets"]:
                    errors.append(
                        f"experiment {exp_name} target {target} must exist in either predefined  or custom datasets"
                    )
            for target_cni in exp["targets_cni"]:
                if target_cni != "none" and target_cni not in dataset["covariates"]:
                    errors.append(
                        f"experiment {exp_name} targets_cni {target_cni} must exist in either predefined  or custom datasets"
                    )
            for matching in exp["matching"]:
                if (
                    matching not in ["none", "balanced"]
                    and matching not in dataset["covariates"]
                ):
                    errors.append(
                        f"experiment {exp_name} matching {matching} must exist in either predefined or custom datasets"
                    )

        # check that model exists in MODELS
        for model in exp["models"]:
            if model not in MODELS:
                errors.append(
                    f"experiment {exp_name}['models'] should exist in: {MODELS}"
                )

        # check that grid has a corresponding yaml file in config/grids
        # and that all models have a key in the grid
        grid_path = os.path.join("config", "grids", exp["grid"] + ".yaml")
        if not os.path.exists(grid_path):
            errors.append(
                f"experiment {exp_name}['grid'] should be a string from: {os.listdir('config/grids')}"
            )
        else:
            grid = yaml.safe_load(open(grid_path))
            for model in exp["models"]:
                if model not in grid:
                    errors.append(
                        f"experiment {exp_name}['grid'] should have hyperparameters defined for model: {model}"
                    )

        # if target_cni is not none, then matching must be none and model must be regression
        if exp["targets_cni"] is not ["none"]:
            if exp["matching"] is not ["none"]:
                errors.append(
                    f"experiment {exp_name} matching must be 'none' when {exp_name} targets_cni is not 'none'"
                )
            for model in exp["models"]:
                if not isinstance(MODELS[model], RegressionModel):
                    errors.append(
                        f"experiment {exp_name} model {model} must be regression model when {exp_name} targets_cni is not 'none'"
                    )

    # check that each path exists
    for dataset_name, dataset in config["custom_datasets"].items():
        for feature in dataset["features"]:
            if not os.path.exists(dataset["features"][feature]):
                errors.append(
                    f"dataset {dataset_name} feature {feature} contains a feature file that does not exist: {feature}"
                )
        for target in dataset["targets"]:
            if not os.path.exists(dataset["targets"][target]):
                errors.append(
                    f"dataset {dataset_name} target {target} contains a target file that does not exist: {target}"
                )

    return errors


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.load(f)
    errors = validate_details(config)
    for error in errors:
        print(error)
