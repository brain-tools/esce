# Empirical Sample Complexity Estimator

The Empirical Sample Complexity Estimator provides a Snakemake workflow to analyze the performance of machine learning models as the sample size increases. The goal of the workflow is to make it easy to compare the scaling behaviour of various machine learning models, feature sets, and target variables.

For more information, refer to the following publication:

[Schulz, M. A., Bzdok, D., Haufe, S., Haynes, J. D., & Ritter, K. (2022). Performance reserves in brain-imaging-based phenotype prediction. BioRxiv, 2022-02.](https://biorxiv.org/content/10.1101/2022.02.23.481601v1.full)

---

## Installing Snakemake

In order to use this workflow, you must have either Anaconda or Miniconda installed and Snakemake must be installed. To install Snakemake, run the following command:

```
conda install -c bioconda snakemake
```

---

## Example Workflow

To try the example workflow, check out the [example/](example/) directory and run the following command:

```
snakemake --cores 1 --configfile example/example_config.yaml --rerun-triggers mtime --use-conda --rerun-incomplete all
```

Once the workflow has completed, you can view the results in the `results/example-dataset/statistics` directory and the plots in `results/example-dataset/plots`.

---

## Configuration

For more information on the configuration file, see [here](config/README.md).

---

## Workflow Structure

The workflow is designed to process datasets that include multiple feature sets, multiple target variables, and covariates-of-no-interest. The features, targets, and covariates are standardized (`rule: prepare_features_and_targets` and `rule prepare_covariates`). Additional feature sets are created by correcting for covariates (`rule: confound_regression`).

For each combination of feature set, target variable, and covariate, the workflow removes rows with missing or NaN values and creates train/validation/test splits for each sample size and random seed defined in the configuration file (`rule: split`). The splits are saved to `results/dataset/splits` so they can be reused if additional models or repetition seeds are added to the configuration.

The workflow then fits and evaluates machine learning models (`rule: fit`) with a range of hyperparameters. The results are saved to `results/dataset/fits`.

The workflow collects the accuracy estimates for the best-performing hyperparameter configurations and writes them to summary files in `results/dataset/scores` (`rule: aggregate`). The accuracy estimates are then used to fit power laws to the data (`results/dataset/statistics/*.stat.json`, `rule: extrapolate`), using bootstrapping to estimate uncertainties (`results/dataset/statistics/*.bootstrap.json`).

Finally, the workflow creates summary figures based on the `.stat.json` and `.bootstrap.json` files. There are five types of figures: individual learning curves for each prediction setup (`plot_individually`), figures aggregating over all feature sets (`plot_by_features`), figures aggregating over all target variables (`plot_by_targets`), figures aggregating over all machine learning models (`plot_by_features`), and figures aggregating over all confound corrections approaches (`plot_by_cni`).


Here is a visualisation of the workflow

<img src="resources/dag.png" width="500">

it was created via `snakemake --forceall --rulegraph --configfile example/example_config.yaml | dot -Tpng > dag.png` (requires graphviz)


## Testing

To make it easier to verify the functionality of the code, and for the simplification of future contributions, we added unit tests so that we can confirm that we are getting what we expect back from important methods.

As the pytest package should be installed when you install the environment, simply run the following line to run all tests:

```
pytest
```

Or run the following line with the name(s) of a file you want to test separately as some test files (files that invovle generating graphs) take slightly longer to run:

```
pytest tests/test_<file_name>.py 
```