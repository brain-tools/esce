Configuration File
==================

Note: Custom names should only use hyphens and never use underscores.

General Configuration
---------------------

This section sets parameters that will be used throughout the workflow
for all evaluated machine learning scenarios. The most important
parameters are:

-  ``sample_sizes``: Train set sizes you want to evaluate. Depending on
   your dataset size, you may need to change the default values.
-  ``seeds``: The number of times a model evaluation should be repeated
   on different subsets of the data. Start with a single seed to ensure
   everything works and then increase the number of seeds for higher
   quality results (10-20 is a reasonable default).

.. code:: yaml

   val_test_frac: .5
   # (float) Ratio of train set size to validation/test set size (in this case, 2:1:1)

   val_test_max: 1000
   # (int) Maximum validation/test set size, to avoid making it unnecessarily large for very large train sets.

   bootstrap_repetitions: 100
   # (int) Number of bootstrap repetitions for fitting power laws to learning curves.

   stratify: False
   # (bool) Stratify classes when splitting into train/val/test sets.

   seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   # (list of int) Random seeds for splits. In this case, all analyses will be repeated 10 times with different (Monte Carlo) train/val/test splits.

   sample_sizes: [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
   # (list of int) Train sample sizes to evaluate.

Experiment Definitions
----------------------

You can define multiple experiments, which are sets of feature sets,
targets, covariates-of-no-interest, and machine learning models to be
evaluated in every possible combination. Some combinations may not make
sense, such as using the same models for regression and classification
targets. In such cases, simply define multiple experiments.

.. code:: yaml

   experiments:
   # Defines experimental setups.
       mnist_benchmark:
       # (str) Name of first experiment.
           dataset: "mnist"
           # (str) Name of dataset to use.
           features: ["pixel",]
           # (list of str) List of feature sets to use.
           features_cni: ["none",]
           # (list of str) List of confounder sets to remove from the features.
           targets: ["ten-digits","odd-even"]
           # (list of str) List of target variables to use.
           targets_cni: ["none",] ]
           # (list of str) List of confounder sets to remove from the targets.
           matching: ["none",]
           # (list of str) List of confounder sets to create matched cohorts.
           models: ["ridge-cls",]
           # (list of str) List of models to evaluate.
           grid: "default"
           # (str) Hyperparameter grid to use.  
       another_set_of_experiments:
       # (str) Name of second experiment.
           [...]

Dataset Definitions
-------------------

In each dataset, you must provide three types of files:

1. Feature set files: The dimensions of the feature set files should be
   ``n_samples x n_features``. The features will be standardized
   automatically.
2. Target files: The dimensions of the target files should be
   ``n_samples x 1``.
3. covariates-of-no-interest (optional): If you choose to include
   covariates-of-no-interest (CNI) files, their dimensions should be
   ``n_samples x n_cni``. During matching, the CNIs will be standardized
   and weighted equally. If you require a different weighting, calculate
   the combined CNI values and save them as a single row CNI file.

Files should be saved in one of the following formats: - Comma-separated
values (``.csv``) - Tab-separated values (``.tsv``) - Numpy array file
(``.npy``)

It’s important to note that within a single dataset, all files must have
the same number of rows and the same samples must be consistently
referred to in all files.

.. code:: yaml

   custom_datasets:
   # Defines custom datasets
       dataset_name:
       # (str) Name of the first custom dataset
           features: {'NAME':'PATH_TO_CSV_TSV_NPY','NAME':'PATH_TO_CSV_TSV_NPY'}
           # (dict of str) Names and paths of the feature set files
           targets: {'NAME':'PATH_TO_CSV_TSV_NPY','NAME':'PATH_TO_CSV_TSV_NPY'}
           # (dict of str) Names and paths of the target files
           covariates: {'NAME':'PATH_TO_CSV_TSV_NPY','NAME':'PATH_TO_CSV_TSV_NPY'}
           # (dict of str) Names and paths of the covariate set files
       another_dataset_name:
       # (str) Name of the second custom dataset
           [...]
