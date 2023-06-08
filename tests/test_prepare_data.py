from workflow.scripts.prepare_data import prepare_data
import numpy as np
import pandas as pd
from pathlib import Path

def test_prepare_data():

    dataset = "dummy"
    variant = "default"
    dummy_datasets = {
        "dummy": {
            "features": {
                "default": "tests/data/prepare_data_features.npy",
            },
            "targets": {
                "default": "tests/data/prepare_data_targets.npy",
            },
            "covariates": {
                "default": "tests/data/prepare_data_covariates.npy",
            },
        },
    }

    ###
    # Test Case 1: features
    ###
    out_path = "tests/data/prepare_data_output_features.npy"
    features_targets_covariates = "features"

    data_file = dummy_datasets["dummy"]["features"]["default"]
    np.save(data_file, np.array([[0.43569162, 0.8024389 , 0.96880062, 0.75306636, 0.85472289,
        0.20791024, 0.22085722, 0.49857798, 0.19380115, 0.7353865 ],
       [0.14745387, 0.08471296, 0.92972959, 0.84134661, 0.05981211,
        0.0717563 , 0.45085172, 0.60922316, 0.121587  , 0.31818272],
       [0.51009147, 0.11000623, 0.0143975 , 0.92334511, 0.59657872,
        0.94151463, 0.36702353, 0.60854194, 0.43687264, 0.97424772]]))

    prepare_data(out_path, dataset, features_targets_covariates, variant, dummy_datasets)

    prepared_data = np.load(out_path)

    expected_output = np.array([[0.43569162, 0.8024389 , 0.96880062, 0.75306636, 0.85472289,
        0.20791024, 0.22085722, 0.49857798, 0.19380115, 0.7353865 ],
       [0.14745387, 0.08471296, 0.92972959, 0.84134661, 0.05981211,
        0.0717563 , 0.45085172, 0.60922316, 0.121587  , 0.31818272],
       [0.51009147, 0.11000623, 0.0143975 , 0.92334511, 0.59657872,
        0.94151463, 0.36702353, 0.60854194, 0.43687264, 0.97424772]])

    assert np.allclose(prepared_data, expected_output), 'features data preparation failed'

    for file_path in [out_path, data_file]:
        Path(file_path).unlink(missing_ok=True)
        
    ###
    # Test Case 2: targets
    ###
    # Generate dummy data for testing
    out_path = "tests/data/prepare_data_output_targets.npy"
    features_targets_covariates = "targets"

    data_file = dummy_datasets["dummy"]["targets"]["default"]
    np.save(data_file, np.array([0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
       0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.,
       1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1.]))

    prepare_data(out_path, dataset, features_targets_covariates, variant, dummy_datasets)

    prepared_data = np.load(out_path)

    expected_output = np.array([0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
       0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.,
       1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1.])

    assert np.allclose(prepared_data, expected_output), 'targets data preparation failed'
    
    for file_path in [out_path, data_file]:
        Path(file_path).unlink(missing_ok=True)
    ###
    # Test Case 3: prepare covariates
    ###
    # Generate dummy data for testing
    out_path = "tests/data/prepare_data_output_covariates.npy"
    features_targets_covariates = "covariates"

    data_file = dummy_datasets["dummy"]["covariates"]["default"]
    np.save(data_file, np.array([[0.9178436 , 0.25769881, 0.7951604 ],
       [0.78043125, 0.5019076 , 0.7911015 ],
       [0.18629525, 0.98264458, 0.25982452],
       [0.29042939, 0.68576042, 0.76633173],
       [0.78591483, 0.01236143, 0.90869882],
       [0.97171503, 0.10690087, 0.25087941],
       [0.79621753, 0.84162531, 0.09957807]]))

    prepare_data(out_path, dataset, features_targets_covariates, variant, dummy_datasets)

    prepared_data = np.load(out_path)

    expected_output = np.array([[0.9178436 , 0.25769881, 0.7951604 ],
       [0.78043125, 0.5019076 , 0.7911015 ],
       [0.18629525, 0.98264458, 0.25982452],
       [0.29042939, 0.68576042, 0.76633173],
       [0.78591483, 0.01236143, 0.90869882],
       [0.97171503, 0.10690087, 0.25087941],
       [0.79621753, 0.84162531, 0.09957807]])

    assert np.allclose(prepared_data, expected_output), 'covariates data preparation failed'

    
    # Remove temporary files
    for file_path in [out_path, data_file]:
        Path(file_path).unlink(missing_ok=True)
