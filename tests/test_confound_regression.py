from workflow.scripts.confound_regression import confound_regression
import numpy as np
import os

def test_confound_regression(tmp_path):
    # Create temporary files for testing
    data_path = "tests/data/confound_regression_features.npy"
    confounds_path = "tests/data/confound_regression_confounds.npy"
    out_path = "tests/data/confound_regression_output.npy"

    # Generate sample data
    data_raw = np.array([1, 2, 3, 4, 5])
    confounds = np.array([0, 1, 0, 1, 0])
    
    np.save(data_path, data_raw)
    np.save(confounds_path, confounds)

    # Call the confound_regression function
    confound_regression(str(data_path), str(confounds_path), str(out_path))

    # Verify the existence of the output file
    assert os.path.exists(out_path), 'output file doesn\'t exist'

    data_corrected = np.load(out_path).flatten()

    # Verify the correctness of the correction
    expected_corrected = [-2, -1,  0,  1,  2]
    
    assert np.array_equal(data_corrected, expected_corrected), 'correction made regarding confounders is incorrect'

    files_to_remove = [data_path, confounds_path, out_path]
    for file_path in files_to_remove:
        os.remove(file_path)