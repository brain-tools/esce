from workflow.scripts.generate_splits import generate_random_split, generate_matched_split, write_splitfile
import os
import json
import numpy as np
from sklearn.utils import shuffle
from tempfile import NamedTemporaryFile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple


def test_generate_random_split():
    # Sample data
    y = np.arange(1000)
    n_train = 700
    n_val = 150
    n_test = 150
    seed = 0
    do_stratify = False

    split = generate_random_split(y, n_train, n_val, n_test, do_stratify, seed, mask = False)

    assert len(split["idx_train"]) == n_train
    assert len(split["idx_val"]) == n_val
    assert len(split["idx_test"]) == n_test

    assert split["samplesize"] == n_train
    assert split["seed"] == seed
    assert split["stratify"] == do_stratify

    # Check the indices are within the range
    assert np.all((split["idx_train"] >= 0) & (split["idx_train"] < len(y)))
    assert np.all((split["idx_val"] >= 0) & (split["idx_val"] < len(y)))
    assert np.all((split["idx_test"] >= 0) & (split["idx_test"] < len(y)))

# DEBUGGING! test_generate_matched_split
def test_generate_matched_split():
    # Sample dataset
    n_train = 5
    n_val = 2
    n_test = 2
    do_stratify = False
    seed = 0

    x = np.array([[ 6.07495227e-02,  3.17397108e-01,  3.96117838e-01,
         3.52398709e-01,  3.70560179e-01, -2.76183696e-01,
        -2.12327711e-01, -2.37740466e-02, -2.99279555e-01,
         2.98851471e-01],
       [-3.11822933e-01, -4.31645885e-01,  3.88923484e-01,
         3.86021866e-01, -4.69723844e-01, -3.78089017e-01,
         6.80322522e-02,  9.17847357e-02, -3.21037578e-01,
        -9.88474115e-02],
       [-1.82432419e-01, -3.92936576e-01, -4.98293632e-01,
         2.70046835e-01, -1.71265311e-02,  4.98123994e-01,
        -9.44987680e-02,  9.41119303e-02, -1.05254538e-02,
         5.04282091e-01],
       [ 1.03601936e-01,  2.51767844e-01, -3.45475262e-01,
         3.16369629e-01,  2.17670820e-01,  3.63753819e-01,
         3.80503160e-01, -5.59233605e-02,  3.86685094e-01,
        -2.64690040e-02],
       [-2.95405356e-01,  1.77332219e-01,  3.39197894e-04,
        -2.59671463e-01, -2.05334392e-01,  7.68400645e-02,
         2.54258948e-01,  4.22848670e-01, -4.10427993e-01,
        -2.57022640e-01],
       [-2.10620572e-01, -2.18800413e-02,  3.00683178e-02,
        -3.76087421e-01, -3.24471845e-01, -1.83666736e-01,
         6.47103182e-02, -2.15033222e-01, -1.26601721e-01,
         1.61359643e-01],
       [-4.85687747e-01, -9.25150977e-02,  1.85041668e-01,
         1.39315276e-01, -3.41536566e-01,  3.37182434e-01,
        -1.73090710e-02, -2.15437234e-01,  5.49335953e-01,
         4.09269626e-01],
       [-2.49164162e-01, -2.50475050e-01,  2.83612462e-01,
         1.68928255e-01,  4.17997842e-02, -2.70878138e-01,
        -4.87344642e-02,  4.18221568e-01,  1.61212537e-01,
         7.70187792e-02],
       [ 2.68558391e-01, -1.04372972e-01,  2.91919310e-01,
        -2.60242695e-01, -2.52616658e-01, -3.08419460e-01,
        -8.48639287e-02,  3.88898066e-01,  2.55569844e-01,
        -5.04933978e-01]])
    x_mask = np.all(np.isfinite(x), 1)
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0]).reshape(-1)
    y_mask = np.isfinite(y)
    xy_mask = np.logical_and(x_mask, y_mask)
    print('xy_mask',xy_mask)
    matching = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0]).reshape(-1,1)
    m_mask = np.isfinite(matching)
    print('matching:', matching)
    mask = np.logical_and(xy_mask, m_mask)
    match = StandardScaler().fit_transform(matching)
    print(match)
    split = generate_matched_split(
        y=y,
        match = matching,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        do_stratify=do_stratify,
        mask=mask,
        seed=seed)


    assert len(split["idx_train"]) == n_train * 2
    assert len(split["idx_val"]) == n_val * 2
    assert len(split["idx_test"]) == n_test * 2
    assert split["samplesize"] == n_train * 2
    assert split["seed"] == seed
    assert split["stratify"] == do_stratify
    assert "average_matching_score" in split

    # Verify the indices are within the range of the original dataset
    assert np.all((split["idx_train"] >= 0) & (split["idx_train"] < len(y)))
    assert np.all((split["idx_val"] >= 0) & (split["idx_val"] < len(y)))
    assert np.all((split["idx_test"] >= 0) & (split["idx_test"] < len(y)))

# def test_write_splitfile():
#     # Mock the necessary inputs
#     features_path = "features.npy"
#     targets_path = "targets.npy"
#     split_path = "split.json"
#     sampling_path = "matching_variables.npy"
#     sampling_type = "balanced"
#     n_train = 100
#     n_val = 20
#     n_test = 20
#     seed = 0
#     stratify = False

#     # Sample data
#     num_samples = 1000
#     num_features = 10
#     num_classes = 2
#     x = np.random.randn(num_samples, num_features)
#     y = np.random.randint(num_classes, size=num_samples)

#     x, y = shuffle(x, y, random_state=seed)

#     np.save(features_path, x)
#     np.save(targets_path, y)

#     # Dummy matching variable data
#     matching = np.random.randn(num_samples)
#     np.save(sampling_path, matching)

#     write_splitfile(
#         features_path,
#         targets_path,
#         split_path,
#         sampling_path,
#         sampling_type,
#         n_train,
#         n_val,
#         n_test,
#         seed,
#         stratify,
#     )

#     with open(split_path, "r") as f:
#         split_dict = json.load(f)

#     assert "idx_train" in split_dict
#     assert "idx_val" in split_dict
#     assert "idx_test" in split_dict
#     assert "samplesize" in split_dict
#     assert "seed" in split_dict
#     assert "stratify" in split_dict

#     assert len(split_dict["idx_train"]) == n_train
#     assert len(split_dict["idx_val"]) == n_val
#     assert len(split_dict["idx_test"]) == n_test

#     # Verify that the indices are within the range of the original data
#     assert np.all((split_dict["idx_train"] >= 0) & (split_dict["idx_train"] < num_samples))
#     assert np.all((split_dict["idx_val"] >= 0) & (split_dict["idx_val"] < num_samples))
#     assert np.all((split_dict["idx_test"] >= 0) & (split_dict["idx_test"] < num_samples))

#     # Remove temporary files
#     for path in [features_path, targets_path, sampling_path, split_path]:
#         os.remove(path)
