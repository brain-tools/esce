import json

import h5py
import numpy as np
import pytest

from workflow.scripts.generate_splits import (
    generate_matched_split,
    generate_random_split,
    write_splitfile,
)

PARAMETERS = "y, match, n_train, n_val, n_test, do_stratify, seed, mask"
TEST_CASES = [
    (
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]]),
        2,
        2,
        2,
        False,
        0,
        np.array([True, True, True, True, True, True, True, True]),
    ),
    (
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        np.array([[0], [3], [1], [2], [2], [1], [3], [0]]),
        2,
        2,
        2,
        False,
        0,
        np.array([True, True, True, True, True, True, True, True]),
    ),
    (
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        np.array([[0], [3], [1], [2], [2], [1], [3], [0]]),
        2,
        2,
        2,
        False,
        0,
        np.array([False, True, True, True, True, True, True, False]),
    ),
    (
        np.array([1, 1, 1, 1, 0, 0, 0, 0]),
        np.array([[1], [2], [3], [4], [9], [8], [7], [6]]),
        2,
        2,
        2,
        False,
        0,
        np.array([True, True, True, True, True, True, True, True]),
    ),
    (
        np.array([1, 1, 1, 1, 0, 0, 0, 0]),
        np.array([[1], [2], [3], [4], [9], [8], [7], [6]]),
        2,
        2,
        2,
        True,
        0,
        np.array([True, True, True, True, True, True, True, True]),
    ),
]


@pytest.mark.parametrize(PARAMETERS, TEST_CASES)
def test_matching_in_generate_matched_split(
    y, match, n_train, n_val, n_test, do_stratify, seed, mask
):
    split = generate_matched_split(
        y, match, n_train, n_val, n_test, do_stratify, seed, mask
    )

    for idx_set in ["idx_train", "idx_val", "idx_test"]:
        indices = split[idx_set]
        assert indices.max() < len(y), f"Index out of bounds in {idx_set}"
        assert len(indices) == len(set(indices)), f"Duplicated indices in {idx_set}"

        if mask is not None:
            assert np.all(mask[indices]), f"Mask compliance failed in {idx_set}"

        mid_point = len(indices) // 2
        patient_indices = indices[:mid_point]
        control_indices = indices[mid_point:]

        patient_confounds = match[patient_indices]
        control_confounds = match[control_indices]

        for i, patient_confound in enumerate(patient_confounds):
            differences = np.abs(control_confounds - patient_confound)
            min_difference_idx = np.argmin(differences)
            assert (
                min_difference_idx == i
            ), f"Patient's closest control is not its matched control in {idx_set}"

    assert len(split["idx_train"]) == n_train, "Incorrect size for training set"
    assert len(split["idx_val"]) == n_val, "Incorrect size for validation set"
    assert len(split["idx_test"]) == n_test, "Incorrect size for test set"

    if do_stratify:
        train_dist = np.bincount(y[split["idx_train"]]) / n_train
        val_dist = np.bincount(y[split["idx_val"]]) / n_val
        test_dist = np.bincount(y[split["idx_test"]]) / n_test
        assert np.allclose(train_dist, val_dist) and np.allclose(
            val_dist, test_dist
        ), "Stratification failed"


@pytest.mark.parametrize(PARAMETERS, TEST_CASES)
def test_generate_random_split(
    y, match, n_train, n_val, n_test, do_stratify, seed, mask
):
    split = generate_random_split(y, n_train, n_val, n_test, do_stratify, seed, mask)

    for idx_set in ["idx_train", "idx_val", "idx_test"]:
        indices = split[idx_set]
        assert indices.max() < len(y), f"Index out of bounds in {idx_set}"
        assert len(indices) == len(set(indices)), f"Duplicated indices in {idx_set}"

        if mask is not None:
            assert np.all(mask[indices]), f"Mask compliance failed in {idx_set}"

    assert len(split["idx_train"]) == n_train, "Incorrect size for training set"
    assert len(split["idx_val"]) == n_val, "Incorrect size for validation set"
    assert len(split["idx_test"]) == n_test, "Incorrect size for test set"

    if do_stratify:
        train_dist = np.bincount(y[split["idx_train"]]) / n_train
        val_dist = np.bincount(y[split["idx_val"]]) / n_val
        test_dist = np.bincount(y[split["idx_test"]]) / n_test
        assert np.allclose(train_dist, val_dist) and np.allclose(
            val_dist, test_dist
        ), "Stratification failed"


def test_write_splitfile_random(tmpdir):
    features = np.random.rand(100, 5)
    targets = np.random.randint(0, 2, 100)
    matching = []

    features_path = str(tmpdir.join("features.npy"))
    targets_path = str(tmpdir.join("targets.npy"))
    matching_path = str(tmpdir.join("matching.npy"))

    with h5py.File(features_path, "w") as f:
        f.create_dataset("data", data=features)
        f.create_dataset("mask", data=np.isfinite(features).all(axis=1))

    with h5py.File(targets_path, "w") as f:
        f.create_dataset("data", data=targets)
        f.create_dataset("mask", data=np.isfinite(targets))

    with h5py.File(matching_path, "w") as f:
        f.create_dataset("data", data=matching)
        f.create_dataset("mask", data=matching)

    split_path = str(tmpdir.join("split.json"))

    write_splitfile(
        features_path,
        targets_path,
        split_path,
        matching_path,
        "none",
        n_train=60,
        n_val=20,
        n_test=20,
        seed=0,
    )

    # Load the split file and check if it is correct
    with open(split_path) as f:
        split_dict = json.load(f)

    assert "idx_train" in split_dict
    assert "idx_val" in split_dict
    assert "idx_test" in split_dict
    assert len(split_dict["idx_train"]) == 60
    assert len(split_dict["idx_val"]) == 20
    assert len(split_dict["idx_test"]) == 20


def test_write_splitfile_balanced(tmpdir):
    features = np.random.rand(100, 5)
    targets = np.random.randint(0, 2, 100)
    matching = []

    features_path = str(tmpdir.join("features.npy"))
    targets_path = str(tmpdir.join("targets.npy"))
    matching_path = str(tmpdir.join("matching.npy"))

    with h5py.File(features_path, "w") as f:
        f.create_dataset("data", data=features)
        f.create_dataset("mask", data=np.isfinite(features).all(axis=1))

    with h5py.File(targets_path, "w") as f:
        f.create_dataset("data", data=targets)
        f.create_dataset("mask", data=np.isfinite(targets))

    with h5py.File(matching_path, "w") as f:
        f.create_dataset("data", data=matching)
        f.create_dataset("mask", data=matching)

    split_path = str(tmpdir.join("split.json"))

    write_splitfile(
        features_path,
        targets_path,
        split_path,
        matching_path,
        "balanced",
        n_train=20,
        n_val=20,
        n_test=20,
        seed=0,
    )

    # Load the split file and check if it is correct
    with open(split_path) as f:
        split_dict = json.load(f)

    assert "idx_train" in split_dict
    assert "idx_val" in split_dict
    assert "idx_test" in split_dict
    assert len(split_dict["idx_train"]) == 20
    assert len(split_dict["idx_val"]) == 20
    assert len(split_dict["idx_test"]) == 20

    assert targets[split_dict["idx_train"]].mean() == 0.5


def test_write_splitfile_matched(tmpdir):
    features = np.random.rand(100, 5)
    targets = np.random.randint(0, 2, 100)
    matching = np.random.rand(100, 3)

    features_path = str(tmpdir.join("features.npy"))
    targets_path = str(tmpdir.join("targets.npy"))
    matching_path = str(tmpdir.join("matching.npy"))

    with h5py.File(features_path, "w") as f:
        f.create_dataset("data", data=features)
        f.create_dataset("mask", data=np.isfinite(features).all(axis=1))

    with h5py.File(targets_path, "w") as f:
        f.create_dataset("data", data=targets)
        f.create_dataset("mask", data=np.isfinite(targets))

    with h5py.File(matching_path, "w") as f:
        f.create_dataset("data", data=matching)
        f.create_dataset("mask", data=np.isfinite(matching).all(axis=1))

    split_path = str(tmpdir.join("split.json"))

    write_splitfile(
        features_path,
        targets_path,
        split_path,
        matching_path,
        "matching",
        n_train=20,
        n_val=20,
        n_test=20,
        seed=0,
    )

    # Load the split file and check if it is correct
    with open(split_path) as f:
        split_dict = json.load(f)

    assert "idx_train" in split_dict
    assert "idx_val" in split_dict
    assert "idx_test" in split_dict
    assert len(split_dict["idx_train"]) == 20
    assert len(split_dict["idx_val"]) == 20
    assert len(split_dict["idx_test"]) == 20

    assert targets[split_dict["idx_train"]].mean() == 0.5
