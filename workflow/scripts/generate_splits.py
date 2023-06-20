import json
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NpEncoder(json.JSONEncoder):
    """Encode numpy arrays to JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_random_split(
    y: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Optional[np.ndarray] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random split of the data."""
    if mask is False:
        idx_originial = np.arange(len(y))
        idx = np.arange(len(y))
    else:
        idx_originial = np.arange(len(y))[mask]
        idx = np.arange(len(y[mask]))
        y = y[mask]

    stratify = y if do_stratify else None
    idx_train, idx_test = train_test_split(
        idx, test_size=n_test, stratify=stratify, random_state=seed
    )

    stratify = y[idx_train] if do_stratify else None
    idx_train, idx_val = train_test_split(
        idx_train,
        train_size=n_train,
        test_size=n_val,
        stratify=stratify,
        random_state=seed,
    )

    split = {
        "idx_train": idx_originial[idx_train],
        "idx_val": idx_originial[idx_val],
        "idx_test": idx_originial[idx_test],
        "samplesize": n_train,
        "seed": seed,
        "stratify": do_stratify,
    }

    return split


def generate_matched_split(
    y: np.ndarray,
    match: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Optional[np.ndarray] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a matched split of the data.

    Assumes a binary classification target variable, coded as 0 and 1, with 1 being the positive (patient) class and 0 being the negative (control) class.
    Masking allows to only consider a subset of the data for matching, i.e. for exluding participants with similar disorders from the control group. 

    :param y: The target variable in the shape (n_samples,).
    :param match: The covariates to use for matching, in the shape (n_samples, n_features).
    :param n_train: The number of training samples.
    :param n_val: The number of validation samples.
    :param n_test: The number of test samples.
    :param do_stratify: Whether to stratify the split.
    :param seed: The random seed.
    :param mask: The (optional) mask to apply to the data, e.g. for selecting only a subset of samples with a particular feature. Shape (n_samples,).
    :return: The split: list of training, validation and test indices and some additional information.
    :rtype: Dict['idx_train', 'idx_val', 'idx_test', 'samplesize', 'seed', 'stratify','average_matching_score']

    """

    random_state = np.random.RandomState(seed)
    mask_orig = mask.copy()

    # patients
    split = generate_random_split(
        y,
        n_train // 2,
        n_val // 2,
        n_test // 2,
        do_stratify,
        seed,
        np.logical_and(mask, y == 1),
    )
    idx_all = np.arange(len(y))

    mask[y == 1] = False
    assert np.isfinite(match[mask]).all()

    match = StandardScaler().fit_transform(match)
    matching_scores = []
    for idx_set in ["idx_train", "idx_val", "idx_test"]:
        control_group = []
        for idx in split[idx_set]:
            idx_pool = idx_all[mask]
            scores = (match[idx_pool] - match[idx]) ** 2
            scores = np.sum(scores, axis=1)

            t = random_state.permutation(np.column_stack((scores, idx_pool)))
            t_idx = np.nanargmin(t.T[0])

            score_match = t.T[0][t_idx]
            matching_scores.append(score_match)

            idx_match = t.T[1][t_idx].astype(int)
            control_group.append(idx_match)
            mask[idx_match] = False

            assert mask_orig[idx_match], (scores, t, t[t_idx], score_match, idx_match)

        split[idx_set] = np.hstack((split[idx_set], control_group))

    split.update({"average_matching_score": np.mean(matching_scores)})
    split["samplesize"] *= 2

    return split


def write_splitfile(
    features_path,
    targets_path,
    split_path,
    sampling_path,
    sampling_type,
    n_train,
    n_val,
    n_test,
    seed,
    stratify=False,
):
    x = np.load(features_path)
    x_mask = np.all(np.isfinite(x), 1)
    y = np.load(targets_path).reshape(-1)
    y_mask = np.isfinite(y)

    xy_mask = np.logical_and(x_mask, y_mask)

    n_classes = len(np.unique(y[xy_mask]))
    idx_all = np.arange(len(y))

    stratify = True if stratify and (n_classes <= 10) else False

    matching = np.load(sampling_path)
    if sampling_type == "none":
        matching = False
    elif sampling_type == "balanced":
        assert n_classes <= 10, 'For too many classes, balancing strategy (in under sampling) doesn\'t make sense.'
        
        matching = False
        idx_undersampled, _ = RandomUnderSampler(random_state=seed).fit_resample(
            idx_all[xy_mask].reshape(-1,1), y[xy_mask].astype(int)
        )
        idx_undersampled=idx_undersampled.reshape(-1)
        xy_mask[[i for i in idx_all if i not in idx_undersampled]] = False
    elif len(matching) == len(y) and len(matching.shape) > 1:
        assert n_classes == 2
        m_mask = np.all(np.isfinite(matching), 1)
        xy_mask = np.logical_and(xy_mask, m_mask)
    elif len(matching) == len(y) and len(matching.shape) == 1:
        assert n_classes == 2
        m_mask = np.isfinite(matching)
        xy_mask = np.logical_and(xy_mask, m_mask)
    else:
        raise Exception("invalid sampling file")

    if matching is False and (sum(xy_mask) > n_train + n_val + n_test):
        split_dict = generate_random_split(
            y=y,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            do_stratify=stratify,
            mask=xy_mask,
            seed=seed,
        )
    elif matching is not False and (sum(xy_mask[y == 1]) > n_train // 2 + n_val // 2 + n_test // 2):
        split_dict = generate_matched_split(
            y=y,
            match=matching,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            do_stratify=stratify,
            mask=xy_mask,
            seed=seed,
        )
    else:
        split_dict = {"error": "insufficient samples"}

    if not "error" in split_dict:
        assert np.isfinite(x[split_dict["idx_train"]]).all()
        assert np.isfinite(y[split_dict["idx_train"]]).all()

    with open(split_path, "w") as f:
        json.dump(split_dict, f, cls=NpEncoder, indent=0)


n_train = int(snakemake.wildcards.samplesize)
n_val = n_test = min(
    round(n_train * snakemake.params.val_test_frac), snakemake.params.val_test_max
) if snakemake.params.val_test_max else round(n_train * snakemake.params.val_test_frac)
n_val = n_test = max(n_val, snakemake.params.val_test_min) if snakemake.params.val_test_min else n_val
assert n_train > 1 and n_val > 1 and n_test > 1


write_splitfile(
    features_path=snakemake.input.features,
    targets_path=snakemake.input.targets,
    split_path=snakemake.output.split,
    sampling_path=snakemake.input.matching,
    sampling_type=snakemake.wildcards.matching,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
    seed=int(snakemake.wildcards.seed),
    stratify=snakemake.params.stratify,
)
