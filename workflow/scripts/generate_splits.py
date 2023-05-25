"""
generat_splits.py
====================================

"""

    
import json
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NpEncoder(json.JSONEncoder):
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
    """
    Standard random split 
    """
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
    """
    
    1: patient group
    0: control group

    matching array: works similar to confound, 
    """
    random_state = np.random.RandomState(seed)
    mask_orig = mask.copy()

    # taking a subset of patients
    split = generate_random_split(
        y,
        n_train // 2,
        n_val // 2,
        n_test // 2,
        do_stratify,
        seed,
        np.logical_and(mask, y == 1), # 1: patient group
    )
    idx_all = np.arange(len(y))

    mask[y == 1] = False # mask to exclude all patients from all participants
    assert np.isfinite(match[mask]).all()

    match = StandardScaler().fit_transform(match)
    matching_scores = []
    for idx_set in ["idx_train", "idx_val", "idx_test"]: # for each set
        control_group = [] # generate a control group
        for idx in split[idx_set]: # for each patient
            idx_pool = idx_all[mask] # the pool of everyone who's not patients (= potential candidates for control group)
            scores = (match[idx_pool] - match[idx]) ** 2 # calculate score from (matching varialbe for everyone in the pool - matchin var of patients)**2
            scores = np.sum(scores, axis=1) # sum up the characteristic scores

            t = random_state.permutation(np.column_stack((scores, idx_pool))) # shuffle to run on random seed in case we always match only with the first participant in the pool --> to take care of the ties (same characteristics)
            t_idx = np.nanargmin(t.T[0]) # take index of the person with the smallest score (matches the best, approximation of the optimal control invidivual)

            score_match = t.T[0][t_idx]
            matching_scores.append(score_match)

            idx_match = t.T[1][t_idx].astype(int)
            control_group.append(idx_match)
            mask[idx_match] = False # mask out the approximate best control group candidate (without replacement)

            assert mask_orig[idx_match], (scores, t, t[t_idx], score_match, idx_match)

        split[idx_set] = np.hstack((split[idx_set], control_group))

    split.update({"average_matching_score": np.mean(matching_scores)})
    split["samplesize"] *= 2

    return split


def write_splitfile(
    features_path,
    targets_path,
    split_path,
    sampling_path, # the path to the matching variable table (e.g. age, gender, ed level)
    sampling_type, # matching options: [none, balanced, ]
    n_train,
    n_val,
    n_test,
    seed,
    stratify=False,
):
    """
    
    """
    # excluding not all fully available data
    x = np.load(features_path)
    x_mask = np.all(np.isfinite(x), 1)
    y = np.load(targets_path).reshape(-1)
    y_mask = np.isfinite(y)

    xy_mask = np.logical_and(x_mask, y_mask)

    n_classes = len(np.unique(y[xy_mask]))
    idx_all = np.arange(len(y))

    # if stratify flag is on but we have too many classes to perform stratification, we don't do it 
    stratify = True if stratify and (n_classes <= 10) else False

    ###
    # sampling options: [none (random), balanced (undersample the controls to num of patients), (if points to a path with data, we do the matching)]
    ###
    matching = np.load(sampling_path)
    if sampling_type == "none":
        matching = False
    elif sampling_type == "balanced":
        matching = False
        idx_undersampled = RandomUnderSampler(random_state=seed).fit_resample(
            idx_all[xy_mask], y[xy_mask]
        )
        xy_mask[[i for i in idx_all if i not in idx_undersampled]] = False
    # if sampling option points to a path with data, we do the matching
    elif len(matching) == len(y) and len(matching.shape) > 1: # if there are more than 1 matching variable (e.g. age, gender, ed level)
        assert n_classes == 2
        # preparing for matching by excluding the participants with NaN values in the matching variables
        m_mask = np.all(np.isfinite(matching), 1)
        xy_mask = np.logical_and(xy_mask, m_mask)
    
    # preparing for matching when there's only one matching variable
    elif len(matching) == len(y) and len(matching.shape) == 1:
        assert n_classes == 2
        # preparing for matching by excluding the participants with NaN values in the matching variables
        m_mask = np.isfinite(matching)
        xy_mask = np.logical_and(xy_mask, m_mask)
    else:
        raise Exception("invalid sampling file")

    if matching is False and sum(xy_mask) > n_train + n_val + n_test: # if matching is 'none' or 'balanced', random splitting
        split_dict = generate_random_split(
            y=y,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            do_stratify=stratify,
            mask=xy_mask,
            seed=seed,
        )
    elif sum(xy_mask[y == 1]) > n_train // 2 + n_val // 2 + n_test // 2: # if special matching is flagged
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
    else: # if not enough sample for previous sampling options, flag error
        split_dict = {"error": "insufficient samples"}

    if not "error" in split_dict: # doulbe checking all values are valid
        assert np.isfinite(x[split_dict["idx_train"]]).all()
        assert np.isfinite(y[split_dict["idx_train"]]).all()

    with open(split_path, "w") as f:
        json.dump(split_dict, f, cls=NpEncoder, indent=0)


n_train = int(snakemake.wildcards.samplesize)
n_val = min(
    round(n_train * snakemake.params.val_test_frac), snakemake.params.val_test_max
)
n_test = min(
    round(n_train * snakemake.params.val_test_frac), snakemake.params.val_test_max
)

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
