from pathlib import Path
import pickle
import h5py
import hashlib
import requests
from tqdm import tqdm
import json
import yaml
from typing import Dict, Any, Union, Tuple, Optional
import numpy as np


def uniform_noise_matrix(M: int, p: float) -> np.ndarray:
    # p: probability of label changing
    N = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            N[i][j] = 1 - p if i == j else p / (M - 1)
    return N


def apply_noise_matrix(labels: np.ndarray, N: np.ndarray, seed: int) -> np.ndarray:
    out = np.zeros(len(labels))
    unique_labels = np.unique(labels)

    lbl2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    random_state = np.random.RandomState(seed)
    for i, l in enumerate(labels):
        probs = N[lbl2idx[l]]
        out[i] = random_state.choice(unique_labels, 1, p=probs)
    return out


def flip_binary_label(x: np.ndarray, prob: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    indices = np.random.random(x.shape) < prob
    return np.logical_xor(x, indices).astype(int)


def flt2str(f: float, decimals: int = 4) -> str:
    return str(round(f, decimals)).replace(".", "#")


def hash_dict(x: Dict[Any, Any]) -> str:
    return hashlib.md5(json.dumps(x, sort_keys=True).encode("utf-8")).hexdigest()


def load_grid_file(grid_name: str) -> Any:
    """
    Loads a grid from a YAML file.
    Grid YAML files may contain multiple grids.
    These grid can be selected by adding @<entry> to
    the file path, e.g. grid.yaml@grid1

    Arguments:
        grid_name: Path to the YAML file.

    Returns:
        Grid dictionary
    """
    grid_key = None
    if "@" in grid_name:
        grid_path, grid_key = grid_name.split("@")
    else:
        grid_path = grid_name

    grid_file = Path(grid_path)
    if grid_file.is_file():
        with grid_file.open("r") as f:
            grid = yaml.safe_load(f)
            if grid_key is not None:
                grid = grid[grid_key]
            return grid
    else:
        raise ValueError("Invalid grid file path")


def load_dataset(
    data_path: Path, label: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads data and label from data file

    Arguments:
        data_path: Path to h5 or pkl file
        label: Label to use contained in the file

    Returns:
        (x,y) Tuple of data and labels
    """

    if data_path.suffix == ".h5":
        with h5py.File(data_path, "r") as f:
            x = f["/data"][...]
            if label is None:
                return x
            y = f[f"/labels/{label}"][...]
    elif data_path.suffix == ".pkl":
        with data_path.open("rb") as f:
            d = pickle.load(f)
            x = d["data"]
            if label is None:
                return x
            y = d[f"label_{label}"]
    else:
        raise ValueError("Unknown file format")
    return x, y


def load_split(split_path: Path) -> Any:
    """
    Loads a split file from a path.

    Arguments:
        split_path: Path to split file

    Returns:
        Tuple consisting of a seed and the splits
    """

    with split_path.open("rb") as f:
        return pickle.load(f)


def download_file(url: str, path: Path) -> None:
    """
    Download the URL and save it in the given path.

    Arguments:
        url: URL to download
        path: Path to store file to
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    bar = tqdm(total=total, unit="iB", unit_scale=True)
    block_size = 1024
    with path.open("wb") as f:
        for data in resp.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
    bar.close()
    if total != 0 and bar.n != total:
        raise ValueError
