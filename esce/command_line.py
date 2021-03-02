from esce.util import load_dataset, load_split, flip, flt2str
from esce.grid import GRID, load_grid
from esce.vis import hp_plot, sc_plot
from esce.sampling import split_grid
from esce.models import score_splits, MODELS, RegressionModel, precompute_kernels
from esce.data import DATA
from esce.util import hash_dict
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py
from sklearn.model_selection import ParameterGrid

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=ConvergenceWarning)


def precomp(
    data_path: Path,
    grid_name: str = "default",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> None:
    """
    Precomputes the kernel gram matrices for the given models.

    Arguments:
        data_path: Path to the data file.
        grid_name: Grid to use
        include: Models to include
        exclude: Models to exclude
    """

    x = load_dataset(data_path)
    grid = load_grid(grid_name)
    models = MODELS
    if include is not None:
        models = {k: v for k, v in models.items() if k in include}

    if exclude is not None:
        models = {k: v for k, v in models.items() if k not in exclude}

    precompute_kernels(x, models, grid)


def run(
    data_path: Path,
    label: str,
    split_path: Path,
    seeds: List[int],
    samples: List[int],
    grid_name: str = "default",
    warm_start: bool = False,
    cache: bool = False,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    output: Optional[str] = None,
) -> None:
    """
    Performs sample complexity computation.

    Arguments:
        data_path: Path to data file
        label: Label to use in data file
        split_path: Path to split file
        seeds: Which seeds to use
        samples: Which samples to use
        grid: Grid to use / grid file to use
        warm_start: Whether or not to continue previous computation
        cache: Turn caching on or off
    """

    x, y = load_dataset(data_path, label)
    found_seeds, splits = load_split(split_path)
    if found_seeds < len(seeds):
        raise ValueError(
            f"More seeds selected than available, found {found_seeds} seeds."
        )

    for seed in seeds:
        if seed >= found_seeds or seed < 0:
            raise ValueError(
                f"Invalid seed {seed}. Seed must be in [0,{found_seeds-1}]."
            )

    if len(seeds) == 0:
        seeds = list(range(found_seeds))

    if len(samples) > 0:
        new_splits = dict()
        for sample in samples:
            new_splits[sample] = splits[sample]
        splits = new_splits

    models = MODELS
    if include is not None:
        models = {k: v for k, v in models.items() if k in include}

    if exclude is not None:
        models = {k: v for k, v in models.items() if k not in exclude}

    seed_str = "_".join(map(str, seeds))
    sample_str = "_".join(map(str, splits.keys()))

    grid = load_grid(grid_name)
    if output is None:
        outfile = (
            Path("results") / f"{data_path.stem}_{label}_s{seed_str}_t{sample_str}.csv"
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)
    else:
        outfile = Path(output)

    score_splits(outfile, x, y, models, grid, splits, seeds, warm_start, cache)


def datagen(
    dataset: str,
    method: Optional[str],
    n_components: int,
    noise: float = 0.0,
    lbl_noise: Optional[List[float]] = None,
    fmt: str = "hdf5",
) -> None:
    """
    Generates a data file.
    The file will be placed in the 'data' directory.

    Arguments:
        dataset: Pre-defined dataset to load
        method: Dimensionality reduction method to use
        n_components: Number of components used for dimensionality reduction
        noise: Noise factor
        fmt: Data file format (hdf5 or pkl)
    """

    method_str = f"_{method}{n_components}" if method is not None else ""
    noise_str = "_n" + flt2str(noise)
    path = Path("data") / f"{dataset}{method_str}{noise_str}"
    if dataset not in DATA:
        raise ValueError("Unknown dataset")

    x, y = DATA[dataset]()
    if method == "pca":
        x = PCA(n_components=n_components, whiten=True, random_state=0).fit_transform(x)
    elif method == "rp":
        x = GaussianRandomProjection(
            n_components=n_components, random_state=0
        ).fit_transform(x)
    elif method == "tsne":
        x = TSNE(n_components=n_components, random_state=0).fit_transform(x)
    x = StandardScaler().fit_transform(x)

    if noise > 0:
        np.random.seed(0)
        x = x + noise * np.random.randn(*x.shape)
        x = StandardScaler().fit_transform(x)

    # Generate default label and label noise
    labels = {"default": y}
    if lbl_noise is not None:
        for lbl in lbl_noise:
            labels[f"noise_{flt2str(lbl)}"] = flip(y, lbl, 0)

    # Write data and labels to hdf5 or pkl
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "hdf5":
        path = path.with_suffix(".h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("/data", x.shape, data=x)
            for k, v in labels.items():
                f.create_dataset(f"/labels/{k}", v.shape, data=v)
    elif fmt == "pkl":
        path = path.with_suffix(".pkl")
        with path.open("wb") as f:
            d = {"data": x}
            for k, v in labels.items():
                d[f"label_{k}"] = v
            pickle.dump(d, f)
    else:
        raise ValueError("Unknown file format")
    print(f"Generated {dataset} data file '{path}'.")


def splitgen(data_path: Path, label: str, n_seeds: int, samples: List[int]) -> None:
    """
    Generates a split file.
    The file will be placed in the 'splits' directory.

    Arguments:
        data_path: Path to data file
        label: Label to use contained in data file
        n_seeds: Number of seeds to use in train_test_split
        samples: List of sample counts
    """
    sample_str = "_".join(map(str, samples))
    path = Path("splits") / f"{data_path.stem}_{label}_s{n_seeds}_t{sample_str}.split"
    path.parent.mkdir(parents=True, exist_ok=True)

    _, y = load_dataset(data_path, label)
    splits = split_grid(y, n_samples=samples, n_seeds=n_seeds)

    with path.open("wb") as f:
        pickle.dump((n_seeds, splits), f)
    print(f"Generated split file '{path}'")


def retrieve(
    path: Path,
    grid_name: str,
    output: Optional[Path] = None,
    show: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> None:
    """
    Retrieves the results, generates plots and the final accuracy scores.

    Arguments:
        path: Path to the results file
        grid_name: Grid to use
        output: Where to write the accuracy scores to
        show: Show results using matplotlib (all/sc/hp)
    """
    grid = load_grid(grid_name)
    df = pd.read_csv(path, index_col=False)

    model_names = set(grid.keys())
    if include is not None:
        model_names = {k for k in model_names if k in include}

    if exclude is not None:
        model_names = {k for k in model_names if k not in exclude}

    # Select entries for a specified grid
    outer_frames = []
    for model_name in model_names:
        rows_per_model = df[df["model"] == model_name]
        if rows_per_model.empty:
            continue

        # Select relevant grid
        inner_frames = []
        for params in ParameterGrid(grid[model_name]):
            param_hash = hash_dict(params)
            df_ = rows_per_model[rows_per_model["param_hash"] == param_hash]
            inner_frames.append(df_)

        # Select best args
        df_ = pd.concat(inner_frames, ignore_index=True)
        model = MODELS[model_name]
        if isinstance(model, RegressionModel):
            idx = df_.groupby(["model", "n", "s"])["r2_val"].idxmax()
            df_ = df_.loc[idx]
        else:
            idx = df_.groupby(["model", "n", "s"])["acc_val"].idxmax()
            df_ = df_.loc[idx]

        df_.reset_index(drop=True, inplace=True)
        outer_frames.append(df_)

    sc_df = pd.concat(outer_frames, ignore_index=True)
    sc_df.reset_index(inplace=True, drop=True)

    root_path = Path("plots") / path.stem if output is None else output
    root_path.mkdir(exist_ok=True, parents=True)

    sc_df.to_csv(root_path / f"scores_{path.stem}.csv", index=False)

    show_hp = show == "all" or show == "hp"
    show_sc = show == "all" or show == "sc"
    hp_plot(root_path, path.stem, df, grid, show_hp)

    regr_missing = sc_df["acc_val"].isnull()
    sc_df.loc[regr_missing, "acc_val"] = np.max(sc_df[regr_missing]["r2_val"], 0)
    sc_df.loc[regr_missing, "acc_test"] = np.max(sc_df[regr_missing]["r2_test"], 0)
    sc_plot(root_path, path.stem, sc_df, show_sc)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="functions")
    subparsers.required = True
    subparsers.dest = "command"
    run_parser = subparsers.add_parser(
        "run", help="perform estimation on dataset and split"
    )
    datagen_parser = subparsers.add_parser("datagen", help="generate dataset file")
    splitgen_parser = subparsers.add_parser("splitgen", help="generate split file")
    retrieve_parser = subparsers.add_parser("retrieve", help="retrieve results")
    precomp_parser = subparsers.add_parser("precomp", help="precompute kernel matrices")
    parser.set_defaults(
        run=False, retrieve=False, datagen=False, splitgen=False, precomp=False
    )

    run_parser.add_argument("data", type=str, help="dataset file to use")
    run_parser.add_argument(
        "--label", default="default", type=str, help="which label to use"
    )
    run_parser.add_argument(
        "--split", type=str, help="split file to use", required=True
    )
    run_parser.add_argument(
        "--seeds", nargs="+", help="seeds to use", default=[], type=int
    )
    run_parser.add_argument(
        "--samples", nargs="+", help="select a subset of samples", default=[], type=int
    )
    run_parser.add_argument("--grid", type=str, help="grid to use", default="default")
    run_parser.add_argument("--warm", action="store_true", help="warm start")
    run_parser.add_argument("--cache", action="store_true", help="cache")
    run_parser.add_argument(
        "--include",
        nargs="+",
        help="include only the specified models",
        default=None,
        type=str,
    )
    run_parser.add_argument(
        "--exclude",
        nargs="+",
        help="exclude models from computation",
        default=None,
        type=str,
    )
    run_parser.add_argument("--output", type=str, help="write to specified file")
    run_parser.set_defaults(run=True)

    precomp_parser.add_argument("data", type=str, help="dataset file to use")
    precomp_parser.add_argument(
        "--grid", type=str, help="grid to use", default="default"
    )
    precomp_parser.add_argument(
        "--include", nargs="+", help="include only the specified models", default=None
    )
    precomp_parser.add_argument(
        "--exclude", nargs="+", help="exclude models from computation", default=None
    )
    precomp_parser.set_defaults(precomp=True)

    datagen_parser.add_argument(
        "dataset",
        default="mnist",
        type=str,
        help="dataset to use (mnist,fashion,superconductivity,higgs)",
    )
    datagen_parser.add_argument(
        "--method",
        default=None,
        type=str,
        help="dimensionality reduction method (pca,rp,tsne)",
    )
    datagen_parser.add_argument(
        "--components",
        default=2,
        type=int,
        help="number of components used in dimensionality reduction",
    )
    datagen_parser.add_argument(
        "--noise", default=0.0, type=float, help="whether or not to add noise"
    )
    datagen_parser.add_argument(
        "--lbl_noise",
        default=None,
        nargs="+",
        type=float,
        help="add labels with noise given the flip probability",
    )
    datagen_parser.add_argument(
        "--format", default="hdf5", type=str, help="output file format (hdf5,pkl)"
    )
    datagen_parser.set_defaults(datagen=True)

    splitgen_parser.add_argument("data", type=str, help="dataset file to use")
    splitgen_parser.add_argument(
        "--label", default="default", type=str, help="label to use"
    )
    splitgen_parser.add_argument(
        "--seeds", type=int, help="number of seeds to use", required=True
    )
    splitgen_parser.add_argument(
        "--samples", nargs="+", help="list number of samples", required=True, type=int
    )
    splitgen_parser.set_defaults(splitgen=True)

    retrieve_parser.add_argument(
        "path", type=str, help="file/directory containing the results to retrieve"
    )
    retrieve_parser.add_argument(
        "--grid", type=str, help="grid to analyse", default="default"
    )
    retrieve_parser.add_argument(
        "--show", type=str, help="show results (all/hp/sc)", default=None
    )
    retrieve_parser.add_argument(
        "--output", type=str, help="output file location", default=None
    )
    retrieve_parser.add_argument(
        "--include", nargs="+", help="include only the specified models", default=None
    )
    retrieve_parser.add_argument(
        "--exclude", nargs="+", help="exclude models from computation", default=None
    )
    retrieve_parser.set_defaults(retrieve=True)
    args = parser.parse_args()

    if args.run:
        run(
            Path(args.data),
            args.label,
            Path(args.split),
            args.seeds,
            args.samples,
            args.grid,
            args.warm,
            args.cache,
            args.include,
            args.exclude,
            args.output,
        )
    elif args.datagen:
        datagen(
            args.dataset,
            args.method,
            args.components,
            args.noise,
            args.lbl_noise,
            args.format,
        )
    elif args.splitgen:
        splitgen(Path(args.data), args.label, args.seeds, args.samples)
    elif args.retrieve:
        out_path = Path(args.output) if args.output is not None else None
        retrieve(
            Path(args.path), args.grid, out_path, args.show, args.include, args.exclude
        )
    elif args.precomp:
        precomp(Path(args.data), args.grid, args.include, args.exclude)


if __name__ == "__main__":
    main()
