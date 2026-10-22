# Copyright (c) 2025 takotime808

"""Infer the sampling method of a design whose input distributions are known.

Each continuous input is mapped to [0, 1] through the CDF given in the YAML
(probability integral transform). If the YAML is right, the transformed design
is uniform on the unit cube whatever the marginals, and
:func:`multioutreg.sampling.detect.detect_sampling_method` can then test it
against simulated Random / LHS / Sobol designs on known bounds. Separately, a
per-input KS test (with p-values) checks whether the YAML marginals fit at all.

Marginal KS tests alone cannot tell LHS from random sampling: both have the
specified marginals. The difference is in how the points fill the strata.
"""

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows
import typer
import pandas as pd
import numpy as np
import yaml
from typing import Any, Dict, List, Optional
from scipy.stats import kstest, norm, gamma, beta
from scipy.stats import qmc  # for LHS
from itertools import product

from multioutreg.sampling.detect import detect_sampling_method


app = typer.Typer(help="Compare sampling distributions for uploaded data.")


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(csv_path)


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Dictionary parsed from the YAML file.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _stat(value: Any) -> Any:
    """YAML files in the wild spell missing values as the string ``"None"``."""
    return None if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null", "")) else value


def build_bounds_dists(yaml_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construct a list of variable distribution descriptors from YAML config.

    Args:
        yaml_cfg: Parsed YAML configuration dictionary.

    Returns:
        List of dictionaries, each describing one variable's distribution.
    """
    dep_vars = yaml_cfg["dependent_variables"]
    dists = yaml_cfg["distributions"]
    stat1 = yaml_cfg["stat1"]
    stat2 = yaml_cfg["stat2"]
    stat3 = yaml_cfg["stat3"]
    results = []
    for i, var in enumerate(dep_vars):
        distrib = dists[i]
        s1, s2, s3 = stat1[i], stat2[i], stat3[i]
        results.append(dict(name=var, dist=distrib, stat1=_stat(s1), stat2=_stat(s2), stat3=_stat(s3)))
    return results


def _rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _is_categorical(desc: Dict[str, Any]) -> bool:
    return isinstance(desc["dist"], list)


def marginal_cdf(desc: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """CDF of a continuous variable's specified distribution, evaluated at ``x``."""
    x = np.asarray(x, dtype=float)
    if desc["dist"] == "uniform":
        low, high = float(desc["stat1"]), float(desc["stat2"])
        return np.clip((x - low) / (high - low), 0.0, 1.0)
    if desc["dist"] == "normal":
        return norm.cdf(x, loc=desc["stat1"], scale=np.sqrt(desc["stat2"]))
    if desc["dist"] == "gamma":
        return gamma.cdf(x, a=desc["stat3"], loc=desc["stat1"], scale=desc["stat2"])
    if desc["dist"] == "beta":
        loc = desc["stat3"] if desc["stat3"] is not None else 0
        return beta.cdf(x, desc["stat1"], desc["stat2"], loc=loc)
    raise ValueError(f"Unsupported distribution: {desc['dist']}")


def _marginal_ppf(desc: Dict[str, Any], u: np.ndarray) -> np.ndarray:
    if desc["dist"] == "uniform":
        low, high = desc["stat1"], desc["stat2"]
        return u * (high - low) + low
    if desc["dist"] == "normal":
        return norm.ppf(u, loc=desc["stat1"], scale=np.sqrt(desc["stat2"]))
    if desc["dist"] == "gamma":
        return gamma.ppf(u, a=desc["stat3"], loc=desc["stat1"], scale=desc["stat2"])
    if desc["dist"] == "beta":
        loc = desc["stat3"] if desc["stat3"] is not None else 0
        return beta.ppf(u, desc["stat1"], desc["stat2"], loc=loc)
    raise ValueError(f"Unsupported distribution: {desc['dist']}")


def sample_from_dist(desc: Dict[str, Any], n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample from a single distribution based on its descriptor.

    Args:
        desc: Dictionary describing the distribution.
        n: Number of samples.
        rng: Random generator (seed it for reproducible output).

    Returns:
        Array of samples.
    """
    rng = _rng(rng)
    if _is_categorical(desc):
        values = np.array(desc["dist"])
        return values[rng.integers(len(values), size=n)]
    return _marginal_ppf(desc, rng.random(n))


def create_lhs_samples(
    var_descs: List[Dict[str, Any]], n_samples: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate Latin Hypercube Samples (LHS) for given variable descriptors.

    Args:
        var_descs: List of variable distribution descriptors.
        n_samples: Number of samples to generate.
        rng: Random generator (seed it for reproducible output).

    Returns:
        Sampled values as a 2D array.
    """
    lhs_points = qmc.LatinHypercube(d=len(var_descs), seed=_rng(rng)).random(n=n_samples)
    result = []
    for i, desc in enumerate(var_descs):
        if _is_categorical(desc):
            vals = np.array(desc["dist"])
            idxs = (lhs_points[:, i] * len(vals)).astype(int).clip(0, len(vals) - 1)
            result.append(vals[idxs])
        else:
            result.append(_marginal_ppf(desc, lhs_points[:, i]))
    return np.column_stack(result)


def create_grid_samples(
    var_descs: List[Dict[str, Any]], n_samples: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate grid samples for given variable descriptors.

    Each continuous variable gets ``floor(m ** (1/d))`` levels (at least 2),
    where ``d`` is the number of continuous variables and ``m`` is
    ``n_samples`` divided by the number of categorical combinations, placed at
    the centres of equal-probability strata of its distribution. Categorical
    variables use all their values. If the grid has more than ``n_samples``
    points, a random subset is returned.

    Args:
        var_descs: List of variable distribution descriptors.
        n_samples: Desired number of samples.
        rng: Random generator (seed it for reproducible output).

    Returns:
        Sampled values as a 2D array.
    """
    n_cat_combos = int(np.prod([len(d["dist"]) for d in var_descs if _is_categorical(d)]))
    d_cont = sum(not _is_categorical(d) for d in var_descs)
    k = max(2, int(np.floor((max(n_samples, 1) / n_cat_combos) ** (1.0 / max(d_cont, 1)) + 1e-9)))
    centres = (np.arange(k) + 0.5) / k
    grids = []
    for desc in var_descs:
        grids.append(np.array(desc["dist"]) if _is_categorical(desc) else _marginal_ppf(desc, centres))
    grid_points = np.array(list(product(*grids)))
    if len(grid_points) > n_samples:
        idx = _rng(rng).choice(len(grid_points), n_samples, replace=False)
        grid_points = grid_points[idx]
    return grid_points


def create_random_samples(
    var_descs: List[Dict[str, Any]], n_samples: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate random samples for given variable descriptors.

    Args:
        var_descs: List of variable distribution descriptors.
        n_samples: Number of samples to generate.
        rng: Random generator (seed it for reproducible output).

    Returns:
        Sampled values as a 2D array.
    """
    rng = _rng(rng)
    return np.column_stack([sample_from_dist(desc, n_samples, rng) for desc in var_descs])


def marginal_ks_tests(data: pd.DataFrame, var_descs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    KS test of each continuous input against its specified distribution.

    A small p-value means the YAML distribution does not describe that input,
    so the sampling-method test (which relies on it) is not trustworthy. The
    test says nothing about *how* the points were sampled.

    Args:
        data: User data with one column per variable.
        var_descs: List of variable distribution descriptors.

    Returns:
        DataFrame with columns ``variable``, ``distribution``, ``D`` and ``p_value``.
    """
    rows = []
    for desc in var_descs:
        if _is_categorical(desc):
            continue
        x = data[desc["name"]].dropna().to_numpy(dtype=float)
        res = kstest(x, lambda v, desc=desc: marginal_cdf(desc, v))
        rows.append(dict(variable=desc["name"], distribution=desc["dist"], D=res.statistic, p_value=res.pvalue))
    return pd.DataFrame(rows, columns=["variable", "distribution", "D", "p_value"])


def probability_integral_transform(data: pd.DataFrame, var_descs: List[Dict[str, Any]]) -> np.ndarray:
    """Map each continuous input to [0, 1] through its specified CDF (categorical inputs are skipped)."""
    cols = [marginal_cdf(desc, data[desc["name"]].to_numpy(dtype=float))
            for desc in var_descs if not _is_categorical(desc)]
    return np.column_stack(cols) if cols else np.empty((len(data), 0))


@app.command()
def compare_sample_methods(
    csv: str = typer.Option(..., help="CSV file path with user data"),
    yaml_path: str = typer.Option(..., "--yaml", help="YAML config with bounds and distributions"),
    seed: int = typer.Option(0, "--seed", help="Seed for the simulated reference designs"),
    n_sim: Optional[int] = typer.Option(None, "--n-sim", help="Simulated designs per method (default depends on n)"),
) -> None:
    """
    Infer whether the data was sampled by Grid, LHS, Sobol or Random sampling.

    Inputs are mapped to [0, 1] with the YAML distributions' CDFs and compared
    with simulated designs of the same size; per-input KS tests check that the
    YAML distributions fit.
    """
    user_df = load_csv(csv)
    yaml_cfg = load_yaml(yaml_path)
    var_descs = build_bounds_dists(yaml_cfg)
    data = user_df[[d["name"] for d in var_descs]].dropna()

    ks = marginal_ks_tests(data, var_descs)
    if not ks.empty:
        typer.echo("Marginal fit (KS test of each input against its YAML distribution):")
        typer.echo(ks.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        bad = ks.loc[ks["p_value"] < 0.05, "variable"].tolist()
        if bad:
            typer.secho(
                f"Warning: {', '.join(bad)} do not fit their YAML distribution (p < 0.05); "
                "the sampling-method result below may be unreliable.",
                err=True, fg=typer.colors.YELLOW,
            )
    skipped = [d["name"] for d in var_descs if _is_categorical(d)]
    if skipped:
        typer.echo(f"Categorical inputs not used for the sampling-method test: {', '.join(skipped)}")

    U = probability_integral_transform(data, var_descs)
    if U.shape[1] == 0:
        typer.secho("No continuous inputs to test.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    try:
        result = detect_sampling_method(U, bounds=(0.0, 1.0), n_sim=n_sim, random_state=seed)
    except ValueError as exc:
        typer.secho(f"Cannot test sampling method: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if result.p_values:
        typer.echo("\nMonte Carlo p-values: " + ", ".join(f"{m}={p:.3f}" for m, p in result.p_values.items()))
    typer.echo(f"Justification: {result.explanation}")
    if result.method == "Uncertain":
        typer.echo("\nThe uploaded data does not clearly match Grid, LHS, Sobol or Random sampling.")
    else:
        typer.echo(f"\nThe uploaded data most closely matches the '{result.method}' sampling technique.")
