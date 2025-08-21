# Copyright (c) 2025 takotime808

import typer
import pandas as pd
import numpy as np
import yaml
from typing import Any, Dict, List, Optional
from scipy.stats import kstest, norm, gamma, beta
from scipy.stats import qmc  # for LHS
from sklearn.preprocessing import MinMaxScaler
from itertools import product


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
        results.append(dict(name=var, dist=distrib, stat1=s1, stat2=s2, stat3=s3))
    return results


def sample_from_dist(desc: Dict[str, Any], n: int) -> np.ndarray:
    """
    Sample from a single distribution based on its descriptor.

    Args:
        desc: Dictionary describing the distribution.
        n: Number of samples.

    Returns:
        Array of samples.
    """
    # Handles one variable: uniform/normal/gamma/beta/categorical
    if isinstance(desc["dist"], list):  # categorical
        values = np.array(desc["dist"])
        idxs = np.random.choice(len(values), size=n, replace=True)
        return values[idxs]
    elif desc["dist"] == "uniform":
        return np.random.uniform(desc["stat1"], desc["stat2"], size=n)
    elif desc["dist"] == "normal":
        return np.random.normal(loc=desc["stat1"], scale=np.sqrt(desc["stat2"]), size=n)
    elif desc["dist"] == "gamma":
        return gamma.rvs(a=desc["stat3"], loc=desc["stat1"], scale=desc["stat2"], size=n)
    elif desc["dist"] == "beta":
        # Assume stat1 and stat2 are alpha and beta params. stat3 = loc, optional.
        a = desc["stat1"]
        b = desc["stat2"]
        loc = desc["stat3"] if desc["stat3"] is not None else 0
        return beta.rvs(a, b, loc=loc, size=n)
    else:
        raise ValueError(f"Unsupported distribution: {desc['dist']}")


def create_lhs_samples(var_descs: List[Dict[str, Any]], n_samples: int) -> np.ndarray:
    """
    Generate Latin Hypercube Samples (LHS) for given variable descriptors.

    Args:
        var_descs: List of variable distribution descriptors.
        n_samples: Number of samples to generate.

    Returns:
        Sampled values as a 2D array.
    """
    # Only handles continuous and integer variables
    num_vars = len(var_descs)
    sampler = qmc.LatinHypercube(d=num_vars)
    lhs_points = sampler.random(n=n_samples)
    result = []
    for i, desc in enumerate(var_descs):
        if isinstance(desc["dist"], list):  # categorical
            vals = np.array(desc["dist"])
            idxs = (lhs_points[:, i] * len(vals)).astype(int).clip(0, len(vals)-1)
            result.append(vals[idxs])
        elif desc["dist"] == "uniform":
            low, high = desc["stat1"], desc["stat2"]
            vals = lhs_points[:, i] * (high - low) + low
            result.append(vals)
        elif desc["dist"] == "normal":
            # In LHS, assign quantiles to normal, then invert CDF
            vals = norm.ppf(lhs_points[:, i], loc=desc["stat1"], scale=np.sqrt(desc["stat2"]))
            result.append(vals)
        elif desc["dist"] == "gamma":
            vals = gamma.ppf(lhs_points[:, i], a=desc["stat3"], loc=desc["stat1"], scale=desc["stat2"])
            result.append(vals)
        elif desc["dist"] == "beta":
            a, b = desc["stat1"], desc["stat2"]
            loc = desc["stat3"] if desc["stat3"] is not None else 0
            vals = beta.ppf(lhs_points[:, i], a, b, loc=loc)
            result.append(vals)
        else:
            raise ValueError(f"LHS not supported for {desc['dist']}")
    return np.column_stack(result)


def create_grid_samples(var_descs: List[Dict[str, Any]], n_samples: int) -> np.ndarray:
    """
    Generate grid samples for given variable descriptors.

    Args:
        var_descs: List of variable distribution descriptors.
        n_samples: Desired number of samples.

    Returns:
        Sampled values as a 2D array.
    """
    # works well for 1d to ~3d, else n_samples needs to be very low
    grids = []
    for desc in var_descs:
        if isinstance(desc["dist"], list):  # categorical
            vals = np.array(desc["dist"])
        elif desc["dist"] == "uniform":
            vals = np.linspace(desc["stat1"], desc["stat2"], int(np.cbrt(n_samples)))
        elif desc["dist"] == "normal":
            # Use central portion of normal: mean +/- 2 sigma
            mean, std = desc["stat1"], np.sqrt(desc["stat2"])
            vals = np.linspace(mean - 2*std, mean + 2*std, int(np.cbrt(n_samples)))
        elif desc["dist"] == "gamma":
            a = desc["stat3"]
            mean = desc["stat1"] + desc["stat2"]*a
            std = np.sqrt(a)*desc["stat2"]
            vals = np.linspace(mean - 2*std, mean + 2*std, int(np.cbrt(n_samples)))
        elif desc["dist"] == "beta":
            a, b = desc["stat1"], desc["stat2"]
            loc = desc["stat3"] if desc["stat3"] is not None else 0
            mean = a/(a+b)
            std = np.sqrt(a*b/((a+b)**2*(a+b+1)))
            vals = np.linspace(mean - 2*std, mean + 2*std, int(np.cbrt(n_samples)))
        else:
            raise ValueError(f"Grid not supported for {desc['dist']}")
        grids.append(vals)
    grid_points = np.array(list(product(*grids)))
    # Subsample if too many grid points
    if len(grid_points) > n_samples:
        idx = np.random.choice(len(grid_points), n_samples, replace=False)
        grid_points = grid_points[idx]
    return grid_points


def create_random_samples(var_descs: List[Dict[str, Any]], n_samples: int) -> np.ndarray:
    """
    Generate random samples for given variable descriptors.

    Args:
        var_descs: List of variable distribution descriptors.
        n_samples: Number of samples to generate.

    Returns:
        Sampled values as a 2D array.
    """
    return np.column_stack([sample_from_dist(desc, n_samples) for desc in var_descs])


def ks_multivariate_test(data: np.ndarray, sampled: np.ndarray) -> float:
    """
    Perform multivariate Kolmogorov-Smirnov test.

    Args:
        data: Original user data (scaled).
        sampled: Sampled data (scaled).

    Returns:
        Sum of KS statistics across dimensions.
    """
    D_sum = 0
    for i in range(data.shape[1]):
        try:
            D, _ = kstest(data[:, i], sampled[:, i])
        except Exception:
            # For categorical, use crude histogram matching
            vals = np.unique(np.concatenate([data[:, i], sampled[:, i]]))
            freq_data = np.array([np.sum(data[:, i] == x)/len(data) for x in vals])
            freq_sampled = np.array([np.sum(sampled[:, i] == x)/len(sampled) for x in vals])
            D = np.max(np.abs(np.cumsum(freq_data) - np.cumsum(freq_sampled)))
        D_sum += D
    return D_sum


@app.command()
def compare_sample_methods(
    csv: str = typer.Option(..., help="CSV file path with user data"),
    yaml_path: str = typer.Option(..., "--yaml", help="YAML config with bounds and distributions"),
    samples: Optional[int] = typer.Option(None, "--samples", help="Number of samples to generate (default: number of CSV rows)")
) -> None:
    """
    Compare sampling methods (LHS, Grid, Random) to determine which best matches the input data.
    """
    user_df = load_csv(csv)
    yaml_cfg = load_yaml(yaml_path)
    dep_vars = yaml_cfg["dependent_variables"]
    var_descs = build_bounds_dists(yaml_cfg)
    user_data = user_df[dep_vars].values
    n_samples = samples if samples is not None else len(user_df)
    scaler = MinMaxScaler()
    user_data_scaled = scaler.fit_transform(user_data.astype(float))

    methods: Dict[str, np.ndarray] = {}
    try:
        methods["LHS"] = create_lhs_samples(var_descs, n_samples)
    except Exception as e:
        typer.secho(f"Could not create LHS samples: {e}", err=True, fg=typer.colors.RED)
    try:
        methods["Grid"] = create_grid_samples(var_descs, n_samples)
    except Exception as e:
        typer.secho(f"Could not create grid samples: {e}", err=True, fg=typer.colors.RED)
    try:
        methods["Random"] = create_random_samples(var_descs, n_samples)
    except Exception as e:
        typer.secho(f"Could not create random samples: {e}", err=True, fg=typer.colors.RED)

    results: Dict[str, float] = {}
    for key, data in methods.items():
        if np.issubdtype(data.dtype, np.number):
            data_scaled = scaler.transform(data)
            results[key] = ks_multivariate_test(user_data_scaled, data_scaled)
        else:
            results[key] = ks_multivariate_test(user_data.astype(str), data.astype(str))

    typer.echo(f"KS D-statistics: {results}")
    best = min(results, key=results.get)
    typer.echo(f"\nThe uploaded data most closely matches the '{best}' sampling technique.")

