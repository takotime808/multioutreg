# Copyright (c) 2025 takotime808

"""DoE Pair Plots: Visualization utility for generating PairGrid plots of numerical variables from a DataFrame."""

import base64
import io
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
# from seaborn.axisgrid import PairGrid


def make_doe_plot(df: pd.DataFrame, numeric_cols: List[str]) -> sns.PairGrid:
    """
    Generate a Seaborn PairGrid plot for the selected numeric columns of a DataFrame.
    
    The plot includes:
        - Scatter plots in the upper triangle
        - Kernel density estimates in the lower triangle
        - Histograms with KDE on the diagonal

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        numeric_cols (List[str]): List of column names to include in the pair plot.

    Returns:
        PairGrid: A Seaborn PairGrid object for further customization or rendering.
    """
    sampled_df = df[numeric_cols].sample(min(200, len(df)), random_state=0)

    # Create the PairGrid
    g = sns.PairGrid(sampled_df)

    # Upper triangle: scatterplot
    g.map_upper(sns.scatterplot, s=15)

    # Lower triangle: kdeplot
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues")

    # Diagonal: hist
    g.map_diag(sns.histplot, kde=True)
    return g

INPUT_COLOR, OUTPUT_COLOR = "#1f6fb4", "#c2410c"


def make_pair_plot(
    df: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    fidelity_col: Optional[str] = None,
    layout: str = "full",
    max_rows: int = 1000,
    height: float = 1.6,
    seed: int = 0,
) -> sns.PairGrid:
    """
    Pair plot of inputs and outputs, fast enough to draw on any dataset size.

    The "fast" grid from ``docs/pair_plot_report.md``: scatter above the
    diagonal, 2-D histogram below and histogram on the diagonal, with no KDE, so
    the time depends on the number of panels and not on the number of rows.
    Input axis labels are blue, output labels orange.

    Parameters:
        df (pd.DataFrame): Data holding every column named below.
        input_cols (Sequence[str]): Independent variables, drawn first.
        output_cols (Sequence[str]): Dependent variables.
        fidelity_col (str, optional): Colour points by this column (level 0 is
            the lowest fidelity) instead of plotting it as a variable.
        layout (str): "full" (n x n), "corner" (lower triangle) or "io"
            (outputs vs inputs only, ``n_out x n_in`` panels).
        max_rows (int): Rows above this are sampled (stratified by fidelity).
        height (float): Height of each panel in inches.
        seed (int): Seed for the row sample.

    Returns:
        PairGrid: The drawn grid.
    """
    input_cols, output_cols = list(input_cols), list(output_cols)
    cols = input_cols + [c for c in output_cols if c not in input_cols]
    sub = df[cols + ([fidelity_col] if fidelity_col else [])]
    if len(sub) > max_rows:
        if fidelity_col:  # stratified, so no level disappears
            frac = max_rows / len(sub)
            sub = sub.groupby(fidelity_col, group_keys=False).apply(
                lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed))
        else:
            sub = sub.sample(max_rows, random_state=seed)

    hue_kw = {}
    if fidelity_col:
        levels = sorted(sub[fidelity_col].unique())
        sub = sub.assign(**{fidelity_col: sub[fidelity_col].astype(str)})
        hue_kw = dict(hue=fidelity_col, hue_order=[str(lv) for lv in levels],
                      palette=sns.color_palette("viridis", len(levels)))

    scatter = dict(s=8, alpha=0.6, linewidth=0, rasterized=True)
    if layout == "io":
        g = sns.PairGrid(sub, x_vars=input_cols, y_vars=output_cols, height=height, **hue_kw)
        g.map(sns.scatterplot, **scatter)
    else:
        g = sns.PairGrid(sub, vars=cols, corner=(layout == "corner"),
                         diag_sharey=False, height=height, **hue_kw)
        if layout == "full":
            g.map_upper(sns.scatterplot, **scatter)
        if fidelity_col:
            g.map_lower(sns.scatterplot, **scatter)
            g.map_diag(sns.histplot, element="step", fill=False,
                       common_norm=False, stat="density", bins=20)
        else:
            g.map_lower(sns.histplot, bins=20, cmap="Blues")
            g.map_diag(sns.histplot, bins=20)

    role = {**{c: INPUT_COLOR for c in input_cols}, **{c: OUTPUT_COLOR for c in output_cols}}
    for ax in g.axes.flat:
        if ax is None:
            continue
        ax.tick_params(labelsize=7)
        # Horizontal y labels, so long column names don't run into each other.
        ax.yaxis.label.set(rotation=0, ha="right", va="center")
        for lab in (ax.xaxis.label, ax.yaxis.label):
            lab.set_fontsize(8)
            if lab.get_text() in role:
                lab.set_color(role[lab.get_text()])
                lab.set_fontweight("bold")
    g.figure.legend(
        handles=[Line2D([], [], color=INPUT_COLOR, lw=6, label="input (independent)"),
                 Line2D([], [], color=OUTPUT_COLOR, lw=6, label="output (dependent)")],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    if fidelity_col:
        g.add_legend(title="fidelity")
    return g


def pair_plot_b64(
    df: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    fidelity_col: Optional[str] = None,
    max_rows: int = 1000,
    max_vars: int = 10,
    dpi: int = 80,
) -> Dict[str, Optional[str]]:
    """
    Draw :func:`make_pair_plot` as a base64 PNG, choosing a layout that stays fast.

    Up to ``max_vars`` numeric variables get the full matrix. Above that only
    the outputs-vs-inputs panels are drawn, and if even those number more than
    ``max_vars ** 2`` the plot is skipped. Never raises: failures come back as
    a note, so a caller running this in the background can always show
    something.

    Returns:
        dict: ``img_b64`` (None when skipped or failed), ``layout`` and ``note``.
    """
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    inputs = [c for c in input_cols if c in numeric and c != fidelity_col]
    outputs = [c for c in output_cols if c in numeric and c != fidelity_col and c not in inputs]
    n_vars = len(inputs) + len(outputs)
    if n_vars < 2:
        return {"img_b64": None, "layout": None,
                "note": "Pair plot skipped: it needs at least two numeric columns."}
    if n_vars <= max_vars:
        layout, note = "full", None
    elif inputs and outputs and len(inputs) * len(outputs) <= max_vars ** 2:
        layout = "io"
        note = (f"{n_vars} variables is over the {max_vars}-variable limit for the full "
                "matrix, so only outputs vs inputs are shown.")
    else:
        return {"img_b64": None, "layout": None,
                "note": f"Pair plot skipped: {n_vars} variables, limit {max_vars}."}
    if len(df) > max_rows:
        sampled = f"A random sample of {max_rows:,} of the {len(df):,} rows is shown."
        note = f"{note} {sampled}" if note else sampled

    try:
        g = make_pair_plot(df, inputs, outputs, fidelity_col=fidelity_col,
                           layout=layout, max_rows=max_rows)
        buf = io.BytesIO()
        g.figure.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(g.figure)
    except Exception as exc:  # a picture must never break the caller
        plt.close("all")
        return {"img_b64": None, "layout": layout, "note": f"Could not draw the pair plot: {exc}"}
    return {"img_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "layout": layout, "note": note}
