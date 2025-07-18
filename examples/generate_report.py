# Copyright (c) 2025 takotime808

import os
from jinja2 import Environment, FileSystemLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from multioutreg.performance.metrics_generalized_api import get_uq_performance_metrics_flexible
from multioutreg.figures.prediction_plots import plot_predictions_with_error_bars
from multioutreg.figures.residuals import plot_residuals_multioutput
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.uncertainty_toolbox_extension import (
    plot_uct_intervals_ordered_multioutput,
)
from multioutreg.figures.performance_metric_figures import plot_uq_metrics_bar
from multioutreg.utils.surrogate_utils import predict_with_std

def make_data(random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.rand(200, 5)
    Y = np.dot(X, rng.rand(5, 3)) + rng.randn(200, 3) * 0.1
    return train_test_split(X, Y, test_size=0.25, random_state=random_state)


def train_model(X_train, Y_train):
    base = GaussianProcessRegressor(random_state=0)
    model = MultiOutputRegressor(base)
    model.fit(X_train, Y_train)
    return model


def make_plots(model, X_test, Y_test, out_dir="."):
    try:
        preds, std = model.predict(X_test, return_std=True)
    except TypeError:
        # if return_std is not an option
        preds, std = predict_with_std(model, X_test)

    os.makedirs(out_dir, exist_ok=True)

    fig = plot_predictions_with_error_bars(Y_test, preds, std)
    pred_path = os.path.join(out_dir, "predictions.png")
    plt.savefig(pred_path)
    plt.close(fig.figure if hasattr(fig, 'figure') else fig)

    axes = plot_residuals_multioutput(preds, Y_test, savefig=False)
    res_path = os.path.join(out_dir, "residuals.png")
    plt.savefig(res_path)
    plt.close()

    plot_coverage(Y_test, preds, std)
    cov_path = os.path.join(out_dir, "coverage.png")
    plt.savefig(cov_path)
    plt.close()

    plot_uct_intervals_ordered_multioutput(preds, std, Y_test)
    int_path = os.path.join(out_dir, "intervals.png")
    plt.savefig(int_path)
    plt.close()

    metrics_df, _ = get_uq_performance_metrics_flexible(model, X_test, Y_test)
    ax = plot_uq_metrics_bar(metrics_df)
    met_path = os.path.join(out_dir, "metrics.png")
    plt.savefig(met_path)
    plt.close(ax.figure)

    return preds, std, metrics_df, {
        "preds": pred_path,
        "residuals": res_path,
        "coverage": cov_path,
        "intervals": int_path,
        "metrics": met_path,
    }


def render_report(
        metrics_df, # pd.DataFrame
        overall_metrics,
        paths,
        template_dir: str = "../templates",
        template_filename: str = "report_template.html",
        output_path: str = "report.html",
):
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_filename)
    html = template.render(
        metrics=metrics_df.to_dict(orient="records"),
        overall=overall_metrics,
        preds_plot=paths["preds"],
        residuals_plot=paths["residuals"],
        coverage_plot=paths["coverage"],
        intervals_plot=paths["intervals"],
        metrics_plot=paths["metrics"],
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html


def main(
        template_dir: str,
        template_filename: str,
        output_path: str,
):
    X_train, X_test, Y_train, Y_test = make_data()
    model = train_model(X_train, Y_train)
    preds, std, metrics_df, paths = make_plots(model, X_test, Y_test)
    metrics_df2, overall = get_uq_performance_metrics_flexible(model, X_test, Y_test)
    render_report(
        metrics_df=metrics_df2,
        overall_metrics=overall,
        paths=paths,
        template_dir=template_dir,
        template_filename=template_filename,
        output_path=output_path,
    )
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main(
        template_dir="multioutreg/report",
        template_filename = "report_template.html",
        output_path = "outputs/output_from_example_generate_report.html",
    )