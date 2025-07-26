# Copyright (c) 2025 takotime808

import pytest
import os
import numpy as np
import pandas as pd
from multioutreg.report import generate_report as gr

def test_make_data_shapes():
    X_train, X_test, Y_train, Y_test = gr.make_data()
    assert X_train.shape[1] == 5
    assert Y_train.shape[1] == 3
    assert X_test.shape[1] == 5
    assert Y_test.shape[1] == 3
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0

def test_train_model_fits_and_predicts():
    X_train, X_test, Y_train, Y_test = gr.make_data()
    model = gr.train_model(X_train, Y_train)
    preds = model.predict(X_test)
    assert preds.shape == Y_test.shape

def test_make_plots_outputs(tmp_path):
    X_train, X_test, Y_train, Y_test = gr.make_data()
    model = gr.train_model(X_train, Y_train)
    out_dir = tmp_path / "figs"
    preds, std, metrics_df, paths = gr.make_plots(model, X_test, Y_test, out_dir=out_dir)
    # Check returned arrays
    assert preds.shape == Y_test.shape
    assert std.shape == Y_test.shape
    # Check files exist and are images
    for key, path in paths.items():
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        assert path.endswith('.png')
    # Check metrics_df is a DataFrame
    assert isinstance(metrics_df, pd.DataFrame)

def test_render_report_creates_html(tmp_path):
    # Create a dummy template
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "report_template.html"
    template_file.write_text("""
        <html>
        <body>
            <h1>Report</h1>
            <div>{{ metrics|length }}</div>
            <div>{{ overall.keys()|list }}</div>
            <img src="{{ preds_plot }}">
            <img src="{{ residuals_plot }}">
        </body>
        </html>
    """)
    # Dummy data
    metrics_df = pd.DataFrame([{"metric":"r2", "value":0.9}])
    overall_metrics = {"rmse": 0.2, "mae": 0.1}
    paths = {k: f"{k}.png" for k in ["preds", "residuals", "coverage", "intervals", "metrics"]}
    output_path = tmp_path / "report.html"
    html = gr.render_report(
        metrics_df,
        overall_metrics,
        paths,
        template_dir=str(template_dir),
        template_filename="report_template.html",
        output_path=str(output_path),
    )
    assert output_path.exists()
    text = output_path.read_text()
    assert "Report" in text
    assert "<img src=" in text
    assert str(paths["preds"]) in text

def test_main_runs_and_generates_report(monkeypatch, tmp_path):
    # Patch the template dir and filename to our dummy
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "report_template.html"
    template_file.write_text("""
        <html>
        <body>
            <h1>Test Report</h1>
            <div>{{ metrics|length }}</div>
            <img src="{{ preds_plot }}">
        </body>
        </html>
    """)
    output_path = tmp_path / "test_main_report.html"
    monkeypatch.chdir(tmp_path)
    gr.main(
        template_dir=str(template_dir),
        template_filename="report_template.html",
        output_path=str(output_path),
    )
    assert output_path.exists()
    assert "Test Report" in output_path.read_text()
