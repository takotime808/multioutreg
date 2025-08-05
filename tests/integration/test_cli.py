# Copyright (c) 2025 takotime808

import pytest
from typer.testing import CliRunner

from mor_cli.example import app as example_app
from mor_cli.infer_sampling import app as infer_app


def test_cli_example_echo():
    runner = CliRunner()
    result = runner.invoke(example_app, ["example", "hello"])
    assert result.exit_code == 0
    assert "Response:" in result.stdout


# def test_cli_infer_sampling(tmp_path):
#     csv_path = tmp_path / "rand.csv"
#     import numpy as np
#     import pandas as pd
#     rng = np.random.default_rng(0)
#     df = pd.DataFrame(rng.uniform(size=(20, 2)), columns=["x1", "x2"])
#     df.to_csv(csv_path, index=False)

#     runner = CliRunner()
#     result = runner.invoke(infer_app, ["infer_sampling", str(csv_path)])
#     assert result.exit_code == 0
#     assert "Inferred Method:" in result.stdout