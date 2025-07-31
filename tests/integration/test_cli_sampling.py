# Copyright (c) 2025 takotime808

import pandas as pd
import numpy as np
from typer.testing import CliRunner

from mor_cli.detect_sampling import app as detect_app


def test_cli_detect_sampling(tmp_path):
    df = pd.DataFrame(np.random.rand(30, 3), columns=["a", "b", "c"])
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(detect_app, [str(csv_path)])
    assert result.exit_code == 0
    assert "Detected sampling: LHS" in result.stdout
