# Copyright (c) 2025 takotime808

import pytest
from typer.testing import CliRunner

from mor_cli.example import app as example_app


def test_cli_example_echo():
    runner = CliRunner()
    result = runner.invoke(example_app, ["example", "hello"])
    assert result.exit_code == 0
    assert "Response:" in result.stdout