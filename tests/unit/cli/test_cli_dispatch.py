# Copyright (c) 2026 takotime808

"""The ``multioutreg`` entry point: command table, dispatch and the docs click group."""

import sys

import pytest

import mor_cli

EXPECTED = {
    "run_example", "print_fonts", "detect_sampling", "grid_search",
    "grid_search_auto_detect", "compare_sample_methods", "ts-forecast", "ts-pipeline",
}


def test_commands_table_lists_every_cli():
    commands = mor_cli.commands()
    assert set(commands) == EXPECTED
    assert all(callable(fn) for fn in commands.values())


def test_click_group_registers_every_command():
    import click

    group = mor_cli.click_group()
    assert isinstance(group, click.Group)
    assert set(group.commands) == EXPECTED


def test_docs_cli_exposes_the_click_group():
    from mor_cli import docs_cli

    assert set(docs_cli.cli.commands) == EXPECTED


@pytest.mark.parametrize("argv, code", [([], 1), (["--help"], 0), (["-h"], 0), (["nope"], 1)])
def test_main_without_a_known_command_lists_commands_and_exits(monkeypatch, capsys, argv, code):
    monkeypatch.setattr(sys, "argv", ["multioutreg", *argv])
    with pytest.raises(SystemExit) as exc:
        mor_cli.main()
    assert exc.value.code == code
    out = capsys.readouterr().out
    assert "grid_search" in out and "ts-pipeline" in out
    if argv == ["nope"]:
        assert "Unknown Command: nope" in out


def test_main_dispatches_to_the_named_command(monkeypatch):
    import typer

    ran = []
    monkeypatch.setattr(typer, "run", ran.append)
    monkeypatch.setattr(sys, "argv", ["multioutreg", "grid_search", "data.csv", "x", "y"])
    mor_cli.main()
    assert ran == [mor_cli.commands()["grid_search"]]
    # The command name is consumed; its own arguments are left for typer.
    assert sys.argv == ["multioutreg grid_search", "data.csv", "x", "y"]
