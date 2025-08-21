# Copyright (c) 2025 takotime808

"""Main CLI script."""


def main():
    """Docstring explaining why typer messes up if imports arent in the fucntion."""
    import sys
    import typer
    from wasabi import msg
    from mor_cli.example import example
    from mor_cli.print_fonts import print_fonts
    from mor_cli.detect_sampling import detect_sampling
    from mor_cli.grid_search import grid_search
    from mor_cli.grid_search_auto_detect import grid_search_auto_detect
    # from .infer_sampling import infer_sampling
    from mor_cli.use_ks_for_sample_bias import compare_sample_methods
    # from .generate_report import generate_report
    from mor_cli.ts_forecast import ts_forecast

    commands = {
        "run_example": example,
        "print_fonts": print_fonts,
        "detect_sampling": detect_sampling,
        "grid_search": grid_search,
        "grid_search_auto_detect": grid_search_auto_detect,
        # "infer_sampling": infer_sampling,
        "compare_sample_methods": compare_sample_methods,
        # "generate_report": generate_report,
        "ts-forecast": ts_forecast,
    }
    if len(sys.argv) == 1:
        msg.info("Available commands", ", ".join(commands), exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = f"multioutreg {command}"
    if command in commands:
        typer.run(commands[command])
    else:
        available = "Available: {}".format(", ".join(commands))
        msg.fail("Unknown Command: {}".format(command), available, exits=1)
