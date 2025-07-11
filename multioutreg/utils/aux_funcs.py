# Copyright (c) 2025 takotime808

"""Utility and auxiliary functions, `multioutreg.utils.aux_funcs`, for multioutreg library."""

import subprocess
from terminology import in_blue

def execute_command(command: str) -> subprocess.CompletedProcess:
    """Execute provided command.

    This function exists for sphinx docs demonstration.

    Args:
        command (str): Command to be executed.

    Returns:
        subprocess.CompletedProcess: Executed command details.
    """

    uninstall_command_return = subprocess.run(
        command,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    return uninstall_command_return


def parse_out_response(
    uninstall_command_return: subprocess.CompletedProcess,
) -> str:
    """Parse out response from executed command.

    This function exists for sphinx docs demonstration.

    Args:
        uninstall_command_return (subprocess.CompletedProcess): Executed command details.

    Returns:
        str: Response from executing command.
    """

    # Parse and display results.
    if uninstall_command_return.stderr != b"":
        response = uninstall_command_return.stderr
        resp_text = "Returned Error: "
    else:
        response = uninstall_command_return.stdout.decode("utf-8")
        resp_text = "Response:"
    
    print(in_blue(resp_text).in_bold().underlined())
    print(f"{response}")

    return response
