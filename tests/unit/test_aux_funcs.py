# Copyright (c) 2025 takotime808

import subprocess

from multioutreg.utils.aux_funcs import execute_command, parse_out_response


def test_execute_and_parse_echo():
    cp = execute_command('echo hello')
    assert isinstance(cp, subprocess.CompletedProcess)
    out = parse_out_response(cp)
    if isinstance(out, (bytes, bytearray)):
        text = out.decode()
    else:
        text = str(out)
    assert "hello" in text