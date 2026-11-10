#!/usr/bin/env python3
"""Block pushes that would publish strings specific to the local machine.

Scans (1) every text file in the HEAD tree and (2) every line *added* by the
commits about to be pushed (so a path committed and later deleted is still
caught, since it would live on in history).

Identity terms are derived at runtime (username, $HOME, hostname, git email
local-part) so this file itself contains nothing personal and is safe to
commit. Extra private terms can be listed one per line in
``.claude/leak-terms.local`` (gitignored).

Exit 0 = clean, exit 2 = leaks found (Claude Code treats 2 as "block").
Usable as a Claude Code PreToolUse hook or a plain ``.git/hooks/pre-push``.
"""
from __future__ import annotations

import getpass
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

SELF = ".claude/hooks/check_local_leaks.py"
LOCAL_TERMS_FILE = ".claude/leak-terms.local"

# Placeholder home dirs that are fine to appear in docs / CI logs.
_PLACEHOLDERS = r"(?:you|user|username|name|me|runner|example|your[-_]?name|<[^>/]+>|\$USER|\{[^}/]+\})"
GENERIC_PATTERNS = {
    "macOS home path": rf"/Users/(?!{_PLACEHOLDERS}/|Shared/)[A-Za-z0-9._-]+/",
    "Linux home path": rf"/home/(?!{_PLACEHOLDERS}/)[A-Za-z0-9._-]+/",
    "Windows home path": rf"[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)(?!{_PLACEHOLDERS}(?:\\|/))[A-Za-z0-9._-]+",
    "macOS temp dir": r"/(?:private/)?var/folders/[A-Za-z0-9_+-]{2}/",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, errors="replace"
    ).stdout


def identity_terms(root: Path) -> list[str]:
    terms: set[str] = set()
    try:
        terms.add(getpass.getuser())
    except Exception:
        pass
    home = os.path.expanduser("~")
    if home not in ("/", ""):
        terms.add(home)
    host = socket.gethostname().split(".")[0]
    if host and host.lower() not in {"localhost", "local"}:
        terms.add(host)
    try:
        email = git("config", "user.email").strip()
        local = email.split("@")[0].split("+")[-1]
        terms.add(local)
    except Exception:
        pass
    extra = root / LOCAL_TERMS_FILE
    if extra.is_file():
        for line in extra.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.add(line)
    # Very short terms produce noise; require 4+ chars.
    return sorted(t for t in terms if len(t) >= 4)


def build_patterns(terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    pats = [(label, re.compile(rx)) for label, rx in GENERIC_PATTERNS.items()]
    for t in terms:
        pats.append(("local identity term", re.compile(re.escape(t), re.IGNORECASE)))
    return pats


def push_base() -> str | None:
    """Commit the pushed range starts from: upstream, else merge-base w/ default."""
    for ref in ("@{upstream}", "origin/HEAD", "origin/main", "origin/master"):
        try:
            return git("merge-base", "HEAD", ref).strip()
        except subprocess.CalledProcessError:
            continue
    return None


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:8000]


def scan_head(pats) -> list[str]:
    hits: list[str] = []
    entries = git("ls-tree", "-r", "-z", "HEAD").split("\0")
    objs = []
    for e in entries:
        if not e:
            continue
        meta, path = e.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind == "blob" and mode != "160000" and path != SELF:
            objs.append((sha, path))
    if not objs:
        return hits
    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    proc.stdin.write("".join(f"{sha}\n" for sha, _ in objs).encode())
    proc.stdin.close()
    for sha, path in objs:
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)  # trailing newline
        if is_binary(data):
            continue
        text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for label, rx in pats:
                m = rx.search(line)
                if m:
                    hits.append(f"{path}:{lineno}: [{label}] ...{snippet(line, m)}...")
                    break
    proc.wait()
    return hits


def scan_history(pats, base: str | None) -> list[str]:
    if base is None:
        return []
    rng = f"{base}..HEAD"
    out = git("log", "-p", "--no-color", "--no-ext-diff", "--format=@@COMMIT %h", rng)
    hits, commit, path = [], "?", "?"
    for line in out.splitlines():
        if line.startswith("@@COMMIT "):
            commit = line.split()[1]
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and path != SELF:
            for label, rx in pats:
                m = rx.search(line)
                if m:
                    hits.append(f"{commit} {path}: [{label}] ...{snippet(line[1:], m)}...")
                    break
    return hits


def snippet(line: str, m: re.Match[str], width: int = 40) -> str:
    s, e = max(0, m.start() - width), min(len(line), m.end() + width)
    return line[s:e].strip()


PUSH_RX = re.compile(r"\bgit\b(?:\s+-[Cc]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+push\b")


def invoked_for_non_push() -> bool:
    """True when run as a Claude hook for a Bash command that isn't a git push.

    Claude's `if: Bash(git push*)` filter can fail open on compound or
    variable-expanding commands, so re-check the real command here. When run
    as a git pre-push hook, stdin is ref lines (not JSON) and we always scan.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return False
    raw = sys.stdin.read()
    try:
        import json

        payload = json.loads(raw)
    except Exception:
        return False
    if not isinstance(payload, dict) or "tool_input" not in payload:
        return False
    cmd = str(payload.get("tool_input", {}).get("command", ""))
    return not PUSH_RX.search(cmd)


def main() -> int:
    if invoked_for_non_push():
        return 0
    try:
        root = Path(git("rev-parse", "--show-toplevel").strip())
    except Exception:
        return 0  # not a git repo; nothing to check
    os.chdir(root)
    pats = build_patterns(identity_terms(root))
    base = push_base()
    head_hits = scan_head(pats)
    hist_hits = scan_history(pats, base)
    if not head_hits and not hist_hits:
        print("local-leak check: clean", file=sys.stderr)
        return 0
    lines = ["PUSH BLOCKED: local-machine-specific strings found."]
    if head_hits:
        lines.append(f"\nIn files at HEAD ({len(head_hits)}):")
        lines += [f"  {h}" for h in head_hits[:50]]
    if hist_hits:
        lines.append(f"\nAdded by unpushed commits {base[:8] if base else ''}..HEAD ({len(hist_hits)}):")
        lines += [f"  {h}" for h in hist_hits[:50]]
    lines.append(
        "\nFix: scrub the strings (for notebooks, clear outputs or replace absolute paths), "
        "amend/rebase so they're gone from the unpushed commits, then push again."
    )
    print("\n".join(lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
