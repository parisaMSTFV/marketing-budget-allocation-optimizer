"""Fail CI when public project files contain common secret or private-system markers."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "generated"}
EXCLUDED_FILES = {Path(__file__).resolve()}

SENSITIVE_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic API key": re.compile(r"(?i)(?:api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]{8,}"),
    "internal connection": re.compile(r"(?i)(?:server|host|database)\s*=\s*[^\s;]{3,}"),
    "private network address": re.compile(
        r"(?i)(?:https?://)?(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2})(?::\d+)?"
    ),
    "internal domain": re.compile(r"(?i)\b[a-z0-9.-]+\.(?:corp|internal|intranet)\b"),
}


def candidate_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.resolve() not in EXCLUDED_FILES
        and path.suffix.lower() in TEXT_SUFFIXES
        and not any(part in EXCLUDED_PARTS for part in path.parts)
    ]


def main() -> None:
    findings: list[str] = []
    for path in candidate_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {label}")
    if findings:
        raise SystemExit("Sensitive-content check failed:\n" + "\n".join(findings))
    print(f"Sensitive-content check passed for {len(candidate_files())} public text files.")


if __name__ == "__main__":
    main()
