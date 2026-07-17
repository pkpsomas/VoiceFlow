"""Minimal .env loader (no external dependency).

Loads KEY=VALUE pairs into os.environ without overriding variables that are
already set. Values may carry inline comments ("soniox   # note") and simple
single/double quoting.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith(("'", '"')) and value.endswith(value[0]) and len(value) >= 2:
        return value[1:-1]
    # Strip inline comments (whitespace followed by #) from unquoted values.
    for idx, ch in enumerate(value):
        if ch == "#" and (idx == 0 or value[idx - 1].isspace()):
            value = value[:idx]
            break
    return value.strip()


def load_dotenv(paths: Optional[Iterable[Path]] = None) -> int:
    """Load the first .env file found; returns number of variables set."""
    candidates = list(paths) if paths else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",  # repo root in source runs
    ]
    loaded = 0
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                key = key.strip()
                value = _parse_value(raw)
                if key and value and key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
            logger.info(f"Loaded {loaded} env vars from {path}")
            return loaded
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    return loaded
