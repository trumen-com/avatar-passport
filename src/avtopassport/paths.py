"""Repository paths. REPO_ROOT is found by walking up to pyproject.toml (override with AVTO_REPO_ROOT)."""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    env = os.environ.get("AVTO_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"
SYNTHETIC = DATA / "synthetic"
MODELS = REPO_ROOT / "models"
OUTPUTS = REPO_ROOT / "outputs"
