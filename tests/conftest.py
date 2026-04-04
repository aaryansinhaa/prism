from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_generated_uploads_after_tests() -> Iterator[None]:
    yield

    repo_root = Path(__file__).resolve().parents[1]
    uploads_dir = repo_root / "model_store" / "uploads"

    if not uploads_dir.exists():
        return

    for child in uploads_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
