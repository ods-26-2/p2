from pathlib import Path

import pytest

from helpers.model_package import build_engine
from p2.api import BACKEND_ENV
from p2.manifest import ENGINE_DIR_ENV


@pytest.fixture
def engine_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "engines"
    root.mkdir()
    monkeypatch.setenv(ENGINE_DIR_ENV, str(root))
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    return root


@pytest.fixture
def synth_engine(engine_root: Path, tmp_path: Path) -> Path:
    return build_engine(tmp_path / "pkg", engine_root)
