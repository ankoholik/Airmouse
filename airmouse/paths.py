from __future__ import annotations

import os
import sys
from pathlib import Path

def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _install_dir() -> Path:
    # When packaged (PyInstaller), sys.executable points to the installed exe.
    return Path(sys.executable).resolve().parent


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _local_app_data() -> Path:
    # Prefer per-user writeable directory on Windows.
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base)
    return Path.home() / "AppData" / "Local"


APP_NAME = "AirMouse"

# Read-only assets live next to the exe when installed, or in repo during dev.
PROJECT_ROOT = _install_dir() if _is_frozen() else _repo_root()
WEIGHTS_DIR = PROJECT_ROOT / "models"

# Writeable runtime data lives in LocalAppData to avoid Program Files permissions.
RUNTIME_DIR = _local_app_data() / APP_NAME
LOGS_DIR = RUNTIME_DIR / "logs"
DATASETS_DIR = RUNTIME_DIR / "data"


def ensure_runtime_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    # NOTE: weights dir is expected to exist in install dir (bundled with app);
    # create it in dev to make local runs smoother.
    if not _is_frozen():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def model_weights_path() -> Path:
    return WEIGHTS_DIR / "gesture_model.xml"


def app_log_path() -> Path:
    return LOGS_DIR / "airmouse.log"
