from __future__ import annotations

import os
from pathlib import Path

def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

def drive_mounted() -> bool:
    return Path("/content/drive/MyDrive").exists()

def mount_drive_if_needed() -> None:
    """Mount Google Drive only when running in Colab and only if not already mounted."""
    if not running_in_colab():
        return
    if drive_mounted():
        return
    from google.colab import drive
    drive.mount("/content/drive")

def project_root() -> Path:
    """Return canonical project root depending on environment."""
    if running_in_colab():
        mount_drive_if_needed()
        return Path("/content/drive/MyDrive/Data Science Portfolio/churn-prediction")
    else:
        return Path.home() / "Desktop/data-science-projects/Data-Science-Portfolio/churn-prediction"

# Compute roots via function so mounting is controlled
PROJECT_ROOT = project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

def ensure_dirs() -> None:
    for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def debug_print() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Raw dir exists:", RAW_DIR.exists(), "->", RAW_DIR)
    print("Interim dir exists:", INTERIM_DIR.exists(), "->", INTERIM_DIR)

