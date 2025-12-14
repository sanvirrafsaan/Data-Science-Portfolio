from pathlib import Path

# --------------------------------------------------
# Environment detection
# --------------------------------------------------

def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------
# Project root
# --------------------------------------------------

if running_in_colab():
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/drive/MyDrive/Data Science Portfolio/churn-prediction")
else:
    PROJECT_ROOT = Path.home() / "Desktop/data-science-projects/Data-Science-Portfolio/churn-prediction"


# --------------------------------------------------
# Directory structure
# --------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# --------------------------------------------------
# Ensure directories exist
# --------------------------------------------------

def ensure_dirs():
    for p in [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Debug helper
# --------------------------------------------------

def debug_print():
    print("Project root:", PROJECT_ROOT)
    print("Raw data:", list(RAW_DIR.iterdir()) if RAW_DIR.exists() else "Missing")
    print("Interim data:", list(INTERIM_DIR.iterdir()) if INTERIM_DIR.exists() else "Missing")

