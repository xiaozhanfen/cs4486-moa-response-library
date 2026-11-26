"""Portable paths for the final project. No dependency on historical directories."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def project_path(variable, default):
    value = Path(os.environ.get(variable, default)).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


DATA_DIR = project_path("MOA_DATA_DIR", "data/raw")
RUNS_DIR = project_path("MOA_RUNS_DIR", "runs")
MODELS_DIR = project_path("MOA_MODELS_DIR", "models")
FOLD_FILE = ROOT / "data/folds.csv"
CV_SEED = 42
N_FOLDS = 5
GENE_COLS = [f"g-{i}" for i in range(772)]
CELL_COLS = [f"c-{i}" for i in range(100)]
BIO_COLS = GENE_COLS + CELL_COLS
COND_COLS = ["cp_time", "cp_dose"]
CLIP = 1e-15
