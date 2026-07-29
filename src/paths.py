from __future__ import annotations
from .common import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR     = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"
PLOTS_DIR       = PROJECT_ROOT / "plots"

for _d in [CONFIGS_DIR, EXPERIMENTS_DIR, OUTPUTS_DIR, DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
