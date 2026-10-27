"""Paths, seed and hyperparameters. Every other module reads its settings from here."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"
RESULT_DIR = ROOT / "outputs" / "results"

MODEL_PATH = MODEL_DIR / "iqa_mobilenetv2.keras"
LABEL_THRESHOLDS_PATH = MODEL_DIR / "label_thresholds.json"
DIAGNOSTIC_THRESHOLDS_PATH = MODEL_DIR / "diagnostic_thresholds.json"

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# KonIQ-10k ships MOS on a 0-100 scale in koniq10k_distributions_sets.csv.
# data.load_scores() checks the file and overrides these if a 1-5 file is used.
MOS_MIN = 0.0
MOS_MAX = 100.0

EPOCHS_PHASE1 = 12
EPOCHS_PHASE2 = 10
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
FINE_TUNE_AT = 100
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 3

# Resolution the OpenCV diagnostics are measured at, and the percentiles that
# cut Poor/Average/Good and Low/Normal/High.
DIAGNOSTIC_SIZE = (512, 384)
TERTILES = (33, 67)


def make_dirs():
    for path in [RAW_DIR, SPLIT_DIR, MODEL_DIR, FIGURE_DIR, LOG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
