"""
Centralised constants and paths.
All path roots are absolute so the project can be launched from any directory.
"""

from pathlib import Path
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent          # project root
DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

# Files and directories 
DATASET_EXCEL    = DATA_DIR / "datasetInfo.xlsx"
VIDEO_DIR        = DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideos"
GT_DIR           = DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideosGT"

# Dataset / sampling
ACTIVITY_LABELS           = ["LPP", "PO", "PW", "PPP", "RK", "FG", "PR", "WL"]
DISTORTION_TYPES          = ["Pri", "Exp", "Fo", "ExFo"]
RANDOM_SEED               = 42
N_RANDOM_VIDEOS_PER_GROUP = 5

# Preprocessing
FRAME_RESIZE = (640, 480)   # (width, height)
BLUR_KERNEL  = (5, 5)
BLUR_SIGMA   = 0

# MOG2 background subtraction
MOG2_HISTORY        = 20
MOG2_VAR_THRESHOLD  = 25
MOG2_DETECT_SHADOWS = False
MORPH_CLOSE_ITER    = 2
MORPH_OPEN_ITER     = 1
MIN_CONTOUR_AREA    = 500   # px²; smaller blobs are ignored

# Tracking
MAX_DISAPPEARED    = 30
MAX_TRAJECTORY_LEN = 300
MIN_TRAJECTORY_LEN = 30   # frames; tracks shorter than this are discarded

# Optical flow (Lucas‑Kanade)
LK_WIN_SIZE     = (15, 15)
LK_MAX_LEVEL    = 2
LK_CRITERIA     = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
MAX_CORNERS     = 200
CORNER_QUALITY  = 0.3
CORNER_MIN_DIST = 7

# Classification
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.15
TEST_SIZE  = 0.15
CV_FOLDS   = 5      # max; clamped when a class has fewer samples

# Display
DISPLAY_EVERY_N_FRAMES = 1
WAIT_MS     = 30
TAIL_LENGTH = 40

TRACK_PALETTE = [
    (255, 100,  60),
    ( 60, 200, 255),
    ( 60, 255, 130),
    (200,  60, 255),
    (255, 220,  60),
    ( 60, 100, 255),
]