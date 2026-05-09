"""
    Set up for the tunable parameter in the project.
"""

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("data")

DATASET_EXCEL    = DATA_DIR / "datasetInfo.xlsx"
VIDEO_DIR        = DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideos"

# AD-SVD ground-truth bounding boxes (VOT format: x,y,w,h per line per frame).
# One .txt file per video, stored at:
#   <GT_DIR>/<VideoStem>.txt
# Coordinates are in the original (un-resized) video resolution and are
# scaled automatically in detector.load_gt_bboxes().
GT_DIR = DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideosGT"

# dataset / sampling 
ACTIVITY_LABELS           = ["LPP", "PO", "PW", "PPP", "RK", "FG", "PR", "WL"]
DISTORTION_TYPES          = ["Pri", "Exp", "Fo", "ExFo"]
RANDOM_SEED               = 42
N_RANDOM_VIDEOS_PER_GROUP = 5

# preprocessing configurations
FRAME_RESIZE = (640, 480)   # (width, height) — all frames resized to this
BLUR_KERNEL  = (5, 5)
BLUR_SIGMA   = 0

# background subtraction (MOG2) configurations
MOG2_HISTORY        = 20
MOG2_VAR_THRESHOLD  = 25
MOG2_DETECT_SHADOWS = False
MORPH_KERNEL_SIZE   = (5, 5)
MORPH_CLOSE_ITER    = 2
MORPH_OPEN_ITER     = 1
MIN_CONTOUR_AREA    = 500   # px² in resized frame; blobs below this are ignored

# tracking configurations
MAX_DISAPPEARED    = 30
MAX_TRAJECTORY_LEN = 300
MIN_TRAJECTORY_LEN = 30     # frames; tracks shorter than this are discarded

# optical flow configurations
LK_WIN_SIZE     = (15, 15)
LK_MAX_LEVEL    = 2
LK_CRITERIA     = (3, 10, 0.03)
MAX_CORNERS     = 200
CORNER_QUALITY  = 0.3
CORNER_MIN_DIST = 7

# classification configurations
TRAIN_SIZE = 0.70   # stratified; train / val / test must sum to 1.0
VAL_SIZE   = 0.15
TEST_SIZE  = 0.15
CV_FOLDS   = 5      # maximum; clamped when a class has fewer samples

# display configurations
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