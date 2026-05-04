from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/home/user/Documents/BScHons-Computer_Science/ResearchImpl/NewSysDevParadigm/suspicious-activity-detection")

DATASET_EXCEL = DATA_DIR / "datasetInfo.xlsx"
VIDEO_DIR     = DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideos"
GT_SEQUENCES_DIR = DATA_DIR / "surveillanceVideosDataset" / "sequences"

ACTIVITY_LABELS  = ["LPP", "PO", "PW", "PPP", "RK", "FG", "PR", "WL"]
DISTORTION_TYPES = ["Pri", "Exp", "Fo", "ExFo"]
RANDOM_SEED      = 42
N_RANDOM_VIDEOS_PER_GROUP = 5

FRAME_RESIZE = (640, 480)
BLUR_KERNEL  = (5, 5)
BLUR_SIGMA   = 0

MOG2_HISTORY        = 20
MOG2_VAR_THRESHOLD  = 25
MOG2_DETECT_SHADOWS = False
MORPH_KERNEL_SIZE   = (5, 5)
MORPH_CLOSE_ITER    = 2
MORPH_OPEN_ITER     = 1
MIN_CONTOUR_AREA    = 500

MAX_DISAPPEARED    = 30
MAX_TRAJECTORY_LEN = 300
MIN_TRAJECTORY_LEN = 30

LK_WIN_SIZE     = (15, 15)
LK_MAX_LEVEL    = 2
LK_CRITERIA     = (3, 10, 0.03)
MAX_CORNERS     = 200
CORNER_QUALITY  = 0.3
CORNER_MIN_DIST = 7

TRAIN_SIZE = 0.70
VAL_SIZE   = 0.15
TEST_SIZE  = 0.15
CV_FOLDS   = 5

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