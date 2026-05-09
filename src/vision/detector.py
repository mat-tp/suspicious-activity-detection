"""
Two detection modes:
 
  1. MOG2 (default)
     Background subtraction → morphological cleanup → contour extraction.
     Returns one bbox dict per foreground blob per frame.
 
  2. Ground-Truth (GT)
     load_gt_bboxes() reads the AD-SVD annotation file for a video.
     Format: one line per frame, "x,y,w,h" in the ORIGINAL video resolution.
     Coordinates are scaled to the resized frame (cfg.FRAME_RESIZE) so they
     can be fed directly into the tracker alongside MOG2 detections.
 
Both modes produce the same bbox dict schema:
    {"x", "y", "w", "h", "cx", "cy", "centroid", "area"}
so the tracker and feature extractor work identically for both.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure

import config as cfg

STRUCT = generate_binary_structure(2, 2)   # 3×3 full connectivity
 
 
# ── bbox helper (shared by MOG2 and GT) ───────────────────────────────────────
 
def make_bbox(x, y, w, h):
    """Return a detection dict from integer x,y,w,h coordinates."""
    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "cx":       int(x + w // 2),
        "cy":       int(y + h // 2),
        "centroid": (int(x + w // 2), int(y + h // 2)),
        "area":     int(w * h),
    }
 
 
# ── ground-truth loader ───────────────────────────────────────────────────────
 
def load_gt_bboxes(video_name, orig_w, orig_h):
    """
    Load per-frame GT bounding boxes from the AD-SVD annotation file.
 
    The dataset stores one .txt file per video:
        <GT_DIR>/<VideoStem>.txt
    Each line: x,y,w,h  (comma-separated integers, one line per frame)
    Coordinates are in the ORIGINAL video resolution (orig_w × orig_h).
 
    This function scales them to cfg.FRAME_RESIZE so they align with the
    resized frames that the rest of the pipeline operates on.
 
    Parameters
    ----------
    video_name : str — e.g. "0001Pri_IndWL_MQ_C4.mp4"
    orig_w     : int — original video width from cv2.VideoCapture
    orig_h     : int — original video height from cv2.VideoCapture
 
    Returns
    -------
    list of bbox dicts (one per frame), or [] if the file does not exist.
    """
    stem    = Path(video_name).stem
    gt_path = cfg.GT_DIR / f"{stem}.txt"
 
    if not gt_path.exists():
        return []
 
    dst_w, dst_h = cfg.FRAME_RESIZE   # (640, 480)
    scale_x = dst_w / orig_w
    scale_y = dst_h / orig_h
 
    bboxes = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(v) for v in line.replace(" ", ",").split(",")]
            x, y, w, h = parts[0], parts[1], parts[2], parts[3]
 
            # Scale from original resolution to resized frame
            sx = int(x * scale_x)
            sy = int(y * scale_y)
            sw = max(1, int(w * scale_x))
            sh = max(1, int(h * scale_y))
 
            bboxes.append(make_bbox(sx, sy, sw, sh))
 
    return bboxes
 
 
# ── MOG2 detector ─────────────────────────────────────────────────────────────
 
class ForegroundDetector:
    """
    MOG2 background subtractor with scipy morphological cleanup.
    Returns bbox dicts in the same format as load_gt_bboxes().
    """
 
    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history       = cfg.MOG2_HISTORY,
            varThreshold  = cfg.MOG2_VAR_THRESHOLD,
            detectShadows = cfg.MOG2_DETECT_SHADOWS,
        )
 
    def clean_mask(self, raw_mask):
        """Binary close then open via scipy.ndimage."""
        binary = raw_mask > 0
        closed = binary_closing(binary, structure=STRUCT,
                                iterations=cfg.MORPH_CLOSE_ITER)
        opened = binary_opening(closed, structure=STRUCT,
                                iterations=cfg.MORPH_OPEN_ITER)
        return (opened.astype(np.uint8) * 255)
 
    def apply(self, gray_frame):
        """
        Detect foreground blobs in a resized grayscale frame.
 
        Returns
        -------
        mask       : cleaned uint8 binary mask (0/255)
        detections : list of bbox dicts, one per valid contour
        """
        raw_mask    = self.mog2.apply(gray_frame)
        mask        = self.clean_mask(raw_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= cfg.MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return mask, detections
 
    def reset(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history       = cfg.MOG2_HISTORY,
            varThreshold  = cfg.MOG2_VAR_THRESHOLD,
            detectShadows = cfg.MOG2_DETECT_SHADOWS,
        )
 