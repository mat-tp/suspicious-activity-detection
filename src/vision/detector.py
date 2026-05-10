"""
Two detection modes:
  1. MOG2 (default) – background subtraction, morphological cleanup, contour extraction.
  2. Ground‑Truth (GT) – reads AD‑SVD annotation files, scales coordinates to FRAME_RESIZE.

Both modes produce bbox dicts with the schema:
    {"x", "y", "w", "h", "cx", "cy", "centroid", "area"}
"""

import cv2
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure

import config as cfg

STRUCT = generate_binary_structure(2, 2)   # 3×3 full connectivity


def make_bbox(x, y, w, h):
    """Return a detection dict from integer x,y,w,h coordinates."""
    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "cx":       int(x + w // 2),
        "cy":       int(y + h // 2),
        "centroid": (int(x + w // 2), int(y + h // 2)),
        "area":     int(w * h),
    }


def load_gt_bboxes(video_name, orig_w, orig_h):
    """
    Load per‑frame GT bounding boxes from the AD‑SVD annotation file.
    File: <GT_DIR>/<VideoStem>_gt.txt (one line per frame: x,y,w,h).
    Coordinates are scaled from original resolution to cfg.FRAME_RESIZE.
    Returns a list of bbox dicts (one per frame) or [] if the file does not exist.
    """
    stem    = Path(video_name).stem
    gt_path = cfg.GT_DIR / f"{stem}_gt.txt"
    if not gt_path.exists():
        return []

    dst_w, dst_h = cfg.FRAME_RESIZE
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

            sx = int(x * scale_x)
            sy = int(y * scale_y)
            sw = max(1, int(w * scale_x))
            sh = max(1, int(h * scale_y))

            bboxes.append(make_bbox(sx, sy, sw, sh))
    return bboxes


class ForegroundDetector:
    """MOG2 background subtractor with scipy morphological cleanup."""

    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history       = cfg.MOG2_HISTORY,
            varThreshold  = cfg.MOG2_VAR_THRESHOLD,
            detectShadows = cfg.MOG2_DETECT_SHADOWS,
        )

    def clean_mask(self, raw_mask):
        """Binary close → open using scipy.ndimage."""
        binary = raw_mask > 0
        closed = binary_closing(binary, structure=STRUCT,
                                iterations=cfg.MORPH_CLOSE_ITER)
        opened = binary_opening(closed, structure=STRUCT,
                                iterations=cfg.MORPH_OPEN_ITER)
        return (opened.astype(np.uint8) * 255)

    def apply(self, gray_frame):
        """Detect foreground blobs in a resized grayscale frame.
        Returns (mask, detections) where mask is a uint8 binary image (0/255)
        and detections is a list of bbox dicts.
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