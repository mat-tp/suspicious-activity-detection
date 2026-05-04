import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure

import settings as cfg

STRUCT = generate_binary_structure(2, 2)

def make_bbox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return {
        "x": x, "y": y, "w": w, "h": h,
        "cx": x + w // 2,
        "cy": y + h // 2,
        "centroid": (x + w // 2, y + h // 2),
        "area": w * h,
    }

class ForegroundDetector:
    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history       = cfg.MOG2_HISTORY,
            varThreshold  = cfg.MOG2_VAR_THRESHOLD,
            detectShadows = cfg.MOG2_DETECT_SHADOWS,
        )

    def clean_mask(self, raw_mask):
        binary = raw_mask > 0
        closed = binary_closing(binary, structure=STRUCT,
                                iterations=cfg.MORPH_CLOSE_ITER)
        opened = binary_opening(closed, structure=STRUCT,
                                iterations=cfg.MORPH_OPEN_ITER)
        return (opened.astype(np.uint8) * 255)

    def apply(self, gray_frame):
        raw_mask   = self.mog2.apply(gray_frame)
        mask       = self.clean_mask(raw_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections = [
            make_bbox(c) for c in contours
            if cv2.contourArea(c) >= cfg.MIN_CONTOUR_AREA
        ]
        return mask, detections

    def reset(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history       = cfg.MOG2_HISTORY,
            varThreshold  = cfg.MOG2_VAR_THRESHOLD,
            detectShadows = cfg.MOG2_DETECT_SHADOWS,
        )