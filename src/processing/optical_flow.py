from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from .preprocessing import *
from .detectors import *
from .trackers import *

class OpticalFlowEstimator:
    def __init__(self, lk_win_size=(15, 15), lk_max_level=2, lk_criteria_eps=0.03,
                 lk_criteria_count=10, max_corners=200, corner_quality=0.3, corner_min_dist=7):
        self.lk_win_size = tuple(lk_win_size)
        self.lk_max_level = lk_max_level
        self.lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_criteria_count, lk_criteria_eps)
        self.corner_params = dict(maxCorners=max_corners, qualityLevel=corner_quality, minDistance=corner_min_dist, blockSize=7)

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> OpticalFlowResult:
        corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.corner_params)
        if corners is None:
            return OpticalFlowResult(0.0, 0.0)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None,
                                                       winSize=self.lk_win_size, maxLevel=self.lk_max_level,
                                                       criteria=self.lk_criteria)
        good_prev = corners[status == 1]
        good_next = next_pts[status == 1]
        if len(good_prev) == 0:
            return OpticalFlowResult(0.0, 0.0)
        flow_vecs = good_next - good_prev
        magnitudes = np.linalg.norm(flow_vecs, axis=1)
        angles_deg = np.degrees(np.arctan2(flow_vecs[:, 1], flow_vecs[:, 0])) % 360
        return OpticalFlowResult(mean_magnitude=float(magnitudes.mean()), mean_angle_deg=float(angles_deg.mean()),
                                 flow_vectors=flow_vecs, prev_corners=good_prev)

    @classmethod
    def from_config(cls, cfg) -> "OpticalFlowEstimator":
        of = cfg.optical_flow
        return cls(lk_win_size=tuple(of.lk_win_size), lk_max_level=of.lk_max_level,
                   lk_criteria_eps=of.lk_criteria_eps, lk_criteria_count=of.lk_criteria_count,
                   max_corners=of.max_corners, corner_quality=of.corner_quality, corner_min_dist=of.corner_min_dist)
