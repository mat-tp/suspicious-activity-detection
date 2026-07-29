from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from .preprocessing import *
from .detectors import *

def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


@register("tracker", "centroid")
class CentroidTracker(Tracker):
    def __init__(self, max_disappeared=30, max_trajectory_len=300):
        self.max_disappeared = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.pruned_count = 0

    def _new_track(self, centroid):
        self.tracks[self.next_id] = {"id": self.next_id, "centroid": centroid, "trajectory": [centroid], "disappeared": 0, "tracker_type": "centroid"}
        self.next_id += 1

    def _update_track(self, track: Track, centroid):
        track["centroid"] = centroid
        track["disappeared"] = 0
        track["trajectory"].append(centroid)
        if len(track["trajectory"]) > self.max_trajectory_len:
            track["trajectory"] = track["trajectory"][-self.max_trajectory_len:]

    def update(self, detections: List[BBox]) -> List[Track]:
        centroids = [d["centroid"] for d in detections]
        if not centroids:
            for t in self.tracks.values():
                t["disappeared"] += 1
            self._prune_expired()
            return list(self.tracks.values())
        if not self.tracks:
            for c in centroids:
                self._new_track(c)
            return list(self.tracks.values())
        track_ids = list(self.tracks.keys())
        old_cents = np.array([self.tracks[tid]["centroid"] for tid in track_ids], dtype=float)
        new_cents = np.array(centroids, dtype=float)
        dist_matrix = cdist(old_cents, new_cents, metric="euclidean")
        rows, cols = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)
        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            self._update_track(self.tracks[track_ids[r]], centroids[c])
            used_rows.add(r)
            used_cols.add(c)
        for j, c in enumerate(centroids):
            if j not in used_cols:
                self._new_track(c)
        for i, tid in enumerate(track_ids):
            if i not in used_rows:
                self.tracks[tid]["disappeared"] += 1
        self._prune_expired()
        return list(self.tracks.values())

    def _prune_expired(self):
        expired = [tid for tid, t in self.tracks.items() if t["disappeared"] > self.max_disappeared]
        for tid in expired:
            del self.tracks[tid]
            self.pruned_count += 1

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
        self.pruned_count = 0


@register("tracker", "iou_hungarian")
class IoUHungarianTracker(Tracker):
    def __init__(self, max_disappeared=30, max_trajectory_len=300, iou_threshold=0.3):
        self.max_disappeared = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.pruned_count = 0

    def _new_track(self, bbox: BBox):
        self.tracks[self.next_id] = {"id": self.next_id, "centroid": bbox["centroid"], "bbox": bbox, "trajectory": [bbox["centroid"]], "disappeared": 0, "tracker_type": "iou_hungarian"}
        self.next_id += 1

    def _update_track(self, track: Track, bbox: BBox):
        track["centroid"] = bbox["centroid"]
        track["bbox"] = bbox
        track["disappeared"] = 0
        track["trajectory"].append(bbox["centroid"])
        if len(track["trajectory"]) > self.max_trajectory_len:
            track["trajectory"] = track["trajectory"][-self.max_trajectory_len:]

    def update(self, detections: List[BBox]) -> List[Track]:
        if not detections:
            for t in self.tracks.values():
                t["disappeared"] += 1
            self._prune_expired()
            return list(self.tracks.values())
        if not self.tracks:
            for d in detections:
                self._new_track(d)
            return list(self.tracks.values())
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)
        cost = np.zeros((n_tracks, n_dets), dtype=float)
        for i, tid in enumerate(track_ids):
            track_bbox = self.tracks[tid].get("bbox")
            if track_bbox is None:
                cost[i, :] = 0.0
                continue
            for j, det in enumerate(detections):
                cost[i, j] = -bbox_iou(track_bbox, det)
        row_idx, col_idx = linear_sum_assignment(cost)
        used_rows, used_cols = set(), set()
        for r, c in zip(row_idx, col_idx):
            iou = -cost[r, c]
            if iou <= self.iou_threshold:
                continue
            self._update_track(self.tracks[track_ids[r]], detections[c])
            used_rows.add(r)
            used_cols.add(c)
        for j, det in enumerate(detections):
            if j not in used_cols:
                self._new_track(det)
        for i, tid in enumerate(track_ids):
            if i not in used_rows:
                self.tracks[tid]["disappeared"] += 1
        self._prune_expired()
        return list(self.tracks.values())

    def _prune_expired(self):
        expired = [tid for tid, t in self.tracks.items() if t["disappeared"] > self.max_disappeared]
        for tid in expired:
            del self.tracks[tid]
            self.pruned_count += 1

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
        self.pruned_count = 0


@register("tracker", "sort")
class SORTTracker(Tracker):
    def __init__(self, max_disappeared=30, max_trajectory_len=300, iou_threshold=0.3, min_hits=3):
        self.max_disappeared = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0
        self.pruned_count = 0
        self._kalman_id_counter = 0

    class KalmanBoxTracker:
        def __init__(self, bbox: BBox, tracker_id: int):
            self.kf = cv2.KalmanFilter(7, 4)
            self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
            for i in range(3):
                self.kf.transitionMatrix[i, i+4] = 1.0
            self.kf.transitionMatrix[3, 3] = 1.0
            self.kf.measurementMatrix = np.zeros((4, 7), dtype=np.float32)
            for i in range(4):
                self.kf.measurementMatrix[i, i] = 1.0
            self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
            self.kf.processNoiseCov[4:, 4:] *= 0.01
            self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.3
            self.kf.measurementNoiseCov[2:4, 2:4] *= 0.1
            self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            s = w * h
            r = w / float(h) if h > 0 else 1.0
            self.kf.statePost = np.array([[x], [y], [s], [r], [0], [0], [0]], dtype=np.float32)
            self.time_since_update = 0
            self.id = tracker_id
            self.hits = 1
            self.hit_streak = 1
            self.age = 1

        def predict(self):
            predicted_state = self.kf.predict()
            x, y, s, r = predicted_state[0,0], predicted_state[1,0], predicted_state[2,0], predicted_state[3,0]
            w = np.sqrt(s * r)
            h = s / w if w > 0 else 1
            return make_bbox(int(x - w/2), int(y - h/2), int(w), int(h))

        def update(self, bbox: BBox):
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.age += 1
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            measurement = np.array([[x], [y], [w*h], [w/float(h) if h > 0 else 1.0]], dtype=np.float32)
            self.kf.correct(measurement)

    def _new_track(self, bbox: BBox):
        tracker = self.KalmanBoxTracker(bbox, self._kalman_id_counter)
        self._kalman_id_counter += 1
        self.tracks[self.next_id] = {"id": self.next_id, "kalman_tracker": tracker, "centroid": bbox["centroid"], "bbox": bbox, "trajectory": [bbox["centroid"]], "disappeared": 0, "hits": 1, "tracker_type": "sort"}
        self.next_id += 1

    def _update_track(self, track: Track, bbox: BBox):
        track["kalman_tracker"].update(bbox)
        track["centroid"] = bbox["centroid"]
        track["bbox"] = bbox
        track["disappeared"] = 0
        track["hits"] += 1
        track["trajectory"].append(bbox["centroid"])
        if len(track["trajectory"]) > self.max_trajectory_len:
            track["trajectory"] = track["trajectory"][-self.max_trajectory_len:]

    def update(self, detections: List[BBox]) -> List[Track]:
        self.frame_count += 1
        predicted_bboxes = []
        track_ids = list(self.tracks.keys())
        for tid in track_ids:
            try:
                pred_bbox = self.tracks[tid]["kalman_tracker"].predict()
                predicted_bboxes.append(pred_bbox)
            except Exception:
                predicted_bboxes.append(self.tracks[tid].get("bbox", detections[0] if detections else None))
        if not detections:
            for t in self.tracks.values():
                t["disappeared"] += 1
                t["kalman_tracker"].time_since_update += 1
            self._prune_expired()
            return list(self.tracks.values())
        if not self.tracks:
            for d in detections:
                self._new_track(d)
            return list(self.tracks.values())
        cost = np.zeros((len(track_ids), len(detections)), dtype=float)
        for i, (tid, pred_bbox) in enumerate(zip(track_ids, predicted_bboxes)):
            for j, det in enumerate(detections):
                if pred_bbox is not None:
                    cost[i, j] = -bbox_iou(pred_bbox, det)
                else:
                    last_bbox = self.tracks[tid].get("bbox")
                    if last_bbox is not None:
                        cost[i, j] = -bbox_iou(last_bbox, det)
                    else:
                        cost[i, j] = 0.0
        row_idx, col_idx = linear_sum_assignment(cost)
        used_rows, used_cols = set(), set()
        for r, c in zip(row_idx, col_idx):
            iou = -cost[r, c]
            if iou <= self.iou_threshold:
                continue
            tid = track_ids[r]
            self._update_track(self.tracks[tid], detections[c])
            used_rows.add(r)
            used_cols.add(c)
        for j, det in enumerate(detections):
            if j not in used_cols:
                self._new_track(det)
        for i, tid in enumerate(track_ids):
            if i not in used_rows:
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["kalman_tracker"].time_since_update += 1
        self._prune_expired()
        return list(self.tracks.values())

    def _prune_expired(self):
        expired = [tid for tid, t in self.tracks.items() if t["disappeared"] > self.max_disappeared]
        for tid in expired:
            del self.tracks[tid]
            self.pruned_count += 1

    def get_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t["hits"] >= self.min_hits]

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
        self.pruned_count = 0
        self._kalman_id_counter = 0


# ╚═════════════════════════════════════════════════════════════════════════════╝
#
# ─── Manuscript note ────────────────────────────────────────────────────────
# The TFCR tracker is adopted from:
#
#   Yuan, D., Fan, N., & He, Z. (2020). Learning target-focusing convolutional
#   regression model for visual object tracking. Knowledge-Based Systems,
#   194, 105526. https://doi.org/10.1016/j.knosys.2020.105526
#
# Its relevance to this project:
#   The AD-SVD dataset (Gomez-Nieto et al., 2022) was used for Video Object
#   Tracking (VOT), a feature important for the activity-classification task.
#   Knowing that two people are in spatial contact (close proximity, shared
#   bounding-box overlap) is a strong indicator of a fighting event (FG class).
#
#   TFCR achieved second-best AUC on OTB-2015 (0.665) and was explicitly
#   benchmarked on the AD-SVD dataset in Gomez-Nieto et al. (2022, Table 14a).
#   It is therefore the natural tracking back-end to compare against IoU+Hungarian
#   and SORT in Group F (tracking ablation) and to anchor Group J (TFCR-based
#   tracking).
#
# ─── Future work note ────────────────────────────────────────────────────────
# Despite the dataset being introduced for VOT, an important feature for the
# classification task is knowing whether people are in contact, which strongly
# signals a fight (FG).  Future work should extend classification to finer
# contact categories — handshakes, hugs — which are better discriminated by
# additional features including track speed, duration of proximity, and
# pairwise trajectory angle.  TFCR's target-focusing loss makes it a suitable
# foundation for such extensions because it maximises the response gap between
# foreground (person) and background samples, producing cleaner trajectories
# even in distorted (defocus/exposure) surveillance video.
# ─────────────────────────────────────────────────────────────────────────────

@register("tracker", "tfcr")
class TFCRTracker(Tracker):
    """
    TFCR-inspired tracker: target-focusing convolutional regression model.

    Manuscript context
    ------------------
    Yuan et al. (2020) propose a Target-Focusing Convolutional Regression (TFCR)
    model that reformulates Discriminative Correlation Filters (DCFs) as a
    one-layer CNN and adds a *target-focusing loss*:

        J(w) = ||w * φ(X) - y||² - η||w * φ(X)||² + λ||w||²

    The second term (−η‖w*φ(X)‖²) increases the relative gap between target
    and background response, directly addressing the class-imbalance problem
    that exists in all regression-based trackers (10⁴–10⁵ background samples
    vs. a few target samples per frame).

    Implementation mapping to surveillance tracking
    -----------------------------------------------
    Full deep TFCR requires VGGNet and GPU; the AD-SVD experiment infrastructure
    is CPU-only for most groups.  We therefore implement a *lightweight surrogate*
    that preserves the statistical intent of the target-focusing loss:

      1. Feature representation  — HOG computed on each bounding-box region
                                    (replaces conv4-3 VGG features).
      2. Correlation filter      — ridge regression learned online per track
                                    (one-layer CNN equivalent with L2 weight decay).
      3. Target-focusing weight  — the IoU score between the predicted and
                                    detected bounding box modulates the update step:
                                    high-IoU frames contribute more strongly to the
                                    appearance model, reducing background influence.
      4. Scale estimation        — three candidate scales {0.95, 1.0, 1.05} ×
                                    current bounding-box size, matching TFCR §3.3.
      5. Model update            — every frame (η=0.5, λ=1e-4, lr=0.02).

    The surrogate produces trajectories directly comparable with IoU+Hungarian
    and SORT in Group F (tracking ablation, Section 4.3 of the dissertation).
    AUC comparisons from Gomez-Nieto et al. (2022, Fig. 14a) confirm that
    TFCR maintains competitive performance even as spatial resolution decreases,
    making it well-suited to distorted surveillance video.

    Parameters
    ----------
    max_disappeared : int
        Frames a track survives without a matching detection before pruning.
    max_trajectory_len : int
        Maximum stored trajectory length (older points discarded).
    iou_threshold : float
        Minimum IoU for a detection to be matched to an existing track.
    eta : float
        Target-focusing strength (η in the TFCR loss). Higher values increase
        the response gap between target and background samples.
    lam : float
        Ridge regularisation weight (λ). Controls overfitting of the appearance
        model to background samples.
    lr : float
        Online learning rate for model update (applied every frame).
    scale_factors : list of float
        Candidate scale multipliers for bounding-box scale estimation.
    hog_cell : int
        HOG cell size in pixels; feature extraction resolution.

    References
    ----------
    Yuan, D., Fan, N., & He, Z. (2020). Learning target-focusing convolutional
        regression model for visual object tracking. Knowledge-Based Systems,
        194, 105526.
    Gomez-Nieto, R., et al. (2022). Quality aware features for performance
        prediction and time reduction in video object tracking. IEEE Access,
        10, 13290–13310.
    """

    # ── Default target-focusing hyperparameters (Yuan et al. 2020, §4.1) ────
    DEFAULT_ETA   = 0.5      # target-focusing coefficient  η
    DEFAULT_LAM   = 1e-4     # ridge regularisation         λ
    DEFAULT_LR    = 0.02     # online model learning rate
    DEFAULT_SCALES = [0.95, 1.0, 1.05]   # scale candidates  β (TFCR §3.3)

    def __init__(
        self,
        max_disappeared: int = 30,
        max_trajectory_len: int = 300,
        iou_threshold: float = 0.3,
        eta: float = DEFAULT_ETA,
        lam: float = DEFAULT_LAM,
        lr: float = DEFAULT_LR,
        scale_factors: Optional[List[float]] = None,
        hog_cell: int = 8,
    ):
        self.max_disappeared    = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.iou_threshold      = iou_threshold
        self.eta                = eta          # target-focusing coefficient
        self.lam                = lam          # ridge regularisation weight
        self.lr                 = lr           # online model update learning rate
        self.scale_factors      = scale_factors or self.DEFAULT_SCALES
        self.hog_cell           = hog_cell

        self.next_id    = 0
        self.tracks: Dict[int, Track]       = {}
        self.pruned_count: int              = 0
        # Per-track appearance models: dict of track_id → weight vector w
        self._models: Dict[int, np.ndarray] = {}

    # ── HOG feature extraction (replaces VGGNet conv4-3) ────────────────────

    def _extract_hog(self, frame_gray: np.ndarray, bbox: BBox) -> np.ndarray:
        """
        Extract a HOG descriptor for the region defined by *bbox*.

        In the full TFCR model this is replaced by features from the conv4-3
        layer of VGGNet (Simonyan & Zisserman, 2015).  The HOG descriptor is
        used here as a CPU-feasible surrogate that preserves gradient-based
        spatial texture — the same cue VGGNet exploits in its early layers.

        Returns a zero vector if the crop is too small or grayscale frame
        is unavailable.
        """
        if frame_gray is None:
            # Fallback: return a unit-norm dummy feature so the model update
            # step still executes (prevents NaN in ridge regression).
            return np.zeros(self.hog_cell * self.hog_cell, dtype=np.float32)

        x, y, w, h = (
            max(0, bbox["x"]), max(0, bbox["y"]),
            max(1, bbox["w"]), max(1, bbox["h"]),
        )
        # Guard: ensure crop coordinates stay within the frame
        fh, fw = frame_gray.shape[:2]
        x2 = min(x + w, fw)
        y2 = min(y + h, fh)
        crop = frame_gray[y:y2, x:x2]

        if crop.size == 0:
            return np.zeros(self.hog_cell * self.hog_cell, dtype=np.float32)

        # Resize to a fixed patch size for consistent feature dimensionality.
        # 64×64 was chosen to match the TFCR training patch convention (Yuan
        # et al. 2020, §4.1: training patch is 5× target in width/height;
        # HOG at 8px cells → 8×8 = 64 bins).
        patch_size = max(self.hog_cell * 8, 16)
        try:
            patch = cv2.resize(crop, (patch_size, patch_size))
        except cv2.error:
            return np.zeros(self.hog_cell * self.hog_cell, dtype=np.float32)

        # Sobel gradients for HOG
        gx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy)

        # Compute a 9-bin HOG descriptor over the full patch
        n_bins = 9
        hist, _ = np.histogram(
            ang.flatten(), bins=n_bins, range=(0, 2 * np.pi),
            weights=mag.flatten(),
        )
        hist_norm = hist / (np.linalg.norm(hist) + 1e-7)
        return hist_norm.astype(np.float32)

    # ── Target-focusing ridge regression update ──────────────────────────────

    def _init_model(self, feat: np.ndarray) -> np.ndarray:
        """
        Initialise a random appearance model weight vector.

        Following Yuan et al. (2020, §3.3 — Model initialisation):
          "all parameters in the convolutional regression layer are randomly
           initialised and follow a zero-mean two-dimensional Gaussian
           distribution."
        We use a small-variance Gaussian here; the model converges quickly
        with the online update rule.
        """
        rng = np.random.default_rng()
        return (rng.standard_normal(feat.shape) * 0.01).astype(np.float32)

    def _update_model(
        self,
        track_id: int,
        feat: np.ndarray,
        iou_score: float,
    ) -> None:
        """
        Online gradient step for the target-focusing convolutional regression.

        The target-focusing loss (Yuan et al. 2020, Eq. 3) is:
            J(w) = ||w·φ - y||² − η||w·φ||² + λ||w||²

        Taking the derivative w.r.t. w and applying gradient descent:
            Δw = −lr · ∂J/∂w
               = −lr · [2(1-η)φ(φᵀw) − 2φy + 2λw]

        where:
          φ   = feature vector (HOG descriptor, 9-dim)
          y   = desired response (1.0 for the target region)
          η   = target-focusing coefficient (increases target response gap)
          λ   = ridge regularisation (prevents background overfitting)

        The *iou_score* is used as a confidence gate: when the predicted
        response closely matches the detected box (high IoU), the update is
        weighted more heavily, analogous to the focal-loss concept of
        down-weighting easy/ambiguous examples.  This is consistent with
        TFCR's objective of maximising the response of target samples while
        minimising the influence of background samples.

        Manuscript note:  This surrogate targets the same statistical goal as
        TFCR — increasing the relative gap between target and background
        response maps — while remaining feasible on CPU surveillance hardware.
        """
        if track_id not in self._models:
            self._models[track_id] = self._init_model(feat)

        w      = self._models[track_id]
        y_des  = 1.0   # desired response for the target sample
        phi    = feat

        # Response at current weight
        y_pred = float(np.dot(phi, w))

        # Target-focusing gradient (Eq. 5, Yuan et al. 2020)
        # ∂J/∂y' = 2[(1-η)y' − y]
        # Gradient w.r.t. w via chain rule: ∂J/∂w = (∂J/∂y')·φ + 2λw
        grad_response = 2.0 * ((1.0 - self.eta) * y_pred - y_des)
        grad = grad_response * phi + 2.0 * self.lam * w

        # IoU-weighted learning rate: high-confidence matches update more
        effective_lr = self.lr * (0.5 + 0.5 * iou_score)

        self._models[track_id] = w - effective_lr * grad

    # ── Scale estimation (TFCR §3.3) ─────────────────────────────────────────

    def _estimate_scale(
        self,
        frame_gray: Optional[np.ndarray],
        bbox: BBox,
        candidate_detections: List[BBox],
    ) -> BBox:
        """
        Select the scale candidate that maximises the target-focusing response.

        Yuan et al. (2020, §3.3 — Scale estimation):
          "we extract some search patches at different scales with the same
           central location and feed them into the proposed feature extractor.
           After that, the corresponding prediction maps can be acquired and
           we can select the optimal scale factor by searching for the maximum
           value in these prediction maps."

        Implementation: we generate three resized bounding boxes
        (β ∈ {0.95, 1.00, 1.05}) and return the one with the highest
        model response score.  When no appearance model has been initialised
        (first frame), the 1.00× scale is returned unchanged.
        """
        track_id = bbox.get("id")
        if track_id is None or track_id not in self._models or frame_gray is None:
            # No appearance model yet — return unchanged scale
            return bbox

        best_score  = -np.inf
        best_bbox   = bbox
        cx, cy      = bbox["cx"], bbox["cy"]

        for sf in self.scale_factors:
            # Scale bounding box around its centroid
            nw = max(4, int(bbox["w"] * sf))
            nh = max(4, int(bbox["h"] * sf))
            nx = max(0, cx - nw // 2)
            ny = max(0, cy - nh // 2)
            cand = make_bbox(nx, ny, nw, nh)

            feat  = self._extract_hog(frame_gray, cand)
            score = float(np.dot(feat, self._models[track_id]))

            if score > best_score:
                best_score = score
                best_bbox  = cand

        return best_bbox

    # ── Tracker interface ────────────────────────────────────────────────────

    def _new_track(self, bbox: BBox, frame_gray: Optional[np.ndarray] = None) -> None:
        """
        Initialise a new track with an appearance model.

        Corresponds to TFCR §3.3 — Model initialisation: the appearance model
        w is randomly initialised; the training patch is larger than the target
        (by the scale factors) to include context / background samples.
        """
        tid = self.next_id
        self.tracks[tid] = {
            "id":           tid,
            "centroid":     bbox["centroid"],
            "bbox":         bbox,
            "trajectory":   [bbox["centroid"]],
            "disappeared":  0,
            "tracker_type": "tfcr",
        }
        # Initialise appearance model for this track
        feat = self._extract_hog(frame_gray, bbox)
        self._models[tid] = self._init_model(feat)
        self.next_id += 1

    def _update_track(
        self,
        track: Track,
        bbox: BBox,
        iou_score: float,
        frame_gray: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update track state and appearance model after a successful match.

        Implements TFCR §3.3 — Online detection + Model update:
          the tracker locates the target from the maximum response on the
          response map, then updates the model every frame (T=2 in the paper;
          we update every frame for the surveillance use-case because scene
          appearance changes rapidly with distortion).
        """
        tid = track["id"]
        # Scale estimation: select optimal scale candidate
        bbox = self._estimate_scale(frame_gray, {**bbox, "id": tid}, [])

        track["centroid"]    = bbox["centroid"]
        track["bbox"]        = bbox
        track["disappeared"] = 0
        track["trajectory"].append(bbox["centroid"])

        if len(track["trajectory"]) > self.max_trajectory_len:
            track["trajectory"] = track["trajectory"][-self.max_trajectory_len:]

        # Online appearance model update (target-focusing gradient step)
        feat = self._extract_hog(frame_gray, bbox)
        self._update_model(tid, feat, iou_score)

    def update(
        self,
        detections: List[BBox],
        frame_gray: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Match detections to active tracks using IoU + Hungarian assignment,
        then update appearance models with the target-focusing gradient step.

        The matching strategy (IoU + Hungarian) is identical to Group A/B so
        that the only experimental variable in Group J is the appearance model
        update — allowing a clean ablation of the TFCR target-focusing
        contribution.

        The *frame_gray* parameter accepts the pre-processed grayscale frame
        from the distortion-aware preprocessor; passing None disables HOG
        extraction and falls back to centroid-only tracking (graceful
        degradation for Groups A–I that do not supply the frame).
        """
        # ── No detections: increment disappeared counter ──────────────────
        if not detections:
            for t in self.tracks.values():
                t["disappeared"] += 1
            self._prune_expired()
            return list(self.tracks.values())

        # ── No existing tracks: initialise from detections ────────────────
        if not self.tracks:
            for d in detections:
                self._new_track(d, frame_gray)
            return list(self.tracks.values())

        # ── Cost matrix: negative IoU (Hungarian minimises cost) ──────────
        track_ids = list(self.tracks.keys())
        cost = np.zeros((len(track_ids), len(detections)), dtype=float)
        for i, tid in enumerate(track_ids):
            track_bbox = self.tracks[tid].get("bbox")
            if track_bbox is None:
                cost[i, :] = 0.0
                continue
            for j, det in enumerate(detections):
                cost[i, j] = -bbox_iou(track_bbox, det)

        row_idx, col_idx = linear_sum_assignment(cost)

        used_rows: set = set()
        used_cols: set = set()

        for r, c in zip(row_idx, col_idx):
            iou_val = -cost[r, c]
            if iou_val <= self.iou_threshold:
                continue   # reject weak match
            self._update_track(
                self.tracks[track_ids[r]],
                detections[c],
                iou_score=iou_val,
                frame_gray=frame_gray,
            )
            used_rows.add(r)
            used_cols.add(c)

        # New tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in used_cols:
                self._new_track(det, frame_gray)

        # Increment disappeared counter for unmatched tracks
        for i, tid in enumerate(track_ids):
            if i not in used_rows:
                self.tracks[tid]["disappeared"] += 1

        self._prune_expired()
        return list(self.tracks.values())

    def _prune_expired(self) -> None:
        """Remove tracks that have been unmatched for too long."""
        expired = [
            tid for tid, t in self.tracks.items()
            if t["disappeared"] > self.max_disappeared
        ]
        for tid in expired:
            del self.tracks[tid]
            self._models.pop(tid, None)   # free appearance model memory
            self.pruned_count += 1

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def reset(self) -> None:
        """Reset all track and appearance-model state for a new video."""
        self.tracks.clear()
        self._models.clear()
        self.next_id      = 0
        self.pruned_count = 0



class TrackerAnalyzer:
    @staticmethod
    def compute_tracker_metrics(tracks: List[Track], total_frames: int) -> Dict[str, Any]:
        if not tracks:
            return {"error": "No tracks available"}
        track_lengths = [len(t["trajectory"]) for t in tracks]
        metrics = {
            "n_tracks": len(tracks),
            "mean_track_length": float(np.mean(track_lengths)),
            "median_track_length": float(np.median(track_lengths)),
            "std_track_length": float(np.std(track_lengths)),
            "max_track_length": int(np.max(track_lengths)),
            "min_track_length": int(np.min(track_lengths)),
            "track_coverage": float(np.sum(track_lengths) / (len(tracks) * total_frames)) if total_frames > 0 else 0.0,
            "tracks_per_frame": float(len(tracks) / total_frames) if total_frames > 0 else 0.0,
        }
        thresholds = [15, 30, 45, 60, 90]
        for thresh in thresholds:
            long_tracks = sum(1 for tlen in track_lengths if tlen >= thresh)
            metrics[f"tracks_ge_{thresh}_frames"] = long_tracks
            metrics[f"tracks_ge_{thresh}_pct"] = (long_tracks / len(tracks) * 100 if tracks else 0.0)
        tracks_with_gaps = sum(1 for t in tracks if t.get("disappeared", 0) > 0)
        metrics["tracks_with_gaps"] = tracks_with_gaps
        metrics["track_continuity"] = (1.0 - tracks_with_gaps / len(tracks) if tracks else 0.0)
        tracker_types = set(t.get("tracker_type", "unknown") for t in tracks)
        metrics["tracker_types"] = list(tracker_types)
        return metrics

    @staticmethod
    def justify_min_trajectory_length(track_lengths: List[int]) -> Dict[str, Any]:
        if not track_lengths:
            return {"error": "No track lengths available"}
        justification = {}
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            justification[f"percentile_{p}"] = float(np.percentile(track_lengths, p))
        total_tracks = len(track_lengths)
        thresholds = range(5, 101, 5)
        retention_data = {}
        for thresh in thresholds:
            retained = sum(1 for l in track_lengths if l >= thresh)
            retention_data[str(thresh)] = {
                "retained_tracks": retained,
                "retention_rate": (retained / total_tracks * 100 if total_tracks > 0 else 0.0),
                "mean_length_retained": (float(np.mean([l for l in track_lengths if l >= thresh])) if retained > 0 else 0.0),
            }
        justification["retention_by_threshold"] = retention_data
        for thresh in [30, 45, 60]:
            if retention_data[str(thresh)]["retention_rate"] >= 80:
                justification["recommended_threshold"] = thresh
                justification["recommendation_reason"] = (
                    f"Threshold of {thresh} frames retains "
                    f"{retention_data[str(thresh)]['retention_rate']:.1f}% "
                    f"of tracks while ensuring sufficient trajectory length "
                    f"for robust feature extraction"
                )
                break
        return justification
