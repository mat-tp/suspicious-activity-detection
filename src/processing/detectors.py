from __future__ import annotations
from .. import paths
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from .preprocessing import *

STRUCT = generate_binary_structure(2, 2)

def make_bbox(x: int, y: int, w: int, h: int) -> BBox:
    cx = int(x + w // 2)
    cy = int(y + h // 2)
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "cx": cx, "cy": cy, "centroid": (cx, cy), "area": int(w * h)}


@register("detector", "mog2")
class MOG2Detector(Detector):
    """
    MOG2 background subtraction.

    Shadow handling (tuning pass)
    ------------------------------
    Previously `detect_shadows` defaulted to False, so OpenCV's shadow
    model never ran and cast shadows were classified as solid foreground
    -- a common source of oversized/merged bounding boxes in outdoor or
    side-lit footage. This version:

      * defaults `detect_shadows=True` so MOG2's own shadow model runs,
      * explicitly separates the mask values MOG2 can produce
        (0 = background, 127 = shadow, 255 = sure foreground) instead of
        `raw_mask > 0`, which treated shadow pixels as foreground and
        silently defeated the shadow model even when it was enabled,
      * exposes `shadow_as_foreground_weight` so shadow pixels can be kept
        only where they adjoin sure foreground (the likely dark side of a
        real object) rather than either "always foreground" or "always
        discarded".

    var_threshold default (tuning pass)
    ------------------------------------
    Previously defaulted to 25, an unexplained deviation from the
    literature. `var_threshold=16` is the value derived analytically in
    the algorithm's originating work (Zivkovic & van der Heijden, 2006,
    "Efficient Adaptive Density Estimation per Image Pixel for the Task
    of Background Subtraction") as 4-sigma, and is the value documented
    by OpenCV itself as the typical setting. This is now the default here
    and throughout the configs. If a project-specific value other than 16
    is ever adopted, it should be backed by a sensitivity sweep on the
    actual dataset (see scripts/mog2_threshold_sweep.py) rather than
    asserted, and the justification should be recorded alongside the
    config that sets it.
    """
    def __init__(self, history=20, var_threshold=16, detect_shadows=True,
                 shadow_value=127, shadow_as_foreground_weight=0.0,
                 morph_close_iter=2, morph_open_iter=1, min_contour_area=500,
                 adaptive=False):   # NEW: adaptive flag
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.shadow_value = shadow_value
        # 0.0 -> shadows fully excluded from foreground (cleanest boxes)
        # 1.0 -> shadows treated exactly like sure foreground (old behaviour)
        # (0,1) -> shadows kept only where adjoining sure foreground
        self.shadow_as_foreground_weight = float(np.clip(shadow_as_foreground_weight, 0.0, 1.0))
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.min_contour_area = min_contour_area
        self.adaptive = adaptive
        self._make_mog2()

    def _make_mog2(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=self.var_threshold, detectShadows=self.detect_shadows)

    def clean_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        sure_fg = raw_mask >= 200  # 255 in OpenCV's shadow-detection convention
        if self.detect_shadows and self.shadow_as_foreground_weight > 0.0:
            shadow = np.isclose(raw_mask, self.shadow_value, atol=20) & ~sure_fg
            if self.shadow_as_foreground_weight >= 1.0:
                binary = sure_fg | shadow
            else:
                sure_fg_dilated = binary_closing(sure_fg, structure=STRUCT, iterations=1)
                binary = sure_fg | (shadow & sure_fg_dilated)
        else:
            binary = sure_fg
        closed = binary_closing(binary, structure=STRUCT, iterations=self.morph_close_iter)
        opened = binary_opening(closed, structure=STRUCT, iterations=self.morph_open_iter)
        return (opened.astype(np.uint8) * 255)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        # Adaptive threshold adjustment if enabled
        if self.adaptive:
            gray = _to_gray(frame)
            brightness = float(gray.mean())
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # Relax threshold for dark or blurry frames
            if brightness < 60 or blur_score < 100:
                self.mog2.setVarThreshold(10)
            else:
                self.mog2.setVarThreshold(self.var_threshold)

        raw_mask = self.mog2.apply(frame)
        mask = self.clean_mask(raw_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections

    def reset(self):
        self._make_mog2()


@register("detector", "adaptive_mog2")
class AdaptiveMOG2Detector(MOG2Detector):
    """
    Quality-guided adaptive MOG2 detector (Group I — v2).

    Design: closed-loop adaptation driven by two evidence sources.

      (a) Mask quality  — scored AFTER running MOG2.
                          Output-quality feedback: reacts to what the
                          segmenter actually produced, not to what we
                          predicted the frame would look like.

      (b) Track quality — derived from the PREVIOUS frame's tracker state
                          (1-frame-lag temporal feedback).
                          Directly targets the true objective (stable
                          trajectories) rather than a proxy (clean masks).

    Per-frame pipeline
    ------------------
    1.  Run MOG2 at the nominal var_threshold.
    2.  Score the foreground mask (multiplicative across three axes):
          coverage      — fg pixel fraction in (COVERAGE_LOW, COVERAGE_HIGH)
          fragmentation — blob count relative to FRAG_MAX
          blob_size     — median blob area / frame_area relative to BLOB_SIZE_REL
                          (resolution-invariant; replaces absolute px² constant)
        → mask_quality ∈ [0, 1]
    3.  Blend with the previous frame's track-quality prior:
          combined = mask_weight * mask_quality
                     + (1 - mask_weight) * _track_quality
    4.  If combined < quality_threshold:
          Interpolate threshold proportionally to the quality deficit:
            alpha      = 1 - combined / quality_threshold
            relaxed_t  = var_threshold
                         - (var_threshold - relaxed_var_threshold) * alpha
            relaxed_t  = clip(relaxed_t, THRESHOLD_FLOOR, var_threshold)
          Re-run MOG2; accept only if the re-scored mask is strictly better
          than the original (acceptance guard — monotonic improvement).

    Multiplicative scoring
    ----------------------
    Using a product (not an average) means that a single failing axis
    collapses the overall score:

        coverage=0.9, fragmentation=0.05, blob_size=0.9  →  score ≈ 0.04

    This is the correct behaviour: one broken dimension should trigger
    adaptation regardless of how good the others look.

    Track-quality prior
    -------------------
    The pipeline runner must call  detector.feedback(active_tracks)
    once per frame, AFTER  tracker.update().  The 1-frame lag is explicit
    and honest; no look-ahead is used.

    Threshold floor
    ---------------
    relaxed_t is clipped to THRESHOLD_FLOOR (15.0) even if
    relaxed_var_threshold is set lower.  This prevents the background
    model from becoming so permissive that foreground floods the mask.

    Paper-defensibility
    -------------------
    The adaptation rule reacts to two forms of evidence (mask quality,
    trajectory health) rather than input proxies (brightness, blur).  The
    graduated interpolation replaces a binary jump with a proportional
    response.  The acceptance guard enforces monotonic improvement.
    Together these make Group I a genuine closed-loop adaptive system.
    """

    COVERAGE_LOW    = 0.0005  # 0.05%  — essentially empty foreground
    COVERAGE_HIGH   = 0.25    # 25%    — background model has collapsed
    FRAG_MAX        = 30      # blob count above this → fragmented / noisy
    BLOB_SIZE_REL   = 0.002   # median blob area / frame_area (resolution-invariant)
    # Hard lower bound on relaxed var_threshold. Must stay below
    # relaxed_var_threshold (default 10.0) or the acceptance-guard
    # interpolation can never actually reach the relaxed value -- with
    # the previous THRESHOLD_FLOOR=15.0 and var_threshold=25 that was
    # fine, but it silently broke once var_threshold dropped to the
    # literature default of 16 (a floor of 15 leaves almost no room to
    # relax). Set below relaxed_var_threshold so the full interpolation
    # range stays usable regardless of the nominal var_threshold.
    THRESHOLD_FLOOR = 8.0

    def __init__(self, history: int = 20, var_threshold: float = 16,
                 detect_shadows: bool = True,
                 morph_close_iter: int = 2, morph_open_iter: int = 1,
                 min_contour_area: int = 500,
                 relaxed_var_threshold: float = 10.0,
                 quality_threshold: float = 0.4,
                 mask_weight: float = 0.7,
                 min_track_len: int = 5,
                 ema_alpha: float = 0.8):
        super().__init__(
            history=history, var_threshold=var_threshold,
            detect_shadows=detect_shadows,
            morph_close_iter=morph_close_iter, morph_open_iter=morph_open_iter,
            min_contour_area=min_contour_area,
            adaptive=False,   # disable parent's input-proxy adaptation
        )
        self.relaxed_var_threshold = relaxed_var_threshold
        self.quality_threshold     = quality_threshold
        self.mask_weight           = float(np.clip(mask_weight, 0.0, 1.0))
        self.min_track_len         = max(min_track_len, 1)
        self.ema_alpha             = float(np.clip(ema_alpha, 0.0, 1.0))
        self._track_quality: float = 1.0   # EMA prior; initialised optimistic
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        """Initialise / clear per-video diagnostic accumulators."""
        self._diag: Dict[str, Any] = {
            "mask_quality_sum":   0.0,
            "track_quality_sum":  0.0,
            "threshold_used_sum": 0.0,
            "n_frames":           0,
            "n_triggered":        0,   # frames where combined < quality_threshold
            "n_rerun_accepted":   0,   # reruns accepted by acceptance guard
        }

    def reset(self) -> None:
        """Reset MOG2 background model, EMA prior, and diagnostic counters."""
        super().reset()
        self._track_quality = 1.0   # restart optimistic each video
        self._reset_diagnostics()

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Return a summary of adaptive behaviour for the current video.

        Keys
        ----
        avg_mask_quality      mean _score_mask() output across all frames
        avg_track_quality     mean EMA track prior sampled each frame
        adaptive_trigger_rate fraction of frames where adaptation was attempted
        avg_threshold_used    mean var_threshold of accepted detections
        rerun_accept_rate     fraction of triggered frames where rerun was kept
        """
        n  = max(self._diag["n_frames"],    1)
        nt = max(self._diag["n_triggered"], 1)
        return {
            "avg_mask_quality":      self._diag["mask_quality_sum"]   / n,
            "avg_track_quality":     self._diag["track_quality_sum"]  / n,
            "avg_threshold_used":    self._diag["threshold_used_sum"] / n,
            "adaptive_trigger_rate": self._diag["n_triggered"] / n,
            "rerun_accept_rate":     self._diag["n_rerun_accepted"] / nt,
            "n_frames":              float(self._diag["n_frames"]),
        }

    # ── Track-quality feedback (called by the pipeline runner) ────────────────

    def feedback(self, tracks: List[Track]) -> None:
        """
        Update the EMA track-quality prior from the current frame's tracker state.

        Must be called AFTER tracker.update() in the main loop (1-frame lag).

        raw = clip(mean_traj_len / min_track_len, 0, 1) x (n_matched / n_total)
        _track_quality <- ema_alpha * _track_quality + (1 - ema_alpha) * raw

        EMA (default alpha=0.8) damps oscillation in both directions:
          a single bad frame decays the prior slowly (0.8^n envelope);
          recovery after a bad patch is equally gradual.
        This reduces the risk of a positive-feedback loop where bad tracking
        triggers more relaxation, which yields noisier masks, worsening tracking.
        """
        if not tracks:
            raw = 0.0
        else:
            traj_lens = [len(t["trajectory"]) for t in tracks]
            len_score = float(np.clip(np.mean(traj_lens) / self.min_track_len, 0.0, 1.0))
            n_total   = len(tracks)
            n_matched = sum(1 for t in tracks if t.get("disappeared", 0) == 0)
            raw       = len_score * (n_matched / n_total)
        self._track_quality = float(self.ema_alpha * self._track_quality
                                    + (1.0 - self.ema_alpha) * raw)

    # ── Mask-quality scoring ──────────────────────────────────────────────────

    def _score_mask(self, mask: np.ndarray, contours) -> float:
        """
        Return a mask quality score in [0, 1].

        Multiplicative across three axes so that any single failing dimension
        collapses the overall score (e.g. 0.9 × 0.05 × 0.9 ≈ 0.04).

        Hard gates (coverage extremes) bypass scoring entirely — these
        represent complete MOG2 failure where no partial score is meaningful.
        """
        frame_area = mask.size           # H × W for a 2-D mask
        fg_px      = int(mask.sum() / 255)
        coverage   = fg_px / frame_area

        # Hard gates — complete failure; return 0 immediately
        if coverage < self.COVERAGE_LOW or coverage > self.COVERAGE_HIGH:
            return 0.0

        valid = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        if not valid:
            return 0.0

        # Coverage score: triangular peak at ~2% foreground
        cov_score  = float(np.clip(1.0 - abs(coverage - 0.02) / 0.23, 0.0, 1.0))

        # Fragmentation score: linear decay above FRAG_MAX blobs
        frag_score = float(np.clip(1.0 - len(valid) / self.FRAG_MAX, 0.0, 1.0))

        # Blob-size score: resolution-invariant — median area as fraction of frame
        areas      = np.array([cv2.contourArea(c) for c in valid])
        rel_area   = float(np.median(areas)) / frame_area
        size_score = float(np.tanh(rel_area / self.BLOB_SIZE_REL))

        # Multiplicative: any single failure collapses the overall score
        return float(np.clip(cov_score * frag_score * size_score, 0.0, 1.0))

    def _detections_from_contours(self, contours) -> List[BBox]:
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections

    def detect(self, frame: np.ndarray) -> List[BBox]:
        # ── Step 1: run at nominal threshold ─────────────────────────────────
        self.mog2.setVarThreshold(self.var_threshold)
        raw_mask = self.mog2.apply(frame)
        mask     = self.clean_mask(raw_mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = self._detections_from_contours(contours)

        # ── Step 2: combined quality score (mask + track prior) ─────────────
        mask_quality  = self._score_mask(mask, contours)
        track_quality = self._track_quality   # EMA prior from previous frame
        combined      = (self.mask_weight * mask_quality
                         + (1.0 - self.mask_weight) * track_quality)

        # Accumulate per-frame diagnostics
        self._diag["mask_quality_sum"]  += mask_quality
        self._diag["track_quality_sum"] += track_quality
        self._diag["n_frames"]          += 1

        if combined >= self.quality_threshold:
            self._diag["threshold_used_sum"] += self.var_threshold
            return detections   # good enough — no adaptation needed

        # ── Step 3: graduated threshold interpolation ─────────────────────
        # combined=0 → alpha=1 (full relaxation); combined→threshold → alpha→0
        self._diag["n_triggered"] += 1
        alpha     = 1.0 - (combined / self.quality_threshold)
        relaxed_t = self.var_threshold - (
            (self.var_threshold - self.relaxed_var_threshold) * alpha
        )
        # Safety floor: prevent the threshold from becoming dangerously permissive
        relaxed_t = float(np.clip(relaxed_t, self.THRESHOLD_FLOOR, self.var_threshold))

        self.mog2.setVarThreshold(relaxed_t)
        raw_mask_r    = self.mog2.apply(frame)
        mask_r        = self.clean_mask(raw_mask_r)
        contours_r, _ = cv2.findContours(
            mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections_r  = self._detections_from_contours(contours_r)

        # ── Step 4: acceptance guard — monotonic improvement only ─────────────
        # Compare mask scores only (track prior unchanged between the two runs)
        quality_r = self._score_mask(mask_r, contours_r)
        self.mog2.setVarThreshold(self.var_threshold)   # always restore

        if quality_r > mask_quality:
            self._diag["n_rerun_accepted"]   += 1
            self._diag["threshold_used_sum"] += relaxed_t
            return detections_r
        self._diag["threshold_used_sum"] += self.var_threshold
        return detections

@register("detector", "ground_truth")
class GroundTruthDetector(Detector):
    def __init__(self, gt_dir=None, frame_resize=(640, 480), fallback_to_mog2=False):
        self.gt_dir = (Path(gt_dir) if gt_dir else paths.DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideosGT")
        self.frame_resize = tuple(frame_resize)
        self.fallback_to_mog2 = fallback_to_mog2
        self._loaded_bboxes: List[Optional[BBox]] = []
        self._frame_idx = 0
        self.current_video = ""

    def load_for_video(self, video_name: str, orig_w: int, orig_h: int) -> bool:
        self.current_video = video_name
        stem = Path(video_name).stem

        # Try multiple naming patterns
        patterns = [
            f"{stem}_gt.txt",
            f"{stem}.txt",
            f"{stem}_groundtruth.txt",
            f"{stem}_bbox.txt",
            f"*{stem}*_gt.txt",
            f"*{stem}*",
        ]
        gt_files = []
        if self.gt_dir.exists():
            for pattern in patterns:
                matches = list(self.gt_dir.glob(pattern))
                if matches:
                    gt_files = matches
                    break
            if not gt_files:
                gt_files = list(self.gt_dir.rglob(f"*{stem}*_gt.txt"))
            if not gt_files:
                gt_files = list(self.gt_dir.rglob(f"*{stem}*"))

        if not gt_files:
            print(f"  [GT] No GT file found for {video_name} (stem={stem})")
            self._loaded_bboxes = []
            return False

        gt_path = gt_files[0]
        try:
            bboxes = []
            with open(gt_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        bboxes.append(None)
                        continue
                    # Split on commas or whitespace
                    parts = re.split(r'[,\s]+', line)
                    if len(parts) >= 4:
                        try:
                            x, y, w, h = map(float, parts[:4])
                            scale_x = self.frame_resize[0] / orig_w
                            scale_y = self.frame_resize[1] / orig_h
                            bboxes.append(make_bbox(
                                int(x * scale_x), int(y * scale_y),
                                int(w * scale_x), int(h * scale_y)
                            ))
                        except ValueError:
                            bboxes.append(None)
                    else:
                        bboxes.append(None)

            self._loaded_bboxes = bboxes
            self._frame_idx = 0
            n_bboxes = len([b for b in bboxes if b is not None])
            print(f"  [GT] Loaded {n_bboxes} bboxes over {len(bboxes)} frames from {gt_path.name}")
            if n_bboxes == 0 and len(bboxes) > 0:
                # Show first line for debugging
                first_line = None
                with open(gt_path) as f2:
                    for line in f2:
                        if line.strip():
                            first_line = line.strip()
                            break
                print(f"  [GT] Debug: first line = '{first_line}' (split into {len(re.split(r'[,\s]+', first_line))} parts)")
            return True

        except Exception as e:
            print(f"  [GT] Failed to parse {gt_path}: {e}")
            self._loaded_bboxes = []
            return False

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if self._frame_idx < len(self._loaded_bboxes):
            bbox = self._loaded_bboxes[self._frame_idx]
            self._frame_idx += 1
            return [bbox] if bbox is not None else []
        self._frame_idx += 1
        return []

    def reset(self):
        self._frame_idx = 0



def _to_gray(frame: np.ndarray) -> np.ndarray:
    """
    Convert *frame* to a single-channel grayscale image.

    The pipeline calls detector.detect(pre_result.gray), which is already a
    1-channel uint8 array.  If someone passes a 3- or 4-channel BGR frame
    instead, this function handles both cases so detectors never crash with
    "Bad number of channels".
    """
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        return frame if frame.ndim == 2 else frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

@register("detector", "dense_flow")
class DenseOpticalFlowDetector(Detector):
    """
    Detects moving regions using Farnebäck dense optical flow magnitude,
    robust to gradual illumination changes (Exposure) and spatial blur (Focus).
    """
    def __init__(
        self,
        magnitude_threshold: float = 2.0,
        min_contour_area: int = 500,
        morph_close_iter: int = 2,
        morph_open_iter: int = 1,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        adaptive_threshold: bool = True,
    ):
        self.magnitude_threshold = magnitude_threshold
        self.min_contour_area = min_contour_area
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.adaptive_threshold = adaptive_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._mag_history: List[float] = []

    def reset(self) -> None:
        self._prev_gray = None
        self._frame_count = 0
        self._mag_history = []

    def _compute_flow_mask(self, gray: np.ndarray) -> np.ndarray:
        # Guard: if shapes differ (e.g. corrupted frame decoded at wrong size),
        # reset prev_gray and return an empty mask rather than crashing.
        if self._prev_gray.shape != gray.shape:
            log.warning(
                "DenseOpticalFlowDetector: shape mismatch prev=%s cur=%s — resetting prev_gray",
                self._prev_gray.shape, gray.shape,
            )
            self._prev_gray = gray
            return np.zeros(gray.shape[:2], dtype=np.uint8)

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, self.poly_n, self.poly_sigma, 0,
            )
        except cv2.error as exc:
            log.warning("DenseOpticalFlowDetector: Farneback failed (%s) — skipping frame", exc)
            self._prev_gray = gray
            return np.zeros(gray.shape[:2], dtype=np.uint8)

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Adaptive threshold: use mean + k*std of recent frames
        if self.adaptive_threshold and len(self._mag_history) >= 5:
            mu = float(np.mean(self._mag_history[-20:]))
            sigma = float(np.std(self._mag_history[-20:]))
            threshold = max(self.magnitude_threshold, mu + 0.5 * sigma)
        else:
            threshold = self.magnitude_threshold

        self._mag_history.append(float(magnitude.mean()))

        binary = (magnitude > threshold).astype(np.uint8) * 255
        closed = binary_closing(binary > 0, structure=STRUCT, iterations=self.morph_close_iter)
        opened = binary_opening(closed, structure=STRUCT, iterations=self.morph_open_iter)
        return (opened.astype(np.uint8) * 255)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        gray = _to_gray(frame)
        self._frame_count += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        mask = self._compute_flow_mask(gray)
        self._prev_gray = gray

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections


@register("detector", "frame_diff")
class AdaptiveFrameDiffDetector(Detector):
    """
    Detects motion via absolute frame difference with adaptive threshold,
    works under heavy blur because it does not maintain a background model.
    """
    def __init__(
        self,
        threshold: int = 25,
        min_contour_area: int = 500,
        morph_close_iter: int = 2,
        morph_open_iter: int = 1,
        noise_percentile: float = 75.0,
        adaptive: bool = True,
    ):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.noise_percentile = noise_percentile
        self.adaptive = adaptive
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def detect(self, frame: np.ndarray) -> List[BBox]:
        gray = _to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        # Guard: mismatched shapes from a corrupted frame — reset and skip.
        if self._prev_gray.shape != gray.shape:
            log.warning(
                "AdaptiveFrameDiffDetector: shape mismatch prev=%s cur=%s — resetting",
                self._prev_gray.shape, gray.shape,
            )
            self._prev_gray = gray
            return []

        try:
            diff = cv2.absdiff(self._prev_gray, gray)
        except cv2.error as exc:
            log.warning("AdaptiveFrameDiffDetector: absdiff failed (%s) — skipping frame", exc)
            self._prev_gray = gray
            return []

        self._prev_gray = gray

        if self.adaptive:
            noise_floor = float(np.percentile(diff, self.noise_percentile))
            threshold = max(self.threshold, int(noise_floor * 1.5))
        else:
            threshold = self.threshold

        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        closed = binary_closing(binary > 0, structure=STRUCT, iterations=self.morph_close_iter)
        opened = binary_opening(closed, structure=STRUCT, iterations=self.morph_open_iter)
        mask = (opened.astype(np.uint8) * 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections


@register("detector", "cascaded_fallback")
class CascadedFallbackDetector(Detector):
    """
    Tries MOG2 first; if insufficient detections in warmup, falls back to
    dense optical flow; if that fails, falls back to frame differencing.
    Guarantees maximum detection coverage without ground truth.
    """
    def __init__(
        self,
        mog2_history: int = 20,
        mog2_var_threshold: float = 16,
        mog2_detect_shadows: bool = True,
        flow_magnitude_threshold: float = 2.0,
        flow_adaptive_threshold: bool = True,
        diff_threshold: int = 25,
        min_contour_area: int = 500,
        morph_close_iter: int = 2,
        morph_open_iter: int = 1,
        warmup_frames: int = 10,
        min_detections_per_warmup: int = 3,
    ):
        self._mog2 = MOG2Detector(
            history=mog2_history,
            var_threshold=mog2_var_threshold,
            detect_shadows=mog2_detect_shadows,
            morph_close_iter=morph_close_iter,
            morph_open_iter=morph_open_iter,
            min_contour_area=min_contour_area,
            adaptive=True,  # enable adaptive within MOG2
        )
        self._flow = DenseOpticalFlowDetector(
            magnitude_threshold=flow_magnitude_threshold,
            min_contour_area=min_contour_area,
            morph_close_iter=morph_close_iter,
            morph_open_iter=morph_open_iter,
            adaptive_threshold=flow_adaptive_threshold,
        )
        self._diff = AdaptiveFrameDiffDetector(
            threshold=diff_threshold,
            min_contour_area=min_contour_area,
            morph_close_iter=morph_close_iter,
            morph_open_iter=morph_open_iter,
        )

        self.warmup_frames = warmup_frames
        self.min_detections_per_warmup = min_detections_per_warmup

        self._frame_count: int = 0
        self._active_detector: str = "mog2"
        self._warmup_detection_count: int = 0
        self._detector_switches: List[Dict] = []

    def reset(self) -> None:
        self._mog2.reset()
        self._flow.reset()
        self._diff.reset()
        self._frame_count = 0
        self._active_detector = "mog2"
        self._warmup_detection_count = 0
        self._detector_switches = []

    def _try_fallback(self, reason: str) -> None:
        switch = {
            "frame": self._frame_count,
            "from":  self._active_detector,
            "reason": reason,
        }
        if self._active_detector == "mog2":
            self._active_detector = "dense_flow"
            log.info("Cascaded detector: MOG2 → Dense Flow at frame %d (%s)", self._frame_count, reason)
        elif self._active_detector == "dense_flow":
            self._active_detector = "frame_diff"
            log.info("Cascaded detector: Dense Flow → Frame Diff at frame %d (%s)", self._frame_count, reason)
        switch["to"] = self._active_detector
        self._detector_switches.append(switch)
        self._warmup_detection_count = 0

    def detect(self, frame: np.ndarray) -> List[BBox]:
        self._frame_count += 1

        # Always feed MOG2 so its background model stays warm
        mog2_dets = self._mog2.detect(frame)

        if self._active_detector == "mog2":
            self._warmup_detection_count += len(mog2_dets)
            if self._frame_count == self.warmup_frames:
                if self._warmup_detection_count < self.min_detections_per_warmup:
                    self._try_fallback("insufficient MOG2 detections in warmup")
                else:
                    return mog2_dets
            return mog2_dets

        if self._active_detector == "dense_flow":
            flow_dets = self._flow.detect(frame)
            self._warmup_detection_count += len(flow_dets)
            if self._frame_count == self.warmup_frames * 2:
                if self._warmup_detection_count < self.min_detections_per_warmup:
                    self._try_fallback("insufficient Dense Flow detections")
            return flow_dets

        # Final fallback: frame differencing
        return self._diff.detect(frame)

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "active_detector":    self._active_detector,
            "detector_switches":  self._detector_switches,
            "n_switches":         len(self._detector_switches),
        }