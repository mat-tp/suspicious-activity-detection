from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from .preprocessing import *
from .detectors import *
from .trackers import *
from .optical_flow import *
from .features import *

class DetectionSource(Enum):
    MOG2 = "mog2"
    GROUND_TRUTH = "ground_truth"

class PipelineBuilder:
    @staticmethod
    def build_preprocessor(cfg) -> Preprocessor:
        pre = cfg.preprocessing
        enabled = pre.get("enabled", True)
        if not enabled:
            return build("preprocessor", "passthrough")
        ptype = pre.type
        distortion_aware_flag = pre.get("distortion_aware", True)
        if ptype == "distortion_aware" and not distortion_aware_flag:
            ptype = "standard"
        if ptype == "standard":
            return build("preprocessor", "standard",
                         frame_resize=tuple(pre.frame_resize),
                         blur_kernel=tuple(pre.blur_kernel),
                         blur_sigma=pre.blur_sigma)
        if ptype == "distortion_aware":
            da = pre.distortion_aware_params
            return build("preprocessor", "distortion_aware",
                         frame_resize=tuple(pre.frame_resize),
                         blur_kernel=tuple(pre.blur_kernel),
                         blur_sigma=pre.blur_sigma,
                         blur_threshold=da.blur_threshold,
                         exposure_low=da.exposure_low,
                         exposure_high=da.exposure_high,
                         enhancement_for_blur=da.enhancement_for_blur,
                         enhancement_for_exposure=da.enhancement_for_exposure,
                         unsharp_sigma=da.unsharp_sigma,
                         unsharp_strength=da.unsharp_strength,
                         clahe_clip_limit=da.clahe_clip_limit,
                         clahe_tile_grid=tuple(da.clahe_tile_grid),
                         bilateral_d=da.bilateral_d,
                         bilateral_sigma_color=da.bilateral_sigma_color,
                         bilateral_sigma_space=da.bilateral_sigma_space,
                         use_multi_metric=da.get("use_multi_metric", True),
                         gamma_correction=da.get("gamma_correction", False),
                         gamma_value=da.get("gamma_value", 1.2),
                         contrast_stretching=da.get("contrast_stretching", False))
        raise ValueError(f"Unknown preprocessor type: {ptype}")

    @staticmethod
    def build_detector(cfg, detection_source: Optional[DetectionSource] = None) -> Detector:
        dtype = detection_source.value if detection_source else cfg.detection.type
        if dtype == "mog2":
            mog2_cfg = cfg.detection.mog2
            return build("detector", "mog2",
                         history=mog2_cfg.history,
                         var_threshold=mog2_cfg.var_threshold,
                         detect_shadows=mog2_cfg.detect_shadows,
                         morph_close_iter=cfg.detection.morphology.close_iter,
                         morph_open_iter=cfg.detection.morphology.open_iter,
                         min_contour_area=cfg.detection.min_contour_area,
                         adaptive=mog2_cfg.get("adaptive", False))
        if dtype == "adaptive_mog2":
            mog2_cfg = cfg.detection.mog2
            return build("detector", "adaptive_mog2",
                         history=mog2_cfg.history,
                         var_threshold=mog2_cfg.var_threshold,
                         detect_shadows=mog2_cfg.detect_shadows,
                         morph_close_iter=cfg.detection.morphology.close_iter,
                         morph_open_iter=cfg.detection.morphology.open_iter,
                         min_contour_area=cfg.detection.min_contour_area,
                         relaxed_var_threshold=mog2_cfg.get("relaxed_var_threshold", 10.0),
                         quality_threshold=mog2_cfg.get("quality_threshold", 0.4),
                         mask_weight=mog2_cfg.get("mask_weight", 0.7),
                         min_track_len=mog2_cfg.get("min_track_len", 5),
                         ema_alpha=mog2_cfg.get("ema_alpha", 0.8))
        if dtype == "ground_truth":
            gt = cfg.detection.ground_truth
            return build("detector", "ground_truth",
                         gt_dir=cfg.paths.gt_dir,
                         frame_resize=tuple(cfg.preprocessing.frame_resize),
                         fallback_to_mog2=gt.fallback_to_mog2)
        # NEW: support other detector types
        if dtype == "dense_flow":
            flow_cfg = cfg.detection.dense_flow
            return build("detector", "dense_flow",
                         magnitude_threshold=flow_cfg.magnitude_threshold,
                         min_contour_area=flow_cfg.min_contour_area,
                         morph_close_iter=flow_cfg.morph_close_iter,
                         morph_open_iter=flow_cfg.morph_open_iter,
                         pyr_scale=flow_cfg.pyr_scale,
                         levels=flow_cfg.levels,
                         winsize=flow_cfg.winsize,
                         iterations=flow_cfg.iterations,
                         poly_n=flow_cfg.poly_n,
                         poly_sigma=flow_cfg.poly_sigma,
                         adaptive_threshold=flow_cfg.adaptive_threshold)
        if dtype == "cascaded_fallback":
            cascade_cfg = cfg.detection.cascaded_fallback
            return build("detector", "cascaded_fallback",
                         mog2_history=cascade_cfg.mog2_history,
                         mog2_var_threshold=cascade_cfg.mog2_var_threshold,
                         flow_magnitude_threshold=cascade_cfg.flow_magnitude_threshold,
                         flow_adaptive_threshold=cascade_cfg.flow_adaptive_threshold,
                         diff_threshold=cascade_cfg.diff_threshold,
                         min_contour_area=cascade_cfg.min_contour_area,
                         morph_close_iter=cascade_cfg.morph_close_iter,
                         morph_open_iter=cascade_cfg.morph_open_iter,
                         warmup_frames=cascade_cfg.warmup_frames,
                         min_detections_per_warmup=cascade_cfg.min_detections_per_warmup)
        raise ValueError(f"Unknown detector type: {dtype}")

    @staticmethod
    def build_tracker(cfg) -> Tracker:
        ttype = cfg.tracking.type
        if ttype == "centroid":
            return build("tracker", "centroid",
                         max_disappeared=cfg.tracking.max_disappeared,
                         max_trajectory_len=cfg.tracking.max_trajectory_len)
        if ttype == "iou_hungarian":
            return build("tracker", "iou_hungarian",
                         max_disappeared=cfg.tracking.max_disappeared,
                         max_trajectory_len=cfg.tracking.max_trajectory_len,
                         iou_threshold=cfg.tracking.iou_hungarian.iou_threshold)
        if ttype == "sort":
            sort_cfg = cfg.tracking.get("sort", {})
            return build("tracker", "sort",
                         max_disappeared=cfg.tracking.max_disappeared,
                         max_trajectory_len=cfg.tracking.max_trajectory_len,
                         iou_threshold=sort_cfg.get("iou_threshold", 0.3),
                         min_hits=sort_cfg.get("min_hits", 3))
        # ── TFCR tracker (Yuan et al. 2020) ──────────────────────────────────
        # Manuscript reference: Yuan, D., Fan, N., & He, Z. (2020).
        # Learning target-focusing convolutional regression model for visual
        # object tracking. Knowledge-Based Systems, 194, 105526.
        # Used in Group J and in Group F tracking ablation for comparison
        # against centroid, IoU+Hungarian, and SORT.
        if ttype == "tfcr":
            tfcr_cfg = cfg.tracking.get("tfcr", {})
            return build(
                "tracker", "tfcr",
                max_disappeared=cfg.tracking.max_disappeared,
                max_trajectory_len=cfg.tracking.max_trajectory_len,
                iou_threshold=tfcr_cfg.get("iou_threshold", 0.3),
                eta=tfcr_cfg.get("eta", TFCRTracker.DEFAULT_ETA),
                lam=tfcr_cfg.get("lam", TFCRTracker.DEFAULT_LAM),
                lr=tfcr_cfg.get("lr",  TFCRTracker.DEFAULT_LR),
                scale_factors=tfcr_cfg.get(
                    "scale_factors", TFCRTracker.DEFAULT_SCALES
                ),
                hog_cell=tfcr_cfg.get("hog_cell", 8),
            )
        raise ValueError(f"Unknown tracker type: {ttype}")

    @staticmethod
    def build_optical_flow(cfg) -> OpticalFlowEstimator:
        return OpticalFlowEstimator.from_config(cfg)

    @classmethod
    def build_all(cls, cfg, detection_source: Optional[DetectionSource] = None):
        return (cls.build_preprocessor(cfg), cls.build_detector(cfg, detection_source),
                cls.build_tracker(cfg), cls.build_optical_flow(cfg))



# ── Video-level temporal context features ────────────────────────────────────

def extract_temporal_context(tracks: List[Track], total_frames: int = 0) -> FeatureDict:
    """
    Compute video-level features that capture *when* and *where* tracks were
    active — information lost by per-track mean/max/std aggregation.

    These features are merged directly into the video feature vector alongside
    the aggregated track features so that models can distinguish activities
    characterised by concurrent multi-person motion (FG, PPP) from those that
    are typically single-person (LPP, PO).

    Args:
        tracks:       Active tracks returned by the tracker after video processing.
        total_frames: Number of frames processed (used to normalise coverage).

    Returns:
        FeatureDict with keys prefixed ``temporal_``.
    """
    if not tracks:
        return {}

    # Approximate start / end frame from trajectory length (we don't store
    # absolute frame indices, so we use cumulative order as a proxy)
    traj_lengths = [len(t["trajectory"]) for t in tracks]
    n_tracks = len(tracks)

    # Temporal spread: std of trajectory lengths approximates spread of when
    # tracks were active (long tracks span more of the video)
    temporal_spread = float(np.std(traj_lengths)) if n_tracks > 1 else 0.0

    # Spatial spread: std of all trajectory centroids across all tracks
    all_centroids = np.array(
        [pt for t in tracks for pt in t["trajectory"]], dtype=float
    )
    spatial_spread = float(np.std(all_centroids, axis=0).mean()) if len(all_centroids) > 1 else 0.0

    # Track overlap: naive estimate — count pairs where both are long enough
    # to plausibly co-exist (both ≥ 15 frames)
    long_tracks = [t for t in tracks if len(t["trajectory"]) >= 15]
    n_long = len(long_tracks)
    overlap_count = max(0, n_long * (n_long - 1) // 2)

    # Track density: tracks per frame
    tracks_per_frame = (sum(traj_lengths) / total_frames) if total_frames > 0 else 0.0

    return {
        "temporal_n_tracks": float(n_tracks),
        "temporal_n_long_tracks": float(n_long),
        "temporal_spread": temporal_spread,
        "temporal_spatial_spread": spatial_spread,
        "temporal_overlap_count": float(overlap_count),
        "temporal_tracks_per_frame": tracks_per_frame,
        "temporal_mean_track_len": float(np.mean(traj_lengths)) if traj_lengths else 0.0,
        "temporal_max_track_len": float(max(traj_lengths)) if traj_lengths else 0.0,
    }



def _pool_stat(arr: np.ndarray, stat: str) -> float:
    if arr is None or len(arr) == 0:
        return 0.0
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min":  float(np.min(arr)),  "max": float(np.max(arr))}.get(stat, 0.0)


def extract_diagnostic_features(frame_diagnostics: List[dict]) -> FeatureDict:
    """16 pooled diagnostic features — always available, no detection needed.
    Research: Gillani et al. (2025), Gomez-Nieto et al. (2022), Bai & Reibman (2010)."""
    _Z: FeatureDict = {k: 0.0 for k in [
        "diag_laplacian_mean","diag_laplacian_std","diag_laplacian_min","diag_laplacian_max",
        "diag_tenengrad_mean","diag_tenengrad_std","diag_tenengrad_min","diag_tenengrad_max",
        "diag_brenner_mean",  "diag_brenner_std",  "diag_brenner_min",  "diag_brenner_max",
        "diag_brightness_mean","diag_brightness_var","diag_dynamic_range","diag_low_contrast_ratio",
    ]}
    if not frame_diagnostics:
        return _Z
    laps    = np.array([d.get("laplacian_var", d.get("blur_score", 0.0)) for d in frame_diagnostics], dtype=float)
    tenens  = np.array([d.get("tenengrad",  0.0) for d in frame_diagnostics], dtype=float)
    brens   = np.array([d.get("brenner",    0.0) for d in frame_diagnostics], dtype=float)
    brights = np.array([d.get("brightness", 0.0) for d in frame_diagnostics], dtype=float)
    return {
        "diag_laplacian_mean":     _pool_stat(laps,    "mean"),
        "diag_laplacian_std":      _pool_stat(laps,    "std"),
        "diag_laplacian_min":      _pool_stat(laps,    "min"),
        "diag_laplacian_max":      _pool_stat(laps,    "max"),
        "diag_tenengrad_mean":     _pool_stat(tenens,  "mean"),
        "diag_tenengrad_std":      _pool_stat(tenens,  "std"),
        "diag_tenengrad_min":      _pool_stat(tenens,  "min"),
        "diag_tenengrad_max":      _pool_stat(tenens,  "max"),
        "diag_brenner_mean":       _pool_stat(brens,   "mean"),
        "diag_brenner_std":        _pool_stat(brens,   "std"),
        "diag_brenner_min":        _pool_stat(brens,   "min"),
        "diag_brenner_max":        _pool_stat(brens,   "max"),
        "diag_brightness_mean":    _pool_stat(brights, "mean"),
        "diag_brightness_var":     float(np.var(brights)) if len(brights) > 1 else 0.0,
        "diag_dynamic_range":      float(brights.max()-brights.min()) if len(brights) > 0 else 0.0,
        "diag_low_contrast_ratio": float(np.mean(laps < 50.0)) if len(laps) > 0 else 0.0,
    }


def extract_global_flow_features(gray_frames: List[np.ndarray],
                                  flow_step: int = 5) -> FeatureDict:
    """22 Farnebäck dense-flow features — always available, no detection needed.
    Research: Mauthner et al. (2009), Xiao et al. (2016), Saleh et al. (2022)."""
    _Z: FeatureDict = {
        "flow_mean_mag":0.0,"flow_std_mag":0.0,"flow_max_mag":0.0,
        "flow_prop_moving":0.0,"flow_dir_entropy":0.0,"flow_jerk":0.0,
        **{f"flow_hof_mean_{i}":0.0 for i in range(8)},
        **{f"flow_hof_std_{i}": 0.0 for i in range(8)},
    }
    if len(gray_frames) < 2:
        return _Z
    def _u8g(f):
        if f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.shape[2] >= 3 else f[:,:,0]
        return f.astype(np.uint8)
    TH, TW = 120, 160
    mags, props, hofs = [], [], []
    prev_s = cv2.resize(_u8g(gray_frames[0]), (TW, TH))
    for idx in range(1, len(gray_frames), max(1, flow_step)):
        curr_s = cv2.resize(_u8g(gray_frames[idx]), (TW, TH))
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_s, curr_s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        except cv2.error:
            prev_s = curr_s; continue
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(float(mag.mean()))
        props.append(float((mag > 1.0).mean()))
        if mag.sum() > 0:
            h, _ = np.histogram(ang[mag > 1.0], bins=8, range=(0, 2*np.pi))
            h = h.astype(float) / (h.sum()+1e-7)
        else:
            h = np.zeros(8)
        hofs.append(h); prev_s = curr_s
    if not mags:
        return _Z
    ma = np.array(mags); ha = np.array(hofs) if hofs else np.zeros((1,8))
    mh = ha.mean(0); sh = ha.std(0) if len(ha) > 1 else np.zeros(8)
    p  = mh / (mh.sum()+1e-7)
    ent = float(-np.sum(p[p > 0]*np.log(p[p > 0]+1e-12)))
    res: FeatureDict = {
        "flow_mean_mag":    float(ma.mean()), "flow_std_mag":      float(ma.std()),
        "flow_max_mag":     float(ma.max()),  "flow_prop_moving":  float(np.mean(props)),
        "flow_dir_entropy": ent,              "flow_jerk":         float(np.mean(np.abs(np.diff(ma)))) if len(ma) > 1 else 0.0,
    }
    for i in range(8):
        res[f"flow_hof_mean_{i}"] = float(mh[i])
        res[f"flow_hof_std_{i}"]  = float(sh[i])
    return res


def extract_appearance_features(gray_frames: List[np.ndarray]) -> FeatureDict:
    """34 LBP+HOG appearance features pooled over frames — always available.
    Research: Mauthner et al. (2009), Csurka et al. (2018), Arias & Millán (2018)."""
    _Z: FeatureDict = {
        **{f"app_lbp_mean_{i}":0.0 for i in range(10)},
        **{f"app_lbp_std_{i}": 0.0 for i in range(10)},
        "app_grad_energy_mean":0.0,"app_grad_energy_std":0.0,
        **{f"app_hog_mean_{i}":0.0 for i in range(6)},
        **{f"app_hog_std_{i}": 0.0 for i in range(6)},
    }
    if not gray_frames:
        return _Z
    T = (64, 64)
    def _lbp(g):
        off = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        code = np.zeros(g.shape, dtype=np.uint8)
        for b,(dy,dx) in enumerate(off):
            s = np.roll(np.roll(g,dy,0),dx,1)
            code |= ((g >= s).astype(np.uint8) << b)
        trans = np.zeros(g.shape, dtype=int)
        for b in range(8):
            trans += ((code>>b)&1)^((code>>((b+1)%8))&1)
        unif = trans <= 2
        bc   = sum(((code>>b)&1) for b in range(8))
        h    = np.zeros(10, dtype=float)
        for k in range(9): h[k] = np.sum((bc==k)&unif)
        h[9] = np.sum(~unif)
        return h / (h.sum()+1e-7)
    lh, ge, hh = [], [], []
    for frame in gray_frames:
        g   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        g64 = cv2.resize(g, T)
        lh.append(_lbp(g64))
        gx  = cv2.Sobel(g64.astype(np.float32), cv2.CV_32F, 1, 0)
        gy  = cv2.Sobel(g64.astype(np.float32), cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(np.abs(gy), np.abs(gx))
        ge.append(float(mag.mean()))
        h6, _ = np.histogram(ang.flatten(), bins=6, range=(0, np.pi), weights=mag.flatten())
        hh.append(h6 / (h6.sum()+1e-7))
    la = np.array(lh); ha = np.array(hh); ga = np.array(ge)
    res: FeatureDict = {
        "app_grad_energy_mean": float(ga.mean()),
        "app_grad_energy_std":  float(ga.std()) if len(ga) > 1 else 0.0,
    }
    for i in range(10):
        res[f"app_lbp_mean_{i}"] = float(la.mean(0)[i])
        res[f"app_lbp_std_{i}"]  = float(la.std(0)[i] if len(la) > 1 else 0.0)
    for i in range(6):
        res[f"app_hog_mean_{i}"] = float(ha.mean(0)[i])
        res[f"app_hog_std_{i}"]  = float(ha.std(0)[i] if len(ha) > 1 else 0.0)
    return res


def extract_video_metadata_features(n_frames: int, fps: float,
                                     duration_sec: float) -> FeatureDict:
    """3 temporal scalars appended to every group's feature vector (A–I).
    After normalisation these are contextual cues, not confounds.
    Research: Sargano et al. (2017) — use all available temporal cues."""
    return {
        "meta_num_frames":   float(n_frames),
        "meta_duration_sec": float(duration_sec),
        "meta_fps":          float(fps),
    }


def blend_trajectory_features(track_features: List[FeatureDict],
                               aggregator: "FeatureAggregator"
                               ) -> Tuple[FeatureDict, int, bool]:
    """Aggregate trajectory features; return (dict, n_tracks, indicator_present)."""
    if not track_features:
        return {}, 0, False
    return aggregator.aggregate(track_features) or {}, len(track_features), True



class ExplainabilityVisualizer:
    """
    Renders an annotated diagnostic video overlay showing optical flow arrows,
    bounding boxes, trajectory tails, and a live feature dashboard sidebar.
    Uses only OpenCV drawing (no Matplotlib) for real-time performance.
    """

    def __init__(self, output_path: str, fps: float = 10.0, frame_size: tuple = (640, 480)):
        self.dashboard_width = 300
        self.out_w = frame_size[0] + self.dashboard_width
        self.out_h = frame_size[1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (self.out_w, self.out_h))
        rng = np.random.default_rng(42)
        self.colors = rng.integers(60, 255, (200, 3), dtype=np.uint8)

    # ── Optical flow arrows ───────────────────────────────────────────────────
    def draw_optical_flow(self, frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
        """Draws Farnebaeck dense optical flow as a sparse grid of arrows."""
        h, w = frame.shape[:2]
        ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
        ys = ys.ravel().astype(int)
        xs = xs.ravel().astype(int)
        # clamp to valid range after any resize mismatch
        ys = np.clip(ys, 0, flow.shape[0] - 1)
        xs = np.clip(xs, 0, flow.shape[1] - 1)
        fx = flow[ys, xs, 0]
        fy = flow[ys, xs, 1]
        mag = np.hypot(fx, fy)
        mask = mag > 1.5
        vis = frame.copy()
        for x1, y1, dvx, dvy in zip(xs[mask], ys[mask], fx[mask], fy[mask]):
            x2, y2 = int(x1 + dvx), int(y1 + dvy)
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 230, 230), 1, tipLength=0.35)
        return vis

    # ── Bounding boxes + trajectory tails ────────────────────────────────────
    def overlay_tracks(self, frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        vis = frame.copy()
        for track in tracks:
            tid = track.get("id", 0)
            color = self.colors[tid % len(self.colors)].tolist()

            bbox = track.get("bbox")
            if bbox:
                x, y, w, h = int(bbox.get("x", 0)), int(bbox.get("y", 0)), \
                              int(bbox.get("w", 0)), int(bbox.get("h", 0))
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis, f"ID:{tid}", (x, max(y - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            traj = track.get("trajectory", [])
            if len(traj) > 1:
                pts = np.array(traj, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=2)
            if traj:
                cx, cy = int(traj[-1][0]), int(traj[-1][1])
                cv2.circle(vis, (cx, cy), 5, color, -1)

        return vis

    # ── Side-panel dashboard ──────────────────────────────────────────────────
    def build_dashboard(self, tracks: List[dict], diagnostics: dict) -> np.ndarray:
        dash = np.zeros((self.out_h, self.dashboard_width, 3), dtype=np.uint8)
        W = self.dashboard_width

        def put(text, y, color=(200, 200, 200), scale=0.4, thickness=1):
            cv2.putText(dash, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness, cv2.LINE_AA)

        y = 24
        put("EXPLAINABILITY", y, color=(255, 255, 255), scale=0.55, thickness=2)
        y += 4
        cv2.line(dash, (6, y), (W - 6, y), (180, 180, 180), 1)
        y += 18

        put("[Quality Prior]", y, color=(80, 220, 80), scale=0.42, thickness=1)
        y += 18
        for k, v in diagnostics.items():
            label = k[:18]
            val = f"{v:.2f}" if isinstance(v, float) else str(v)[:10]
            put(f"  {label}: {val}", y)
            y += 16
            if y > self.out_h - 80:
                break

        y += 8
        cv2.line(dash, (6, y), (W - 6, y), (100, 100, 100), 1)
        y += 14

        put("[Active Track]", y, color=(80, 220, 220), scale=0.42, thickness=1)
        y += 18

        if tracks:
            longest = max(tracks, key=lambda t: len(t.get("trajectory", [])))
            tid = longest.get("id", "?")
            traj = longest.get("trajectory", [])
            n_pts = len(traj)

            put(f"  Track ID : {tid}", y); y += 15
            put(f"  Path pts : {n_pts}", y); y += 15

            # Instantaneous speed (px/frame) between last two points
            if n_pts >= 2:
                p1, p2 = traj[-2], traj[-1]
                speed = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                put(f"  Speed    : {speed:.1f} px/fr", y); y += 15

            # Trajectory bounding box displacement
            if n_pts >= 2:
                xs_t = [p[0] for p in traj]
                ys_t = [p[1] for p in traj]
                disp = float(np.hypot(xs_t[-1] - xs_t[0], ys_t[-1] - ys_t[0]))
                put(f"  Displace : {disp:.1f} px", y); y += 15

        y += 6
        put(f"Tracks visible: {len(tracks)}", y, color=(255, 200, 80), scale=0.42)

        return dash

    # ── Composite and write ───────────────────────────────────────────────────
    def write_frame(self, frame_bgr: np.ndarray, tracks: List[dict],
                    diagnostics: dict, flow: Optional[np.ndarray] = None):
        vis = frame_bgr.copy()

        # Resize vis to match declared output dimensions (guard against mismatch)
        target_h, target_w = self.out_h, self.out_w - self.dashboard_width
        if vis.shape[0] != target_h or vis.shape[1] != target_w:
            vis = cv2.resize(vis, (target_w, target_h))

        if flow is not None:
            vis = self.draw_optical_flow(vis, flow)

        vis = self.overlay_tracks(vis, tracks)
        dashboard = self.build_dashboard(tracks, diagnostics)
        final = np.hstack((vis, dashboard))
        self.writer.write(final)

    def release(self):
        self.writer.release()


class PipelineRunner:
    def __init__(self, preprocessor: Preprocessor, detector: Detector,
                 tracker: Tracker, optical_flow: Optional[OpticalFlowEstimator] = None,
                 feature_extractors: Optional[List[FeatureExtractor]] = None,
                 aggregator: Optional[FeatureAggregator] = None,
                 min_trajectory_len: int = 30,
                 use_dense_trajectories: bool = False,
                 enable_multifamily: bool = False):
        self.preprocessor = preprocessor
        self.detector = detector
        self.tracker = tracker
        self.optical_flow = optical_flow
        self.feature_extractors = feature_extractors or []
        self.aggregator = aggregator or FeatureAggregator()
        self.min_trajectory_len = min_trajectory_len
        self.use_dense_trajectories = use_dense_trajectories
        self.enable_multifamily = enable_multifamily

    def process_video(self, video_path: Path,
                      collect_diagnostics: bool = False,
                      frame_sampling: int = 1,
                      visualise_path: Optional[Path] = None,
                      collect_frame_detections: bool = False) -> VideoProcessingResult:
        video_path = Path(video_path)
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                log.error("Cannot open video file: %s", video_path.name)
                return VideoProcessingResult(
                    tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                    n_frames_processed=0, failure=FailureReason(
                        file_missing=True,
                        error_message=f"cv2.VideoCapture could not open {video_path.name}"))
        except Exception as e:
            import traceback as _tb
            tb_str = _tb.format_exc()
            log.error("Failed to open video %s: %s\n%s", video_path.name, e, tb_str)
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=0, failure=FailureReason(
                    unknown_error=True, error_message=str(e), traceback=tb_str))

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Read fps from capture; fall back to 10.0 (AD-SVD dataset default).
        # This value is injected into context["fps"] for every feature extractor
        # so that all normalised features (px/s, px/s², rad/s …) are computed
        # in consistent physical units across groups (mentor v6.0).
        _cap_fps = float(cap.get(cv2.CAP_PROP_FPS))
        _cap_fps = _cap_fps if _cap_fps > 0 else 10.0

        # ── Explainability visualizer (optional) ─────────────────────────────
        _visualizer: Optional[ExplainabilityVisualizer] = None
        if visualise_path is not None:
            try:
                Path(visualise_path).parent.mkdir(parents=True, exist_ok=True)
                _visualizer = ExplainabilityVisualizer(
                    output_path=str(visualise_path),
                    fps=_cap_fps,
                    frame_size=(orig_w, orig_h),
                )
            except Exception as _ve:
                log.warning("ExplainabilityVisualizer init failed for %s: %s", video_path.name, _ve)
                _visualizer = None

        used_gt = False
        if isinstance(self.detector, GroundTruthDetector):
            used_gt = self.detector.load_for_video(video_path.name, orig_w, orig_h)
            if not used_gt and self.detector.fallback_to_mog2:
                print(f"  [GT] Fallback to MOG2 for {video_path.name}")
                self.detector = MOG2Detector(
                    history=20, var_threshold=25, detect_shadows=False,
                    morph_close_iter=2, morph_open_iter=1, min_contour_area=500
                )

        self.detector.reset()
        self.tracker.reset()
        self.preprocessor.reset()

        prev_gray = None
        flow_mags = []
        flow_angles = []
        diagnostics = []
        frame_no = 0
        n_detections = 0
        all_gray_frames = []

        detections_per_frame = []
        tracks_per_frame = []
        frame_detections: List[List[BBox]] = []

        # Determine total number of frames for dynamic min trajectory length
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Dynamic min length: at most 30, at least 10, and at most 60% of video length
        dynamic_min_len = min(self.min_trajectory_len, max(10, int(total_frames * 0.6)))
        actual_min_len = max(10, dynamic_min_len)  # ensure at least 10

        try:
            while True:
                ret, raw = cap.read()
                if not ret:
                    break
                # Guard: cap.read() can return ret=True with a corrupt/empty frame.
                # cv2.resize (inside the preprocessor) will crash on zero-size arrays.
                if raw is None or raw.size == 0:
                    log.warning("process_video: empty frame at frame_no=%d — skipping", frame_no)
                    frame_no += 1
                    continue
                if frame_sampling > 1 and frame_no % frame_sampling != 0:
                    frame_no += 1
                    continue

                pre_result = self.preprocessor.process(raw)
                if collect_diagnostics and pre_result.diagnostics:
                    diagnostics.append(pre_result.diagnostics)

                dets = self.detector.detect(pre_result.gray)
                n_detections += len(dets)
                detections_per_frame.append(len(dets))

                # ── Tracker update ────────────────────────────────────────────
                # Pass the pre-processed grayscale frame so that trackers that
                # maintain appearance models (e.g. TFCRTracker) can compute HOG
                # features from the current frame.  Trackers that do not accept
                # frame_gray (centroid, IoU+Hungarian, SORT) ignore this kwarg
                # gracefully because their update() signatures accept **kwargs.
                # Manuscript note: this single injection point enables a clean
                # ablation of the TFCR appearance model without modifying the
                # pipeline runner logic for other groups.
                if isinstance(self.tracker, TFCRTracker):
                    self.tracker.update(dets, frame_gray=pre_result.gray)
                else:
                    self.tracker.update(dets)
                active = self.tracker.get_active_tracks()
                tracks_per_frame.append(len(active))
                if collect_frame_detections:
                    frame_detections.append([t["bbox"] for t in active if "bbox" in t])
                # Provide track-quality feedback to the adaptive detector (1-frame lag).
                # This is a no-op for all non-adaptive detector types.
                if hasattr(self.detector, "feedback"):
                    self.detector.feedback(active)

                if self.optical_flow and prev_gray is not None:
                    flow_result = self.optical_flow.compute(prev_gray, pre_result.gray)
                    flow_mags.append(flow_result.mean_magnitude)
                    flow_angles.append(flow_result.mean_angle_deg)

                # ── Explainability frame rendering ────────────────────────────
                if _visualizer is not None:
                    _diag_now = pre_result.diagnostics if pre_result.diagnostics else {}
                    _dense_flow: Optional[np.ndarray] = None
                    # Expose dense flow if the detector computed it (DenseOpticalFlowDetector)
                    if prev_gray is not None and hasattr(self.detector, "_prev_gray"):
                        try:
                            _dense_flow = cv2.calcOpticalFlowFarneback(
                                prev_gray, pre_result.gray,
                                None, 0.5, 3, 15, 3, 5, 1.2, 0
                            )
                        except Exception:
                            _dense_flow = None
                    _visualizer.write_frame(
                        frame_bgr=pre_result.bgr,
                        tracks=active,
                        diagnostics=_diag_now,
                        flow=_dense_flow,
                    )

                prev_gray = pre_result.gray
                frame_no += 1
                if self.use_dense_trajectories:
                    all_gray_frames.append(pre_result.gray)

            cap.release()
            if _visualizer is not None:
                _visualizer.release()
                _visualizer = None
        except Exception as e:
            cap.release()
            if _visualizer is not None:
                try:
                    _visualizer.release()
                except Exception:
                    pass
                _visualizer = None
            import traceback as _tb
            tb_str = _tb.format_exc()
            log.error(
                "Video %s failed during frame processing: %s\n%s",
                video_path.name, e, tb_str,
            )
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=frame_no,
                failure=FailureReason(unknown_error=True, error_message=str(e), traceback=tb_str))

        detection_stats = {
            "mean": float(np.mean(detections_per_frame)) if detections_per_frame else 0.0,
            "std": float(np.std(detections_per_frame)) if detections_per_frame else 0.0,
            "max": max(detections_per_frame) if detections_per_frame else 0,
            "zero_frames": sum(1 for d in detections_per_frame if d == 0),
            "total_frames": len(detections_per_frame),
        }
        tracking_stats = {
            "mean_tracks": float(np.mean(tracks_per_frame)) if tracks_per_frame else 0.0,
            "std_tracks": float(np.std(tracks_per_frame)) if tracks_per_frame else 0.0,
            "max_tracks": max(tracks_per_frame) if tracks_per_frame else 0,
            "total_frames": len(tracks_per_frame),
            "pruned_count": getattr(self.tracker, "pruned_count", 0),
        }

        # ── v6: multi-family handles zero-detection gracefully ──────────────
        if n_detections == 0 and frame_no > 0 and not self.enable_multifamily:
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=frame_no, used_gt=used_gt,
                detection_stats=detection_stats, tracking_stats=tracking_stats,
                failure=FailureReason(detection_failure=True),
                frame_detections=frame_detections, frame_size=(orig_w, orig_h))

        tracks = self.tracker.get_active_tracks()
        mean_mag = float(np.mean(flow_mags)) if flow_mags else 0.0
        mean_ang = float(np.mean(flow_angles)) if flow_angles else 0.0
        tracker_metrics = TrackerAnalyzer.compute_tracker_metrics(tracks, frame_no)

        context = {"mean_flow_magnitude": mean_mag, "mean_flow_angle_deg": mean_ang,
                   "all_tracks": tracks, "fps": _cap_fps}
        if self.use_dense_trajectories and len(all_gray_frames) >= 2:
            mid = len(all_gray_frames) // 2
            context["prev_gray"] = all_gray_frames[max(0, mid-1)]
            context["curr_gray"] = all_gray_frames[mid]

        # ── v6: NO min_track_length gate — all tracks contribute ─────────────
        # Research: Mauthner et al. (2009); flow/appearance compensate when
        # tracks are absent in multi-family mode.
        track_feats:   List[FeatureDict] = []
        track_lengths: List[int]         = []

        for track in tracks:
            tlen = len(track["trajectory"])
            track_lengths.append(tlen)
            combined: FeatureDict = {}
            for extractor in self.feature_extractors:
                try:
                    fv = extractor.extract_track(track, **context)
                    if fv:
                        combined.update(fv)
                except Exception:
                    pass
            if combined:
                track_feats.append(combined)

        video_fv: Optional[FeatureDict] = self.aggregator.aggregate(track_feats) if track_feats else None

        # ── Multi-family blending (Group I, enable_multifamily=True) ─────────
        if self.enable_multifamily:
            diag_feats   = extract_diagnostic_features(diagnostics)
            flow_feats   = extract_global_flow_features(all_gray_frames, flow_step=5)
            step         = max(1, len(all_gray_frames) // 30)
            appear_feats = extract_appearance_features(all_gray_frames[::step])
            n_tj   = len(track_feats)
            tblock = video_fv or {}
            tblock["indicator_tracks_present"] = float(1 if n_tj > 0 else 0)
            tblock["meta_n_traj_tracks"]       = float(n_tj)
            temporal_feats = extract_temporal_context(tracks, total_frames=frame_no)
            blended: FeatureDict = {}
            blended.update(diag_feats)
            blended.update(flow_feats)
            blended.update(appear_feats)
            blended.update(tblock)
            blended.update(temporal_feats)
            blended.setdefault("n_valid_tracks", float(n_tj))
            video_fv = blended

        else:
            # Standard path: fail gracefully if no features produced
            if not track_feats:
                fail = (FailureReason(detection_failure=True) if not tracks
                        else FailureReason(feature_failure=True))
                return VideoProcessingResult(
                    tracks=tracks, mean_flow_magnitude=mean_mag,
                    mean_flow_angle_deg=mean_ang, n_frames_processed=frame_no,
                    used_gt=used_gt, preprocessing_diagnostics=diagnostics,
                    track_lengths=track_lengths,
                    detection_stats=detection_stats, tracking_stats=tracking_stats,
                    failure=fail, tracker_metrics=tracker_metrics,
                    frame_detections=frame_detections, frame_size=(orig_w, orig_h))
            if video_fv is None:
                return VideoProcessingResult(
                    tracks=tracks, mean_flow_magnitude=mean_mag,
                    mean_flow_angle_deg=mean_ang, n_frames_processed=frame_no,
                    used_gt=used_gt, preprocessing_diagnostics=diagnostics,
                    track_lengths=track_lengths,
                    detection_stats=detection_stats, tracking_stats=tracking_stats,
                    failure=FailureReason(feature_failure=True),
                    tracker_metrics=tracker_metrics,
                    frame_detections=frame_detections, frame_size=(orig_w, orig_h))
            temporal_feats = extract_temporal_context(tracks, total_frames=frame_no)
            video_fv.update(temporal_feats)

        feat_count = len(video_fv) if video_fv else 0

        return VideoProcessingResult(
            tracks=tracks,
            mean_flow_magnitude=mean_mag,
            mean_flow_angle_deg=mean_ang,
            n_frames_processed=frame_no,
            used_gt=used_gt,
            preprocessing_diagnostics=diagnostics,
            track_features=track_feats,
            video_feature=video_fv,
            track_lengths=track_lengths,
            feature_count=feat_count,
            tracker_metrics=tracker_metrics,
            detection_stats=detection_stats,
            tracking_stats=tracking_stats,
            frame_detections=frame_detections,
            frame_size=(orig_w, orig_h),
        )
