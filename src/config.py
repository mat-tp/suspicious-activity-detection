from __future__ import annotations
from . import paths
from .common import *
from .paths import *
from .interfaces import *
from .registry import *

class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) if isinstance(v, dict) else v for v in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list:
        return list(self._data.keys())

    def items(self):
        return self._data.items()

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(v):
            if isinstance(v, Config):
                return {k: unwrap(val) for k, val in v._data.items()}
            if isinstance(v, list):
                return [unwrap(item) for item in v]
            return v
        return unwrap(self)

    def __repr__(self):
        return f"Config({self._data!r})"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_paths(data: dict, root: Path) -> dict:
    data = copy.deepcopy(data)
    if "paths" in data:
        for key, rel in data["paths"].items():
            if isinstance(rel, str) and not Path(rel).is_absolute():
                data["paths"][key] = str((root / rel).resolve())
    return data


def _load_single_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_with_extends(config_path: Path, configs_dir: Path, visited: set = None) -> dict:
    if visited is None:
        visited = set()
    config_path = config_path.resolve()
    if config_path in visited:
        log.warning("Circular extends detected: %s", config_path)
        return {}
    visited.add(config_path)
    data = _load_single_yaml(config_path)
    if not data:
        return {}
    extends = data.pop("extends", None)
    if not extends:
        return data
    parent_path = configs_dir / extends
    if not parent_path.suffix:
        parent_path = parent_path.with_suffix(".yaml")
    if not parent_path.exists():
        log.warning("Parent config not found: %s", parent_path)
        return data
    parent_data = _load_with_extends(parent_path, configs_dir, visited)
    return _deep_merge(parent_data, data)


def load_config(group_name: str, project_root: Path = None, overrides: dict = None) -> Config:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    configs_dir = project_root / "configs"
    path = Path(group_name)
    if not path.is_absolute():
        if not path.suffix:
            path = path.with_suffix(".yaml")
        path = configs_dir / path
    if not path.exists():
        available_list = [p.stem for p in configs_dir.glob("*.yaml") if p.stem != "base"]
        raise FileNotFoundError(f"Config '{group_name}' not found at {path}.\nAvailable: {available_list}")
    merged = _load_with_extends(path, configs_dir)
    if not merged:
        raise ValueError(f"Failed to load config: {group_name}")
    merged = _resolve_paths(merged, project_root)
    if overrides:
        merged = _deep_merge(merged, overrides)
    config = Config(merged)
    if hasattr(config, 'paths'):
        print(f"  [config] Paths loaded: {list(config.paths.keys())}")
    else:
        log.warning("No 'paths' section in config. Available keys: %s", list(config.keys()))
    return config



class DatasetInfoHandler:
    def __init__(self, dataset_excel_path: str):
        self.dataset_excel_path = Path(dataset_excel_path)
        self.df = None
        self.video_metadata = {}
        self.resolution_stats = {}
        self.fps_stats = {}
        self.duration_stats = {}
        self.distortion_types = []
        self.quality_levels = ["HQ", "MQ", "LQ", "unknown"]
        self._load_and_analyze()

    def _load_and_analyze(self):
        if not self.dataset_excel_path.exists():
            log.warning("Dataset info file not found: %s", self.dataset_excel_path)
            return
        try:
            self.df = pd.read_excel(self.dataset_excel_path)
            col_mapping = self._detect_columns(self.df)
            for _, row in self.df.iterrows():
                video_name = str(row.get(col_mapping.get("name", ""), ""))
                if not video_name:
                    continue
                metadata = {
                    "duration": float(row.get(col_mapping.get("duration", ""), 0) or 0),
                    "fps": float(row.get(col_mapping.get("fps", ""), 0) or 30),
                    "frame_count": int(row.get(col_mapping.get("frames", ""), 0) or 0),
                    "width": int(row.get(col_mapping.get("width", ""), 0) or 1920),
                    "height": int(row.get(col_mapping.get("height", ""), 0) or 956),
                    "activity": row.get(col_mapping.get("activity", ""), ""),
                    "distortion": row.get(col_mapping.get("distortion", ""), ""),
                }
                if metadata["fps"] > 0 and metadata["duration"] > 0:
                    metadata["computed_frames"] = int(metadata["fps"] * metadata["duration"])
                self.video_metadata[video_name] = metadata
            if "distortion" in col_mapping:
                self.distortion_types = self.df[col_mapping["distortion"]].dropna().unique().tolist()
            else:
                self.distortion_types = ["Pri", "Exp", "Fo", "ExFo"]
            self._compute_statistics()
        except Exception as e:
            log.warning("Error loading dataset info: %s", e)

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        columns = df.columns.tolist()
        for col in columns:
            col_lower = col.lower()
            if "name" in col_lower or "video" in col_lower:
                mapping["name"] = col
            elif "duration" in col_lower or "length" in col_lower:
                mapping["duration"] = col
            elif "fps" in col_lower or "frame rate" in col_lower:
                mapping["fps"] = col
            elif "frame" in col_lower and "count" in col_lower:
                mapping["frames"] = col
            elif "width" in col_lower:
                mapping["width"] = col
            elif "height" in col_lower:
                mapping["height"] = col
            elif "activity" in col_lower:
                mapping["activity"] = col
            elif "distortion" in col_lower:
                mapping["distortion"] = col
        return mapping

    def _compute_statistics(self):
        if not self.video_metadata:
            return
        widths = [m["width"] for m in self.video_metadata.values() if m["width"] > 0]
        heights = [m["height"] for m in self.video_metadata.values() if m["height"] > 0]
        if widths and heights:
            resolutions = list(zip(widths, heights))
            res_counter = Counter(resolutions)
            most_common_res = res_counter.most_common(1)[0][0]
            self.resolution_stats = {
                "unique_count": len(res_counter),
                "most_common": most_common_res,
                "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths), "std": np.std(widths)},
                "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights), "std": np.std(heights)},
                "all_same": len(res_counter) == 1
            }
        fps_values = [m["fps"] for m in self.video_metadata.values() if m["fps"] > 0]
        if fps_values:
            self.fps_stats = {"min": min(fps_values), "max": max(fps_values), "mean": np.mean(fps_values), "std": np.std(fps_values), "all_same": len(set(fps_values)) == 1}
        dur_values = [m["duration"] for m in self.video_metadata.values() if m["duration"] > 0]
        if dur_values:
            self.duration_stats = {"min": min(dur_values), "max": max(dur_values), "mean": np.mean(dur_values), "std": np.std(dur_values)}

    def get_video_info(self, video_name: str) -> Dict[str, Any]:
        return self.video_metadata.get(video_name, {})

    def justify_resize_dimensions(self, target_size=(640, 480)) -> str:
        if not self.resolution_stats:
            return "No resolution data available"
        stats = self.resolution_stats
        justification = f"""
Resolution Analysis:
- Unique resolutions: {stats['unique_count']}
- Most common: {stats['most_common'][0]}x{stats['most_common'][1]}
- Width: {stats['width']['min']}-{stats['width']['max']} (μ={stats['width']['mean']:.0f}, σ={stats['width']['std']:.0f})
- Height: {stats['height']['min']}-{stats['height']['max']} (μ={stats['height']['mean']:.0f}, σ={stats['height']['std']:.0f})
Justification for {target_size[0]}x{target_size[1]} resize:
"""
        if stats['all_same']:
            orig_w, orig_h = stats['most_common']
            justification += f"""
All videos share ({orig_w}x{orig_h}).
Resizing to {target_size[0]}x{target_size[1]} is justified because:
1. ~3x speedup
2. Optical flow accuracy plateaus above 640px (Baker et al., IJCV 2011)
3. Standardised dimensions
4. Maintains sufficient detail for surveillance contexts
"""
        else:
            justification += """
Multiple resolutions detected - uniform resize ensures:
1. Consistent feature dimensions
2. Fair comparison across distortion conditions
3. Reduced computational overhead
"""
        return justification

    def justify_min_trajectory_length(self, min_length=30) -> str:
        if not self.fps_stats or not hasattr(self, 'duration_stats'):
            return "Insufficient data"
        avg_fps = self.fps_stats.get('mean', 30)
        avg_dur = self.duration_stats.get('mean', 0)
        time_at_min = min_length / avg_fps if avg_fps > 0 else 1.0
        justification = f"""
Minimum Trajectory Length Justification ({min_length} frames):
- Average FPS: {avg_fps:.1f}
- Average duration: {avg_dur:.1f}s
- {min_length} frames = {time_at_min:.2f}s
Why {min_length} frames:
1. Provides {time_at_min:.2f}s of motion history (Morris & Trivedi, 2008)
2. Sufficient for action cycles
3. Short enough to maintain high recall
"""
        if avg_dur > 0:
            pct = (min_length / (avg_fps * avg_dur)) * 100
            justification += f"4. Represents only {pct:.1f}% of average video\n"
        return justification

    def get_adaptive_frame_sampling(self, video_name: str, target_frames: int = 100) -> int:
        info = self.get_video_info(video_name)
        if info and info.get("frame_count", 0) > target_frames:
            return max(1, info["frame_count"] // target_frames)
        return 1



BASE_CONFIG_DICT = {
    "experiment": {
        "name": "default",
        "description": "",
        "group": "A",
        "seed": 42,
        # Number of worker PROCESSES for the classical video-feature loop
        # (collect_video_features). 1 = legacy sequential behaviour with a
        # single reused stateful pipeline. >1 processes videos concurrently,
        # each worker building its own detector/tracker instances.
        "n_workers": 1,
    },
    "paths": {
        "data_dir":        "data",
        "video_dir":       "data/surveillanceVideosDataset/surveillanceVideos",
        "gt_dir":          "data/surveillanceVideosDataset/surveillanceVideosGT",
        "dataset_excel":   "data/datasetInfo.xlsx",
        "model_dir":       "models",
        "plots_dir":       "plots",
        "experiments_dir": "experiments",
        "outputs_dir":     "outputs",
    },
    "dataset": {
        "activity_labels":          ["LPP", "PO", "PW", "PPP", "RK", "FG", "PR", "WL"],
        "distortion_types":         ["Pri", "Exp", "Fo", "ExFo"],
        "n_random_videos_per_group": 5,
        "use_dataset_info": True,
    },
    "preprocessing": {
        "type": "standard",
        "enabled": True,
        "distortion_aware": True,
        "frame_resize": [640, 480],
        "blur_kernel":  [5, 5],
        "blur_sigma":   0,
        "distortion_aware_params": {
            "blur_threshold":        100.0,
            "exposure_low":           60.0,
            "exposure_high":         195.0,
            "enhancement_for_blur":  "unsharp_mask",
            "enhancement_for_exposure": "clahe",
            "unsharp_sigma":           2.0,
            "unsharp_strength":        1.5,
            "clahe_clip_limit":        2.0,
            "clahe_tile_grid":        [8, 8],
            "bilateral_d":             9,
            "bilateral_sigma_color":  75,
            "bilateral_sigma_space":  75,
            "use_multi_metric": True,
            "gamma_correction": False,
            "gamma_value": 1.2,
            "contrast_stretching": False,
        },
    },
    "detection": {
        "type": "mog2",
        "mog2": {
            "history":        20,
            "var_threshold":  25,
            "detect_shadows": False,
            "adaptive": False,          # NEW: enable adaptive threshold
        },
        "morphology": {
            "close_iter": 2,
            "open_iter":  1,
        },
        "min_contour_area": 500,
        "ground_truth": {
            "fallback_to_mog2": False,
        },
    },
    "tracking": {
        "type":               "iou_hungarian",
        "max_disappeared":     30,
        "max_trajectory_len": 300,
        "min_trajectory_len":  30,
        "iou_hungarian": {
            "iou_threshold": 0.3,
        },
        "sort": {
            "iou_threshold": 0.3,
            "min_hits": 3,
            "kalman_process_noise": 0.03,
            "kalman_measurement_noise": 0.3,
        },
        # ── TFCR tracker hyperparameters (Yuan et al. 2020) ──────────────────
        # η  (eta)  — target-focusing coefficient in J(w) = ||w*φ-y||² − η||w*φ||² + λ||w||²
        #             Set to 0.5 (paper default, §4.1).  Higher values increase
        #             the response gap between target and background samples.
        # λ  (lam)  — ridge regularisation; prevents overfitting to the
        #             ~10⁴ background samples per frame.
        # lr        — online learning rate; lower than the paper (5e-8 ADMM)
        #             because the HOG surrogate feature space is much smaller
        #             (9-dim vs. 64-ch conv4-3).
        # scale_factors — three scale candidates β ∈ {0.95, 1.00, 1.05}
        #             (TFCR §3.3, Eq. 6): (wₜ, hₜ) = β(wₜ₋₁, hₜ₋₁).
        # hog_cell  — HOG cell size in pixels; 8 matches the TFCR training
        #             patch convention (64px patch, 8×8 cells = 64 bins).
        "tfcr": {
            "iou_threshold": 0.3,
            "eta":           0.5,
            "lam":           1e-4,
            "lr":            0.02,
            "scale_factors": [0.95, 1.0, 1.05],
            "hog_cell":      8,
        },
    },
    "optical_flow": {
        "lk_win_size":       [15, 15],
        "lk_max_level":       2,
        "lk_criteria_eps":    0.03,
        "lk_criteria_count": 10,
        "max_corners":       200,
        "corner_quality":    0.3,
        "corner_min_dist":   7,
    },
    "features": {
        "modules":     ["trajectory", "circular_motion", "kinematics",
                        "shape", "wavelet", "interaction"],
        "aggregation": ["mean", "max", "std"],
        "dense_trajectories": {
            "enabled": False,
            "step_size": 5,
            "block_size": 32,
            "feature_types": ["trajectory", "hog", "hof", "mbh"],
        },
    },
    "models": {
        "active": [
            {"name": "random_forest_sklearn",
             "params": {"n_estimators": 200, "max_depth": None,
                        "class_weight": "balanced"}},
            {"name": "svm_rbf",
             "params": {"C": 10, "gamma": "scale", "class_weight": "balanced"}},
            {"name": "knn",
             "params": {"n_neighbors": 5, "metric": "euclidean"}},
            {"name": "random_forest_custom",
             "params": {"n_trees": 200, "max_depth": None,
                        "min_samples_split": 2, "class_weight": "balanced"}},
            {"name": "lightgbm",
             "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}},
            {"name": "xgboost",
             "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}},
        ],
    },
    "training": {
        "train_size": 0.70,
        "val_size":   0.15,
        "test_size":  0.15,
        "cv_folds":    5,
        "stratify":   True,
        "use_cv":     True,          # NEW: use cross-validation instead of fixed split
        "min_per_class": 3,          # NEW: minimum samples per class to train
    },
    "smoothing": {
        "enabled": False,
        "method": "majority_vote",
        "ema_alpha": 0.7,
    },
    "targets":   ["activity", "distortion"],
    "data_path": "tabular",
    "clip": {
        "clip_len": 16,
        "frame_size": [112, 112],
        "sampling": "uniform",
        "batch_size": 8,
    },
    "embedding": {
        "backbone":      "resnet18_2d",
        "pretrained":    False,
        "embedding_dim": 512,
        "pooling":       "mean",
    },
    "neural": {
        "cnn": {
            "backbone":   "resnet18_2d",
            "pretrained": False,
            "lr":          0.001,
            "epochs":     15,
            "batch_size":  8,
        },
        "cnn_lstm": {
            "backbone":     "resnet18_2d",
            "hidden_size":  256,
            "num_layers":   1,
            "lr":           0.0005,
            "epochs":      20,
            "batch_size":   8,
        },
        "cnn3d": {
            "backbone":   "r3d_18",
            "pretrained": False,
            "lr":          0.0005,
            "epochs":     20,
            "batch_size":  4,
        },
    },
}



# ──────────────────────────────────────────────────────────────────────────────
# FULL_MODEL_SUITE — canonical list of all classical classifiers used across
# groups A, B, E, F, G, H, and I.
#
# Having a single definition guarantees:
#   1. All groups are evaluated on the same models → fair comparison.
#   2. A single change updates every group at once.
#   3. Group C (neural) and Group D (feature_fusion) are excluded intentionally
#      because they use different data paths / classifiers.
#
# Research: Sargano et al. (2017) and Csurka et al. (2018) surveys compare
# RF, SVM, and KNN as standard baselines; LightGBM and XGBoost are added as
# modern ensemble comparisons (mentored addition v6.0).
# ──────────────────────────────────────────────────────────────────────────────
FULL_MODEL_SUITE = [
    {"name": "random_forest_sklearn",
     "params": {"n_estimators": 200, "max_depth": None, "class_weight": "balanced"}},
    {"name": "svm_rbf",
     "params": {"C": 10, "gamma": "scale", "class_weight": "balanced"}},
    {"name": "knn",
     "params": {"n_neighbors": 5, "metric": "euclidean"}},
    {"name": "random_forest_custom",
     "params": {"n_trees": 200, "max_depth": None,
                "min_samples_split": 2, "class_weight": "balanced"}},
    {"name": "lightgbm",
     "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}},
    {"name": "xgboost",
     "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}},
]
def create_default_configs():
    configs = {
        "base.yaml": BASE_CONFIG_DICT,
        "group_a_classical.yaml": {
            "extends": "base.yaml",
            "experiment": {"name": "group_a_classical_baseline", "description": "MOG2 + IoU+Hungarian + Trajectory features + Classical models", "group": "A"},
            "preprocessing": {"type": "standard", "enabled": True, "distortion_aware": False},
            "detection": {"type": "mog2", "mog2": {"adaptive": False}},
            "tracking": {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features": {"modules": ["trajectory"]},
            "models": {"active": FULL_MODEL_SUITE},
            "targets": ["activity", "distortion"],
        },
        "group_b_distortion_aware.yaml": {
            "extends": "base.yaml",
            "experiment": {"name": "group_b_distortion_aware_classical", "description": "Distortion-aware + Full features + RF/LightGBM", "group": "B"},
            "preprocessing": {"type": "distortion_aware", "enabled": True, "distortion_aware": True},
            "detection": {"type": "mog2", "mog2": {"adaptive": True}},   # adaptive MOG2 enabled
            "tracking": {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features": {"modules": ["trajectory", "circular_motion", "kinematics", "shape", "wavelet", "interaction"]},
            "models": {"active": FULL_MODEL_SUITE},
            "targets": ["activity", "distortion"],
        },
        "group_c_neural.yaml": {
            "extends": "base.yaml",
            "experiment": {"name": "group_c_neural_baselines", "description": "CNN/CNN+LSTM/3D-CNN — comparison baselines only", "group": "C"},
            "comparison_only": True,
            "preprocessing": {"type": "standard", "enabled": True, "distortion_aware": False},
            "data_path": "clip",
            "features": {"modules": []},
            "models": {"active": [
                {"name": "cnn", "params": {"backbone": "resnet18_2d", "pretrained": False, "lr": 0.001, "epochs": 15, "batch_size": 8}},
                {"name": "cnn_lstm", "params": {"backbone": "resnet34_2d", "pretrained": True, "hidden_size": 256,
                                                "num_layers": 2, "bidirectional": True, "dropout": 0.3,
                                                "lr": 0.0005, "epochs": 20, "batch_size": 8}},
                {"name": "cnn3d", "params": {"backbone": "r3d_18", "pretrained": False, "lr": 0.0005, "epochs": 20, "batch_size": 4}},
            ]},
            "targets": ["activity"],
        },
        "group_d_hybrid.yaml": {
            "extends": "base.yaml",
            "experiment": {"name": "group_d_hybrid_fusion", "description": "Handcrafted features + CNN embeddings hybrid", "group": "D"},
            "preprocessing": {"type": "distortion_aware", "enabled": True, "distortion_aware": True},
            "detection": {"type": "mog2", "mog2": {"adaptive": True}},
            "tracking": {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "data_path": "hybrid",
            "features": {"modules": ["trajectory", "circular_motion", "kinematics", "shape", "wavelet", "interaction"]},
            "models": {"active": [{"name": "feature_fusion", "params": {"classifier": "lightgbm", "n_estimators": 200, "max_depth": 6}}]},
            "targets": ["activity", "distortion"],
        },
        "group_e_gt_upper_bound.yaml": {
            "extends": "base.yaml",
            "experiment": {"name": "group_e_ground_truth_upper_bound", "description": "GT bounding boxes + Full features — pure upper bound", "group": "E"},
            "preprocessing": {"type": "standard", "enabled": True, "distortion_aware": False},
            "detection": {"type": "ground_truth", "ground_truth": {"fallback_to_mog2": False}},
            "tracking": {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features": {"modules": ["trajectory", "circular_motion", "kinematics", "shape", "wavelet", "interaction"]},
            "models": {"active": FULL_MODEL_SUITE},
            "targets": ["activity", "distortion"],
        },
        "group_f_tracking_ablation.yaml": {
            "extends": "group_a_classical.yaml",
            "experiment": {"name": "group_f_tracking_ablation", "description": "Compare tracking methods: Centroid vs IoU+Hungarian vs SORT vs TFCR", "group": "F"},
            # ── Manuscript note ──────────────────────────────────────────────
            # Group F is a tracking-method ablation:  the same MOG2 detector,
            # distortion-aware preprocessor, and full feature set are used
            # across four trackers so that the only variable is the tracking
            # algorithm.  TFCR (Yuan et al. 2020) is included because it was
            # the best-performing external tracker benchmarked on the AD-SVD
            # dataset (Gomez-Nieto et al. 2022, Fig. 14a) and therefore
            # represents the strongest available baseline.
            "tracking_ablation": {"enabled": True, "methods": ["centroid", "iou_hungarian", "sort", "tfcr"], "compare_min_lengths": [15, 30, 45, 60]},
            "models": {"active": FULL_MODEL_SUITE},
        },
        "group_g_feature_comparison.yaml": {
            "extends": "group_a_classical.yaml",
            "experiment": {"name": "group_g_feature_comparison", "description": "Compare blob-based vs dense trajectory features", "group": "G"},
            "features": {
                "modules": ["trajectory"],
                "dense_trajectories": {"enabled": True, "step_size": 5, "block_size": 32, "feature_types": ["trajectory", "hog", "hof", "mbh"]},
                "compare_approaches": True,
            },
            "models": {"active": FULL_MODEL_SUITE},
        },
        # NEW: Optical flow detector group
        "group_h_optical_flow_detector.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name": "group_h_optical_flow_detector",
                "description": "Dense optical flow detection — no background model, no ground truth. Tests robustness to distortion.",
                "group": "H",
            },
            "preprocessing": {"type": "standard", "enabled": True, "distortion_aware": False},
            "detection": {
                "type": "dense_flow",
                "dense_flow": {
                    "magnitude_threshold": 2.0,
                    "min_contour_area": 500,
                    "morph_close_iter": 2,
                    "morph_open_iter": 1,
                    "pyr_scale": 0.5,
                    "levels": 3,
                    "winsize": 15,
                    "iterations": 3,
                    "poly_n": 5,
                    "poly_sigma": 1.2,
                    "adaptive_threshold": True,
                },
            },
            "tracking": {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features": {"modules": ["trajectory", "circular_motion", "kinematics", "shape", "interaction"]},
            "models": {"active": FULL_MODEL_SUITE},
            "targets": ["activity", "distortion"],
        },
        # NEW: Cascaded fallback detector group
        "group_h_cascaded_fallback.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name": "group_h_cascaded_fallback",
                "description": "MOG2 → Dense Optical Flow → Frame Differencing cascade. Maximises detection coverage.",
                "group": "H",
            },
            "preprocessing": {"type": "standard", "enabled": True, "distortion_aware": False},
            "detection": {
                "type": "cascaded_fallback",
                "cascaded_fallback": {
                    "mog2_history": 20,
                    "mog2_var_threshold": 25,
                    "flow_magnitude_threshold": 2.0,
                    "flow_adaptive_threshold": True,
                    "diff_threshold": 25,
                    "min_contour_area": 500,
                    "morph_close_iter": 2,
                    "morph_open_iter": 1,
                    "warmup_frames": 10,
                    "min_detections_per_warmup": 3,
                },
            },
            "tracking": {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features": {"modules": ["trajectory", "circular_motion", "kinematics", "shape", "interaction"]},
            "models": {"active": FULL_MODEL_SUITE},
            "targets": ["activity", "distortion"],
        },
        "group_i_adaptive_mog2.yaml": {
            "extends": "group_a_classical.yaml",
            "experiment": {
                "name": "group_i_adaptive_mog2",
                "description": (
                    "Quality-guided adaptive MOG2 + multi-family feature blending. "
                    "AdaptiveMOG2 dynamically lowers var_threshold when mask quality "
                    "or trajectory stability falls below the configured threshold. "
                    "Multi-family blending (diagnostic 16 + flow 22 + appearance 34 "
                    "+ trajectory+indicator + temporal 3) guarantees a feature vector "
                    "for EVERY video — no video discarded due to short/absent tracks. "
                    "Research: Mauthner 2009, Gillani 2025, Gomez-Nieto 2022, Csurka 2018."
                ),
                "group": "I",
            },
            "preprocessing": {
                "type": "distortion_aware",
                "enabled": True,
                "distortion_aware": True,
            },
            "detection": {
                "type": "adaptive_mog2",
                "mog2": {
                    "history": 20,
                    "var_threshold": 25,
                    "detect_shadows": False,
                    "relaxed_var_threshold": 10.0,
                    "quality_threshold": 0.4,
                    "mask_weight": 0.7,
                    "min_track_len": 5,
                    "ema_alpha": 0.8,
                },
                "morphology": {"close_iter": 2, "open_iter": 1},
                "min_contour_area": 500,
                "ground_truth": {"fallback_to_mog2": False},
            },
            "features": {
                "modules": [
                    "trajectory", "circular_motion", "kinematics", "shape", "interaction",
                ],
                # Activates multi-family blending in PipelineRunner.
                # Also set implicitly when experiment.group == "I".
                "enable_multifamily": True,
            },
            "models": {"active": FULL_MODEL_SUITE},
            "targets": ["activity", "distortion"],
        },
        # ── Group J: TFCR-based tracking ──────────────────────────────────────
        # Manuscript justification:
        #   The AD-SVD dataset (Gomez-Nieto et al. 2022) was originally
        #   introduced for Video Object Tracking (VOT).  Among the trackers
        #   benchmarked on AD-SVD, TFCR (Yuan et al. 2020) achieved the highest
        #   AUC at spatial scale 1/4 (Fig. 14a), making it the best-performing
        #   published tracker on authentically distorted surveillance video.
        #
        #   Group J re-uses the Group B (distortion-aware) pipeline but replaces
        #   the IoU+Hungarian tracker with TFCR, enabling a direct comparison
        #   between the classification-optimised tracker (Group B) and the
        #   VOT-state-of-the-art tracker (Group J).
        #
        #   Future work (noted in run.py §8A):  finer contact-event categories
        #   (handshake, hug) could be distinguished by augmenting TFCR
        #   trajectories with duration-of-proximity and approach-angle features.
        "group_j_tfcr_tracker.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name": "group_j_tfcr_tracker",
                "description": (
                    "TFCR target-focusing convolutional regression tracker "
                    "(Yuan et al. 2020) + distortion-aware preprocessing + full "
                    "feature set.  TFCR was the best-performing tracker on the "
                    "AD-SVD dataset (Gomez-Nieto et al. 2022, Fig. 14a) and is "
                    "used here as the VOT state-of-the-art baseline."
                ),
                "group": "J",
            },
            "preprocessing": {
                "type": "distortion_aware",
                "enabled": True,
                "distortion_aware": True,
            },
            "detection": {
                "type": "mog2",
                "mog2": {"adaptive": True},
            },
            "tracking": {
                "type": "tfcr",
                "max_disappeared": 30,
                "max_trajectory_len": 300,
                "min_trajectory_len": 30,
                # TFCR hyperparameters — matching Yuan et al. (2020) §4.1
                "tfcr": {
                    "iou_threshold": 0.3,
                    "eta":           0.5,    # target-focusing strength
                    "lam":           1e-4,   # ridge regularisation
                    "lr":            0.02,   # online learning rate
                    "scale_factors": [0.95, 1.0, 1.05],   # β (TFCR §3.3)
                    "hog_cell":      8,      # HOG cell size (px)
                },
            },
            "features": {
                "modules": [
                    "trajectory", "circular_motion", "kinematics",
                    "shape", "wavelet", "interaction",
                ],
            },
            "models": {"active": FULL_MODEL_SUITE},
            "targets": ["activity", "distortion"],
        },
        "dataset_analysis.yaml": {
            "extends": "base.yaml",
            "experiment": {"name": "dataset_analysis", "description": "Analyze dataset properties", "group": "ANALYSIS"},
            "dataset": {"use_dataset_info": True, "generate_statistics": True, "justify_parameters": True},
        },
    }
    for name, data in configs.items():
        path = paths.CONFIGS_DIR / name
        if not path.exists():
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"  Created: {path}")
