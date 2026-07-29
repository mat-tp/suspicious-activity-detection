#!/usr/bin/env python3
"""
=============================================================================
Distortion-Aware Suspicious Activity Detection in Authentically Distorted
Surveillance Videos: Complete Production Framework (Enhanced)

Author : Tshephang Phuti-A-Nkutu Matlala | 223004635
UJ Department of Computer Science and Software Engineering

VERSION 6.1 — TFCR tracker + Group J config + manuscript comments:
  • TFCRTracker: target-focusing convolutional regression (Yuan et al. 2020)
    — HOG surrogate for VGGNet conv4-3 features (CPU-feasible)
    — target-focusing loss: J(w)=||w*φ-y||²−η||w*φ||²+λ||w||² (Eq. 3)
    — IoU-weighted online learning rate (confidence gate)
    — scale estimation: β∈{0.95,1.0,1.05} (TFCR §3.3)
  • Group J config: TFCR tracker + distortion-aware preprocessing + full features
  • Group F extended: tracking ablation now includes TFCR (4 methods total)
  • frame_gray injection in PipelineRunner so TFCR receives the preprocessed
    grayscale frame for HOG extraction (backward-compatible with other trackers)
  • Manuscript-ready comments throughout §8A explaining the TFCR loss,
    its relationship to AD-SVD VOT results, and future work on contact events
VERSION 6.0 — Full model suite, normalised features, multi-family blending:
  • Unknown-failure logging with full traceback + grouping by activity class
  • AdaptiveMOG2Detector: per-frame var_threshold relaxation (Group I)
  • compute_min_trajectory_len(): reference only (v6: NOT a hard gate)
  • check_training_viability(): pre-flight guard against degenerate splits
  • evaluate_with_cv(): stratified k-fold CV with mean±std reporting
  • extract_temporal_context(): video-level temporal/spatial features
  • DenseOpticalFlowDetector: annotation-free motion detector (Group H-optical)
  • AdaptiveFrameDiffDetector: frame-difference fallback detector
  • CascadedFallbackDetector: MOG2 → Dense Flow → Frame Diff (Group H)
  • Group I config (adaptive_mog2) and Group H configs added
  v6.0 (mentor review — full model suite + normalisation + multi-family):
  • FULL_MODEL_SUITE: single constant — RF, SVM, KNN, RF-custom, LightGBM,
    XGBoost — applied to ALL groups (A, B, E, F, G, H, I) for a fair comparison.
  • TrajectoryFeatures: speeds → px/s (×fps), displacement/path → px/s,
    track duration → seconds; all per-track normalisation uses track's own fps.
  • KinematicsFeatures: accel → px/s², jerk → px/s³ (×fps and ×fps² resp.).
  • ShapeFeatures: turning_rate (rad/s) replaces raw cumulative angle sum.
  • WaveletFeatures: speed signal normalised to px/s before DWT.
  • InteractionFeatures: relative speed → px/s; proximity events → events/s.
  • fps injected into feature-extractor context["fps"] from CAP_PROP_FPS.
  • extract_diagnostic_features(): 16 diagnostic features (always available).
  • extract_global_flow_features(): 22 Farnebäck HOF features (always).
  • extract_appearance_features(): 34 LBP+HOG features (always, no detection).
  • extract_video_metadata_features(): 3 temporal scalars (all groups A–I).
  • PipelineRunner.enable_multifamily: 120-dim blended vector (Group I).
  • min_track_length gate REMOVED — every video yields a feature vector.
  Research: Sargano 2017, Csurka 2018, Mauthner 2009, Gillani 2025,
            Gomez-Nieto 2022, Xiao 2016.

Experiment groups:
  A  — MOG2 baseline, no distortion-aware preprocessing
  B  — MOG2 + distortion-aware preprocessing + full features
  C  — CNN/CNN+LSTM/3D-CNN neural comparison baselines
  D  — Hybrid handcrafted + CNN-embedding fusion
  E  — Ground-truth upper bound
  F  — Tracking method ablation (centroid vs IoU+Hungarian vs SORT vs TFCR)
  G  — Feature comparison (blob-based vs dense trajectories)
  H  — Dense optical flow detector / Cascaded fallback detector
  I  — Adaptive MOG2 (dynamic var_threshold per frame)
  J  — TFCR tracker (Yuan et al. 2020) + distortion-aware preprocessing
=============================================================================
"""

from __future__ import annotations

import abc
import argparse
import copy
import json
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    log.warning("LightGBM not installed — LightGBM models unavailable")

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    log.warning("PyWavelets not installed — DWT wavelet features unavailable")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not installed — neural comparison models unavailable")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not installed — plots will be skipped")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    log.warning("XGBoost not installed — XGBoost models unavailable")

# Optional FiftyOne
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    log.warning("FiftyOne not installed — video visualisation unavailable")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                      SECTION 1: PROJECT CONFIGURATION                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR     = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"
PLOTS_DIR       = PROJECT_ROOT / "plots"

for _d in [CONFIGS_DIR, EXPERIMENTS_DIR, OUTPUTS_DIR, DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                   SECTION 2: CORE INTERFACES AND DATA TYPES                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

BBox        = Dict[str, Any]
Track       = Dict[str, Any]
FeatureDict = Dict[str, float]


@dataclass
class PreprocessResult:
    bgr: np.ndarray
    gray: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    enhanced: bool = False


@dataclass
class OpticalFlowResult:
    mean_magnitude: float
    mean_angle_deg: float
    flow_vectors: Optional[np.ndarray] = None
    prev_corners: Optional[np.ndarray] = None


@dataclass
class FailureReason:
    file_missing: bool = False
    detection_failure: bool = False
    tracking_failure: bool = False
    track_too_short: bool = False
    feature_failure: bool = False
    unknown_error: bool = False
    low_detection_density: bool = False
    track_pruned_often: bool = False
    error_message: str = ""
    traceback: str = ""


@dataclass
class VideoProcessingResult:
    tracks: List[Track]
    mean_flow_magnitude: float
    mean_flow_angle_deg: float
    n_frames_processed: int
    used_gt: bool = False
    preprocessing_diagnostics: List[dict] = field(default_factory=list)
    track_features: List[FeatureDict] = field(default_factory=list)
    video_feature: Optional[FeatureDict] = None
    track_lengths: List[int] = field(default_factory=list)
    feature_count: int = 0
    failure: Optional[FailureReason] = None
    tracker_metrics: Optional[Dict[str, Any]] = None
    detection_stats: Dict[str, Any] = field(default_factory=dict)
    tracking_stats: Dict[str, Any] = field(default_factory=dict)


# ── Abstract base classes ─────────────────────────────────────────────────────

class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]: ...
    @abc.abstractmethod
    def reset(self) -> None: ...


class Tracker(abc.ABC):
    @abc.abstractmethod
    def update(self, detections: List[BBox]) -> List[Track]: ...
    @abc.abstractmethod
    def get_active_tracks(self) -> List[Track]: ...
    @abc.abstractmethod
    def reset(self) -> None: ...


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def process(self, frame: np.ndarray) -> PreprocessResult: ...
    def reset(self) -> None:
        pass


class FeatureExtractor(abc.ABC):
    @abc.abstractmethod
    def feature_names(self) -> List[str]: ...
    @abc.abstractmethod
    def extract_track(self, track: Track, **context: Any) -> Optional[FeatureDict]: ...


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model": ...
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None
    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return None
    @property
    def classes_(self) -> Optional[np.ndarray]:
        return None


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                      SECTION 3: REGISTRY SYSTEM                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

_REGISTRIES: Dict[str, Dict[str, type]] = {
    "detector": {}, "tracker": {}, "preprocessor": {},
    "feature": {}, "model": {},
}


def register(category: str, name: str) -> Callable:
    if category not in _REGISTRIES:
        raise KeyError(f"Unknown category '{category}'. Valid: {list(_REGISTRIES.keys())}")
    def decorator(cls: type) -> type:
        if name in _REGISTRIES[category]:
            raise ValueError(f"'{name}' already registered under '{category}'")
        _REGISTRIES[category][name] = cls
        return cls
    return decorator


def build(category: str, name: str, **kwargs) -> Any:
    if category not in _REGISTRIES or name not in _REGISTRIES[category]:
        available_list = list(_REGISTRIES.get(category, {}).keys())
        raise KeyError(f"'{name}' not registered under '{category}'. Available: {available_list}")
    return _REGISTRIES[category][name](**kwargs)


def available(category: str) -> list:
    return sorted(_REGISTRIES.get(category, {}).keys())


def is_registered(category: str, name: str) -> bool:
    return name in _REGISTRIES.get(category, {})


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                   SECTION 4: CONFIGURATION MANAGEMENT                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                   SECTION 4B: DATASET INFO HANDLER                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                      SECTION 5: BASE CONFIGURATION                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

BASE_CONFIG_DICT = {
    "experiment": {
        "name": "default",
        "description": "",
        "group": "A",
        "seed": 42,
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
                {"name": "cnn_lstm", "params": {"backbone": "resnet18_2d", "hidden_size": 256, "num_layers": 1, "lr": 0.0005, "epochs": 20, "batch_size": 8}},
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
        path = CONFIGS_DIR / name
        if not path.exists():
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"  Created: {path}")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║              SECTION 6: PREPROCESSORS                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

@register("preprocessor", "passthrough")
class PassthroughPreprocessor(Preprocessor):
    def process(self, frame: np.ndarray) -> PreprocessResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return PreprocessResult(bgr=frame, gray=gray, diagnostics={"enabled": False}, enhanced=False)


@register("preprocessor", "standard")
class StandardPreprocessor(Preprocessor):
    def __init__(self, frame_resize=(640, 480), blur_kernel=(5, 5), blur_sigma=0):
        self.frame_resize = tuple(frame_resize)
        self.blur_kernel = tuple(blur_kernel)
        self.blur_sigma = blur_sigma

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)
        return PreprocessResult(
            bgr=bgr, gray=gray,
            diagnostics={"blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "brightness": float(gray.mean())},
            enhanced=False
        )


@register("preprocessor", "distortion_aware")
class DistortionAwarePreprocessor(Preprocessor):
    def __init__(
        self,
        frame_resize=(640, 480),
        blur_kernel=(5, 5),
        blur_sigma=0,
        blur_threshold=100.0,
        exposure_low=60.0,
        exposure_high=195.0,
        enhancement_for_blur="unsharp_mask",
        enhancement_for_exposure="clahe",
        unsharp_sigma=2.0,
        unsharp_strength=1.5,
        clahe_clip_limit=2.0,
        clahe_tile_grid=(8, 8),
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        use_multi_metric=True,
        gamma_correction=False,
        gamma_value=1.2,
        contrast_stretching=False,
    ):
        self.frame_resize = tuple(frame_resize)
        self.blur_kernel = tuple(blur_kernel)
        self.blur_sigma = blur_sigma
        self.blur_threshold = blur_threshold
        self.exposure_low = exposure_low
        self.exposure_high = exposure_high
        self.enhancement_for_blur = enhancement_for_blur
        self.enhancement_for_exposure = enhancement_for_exposure
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_strength = unsharp_strength
        self.use_multi_metric = use_multi_metric
        self.gamma_correction = gamma_correction
        self.gamma_value = gamma_value
        self.contrast_stretching = contrast_stretching
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tuple(clahe_tile_grid))
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def _detect_blur_multi_metric(self, gray: np.ndarray) -> Dict[str, float]:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        tenengrad = np.sqrt(gx**2 + gy**2).mean()
        brenner = np.mean(np.abs(np.diff(gray.astype(np.float64), axis=1)))
        return {"laplacian_var": lap_var, "tenengrad": tenengrad, "brenner": brenner, "blur_score": 1.0 - min(lap_var / 500.0, 1.0)}

    def _gamma_correct(self, gray: np.ndarray) -> np.ndarray:
        inv_gamma = 1.0 / self.gamma_value
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(gray, table)

    def _contrast_stretch(self, gray: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(gray, (2, 98))
        return cv2.normalize(gray, None, p2, p98, cv2.NORM_MINMAX)

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if self.use_multi_metric:
            blur_metrics = self._detect_blur_multi_metric(gray)
            blur_score = blur_metrics["laplacian_var"]
        else:
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        is_blurry = blur_score < self.blur_threshold
        is_underexposed = brightness < self.exposure_low
        is_overexposed = brightness > self.exposure_high

        if is_blurry and (is_underexposed or is_overexposed):
            distortion_type = "blurry+exposure"
        elif is_blurry:
            distortion_type = "blurry"
        elif is_underexposed:
            distortion_type = "underexposed"
        elif is_overexposed:
            distortion_type = "overexposed"
        else:
            distortion_type = "pristine"

        enhanced = False

        if self.gamma_correction and (is_underexposed or is_overexposed):
            gray = self._gamma_correct(gray)
            enhanced = True

        if is_blurry:
            if self.enhancement_for_blur == "unsharp_mask":
                gray = self._unsharp_mask(gray)
                enhanced = True
            elif self.enhancement_for_blur == "wiener":
                gray = self._wiener_filter(gray)
                enhanced = True

        if is_underexposed or is_overexposed:
            if self.enhancement_for_exposure == "clahe":
                gray = self.clahe.apply(gray)
                enhanced = True
            elif self.enhancement_for_exposure == "bilateral":
                gray = cv2.bilateralFilter(gray, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
                enhanced = True

        if self.contrast_stretching and (is_underexposed or is_overexposed):
            gray = self._contrast_stretch(gray)
            enhanced = True

        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)

        diagnostics = {
            "blur_score": blur_score,
            "brightness": brightness,
            "distortion_type": distortion_type,
            "enhanced": enhanced,
            "is_blurry": is_blurry,
            "is_underexposed": is_underexposed,
            "is_overexposed": is_overexposed,
        }
        if self.use_multi_metric:
            diagnostics.update(blur_metrics)

        return PreprocessResult(bgr=bgr, gray=gray, enhanced=enhanced, diagnostics=diagnostics)

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), self.unsharp_sigma)
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        adaptive_strength = self.unsharp_strength * (1.0 + np.tanh(local_var / 100.0))
        return cv2.addWeighted(gray, 1.0 + adaptive_strength, blurred, -adaptive_strength, 0)

    def _wiener_filter(self, gray: np.ndarray) -> np.ndarray:
        kernel = (5, 5)
        f64 = gray.astype(np.float64)
        mean = cv2.blur(f64, kernel)
        mean_sq = cv2.blur(f64 ** 2, kernel)
        variance = mean_sq - mean ** 2
        noise_mask = variance < np.percentile(variance, 20)
        noise_var = np.mean(variance[noise_mask]) if noise_mask.any() else 100.0
        ratio = np.maximum(0, variance - noise_var) / np.maximum(variance, noise_var)
        result = mean + ratio * (f64 - mean)
        return np.clip(result, 0, 255).astype(np.uint8)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║            SECTION 7: DETECTORS (MOG2, Adaptive, Ground-Truth)             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

STRUCT = generate_binary_structure(2, 2)

def make_bbox(x: int, y: int, w: int, h: int) -> BBox:
    cx = int(x + w // 2)
    cy = int(y + h // 2)
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "cx": cx, "cy": cy, "centroid": (cx, cy), "area": int(w * h)}


@register("detector", "mog2")
class MOG2Detector(Detector):
    def __init__(self, history=20, var_threshold=25, detect_shadows=False,
                 morph_close_iter=2, morph_open_iter=1, min_contour_area=500,
                 adaptive=False):   # NEW: adaptive flag
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter = morph_open_iter
        self.min_contour_area = min_contour_area
        self.adaptive = adaptive
        self._make_mog2()

    def _make_mog2(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=self.var_threshold, detectShadows=self.detect_shadows)

    def clean_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        binary = raw_mask > 0
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
    THRESHOLD_FLOOR = 15.0    # hard lower bound on relaxed var_threshold

    def __init__(self, history: int = 20, var_threshold: float = 25,
                 detect_shadows: bool = False,
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
        self.gt_dir = (Path(gt_dir) if gt_dir else DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideosGT")
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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║            SECTION 7B: NEW DETECTORS (Optical Flow, Frame Diff, Cascade)   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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
        mog2_var_threshold: float = 25,
        mog2_detect_shadows: bool = False,
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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║            SECTION 8: TRACKERS                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║          SECTION 8A: TFCR TRACKER (Target-Focusing Convolutional           ║
# ║                       Regression)                                          ║
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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║            SECTION 8B: TRACKER ANALYSIS UTILITIES                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                       SECTION 9: OPTICAL FLOW                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║           SECTION 10: FEATURE EXTRACTORS                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

@register("feature", "trajectory")
class TrajectoryFeatures(FeatureExtractor):
    """
    Trajectory-level motion descriptors, fully normalised to physical units.

    Normalisation (mentor v6.0 — applied identically to ALL groups A–I):
      speeds    [px/frame]  × fps   → px/s
      path_len, net_disp             ÷ duration_sec → px/s (rates)
      track_duration               = n_frames / fps (seconds)
      dimensionless ratios (straightness, skewness, kurtosis,
        direction_std_deg) — unchanged.

    fps is read from context["fps"] (injected by PipelineRunner from
    CAP_PROP_FPS; default 10.0 for backward compatibility).
    Per-track duration uses the track's own frame count, NOT the total
    video duration, so a short track inside a long video is correctly
    normalised.  Research: mentor review v6.0; Sargano et al. (2017).
    """

    def __init__(self, min_trajectory_len=30):
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return [
            "track_duration_sec",           # track's own duration (s)
            "path_speed_px_s",              # path_length / duration (px/s)
            "net_displacement_speed_px_s",  # net_disp / duration  (px/s)
            "straightness",                 # net_disp / path_len  ∈ [0,1]
            "mean_speed_px_s",              # mean instantaneous speed (px/s)
            "max_speed_px_s",               # max  instantaneous speed (px/s)
            "std_speed_px_s",               # std  of speed (px/s)
            "skew_speed",                   # dimensionless skewness
            "kurt_speed",                   # dimensionless kurtosis
            "mean_direction_deg",           # mean heading (°)
            "direction_std_deg",            # std of heading (°)
            "bbox_area",                    # spatial extent (px²)
            "mean_flow_magnitude",          # video-level LK flow magnitude
            "mean_flow_angle_deg",          # video-level LK flow angle (°)
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None

        fps = float(context.get("fps", 10.0))
        n_frames_tracked = len(traj)
        duration_sec = n_frames_tracked / fps      # this track's own duration

        deltas      = np.diff(traj, axis=0)
        speeds_pf   = np.linalg.norm(deltas, axis=1)   # px / frame
        speeds_ps   = speeds_pf * fps                   # px / s
        angles_deg  = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360

        total_dist_px = float(speeds_pf.sum())
        net_disp_px   = float(euclidean(traj[0], traj[-1]))
        straightness  = net_disp_px / total_dist_px if total_dist_px > 0 else 0.0

        bbox_wh   = traj.max(axis=0) - traj.min(axis=0)
        bbox_area = float(bbox_wh[0] * bbox_wh[1])

        return {
            "track_duration_sec":           duration_sec,
            "path_speed_px_s":              total_dist_px / duration_sec if duration_sec > 0 else 0.0,
            "net_displacement_speed_px_s":  net_disp_px   / duration_sec if duration_sec > 0 else 0.0,
            "straightness":                 straightness,
            "mean_speed_px_s":              float(speeds_ps.mean()),
            "max_speed_px_s":               float(speeds_ps.max()),
            "std_speed_px_s":               float(speeds_ps.std()),
            "skew_speed":                   float(skew(speeds_ps)) if len(speeds_ps) > 2 else 0.0,
            "kurt_speed":                   float(kurtosis(speeds_ps)) if len(speeds_ps) > 2 else 0.0,
            "mean_direction_deg":           float(angles_deg.mean()),
            "direction_std_deg":            float(angles_deg.std()),
            "bbox_area":                    bbox_area,
            "mean_flow_magnitude":          context.get("mean_flow_magnitude", 0.0),
            "mean_flow_angle_deg":          context.get("mean_flow_angle_deg", 0.0),
        }


@register("feature", "circular_motion")
class CircularMotionFeatures(FeatureExtractor):
    def feature_names(self) -> List[str]:
        return ["circular_mean_rad", "circular_variance", "direction_concentration", "angular_dispersion"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None
        deltas = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))
        circ_mean = np.arctan2(mean_sin, mean_cos)
        R = np.sqrt(mean_cos**2 + mean_sin**2)
        circ_var = 1.0 - R
        ang_diff = np.arctan2(np.sin(angles_rad - circ_mean), np.cos(angles_rad - circ_mean))
        return {
            "circular_mean_rad": float(circ_mean),
            "circular_variance": float(circ_var),
            "direction_concentration": float(R),
            "angular_dispersion": float(np.std(ang_diff)),
        }


@register("feature", "kinematics")
class KinematicsFeatures(FeatureExtractor):
    """
    Acceleration and jerk in physical units (mentor v6.0).

      speed     [px/frame]  × fps   = speed  [px/s]
      accel     Δspeed/Δt   × fps   = accel  [px/s²]  (Δt = 1/fps s)
      jerk      Δaccel/Δt   × fps   = jerk   [px/s³]

    Skewness statistics are dimensionless and unchanged.
    fps from context["fps"]; default 10.0.
    """

    def feature_names(self) -> List[str]:
        return [
            "mean_accel_px_s2", "std_accel_px_s2", "max_accel_px_s2", "skew_accel",
            "mean_jerk_px_s3",  "std_jerk_px_s3",  "max_jerk_px_s3",  "skew_jerk",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None
        fps = float(context.get("fps", 10.0))

        deltas     = np.diff(traj, axis=0)
        speeds_pf  = np.linalg.norm(deltas, axis=1)   # px/frame
        accel_ps2  = np.diff(speeds_pf) * fps          # px/s²  (×fps converts Δ/frame to Δ/s)
        jerk_ps3   = np.diff(accel_ps2) * fps          # px/s³

        if len(accel_ps2) < 2 or len(jerk_ps3) < 2:
            return None
        return {
            "mean_accel_px_s2": float(np.mean(accel_ps2)),
            "std_accel_px_s2":  float(np.std(accel_ps2)),
            "max_accel_px_s2":  float(np.max(np.abs(accel_ps2))),
            "skew_accel":       float(skew(accel_ps2)) if len(accel_ps2) > 2 else 0.0,
            "mean_jerk_px_s3":  float(np.mean(jerk_ps3)),
            "std_jerk_px_s3":   float(np.std(jerk_ps3)),
            "max_jerk_px_s3":   float(np.max(np.abs(jerk_ps3))),
            "skew_jerk":        float(skew(jerk_ps3))  if len(jerk_ps3)  > 2 else 0.0,
        }


@register("feature", "shape")
class ShapeFeatures(FeatureExtractor):
    """
    Trajectory shape descriptors (mentor v6.0 normalisation).

      mean/std curvature: |Δangle| per step (rad) — already dimensionless.
      turning_rate_rad_s: total angle turned ÷ duration_sec → rad/s.
        Captures direction-change rate independently of video length.
      convex_hull_area:   spatial extent in px² — length-independent within
        a video; kept raw.
      compactness:        4π·area/perimeter² ∈ [0,1] — dimensionless.
      path_efficiency:    net_disp / path_len ∈ [0,1] — dimensionless.

    fps from context["fps"]; default 10.0.
    """

    def feature_names(self) -> List[str]:
        return [
            "mean_curvature",      # mean |Δangle| per step (rad)
            "std_curvature",       # std  |Δangle| per step (rad)
            "turning_rate_rad_s",  # total angle turned / duration (rad/s)
            "convex_hull_area",    # spatial extent (px²)
            "compactness",         # 4π·area/perimeter²
            "path_efficiency",     # net_disp / path_len ∈ [0,1]
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None

        fps          = float(context.get("fps", 10.0))
        duration_sec = len(traj) / fps

        deltas     = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        curvature  = np.abs(np.diff(angles_rad))
        curvature  = np.minimum(curvature, 2 * np.pi - curvature)

        total_angle   = float(curvature.sum())
        turning_rate  = total_angle / duration_sec if duration_sec > 0 else 0.0

        hull_area   = 0.0
        compactness = 0.0
        try:
            if len(traj) >= 3:
                hull = ConvexHull(traj)
                hull_area  = float(hull.volume)
                hull_perim = float(hull.area)
                if hull_perim > 0:
                    compactness = 4 * np.pi * hull_area / (hull_perim ** 2)
        except Exception:
            pass

        total_dist = float(np.sum(np.linalg.norm(deltas, axis=1)))
        net_disp   = float(euclidean(traj[0], traj[-1]))
        path_eff   = net_disp / total_dist if total_dist > 0 else 0.0

        return {
            "mean_curvature":     float(np.mean(curvature)) if len(curvature) > 0 else 0.0,
            "std_curvature":      float(np.std(curvature))  if len(curvature) > 0 else 0.0,
            "turning_rate_rad_s": turning_rate,
            "convex_hull_area":   hull_area,
            "compactness":        compactness,
            "path_efficiency":    path_eff,
        }


@register("feature", "wavelet")
class WaveletFeatures(FeatureExtractor):
    def __init__(self, wavelet="db4", level=3, min_trajectory_len=30):
        self.wavelet = wavelet
        self.level = level
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return ["dwt_energy_approx", "dwt_energy_detail1", "dwt_energy_detail2",
                "dwt_energy_detail3", "dwt_energy_total", "dwt_entropy"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        if not PYWAVELETS_AVAILABLE:
            return None
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None
        fps    = float(context.get("fps", 10.0))
        deltas = np.diff(traj, axis=0)
        # Normalise to px/s so wavelet energies are in (px/s)², comparable
        # across videos with different frame rates (mentor v6.0).
        # Research: Zhang et al. (2022) — DWT on speed signals.
        speeds = np.linalg.norm(deltas, axis=1) * fps   # px/s
        n = len(speeds)
        pad_len = 2 ** int(np.ceil(np.log2(max(n, 8)))) - n
        if pad_len > 0:
            speeds = np.pad(speeds, (0, pad_len), mode="reflect")
        try:
            coeffs = pywt.wavedec(speeds, self.wavelet, level=self.level)
        except Exception:
            return None
        energies = [float(np.sum(np.array(c) ** 2)) for c in coeffs]
        total_energy = sum(energies)
        eps = 1e-10
        energy_dist = np.array(energies) / (total_energy + eps)
        entropy = float(-np.sum(energy_dist * np.log(energy_dist + eps)))
        result = {"dwt_energy_approx": energies[0] if energies else 0.0}
        for i in range(1, 4):
            result[f"dwt_energy_detail{i}"] = energies[i] if i < len(energies) else 0.0
        result["dwt_energy_total"] = total_energy
        result["dwt_entropy"] = entropy
        return result


@register("feature", "interaction")
class InteractionFeatures(FeatureExtractor):
    def __init__(self, proximity_threshold=100.0, min_trajectory_len=30):
        self.proximity_threshold = proximity_threshold
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return ["mean_pairwise_distance", "min_pairwise_distance", "std_pairwise_distance",
                "mean_relative_speed", "max_relative_speed", "proximity_duration_ratio", "n_proximity_events"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """
        Interaction features with normalised speed and event rate (v6.0).

          relative speed:  px/frame → px/s  (× fps)
          pairwise_dists:  spatial (px) — unchanged (not temporal)
          proximity_duration_ratio: already dimensionless fraction
          proximity_event_rate_s:  n_events / overlap_duration_sec (events/s)

        fps from context["fps"]; default 10.0.
        """
        all_tracks = context.get("all_tracks", [])
        if len(all_tracks) < 2:
            return None
        my_traj = np.array(track["trajectory"], dtype=float)
        if len(my_traj) < self.min_trajectory_len:
            return None

        fps = float(context.get("fps", 10.0))

        pairwise_dists: List[float] = []
        rel_speeds_ps:  List[float] = []
        proximity_frames = 0
        total_frames     = 0
        n_events         = 0

        for other in all_tracks:
            if other["id"] == track["id"]:
                continue
            other_traj = np.array(other["trajectory"], dtype=float)
            min_len = min(len(my_traj), len(other_traj))
            if min_len < 2:
                continue
            in_proximity = False
            for i in range(min_len):
                d = euclidean(my_traj[i], other_traj[i])
                pairwise_dists.append(d)
                total_frames += 1
                if d < self.proximity_threshold:
                    proximity_frames += 1
                    if not in_proximity:
                        n_events += 1
                        in_proximity = True
                else:
                    in_proximity = False
            my_spd_pf    = np.linalg.norm(np.diff(my_traj[:min_len],    axis=0), axis=1)
            other_spd_pf = np.linalg.norm(np.diff(other_traj[:min_len], axis=0), axis=1)
            # Convert differential speed from px/frame to px/s
            rel_speeds_ps.extend(np.abs(my_spd_pf - other_spd_pf) * fps)

        if not pairwise_dists:
            return None

        overlap_sec = total_frames / fps if fps > 0 else 1.0
        return {
            "mean_pairwise_distance":   float(np.mean(pairwise_dists)),
            "min_pairwise_distance":    float(np.min(pairwise_dists)),
            "std_pairwise_distance":    float(np.std(pairwise_dists)),
            "mean_relative_speed_px_s": float(np.mean(rel_speeds_ps)) if rel_speeds_ps else 0.0,
            "max_relative_speed_px_s":  float(np.max(rel_speeds_ps))  if rel_speeds_ps else 0.0,
            "proximity_duration_ratio": proximity_frames / total_frames if total_frames > 0 else 0.0,
            "proximity_event_rate_s":   n_events / overlap_sec if overlap_sec > 0 else 0.0,
        }


@register("feature", "dense_trajectories")
class DenseTrajectoryFeatures(FeatureExtractor):
    def __init__(self, min_trajectory_len=15, step_size=5, block_size=32, feature_types=None):
        self.min_trajectory_len = min_trajectory_len
        self.step_size = step_size
        self.block_size = block_size
        self.feature_types = feature_types or ["trajectory", "hog", "hof", "mbh"]
        self.hog_descriptor = cv2.HOGDescriptor(_winSize=(block_size, block_size), _blockSize=(16, 16),
                                                _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
        self.fast_detector = cv2.FastFeatureDetector_create(threshold=20)

    def feature_names(self) -> List[str]:
        names = []
        if "trajectory" in self.feature_types:
            names.extend(["dense_traj_length", "dense_traj_displacement", "dense_traj_mean_speed",
                          "dense_traj_std_speed", "dense_traj_direction_mean", "dense_traj_direction_std"])
        if "hog" in self.feature_types:
            names.extend([f"dense_hog_bin_{i}" for i in range(9)])
        if "hof" in self.feature_types:
            names.extend([f"dense_hof_bin_{i}" for i in range(9)])
        if "mbh" in self.feature_types:
            names.extend([f"dense_mbhx_bin_{i}" for i in range(9)])
            names.extend([f"dense_mbhy_bin_{i}" for i in range(9)])
        names.extend(["dense_keypoint_density", "dense_flow_magnitude_mean", "dense_flow_magnitude_std",
                      "dense_texture_variance", "dense_edge_density"])
        return names

    def extract_keypoints(self, gray_frame: np.ndarray) -> np.ndarray:
        h, w = gray_frame.shape[:2]
        points = []
        shi_tomasi = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01,
                                             minDistance=self.step_size, blockSize=3)
        if shi_tomasi is not None:
            points.extend(shi_tomasi.reshape(-1, 2))
        kp = self.fast_detector.detect(gray_frame, None)
        points.extend([kp_item.pt for kp_item in kp[:50]])
        for y in range(self.step_size, h - self.step_size, self.step_size):
            for x in range(self.step_size, w - self.step_size, self.step_size):
                patch = gray_frame[y-2:y+3, x-2:x+3]
                if np.std(patch) > 10:
                    points.append([float(x), float(y)])
        if points:
            points = np.array(points)
            if len(points) > 1:
                # Grid-based NMS
                cell = self.step_size
                seen: Dict[Tuple[int, int], bool] = {}
                keep = np.zeros(len(points), dtype=bool)
                for i, (px, py) in enumerate(points):
                    key = (int(px) // cell, int(py) // cell)
                    if key not in seen:
                        seen[key] = True
                        keep[i] = True
                points = points[keep]
        return points if len(points) > 0 else np.array([])

    def compute_local_descriptors(self, gray_patch: np.ndarray, flow_patch: np.ndarray) -> Dict[str, float]:
        features = {}
        if "hog" in self.feature_types and gray_patch.shape[0] >= self.block_size:
            try:
                resized = cv2.resize(gray_patch, (self.block_size, self.block_size))
                hog = self.hog_descriptor.compute(resized)
                if hog is not None:
                    hog = hog.reshape(-1, 9).mean(axis=0)
                    for i in range(min(9, len(hog))):
                        features[f"dense_hog_bin_{i}"] = float(hog[i])
            except Exception:
                for i in range(9):
                    features[f"dense_hog_bin_{i}"] = 0.0
        if "hof" in self.feature_types and flow_patch.size >= 2:
            try:
                mag, ang = cv2.cartToPolar(flow_patch[..., 0], flow_patch[..., 1])
                mask = mag > 1.0
                if mask.sum() > 0:
                    hof, _ = np.histogram(ang[mask], bins=9, range=(0, 2*np.pi))
                    hof = hof.astype(float) / (hof.sum() + 1e-7)
                else:
                    hof = np.zeros(9)
                for i in range(9):
                    features[f"dense_hof_bin_{i}"] = float(hof[i])
            except Exception:
                for i in range(9):
                    features[f"dense_hof_bin_{i}"] = 0.0
        if "mbh" in self.feature_types and flow_patch.size >= 4:
            try:
                grad_x = cv2.Sobel(flow_patch[..., 0], cv2.CV_32F, 1, 0)
                grad_y = cv2.Sobel(flow_patch[..., 1], cv2.CV_32F, 0, 1)
                mbhx, _ = np.histogram(grad_x.flatten(), bins=9, range=(-10, 10))
                mbhy, _ = np.histogram(grad_y.flatten(), bins=9, range=(-10, 10))
                mbhx = mbhx.astype(float) / (mbhx.sum() + 1e-7)
                mbhy = mbhy.astype(float) / (mbhy.sum() + 1e-7)
                for i in range(9):
                    features[f"dense_mbhx_bin_{i}"] = float(mbhx[i])
                    features[f"dense_mbhy_bin_{i}"] = float(mbhy[i])
            except Exception:
                for i in range(9):
                    features[f"dense_mbhx_bin_{i}"] = 0.0
                    features[f"dense_mbhy_bin_{i}"] = 0.0
        return features

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None
        features = {}
        if "trajectory" in self.feature_types:
            deltas = np.diff(traj, axis=0)
            speeds = np.linalg.norm(deltas, axis=1)
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            features.update({
                "dense_traj_length": float(len(traj)),
                "dense_traj_displacement": float(np.linalg.norm(traj[-1] - traj[0])),
                "dense_traj_mean_speed": float(np.mean(speeds)),
                "dense_traj_std_speed": float(np.std(speeds)),
                "dense_traj_direction_mean": float(np.mean(angles)),
                "dense_traj_direction_std": float(np.std(angles))
            })
        prev_gray = context.get("prev_gray")
        curr_gray = context.get("curr_gray")
        if prev_gray is not None and curr_gray is not None:
            h, w = curr_gray.shape[:2]
            keypoints = self.extract_keypoints(curr_gray)
            if len(keypoints) > 0:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_mags = np.linalg.norm(flow, axis=2)
                edges = cv2.Canny(curr_gray, 50, 150)
                features.update({
                    "dense_keypoint_density": float(len(keypoints)),
                    "dense_flow_magnitude_mean": float(np.mean(flow_mags)),
                    "dense_flow_magnitude_std": float(np.std(flow_mags)),
                    "dense_texture_variance": float(np.var(curr_gray)),
                    "dense_edge_density": float(np.sum(edges > 0) / edges.size)
                })
                sampled_kps = keypoints[:20]
                for pt in sampled_kps:
                    x, y = int(pt[0]), int(pt[1])
                    x1 = max(0, x - self.block_size//2)
                    y1 = max(0, y - self.block_size//2)
                    x2 = min(w, x + self.block_size//2)
                    y2 = min(h, y + self.block_size//2)
                    if x2 > x1 and y2 > y1:
                        gray_patch = curr_gray[y1:y2, x1:x2]
                        flow_patch = flow[y1:y2, x1:x2]
                        patch_feats = self.compute_local_descriptors(gray_patch, flow_patch)
                        for key, val in patch_feats.items():
                            features[key] = features.get(key, 0.0) + val
                if sampled_kps and len(sampled_kps) > 0:
                    for key in list(features.keys()):
                        if any(key.startswith(prefix) for prefix in ["dense_hog_", "dense_hof_", "dense_mbh"]):
                            features[key] /= len(sampled_kps)
        return features if features else None


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                SECTION 11: FEATURE AGGREGATION                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class FeatureAggregator:
    def __init__(self, aggregation_fns=None):
        self.aggregation_fns = aggregation_fns or ["mean", "max", "std"]

    def aggregate(self, track_features: List[FeatureDict]) -> Optional[FeatureDict]:
        if not track_features:
            return None
        all_keys = set()
        for tf in track_features:
            all_keys.update(tf.keys())
        video_fv = {"n_valid_tracks": float(len(track_features))}
        for key in sorted(all_keys):
            values = [tf[key] for tf in track_features if key in tf]
            if not values:
                continue
            arr = np.array(values, dtype=float)
            if "mean" in self.aggregation_fns:
                video_fv[f"mean_{key}"] = float(np.mean(arr))
            if "max" in self.aggregation_fns:
                video_fv[f"max_{key}"] = float(np.max(arr))
            if "std" in self.aggregation_fns:
                video_fv[f"std_{key}"] = float(np.std(arr)) if len(arr) > 1 else 0.0
        return video_fv


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                   SECTION 12: PIPELINE BUILDER                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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



# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                    SECTION 13: PIPELINE RUNNER                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║       SECTION 13B: MULTI-FAMILY FEATURE EXTRACTION  (Group I)              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 13B: EXPLAINABILITY VISUALIZER                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

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
                      visualise_path: Optional[Path] = None) -> VideoProcessingResult:
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
                failure=FailureReason(detection_failure=True))

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
                    failure=fail, tracker_metrics=tracker_metrics)
            if video_fv is None:
                return VideoProcessingResult(
                    tracks=tracks, mean_flow_magnitude=mean_mag,
                    mean_flow_angle_deg=mean_ang, n_frames_processed=frame_no,
                    used_gt=used_gt, preprocessing_diagnostics=diagnostics,
                    track_lengths=track_lengths,
                    detection_stats=detection_stats, tracking_stats=tracking_stats,
                    failure=FailureReason(feature_failure=True),
                    tracker_metrics=tracker_metrics)
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
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║              SECTION 14: DECISION TREE + RANDOM FOREST (Scratch)           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class DecisionTree(BaseEstimator):
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_state=42):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.random_state = random_state
        self.root = None
        self.n_total_features = None

    def fit(self, X, y):
        self.n_total_features = X.shape[1]
        self.n_split_features = (X.shape[1] if not self.n_features else min(X.shape[1], self.n_features))
        rng = np.random.default_rng(self.random_state)
        self.root = self._grow(X, y, depth=0, rng=rng)
        return self

    def _grow(self, X, y, depth, rng):
        if X.shape[0] == 0:
            return {"leaf": True, "value": 0}
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or X.shape[0] < self.min_samples_split):
            return {"leaf": True, "value": self._majority(y)}
        feat_idxs = rng.choice(self.n_total_features, self.n_split_features, replace=False)
        best_feat, best_thr = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return {"leaf": True, "value": self._majority(y)}
        left_mask = X[:, best_feat] <= best_thr
        return {"leaf": False, "feature": best_feat, "threshold": best_thr,
                "left": self._grow(X[left_mask], y[left_mask], depth + 1, rng),
                "right": self._grow(X[~left_mask], y[~left_mask], depth + 1, rng)}

    def _best_split(self, X, y, feat_idxs):
        best_gain, best_feat, best_thr = -1.0, None, None
        for feat in feat_idxs:
            col = X[:, feat]
            for thr in np.unique(col):
                gain = self._info_gain(y, col, thr)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr

    def _info_gain(self, y, col, thr):
        left_mask = col <= thr
        if left_mask.sum() < 1 or (~left_mask).sum() < 1:
            return 0.0
        n = len(y)
        n_l = left_mask.sum()
        return (self._entropy(y) - (n_l / n) * self._entropy(y[left_mask]) - ((n - n_l) / n) * self._entropy(y[~left_mask]))

    def _entropy(self, y):
        counts = np.bincount(y)
        ps = counts / len(y)
        return -np.sum(ps[ps > 0] * np.log(ps[ps > 0]))

    def _majority(self, y):
        return int(np.bincount(y).argmax()) if len(y) > 0 else 0

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])

    def compute_importances(self, n_features):
        importances = np.zeros(n_features)
        self._accumulate(self.root, importances)
        return importances

    def _accumulate(self, node, importances):
        if node["leaf"]:
            return
        importances[node["feature"]] += 1.0
        self._accumulate(node["left"], importances)
        self._accumulate(node["right"], importances)


class RandomForestCustom(BaseEstimator):
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, n_features=None, class_weight=None, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth or 1_000_000
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_feat = self.n_features or int(np.sqrt(X.shape[1]))
        self.classes_ = np.unique(y)
        self.trees = []
        sample_weights = None
        if self.class_weight == "balanced":
            class_counts = np.bincount(y)
            class_w = 1.0 / class_counts
            sample_weights = class_w[y]
            sample_weights /= sample_weights.sum()
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                n_features=n_feat, random_state=int(rng.integers(0, 1_000_000)))
            if sample_weights is not None:
                idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True, p=sample_weights)
            else:
                idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        raw = np.sum([t.compute_importances(X.shape[1]) for t in self.trees], axis=0)
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw
        return self

    def predict(self, X):
        all_preds = np.array([t.predict(X) for t in self.trees])
        result = []
        for i in range(all_preds.shape[1]):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            result.append(vals[np.argmax(counts)])
        return np.array(result)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║             SECTION 15: MODEL WRAPPERS                                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def _build_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


class SklearnParamsMixin:
    """Mixin that adds sklearn-compatible get_params / set_params via stored instance attrs."""

    def get_params(self, deep: bool = True) -> dict:
        import inspect
        sig = inspect.signature(self.__class__.__init__)
        return {k: getattr(self, k) for k in sig.parameters if k != "self" and hasattr(self, k)}

    def set_params(self, **params):
        current = self.get_params()
        current.update(params)
        self.__init__(**current)
        return self


class SklearnModelAdapter(ClassifierMixin, BaseEstimator, Model):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        try:
            return self.pipeline.predict_proba(X)
        except Exception:
            return None

    @property
    def feature_importances_(self):
        try:
            return self.pipeline.named_steps["clf"].feature_importances_
        except Exception:
            return None

    @property
    def classes_(self):
        try:
            return self.pipeline.named_steps["clf"].classes_
        except Exception:
            return None


@register("model", "random_forest_sklearn")
class RandomForestSklearnModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, n_estimators=200, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(_build_pipeline(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                                class_weight=class_weight, random_state=random_state, n_jobs=n_jobs)))


@register("model", "random_forest_custom")
class RandomForestCustomModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, n_trees=200, max_depth=None, min_samples_split=2, class_weight="balanced", random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        super().__init__(_build_pipeline(RandomForestCustom(n_trees=n_trees, max_depth=max_depth,
                                                            min_samples_split=min_samples_split, class_weight=class_weight,
                                                            random_state=random_state)))


@register("model", "svm_rbf")
class SVMRBFModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, C=10, gamma="scale", class_weight="balanced", random_state=42):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        super().__init__(_build_pipeline(SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
                                             random_state=random_state, probability=True)))


@register("model", "knn")
class KNNModel(SklearnParamsMixin, SklearnModelAdapter):
    def __init__(self, n_neighbors=5, metric="euclidean", **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        super().__init__(_build_pipeline(KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)))


@register("model", "lightgbm")
class LightGBMModel(SklearnParamsMixin, ClassifierMixin, BaseEstimator, Model):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31, class_weight="balanced", random_state=42):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state
        self._model = None
        self._scaler = StandardScaler()
        self._le = LabelEncoder()
        self._classes = None

    def fit(self, X, y):
        self._le.fit(y)
        y_enc = self._le.transform(y)
        self._classes = self._le.classes_
        X_scaled = self._scaler.fit_transform(X)
        self._model = lgb.LGBMClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                         learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                                         class_weight=self.class_weight, random_state=self.random_state, verbose=-1)
        self._model.fit(X_scaled, y_enc)
        return self

    def predict(self, X):
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._le.inverse_transform(self._model.predict(self._scaler.transform(X)))

    def predict_proba(self, X):
        if self._model is None:
            return None
        return self._model.predict_proba(self._scaler.transform(X))

    @property
    def feature_importances_(self):
        return self._model.feature_importances_ if self._model else None

    @property
    def classes_(self):
        return self._classes


@register("model", "xgboost")
class XGBoostModel(SklearnParamsMixin, ClassifierMixin, BaseEstimator, Model):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None
        self._scaler = StandardScaler()
        self._le = LabelEncoder()
        self._classes = None

    def fit(self, X, y):
        self._le.fit(y)
        y_enc = self._le.transform(y)
        self._classes = self._le.classes_
        X_scaled = self._scaler.fit_transform(X)
        class_counts = np.bincount(y_enc)
        sample_w = np.array([max(class_counts) / class_counts[yi] for yi in y_enc])
        self._model = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        learning_rate=self.learning_rate, random_state=self.random_state,
                                        use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
        self._model.fit(X_scaled, y_enc, sample_weight=sample_w)
        return self

    def predict(self, X):
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._le.inverse_transform(self._model.predict(self._scaler.transform(X)))

    def predict_proba(self, X):
        if self._model is None:
            return None
        return self._model.predict_proba(self._scaler.transform(X))

    @property
    def feature_importances_(self):
        return self._model.feature_importances_ if self._model else None

    @property
    def classes_(self):
        return self._classes


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║              SECTION 16: CNN COMPARISON BASELINES (Group C only)           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

if TORCH_AVAILABLE:
    # Simple in-memory frame cache so repeated epochs don't re-decode every
    # video from disk every time (Group C review note: "full video decoding
    # every sample" — high computational cost, low scientific cost, but easy
    # to fix and meaningfully speeds up training).
    _FRAME_CACHE: Dict[str, list] = {}
    _FRAME_CACHE_MAX_ITEMS = 256

    def _compute_class_weights(y_enc: np.ndarray, n_classes: int) -> "torch.Tensor":
        """Inverse-frequency class weights for CrossEntropyLoss, addressing
        Group C review note 'no class weighting' (AD-SVD is not balanced)."""
        counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (n_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)

    # ── GPU utilisation helpers (Group C review: "GPU sits idle waiting on
    # CPU-bound video decoding") ─────────────────────────────────────────────
    # Video decoding/resizing in VideoClipDataset.__getitem__ runs on CPU no
    # matter how the GPU is used, so the fix is to overlap that CPU work with
    # GPU compute rather than to make the GPU itself do more. This matters
    # *more* on a modest GPU: a weak card finishes each batch's forward/backward
    # pass quickly, so it is proportionally more exposed to CPU decode stalls
    # than a powerful one would be.
    _NUM_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

    def _make_loader(dataset, batch_size, shuffle, num_workers=None):
        """DataLoader factory used by every Group C model (CNN, CNN+LSTM,
        3D CNN) so train/val/predict loaders all get the same treatment:
        background workers decode/augment the next batch on CPU while the
        GPU is busy with the current one, pinned memory speeds up the CPU->GPU
        copy, and persistent_workers avoids re-spawning worker processes every
        epoch (video decoding worker start-up is not free)."""
        nw = _NUM_WORKERS if num_workers is None else num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

    def _to_device(x, device):
        """.to() with non_blocking=True — only actually asynchronous when
        paired with pin_memory=True (see _make_loader), otherwise behaves
        like a normal blocking transfer."""
        return x.to(device, non_blocking=True)

    class _AmpHelper:
        """Automatic Mixed Precision wrapper. On a basic/low-VRAM GPU this is
        usually a bigger win than on a powerful one: fp16 activations roughly
        halve memory use, which both speeds up the conv/LSTM math and leaves
        headroom to raise batch_size instead of being stuck at 4-8. Silently
        becomes a no-op on CPU-only machines."""

        def __init__(self, device: "torch.device", enabled: Optional[bool] = None):
            self.enabled = (enabled if enabled is not None else torch.cuda.is_available())
            self._scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
            self._device_type = "cuda" if device.type == "cuda" else "cpu"

        def autocast(self):
            return torch.autocast(device_type=self._device_type, enabled=self.enabled)

        def backward(self, loss):
            self._scaler.scale(loss).backward()

        def step(self, optimizer):
            self._scaler.step(optimizer)
            self._scaler.update()

    class VideoClipDataset(TorchDataset):
        def __init__(self, video_paths, labels, clip_len=16, frame_size=(112, 112), sampling="uniform",
                     transform=None, train: bool = False, augment: bool = False):
            self.video_paths = video_paths
            self.labels = labels
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.sampling = sampling
            self.train = train
            # Light, video-safe augmentation applied only to the training
            # split (Group C review: "Add lightweight augmentation").
            # Kept deliberately mild — flip / crop / brightness jitter only —
            # so per-frame transforms stay consistent and don't introduce
            # spurious motion artefacts within a clip.
            self.augment = augment and train
            if transform is not None:
                self.transform = transform
            elif self.augment:
                self.transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((int(frame_size[0] * 1.15), int(frame_size[1] * 1.15))),
                    T.RandomCrop(frame_size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = T.Compose([
                    T.ToPILImage(), T.Resize(frame_size), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        def __len__(self): return len(self.video_paths)

        def __getitem__(self, idx):
            path = self.video_paths[idx]
            frames = self._extract_frames(path)
            if len(frames) >= self.clip_len:
                if self.sampling == "uniform":
                    indices = np.linspace(0, len(frames) - 1, self.clip_len, dtype=int)
                else:
                    start = random.randint(0, len(frames) - self.clip_len)
                    indices = range(start, start + self.clip_len)
            else:
                indices = list(range(len(frames)))
                while len(indices) < self.clip_len:
                    indices.append(len(frames) - 1)
            # Apply the same random crop/flip decision across every frame in
            # the clip so augmentation doesn't introduce inter-frame jitter
            # that would look like spurious motion.
            if self.augment:
                seed = random.randint(0, 2**31 - 1)
                frames_out = []
                for i in indices[:self.clip_len]:
                    torch.manual_seed(seed)
                    random.seed(seed)
                    frames_out.append(self.transform(frames[i]))
                clip = torch.stack(frames_out)
            else:
                clip = torch.stack([self.transform(frames[i]) for i in indices[:self.clip_len]])
            return clip, torch.tensor(self.labels[idx])

        def _extract_frames(self, path):
            # Cache key includes frame_size since we now store *resized*
            # frames (see below) rather than raw full-resolution frames.
            cache_key = (path, self.frame_size)
            cached = _FRAME_CACHE.get(cache_key)
            if cached is not None:
                return cached
            cap = cv2.VideoCapture(str(path))
            frames = []
            # Resize target as (width, height) for cv2.resize, derived from
            # frame_size which is (H, W) elsewhere in this class.
            resize_wh = (self.frame_size[1], self.frame_size[0])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Downscale immediately, before converting/appending. The
                # original full-resolution frames (e.g. 1920x956, ~5.25MB
                # each) were being kept in RAM and cached across up to 256
                # videos, which is what was blowing through available
                # memory (numpy/cv2/torch OOM errors) — nothing downstream
                # ever needs more than frame_size pixels per frame.
                frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if not frames:
                frames = [np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)]
            if len(_FRAME_CACHE) < _FRAME_CACHE_MAX_ITEMS:
                _FRAME_CACHE[cache_key] = frames
            return frames

    def _train_val_split_paths(X, y, val_size=0.15, seed=42):
        """Split off a validation set for early stopping. Falls back to a
        non-stratified split (or no split, if too few samples) gracefully,
        mirroring the stratification fallback already used elsewhere in the
        pipeline (Group C review: 'no validation split / early stopping')."""
        y = np.asarray(y)
        n_classes = len(np.unique(y))
        if len(y) < max(4, n_classes * 2):
            # Too small to carve out a validation set — train on everything,
            # skip early stopping for this fold.
            return X, [], y, np.array([])
        try:
            return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
        except ValueError:
            return train_test_split(X, y, test_size=val_size, random_state=seed)

    @register("model", "cnn")
    class CNNModel(Model):
        """2D-CNN baseline.

        Group C review fix (Major issue 1): the previous implementation
        averaged raw frames into a single static image before the backbone
        ever saw them, which discards motion/ordering/acceleration entirely.
        Here each frame is passed through the ResNet backbone individually
        and the resulting *feature vectors* are averaged ("late pooling"),
        which preserves far more of the per-frame visual information while
        keeping the model a lightweight 2D-CNN comparison baseline (the
        ordering-aware comparison is CNNLSTMModel / CNN3DModel).
        """

        def __init__(self, backbone="resnet18_2d", pretrained=False, lr=0.001, epochs=15, batch_size=8,
                     random_state=42, clip_len=16, frame_size=(112, 112),
                     val_size=0.15, patience=5, use_class_weights=True, augment=True):
            self.backbone = backbone
            self.pretrained = pretrained
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.val_size = val_size
            self.patience = patience
            self.use_class_weights = use_class_weights
            self.augment = augment
            self._model = None
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._amp = _AmpHelper(self._device)
            self._le = LabelEncoder()
            self._classes = None

        def _build_backbone(self, n_classes):
            model = tv_models.resnet18(pretrained=self.pretrained)
            model.fc = nn.Linear(512, n_classes)
            return model.to(self._device)

        def _forward_pooled(self, clips):
            # clips: (B, T, C, H, W) -> per-frame backbone features averaged
            # over T, then classified. This is what replaces the old
            # `clips.mean(dim=1)` raw-pixel averaging.
            B, T, C, H, W = clips.shape
            feats = self._features(clips.view(B * T, C, H, W))
            feats = feats.view(B, T, -1).mean(dim=1)
            return self._classifier(feats)

        def fit(self, X, y):
            self._le.fit(y)
            y_enc = self._le.transform(y)
            self._classes = self._le.classes_
            n_classes = len(self._classes)
            backbone = self._build_backbone(n_classes)
            self._features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()).to(self._device)
            self._classifier = nn.Linear(512, n_classes).to(self._device)
            self._model = backbone  # kept for checkpointing compatibility

            X_tr, X_val, y_tr, y_val = _train_val_split_paths(X, y_enc, val_size=self.val_size, seed=self.random_state)
            params = list(self._features.parameters()) + list(self._classifier.parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr)
            weight = _compute_class_weights(y_tr, n_classes).to(self._device) if self.use_class_weights else None
            criterion = nn.CrossEntropyLoss(weight=weight)

            train_ds = VideoClipDataset(X_tr, y_tr, clip_len=self.clip_len, frame_size=self.frame_size,
                                         train=True, augment=self.augment)
            train_loader = _make_loader(train_ds, self.batch_size, shuffle=True)
            val_loader = None
            if len(X_val):
                val_ds = VideoClipDataset(X_val, y_val, clip_len=self.clip_len, frame_size=self.frame_size, train=False)
                val_loader = _make_loader(val_ds, self.batch_size, shuffle=False)

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0
            for epoch in range(self.epochs):
                self._features.train(); self._classifier.train()
                total_loss = 0
                for clips, labels in train_loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    optimizer.zero_grad()
                    with self._amp.autocast():
                        loss = criterion(self._forward_pooled(clips), labels)
                    self._amp.backward(loss)
                    self._amp.step(optimizer)
                    total_loss += loss.item()
                msg = f"    Epoch {epoch+1}/{self.epochs}  train_loss={total_loss/len(train_loader):.4f}"

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader, criterion)
                    msg += f"  val_loss={val_loss:.4f}"
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        best_state = (copy.deepcopy(self._features.state_dict()),
                                      copy.deepcopy(self._classifier.state_dict()))
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    print(msg)
                    if epochs_no_improve >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1} (patience={self.patience})")
                        break
                else:
                    print(msg)

            if best_state is not None:
                self._features.load_state_dict(best_state[0])
                self._classifier.load_state_dict(best_state[1])
            return self

        def _eval_loss(self, loader, criterion):
            self._features.eval(); self._classifier.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for clips, labels in loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    with self._amp.autocast():
                        loss = criterion(self._forward_pooled(clips), labels)
                    total += loss.item() * len(labels)
                    n += len(labels)
            return total / max(n, 1)

        def predict(self, X):
            if self._model is None: raise ValueError("Model not fitted")
            self._features.eval(); self._classifier.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips, self._device)
                    with self._amp.autocast():
                        out = self._forward_pooled(clips)
                    preds.extend(out.argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        def predict_proba(self, X):
            if self._model is None: return None
            self._features.eval(); self._classifier.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            probas = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips, self._device)
                    with self._amp.autocast():
                        logits = self._forward_pooled(clips)
                    probas.extend(torch.softmax(logits, dim=1).float().cpu().numpy())
            return np.array(probas)

        @property
        def classes_(self): return self._classes

    @register("model", "cnn_lstm")
    class CNNLSTMModel(Model):
        def __init__(self, backbone="resnet18_2d", hidden_size=256, num_layers=1, lr=0.0005, epochs=20,
                     batch_size=8, random_state=42, clip_len=16, frame_size=(112, 112),
                     val_size=0.15, patience=5, use_class_weights=True, augment=True):
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.val_size = val_size
            self.patience = patience
            self.use_class_weights = use_class_weights
            self.augment = augment
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._amp = _AmpHelper(self._device)
            self._le = LabelEncoder()
            self._classes = None
            self._cnn = None
            self._lstm = None
            self._classifier = None

        def _build(self, n_classes):
            cnn = tv_models.resnet18(pretrained=False)
            cnn.fc = nn.Identity()
            self._cnn = cnn.to(self._device)
            self._lstm = nn.LSTM(512, self.hidden_size, self.num_layers, batch_first=True).to(self._device)
            self._classifier = nn.Linear(self.hidden_size, n_classes).to(self._device)

        def _forward(self, clips):
            B, T, C, H, W = clips.shape
            feats = self._cnn(clips.view(B * T, C, H, W)).view(B, T, -1)
            out, _ = self._lstm(feats)
            return self._classifier(out[:, -1, :])

        def fit(self, X, y):
            self._le.fit(y)
            y_enc = self._le.transform(y)
            self._classes = self._le.classes_
            self._build(len(self._classes))
            X_tr, X_val, y_tr, y_val = _train_val_split_paths(X, y_enc, val_size=self.val_size, seed=self.random_state)
            params = list(self._cnn.parameters()) + list(self._lstm.parameters()) + list(self._classifier.parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr)
            weight = _compute_class_weights(y_tr, len(self._classes)).to(self._device) if self.use_class_weights else None
            criterion = nn.CrossEntropyLoss(weight=weight)

            train_ds = VideoClipDataset(X_tr, y_tr, clip_len=self.clip_len, frame_size=self.frame_size,
                                         train=True, augment=self.augment)
            train_loader = _make_loader(train_ds, self.batch_size, shuffle=True)
            val_loader = None
            if len(X_val):
                val_ds = VideoClipDataset(X_val, y_val, clip_len=self.clip_len, frame_size=self.frame_size, train=False)
                val_loader = _make_loader(val_ds, self.batch_size, shuffle=False)

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0
            for epoch in range(self.epochs):
                self._cnn.train(); self._lstm.train(); self._classifier.train()
                total_loss = 0
                for clips, labels in train_loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    optimizer.zero_grad()
                    with self._amp.autocast():
                        loss = criterion(self._forward(clips), labels)
                    self._amp.backward(loss)
                    self._amp.step(optimizer)
                    total_loss += loss.item()
                msg = f"    Epoch {epoch+1}/{self.epochs}  train_loss={total_loss/len(train_loader):.4f}"

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader, criterion)
                    msg += f"  val_loss={val_loss:.4f}"
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        best_state = (copy.deepcopy(self._cnn.state_dict()),
                                      copy.deepcopy(self._lstm.state_dict()),
                                      copy.deepcopy(self._classifier.state_dict()))
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    print(msg)
                    if epochs_no_improve >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1} (patience={self.patience})")
                        break
                else:
                    print(msg)

            if best_state is not None:
                self._cnn.load_state_dict(best_state[0])
                self._lstm.load_state_dict(best_state[1])
                self._classifier.load_state_dict(best_state[2])
            return self

        def _eval_loss(self, loader, criterion):
            self._cnn.eval(); self._lstm.eval(); self._classifier.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for clips, labels in loader:
                    clips = _to_device(clips, self._device)
                    labels = _to_device(labels, self._device)
                    with self._amp.autocast():
                        loss = criterion(self._forward(clips), labels)
                    total += loss.item() * len(labels)
                    n += len(labels)
            return total / max(n, 1)

        def predict(self, X):
            if self._classifier is None: raise ValueError("Model not fitted")
            self._cnn.eval(); self._lstm.eval(); self._classifier.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips, self._device)
                    with self._amp.autocast():
                        out = self._forward(clips)
                    preds.extend(out.argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        @property
        def classes_(self): return self._classes

    @register("model", "cnn3d")
    class CNN3DModel(Model):
        def __init__(self, backbone="r3d_18", pretrained=False, lr=0.0005, epochs=20, batch_size=4,
                     random_state=42, clip_len=16, frame_size=(112, 112),
                     val_size=0.15, patience=5, use_class_weights=True, augment=True):
            self.backbone = backbone
            self.pretrained = pretrained
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.clip_len = clip_len
            self.frame_size = frame_size
            self.val_size = val_size
            self.patience = patience
            self.use_class_weights = use_class_weights
            self.augment = augment
            self._model = None
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._amp = _AmpHelper(self._device)
            self._le = LabelEncoder()
            self._classes = None

        def _build(self, n_classes):
            model = tv_models.video.r3d_18(pretrained=self.pretrained)
            model.fc = nn.Linear(512, n_classes)
            return model.to(self._device)

        def fit(self, X, y):
            self._le.fit(y)
            y_enc = self._le.transform(y)
            self._classes = self._le.classes_
            self._model = self._build(len(self._classes))
            X_tr, X_val, y_tr, y_val = _train_val_split_paths(X, y_enc, val_size=self.val_size, seed=self.random_state)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            weight = _compute_class_weights(y_tr, len(self._classes)).to(self._device) if self.use_class_weights else None
            criterion = nn.CrossEntropyLoss(weight=weight)

            train_ds = VideoClipDataset(X_tr, y_tr, clip_len=self.clip_len, frame_size=self.frame_size,
                                         train=True, augment=self.augment)
            train_loader = _make_loader(train_ds, self.batch_size, shuffle=True)
            val_loader = None
            if len(X_val):
                val_ds = VideoClipDataset(X_val, y_val, clip_len=self.clip_len, frame_size=self.frame_size, train=False)
                val_loader = _make_loader(val_ds, self.batch_size, shuffle=False)

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0
            for epoch in range(self.epochs):
                self._model.train()
                total_loss = 0
                for clips, labels in train_loader:
                    clips = _to_device(clips.permute(0, 2, 1, 3, 4), self._device)
                    labels = _to_device(labels, self._device)
                    optimizer.zero_grad()
                    with self._amp.autocast():
                        loss = criterion(self._model(clips), labels)
                    self._amp.backward(loss)
                    self._amp.step(optimizer)
                    total_loss += loss.item()
                msg = f"    Epoch {epoch+1}/{self.epochs}  train_loss={total_loss/len(train_loader):.4f}"

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader, criterion)
                    msg += f"  val_loss={val_loss:.4f}"
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(self._model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    print(msg)
                    if epochs_no_improve >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1} (patience={self.patience})")
                        break
                else:
                    print(msg)

            if best_state is not None:
                self._model.load_state_dict(best_state)
            return self

        def _eval_loss(self, loader, criterion):
            self._model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for clips, labels in loader:
                    clips = _to_device(clips.permute(0, 2, 1, 3, 4), self._device)
                    labels = _to_device(labels, self._device)
                    with self._amp.autocast():
                        loss = criterion(self._model(clips), labels)
                    total += loss.item() * len(labels)
                    n += len(labels)
            return total / max(n, 1)

        def predict(self, X):
            if self._model is None: raise ValueError("Model not fitted")
            self._model.eval()
            dataset = VideoClipDataset(X, [0]*len(X), clip_len=self.clip_len, frame_size=self.frame_size)
            loader = _make_loader(dataset, self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips = _to_device(clips.permute(0, 2, 1, 3, 4), self._device)
                    with self._amp.autocast():
                        out = self._model(clips)
                    preds.extend(out.argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        @property
        def classes_(self): return self._classes


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 17: TEMPORAL SMOOTHER                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TemporalSmoother:
    def __init__(self, method="majority_vote", ema_alpha=0.7):
        self.method = method
        self.ema_alpha = ema_alpha

    def smooth(self, predictions_by_model: Dict[str, np.ndarray], accuracy_by_model: Dict[str, float] = None) -> np.ndarray:
        model_names = list(predictions_by_model.keys())
        if not model_names:
            return np.array([])
        if len(model_names) == 1:
            return predictions_by_model[model_names[0]]
        if self.method == "majority_vote":
            return self._majority_vote(predictions_by_model, model_names)
        elif self.method == "ema":
            return self._ema_vote(predictions_by_model, model_names, accuracy_by_model)
        return predictions_by_model[model_names[0]]

    def _majority_vote(self, predictions_by_model, model_names):
        n_samples = len(predictions_by_model[model_names[0]])
        result = []
        for i in range(n_samples):
            votes = [predictions_by_model[m][i] for m in model_names]
            labels, counts = np.unique(votes, return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.array(result)

    def _ema_vote(self, predictions_by_model, model_names, accuracy_by_model):
        if not accuracy_by_model:
            return self._majority_vote(predictions_by_model, model_names)
        sorted_names = sorted(model_names, key=lambda m: accuracy_by_model.get(m, 0.0))
        weight = 1.0
        weights = {}
        for name in sorted_names:
            weights[name] = weight
            weight *= self.ema_alpha
        n_samples = len(predictions_by_model[sorted_names[0]])
        result = []
        for i in range(n_samples):
            vote_count = {}
            for name in sorted_names:
                label = predictions_by_model[name][i]
                vote_count[label] = vote_count.get(label, 0) + weights[name]
            result.append(max(vote_count, key=vote_count.get))
        return np.array(result)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 18: PLOT MANAGER                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class PlotManager:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check(self) -> bool:
        if not MATPLOTLIB_AVAILABLE:
            print("  [plot] Matplotlib not available — skipping plot")
            return False
        return True

    def confusion_matrix_plot(self, cm: np.ndarray, class_names: List[str], title: str, filename: str):
        if not self._check():
            return
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues', ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def combined_confusion_matrices(self, cm_activity: np.ndarray, classes_activity: List[str],
                                    cm_distortion: np.ndarray, classes_distortion: List[str],
                                    title: str, filename: str):
        if not self._check():
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(cm_activity, annot=True, fmt='d', xticklabels=classes_activity, yticklabels=classes_activity,
                    cmap='Blues', ax=axes[0], cbar=True)
        axes[0].set_title("Activity Confusion Matrix")
        axes[0].set_ylabel("True")
        axes[0].set_xlabel("Predicted")
        sns.heatmap(cm_distortion, annot=True, fmt='d', xticklabels=classes_distortion, yticklabels=classes_distortion,
                    cmap='Oranges', ax=axes[1], cbar=True)
        axes[1].set_title("Distortion Confusion Matrix")
        axes[1].set_ylabel("True")
        axes[1].set_xlabel("Predicted")
        plt.suptitle(title)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def feature_importance_plot(self, feature_names: List[str], importances: np.ndarray, title: str, filename: str, top_n: int = 20):
        if not self._check() or len(importances) != len(feature_names):
            return
        order = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in order]
        top_values = importances[order]
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        ax.barh(range(len(top_names)), top_values, color="steelblue")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def model_comparison_plot(self, results: Dict[str, Dict], metric: str, target: str, filename: str):
        if not self._check():
            return
        model_names = []
        metric_vals = []
        for mname, res in results.items():
            if metric in res and isinstance(res[metric], (int, float)):
                model_names.append(mname)
                metric_vals.append(res[metric])
        if not model_names:
            return
        fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.2), 5))
        bars = ax.bar(range(len(model_names)), metric_vals, color="steelblue")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric} by model — target: {target}")
        for bar, val in zip(bars, metric_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def detector_stats_plot(self, stats_by_distortion: Dict[str, Dict], filename: str):
        if not self._check() or not stats_by_distortion:
            return
        dist_names = list(stats_by_distortion.keys())
        rates = [stats_by_distortion[d].get("success_rate", 0.0) for d in dist_names]
        fig, ax = plt.subplots(figsize=(max(5, len(dist_names) * 1.5), 5))
        bars = ax.bar(dist_names, rates, color=plt.cm.tab10(np.linspace(0, 1, len(dist_names))))
        ax.set_ylabel("Detection success rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Detection success rate by distortion type")
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{rate:.2f}", ha="center", va="bottom")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def track_length_histogram(self, all_track_lengths: List[int], title: str, filename: str):
        if not self._check() or not all_track_lengths:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(all_track_lengths, bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel("Track length (frames)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 19: FIFTYONE VISUALISATION                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class FiftyOneVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}

    def create_video_dataset(self, video_path: Path, tracks: List[Track],
                             features: Optional[Dict[str, float]] = None,
                             detections: Optional[List[BBox]] = None,
                             gt_bboxes: Optional[List[BBox]] = None,
                             video_name: str = ""):
        if not FIFTYONE_AVAILABLE:
            print("  [warn] FiftyOne not installed — skipping visualisation")
            return None
        dataset = fo.Dataset(name=f"{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"  [fiftyone] Created dataset skeleton for {video_name}")
        return dataset

    def launch_session(self, dataset):
        if FIFTYONE_AVAILABLE and dataset is not None:
            session = fo.launch_app(dataset)
            return session
        return None


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 20: EXPERIMENT RUNNER UTILITIES                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def get_next_run_id(experiment_dir: Path) -> str:
    existing_numbers = []
    if experiment_dir.exists():
        for d in experiment_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    existing_numbers.append(int(d.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
    next_num = max(existing_numbers, default=0) + 1
    return f"run_{next_num:03d}"


def get_git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_min_trajectory_len(n_frames: int, base_min: int = 30) -> int:
    """
    Reference trajectory length, scaled to video duration.

    v6.0: NOT a hard gate.  Used only as the denominator in
    AdaptiveMOG2Detector.feedback() and for reporting.  The pipeline
    processes every track regardless of length; multi-family blending
    compensates when no tracks exist (Mauthner et al. 2009).
    """
    return min(base_min, max(10, int(n_frames * 0.6)))


# ── NEW: Training viability check ─────────────────────────────────────────────
def check_training_viability(
    y: np.ndarray,
    n_folds: int = 5,
    min_per_class: int = 3,
) -> Tuple[bool, str]:
    counts = Counter(y)
    n_samples = len(y)
    n_classes = len(counts)

    if n_samples < n_classes * min_per_class:
        return False, (
            f"Only {n_samples} samples for {n_classes} classes "
            f"(need ≥ {n_classes * min_per_class})"
        )

    singleton_classes = [c for c, n in counts.items() if n < min_per_class]
    if singleton_classes:
        return False, (
            f"Classes with < {min_per_class} samples: {singleton_classes}. "
            f"Results would be statistically meaningless."
        )

    if n_samples < n_folds * n_classes:
        return False, (
            f"{n_folds}-fold CV not viable: need ≥ {n_folds * n_classes} samples, "
            f"have {n_samples}"
        )

    return True, "OK"


# ── NEW: Cross‑validation evaluator ──────────────────────────────────────────
def evaluate_with_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    label_encoder: LabelEncoder = None,
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model,
        X, y,
        cv=skf,
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
        return_train_score=True,
        return_estimator=True,
    )
    return {
        "cv_accuracy_mean":  float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_std":   float(cv_results["test_accuracy"].std()),
        "cv_f1_mean":        float(cv_results["test_f1_weighted"].mean()),
        "cv_f1_std":         float(cv_results["test_f1_weighted"].std()),
        "train_accuracy_mean": float(cv_results["train_accuracy"].mean()),
        "overfit_gap":       float(
            cv_results["train_accuracy"].mean() - cv_results["test_accuracy"].mean()
        ),
        "cv_scores": {
            "accuracy": cv_results["test_accuracy"].tolist(),
            "f1_weighted": cv_results["test_f1_weighted"].tolist(),
        },
    }


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 21: EXPERIMENT RUNNER                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class ExperimentRunner:
    def __init__(self, config: Config, experiment_name: str = None,
                 detection_source: Optional[DetectionSource] = None,
                 diagnose_only: bool = False,
                 visualise: bool = False,
                 n_visualise: int = 3):
        self.config = config
        self.experiment_name = experiment_name or config.experiment.name
        self.seed = config.experiment.get("seed", 42)
        self.is_comparison = config.get("comparison_only", False)
        self.detection_source = detection_source
        self.diagnose_only = diagnose_only
        self.visualise = visualise
        self.n_visualise = n_visualise

        if config.dataset.get("use_dataset_info", False):
            self.dataset_info = DatasetInfoHandler(config.paths.dataset_excel)
        else:
            self.dataset_info = None

        exp_dir = EXPERIMENTS_DIR / self.experiment_name
        self.run_id = get_next_run_id(exp_dir)
        self.run_dir = exp_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.plots = PlotManager(self.run_dir / "plots")
        self.fo_visualizer = FiftyOneVisualizer(self.run_dir / "fiftyone")

        random.seed(self.seed)
        np.random.seed(self.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)

        if self.is_comparison:
            print(f"\n  [NOTE] Group {config.experiment.group} is a comparison baseline.")
        if self.dataset_info and config.dataset.get("justify_parameters", False):
            self._print_parameter_justifications()

    def _print_parameter_justifications(self):
        print("\n" + "=" * 70)
        print("  PARAMETER JUSTIFICATIONS (from datasetInfo.xlsx analysis)")
        print("=" * 70)
        print(self.dataset_info.justify_resize_dimensions(tuple(self.config.preprocessing.frame_resize)))
        print(self.dataset_info.justify_min_trajectory_length(self.config.tracking.min_trajectory_len))
        print("=" * 70)

    def load_dataset(self) -> pd.DataFrame:
        if hasattr(self.config, 'paths') and hasattr(self.config.paths, 'dataset_excel'):
            excel_path = self.config.paths.dataset_excel
        else:
            excel_path = "data/datasetInfo.xlsx"
        excel_path = Path(excel_path)
        if not excel_path.is_absolute():
            excel_path = PROJECT_ROOT / excel_path
        if not excel_path.exists():
            raise FileNotFoundError(f"Dataset Excel not found: {excel_path}")
        df = pd.read_excel(excel_path)
        required = {"Activity", "Distortion", "Name of Video Series"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Excel missing columns: {missing}")
        return df

    def prepare_clip_dataset(self, df: pd.DataFrame, use_sample: bool = False) -> Dict:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")
        target_col = self.config.targets[0] if self.config.targets else "activity"
        col_mapping = {"activity": "Activity", "distortion": "Distortion"}
        actual_col = col_mapping.get(target_col, target_col)
        if actual_col not in df.columns:
            raise KeyError(f"Column '{actual_col}' not found. Available: {df.columns.tolist()}")
        groups = {}
        for target_val, grp in df.groupby(actual_col):
            pool = grp["Name of Video Series"].tolist()
            if use_sample:
                n = self.config.dataset.get("n_random_videos_per_group", 5)
                pool = random.sample(pool, min(n, len(pool)))
            groups[target_val] = pool
        video_paths = []
        labels = []
        for label, vnames in groups.items():
            for vname in vnames:
                vpath = Path(self.config.paths.video_dir) / vname
                if vpath.exists():
                    video_paths.append(str(vpath))
                    labels.append(label)
        le = LabelEncoder()
        y_enc = le.fit_transform(labels)
        return {"X": video_paths, "y": y_enc, "label_encoder": le, "classes": le.classes_}

    def train_neural_model(self, clip_data: Dict, target_col: str = "activity") -> Dict:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        X = clip_data["X"]
        y = clip_data["y"]
        le = clip_data["label_encoder"]
        classes = clip_data["classes"]
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.training.test_size,
                                                                stratify=y, random_state=self.seed)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.training.test_size,
                                                                random_state=self.seed)
        results = {}
        clip_len = getattr(self.config.clip, "clip_len", 16)
        frame_size = tuple(getattr(self.config.clip, "frame_size", [112, 112]))
        batch_size = getattr(self.config.clip, "batch_size", 8)

        for model_def in self.config.models.active:
            model_name = model_def.get("name", "")
            params = model_def.get("params", {})
            print(f"\n{'='*60}\n  Neural Model: {model_name}  |  Target: {target_col}")
            try:
                model = build("model", model_name, random_state=self.seed, **params)
                model.clip_len = clip_len
                model.frame_size = frame_size
                model.batch_size = batch_size
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
                self.plots.confusion_matrix_plot(cm, classes.tolist(), f"{model_name} — {target_col}",
                                                 f"cm_{model_name}_{target_col}.png")
                results[model_name] = {"accuracy": acc, "f1_weighted": f1, "precision": prec, "recall": rec,
                                       "confusion_matrix": cm.tolist(), "y_true": y_test, "y_pred": y_pred,
                                       "classes": classes.tolist()}
                if hasattr(model, "_model"):
                    torch.save(model._model.state_dict(), self.run_dir / f"{model_name}_{target_col}.pt")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                results[model_name] = {"error": str(e)}
        with open(self.run_dir / f"results_{target_col}_neural.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    def collect_video_features(self, df: pd.DataFrame, use_sample: bool = False) -> List[Dict]:
        groups = {}
        for (act, dist), grp in df.groupby(["Activity", "Distortion"]):
            pool = grp["Name of Video Series"].tolist()
            if use_sample:
                n = self.config.dataset.get("n_random_videos_per_group", 5)
                pool = random.sample(pool, min(n, len(pool)))
            groups[(act, dist)] = pool

        total = sum(len(v) for v in groups.values())
        print(f"\n  Videos to process: {total}")

        preprocessor, detector, tracker, optical_flow = PipelineBuilder.build_all(self.config, self.detection_source)

        extractors = []
        feature_mods = self.config.features.get("modules", [])
        _ACCEPTS_MIN_LEN = {"trajectory", "wavelet", "interaction", "dense_trajectories"}
        for mod_name in feature_mods:
            if is_registered("feature", mod_name):
                kwargs = {}
                if mod_name in _ACCEPTS_MIN_LEN:
                    kwargs["min_trajectory_len"] = self.config.tracking.min_trajectory_len
                ext = build("feature", mod_name, **kwargs)
                extractors.append(ext)
            else:
                log.warning("Feature module %r not registered — skipping", mod_name)

        use_dense = self.config.features.get("dense_trajectories", {}).get("enabled", False)
        if use_dense:
            dt_cfg = self.config.features.dense_trajectories
            dense_ext = build("feature", "dense_trajectories",
                              min_trajectory_len=dt_cfg.get("step_size", 5),
                              step_size=dt_cfg.get("step_size", 5),
                              block_size=dt_cfg.get("block_size", 32),
                              feature_types=dt_cfg.get("feature_types", ["trajectory", "hog", "hof", "mbh"]))
            extractors.append(dense_ext)

        aggregator = FeatureAggregator(aggregation_fns=self.config.features.get("aggregation", ["mean", "max", "std"]))

        frame_sampling = 1
        if self.dataset_info:
            first_video = list(groups.values())[0][0] if groups else ""
            frame_sampling = self.dataset_info.get_adaptive_frame_sampling(first_video, 100)

        group_name = self.config.experiment.get("group", "")
        enable_mf  = (group_name == "I") or self.config.features.get("enable_multifamily", False)
        runner = PipelineRunner(preprocessor, detector, tracker, optical_flow,
                                extractors, aggregator,
                                self.config.tracking.min_trajectory_len,
                                use_dense_trajectories=use_dense,
                                enable_multifamily=enable_mf)

        video_features = []
        failure_records = []
        all_track_lengths = []
        all_tracker_metrics = []
        all_adaptive_stats: List[Dict] = []
        processed = 0

        # ── Per-class annotated video cap ────────────────────────────────────
        visualised_counts: Dict[str, int] = {
            act: 0 for act in df["Activity"].unique()
        }
        visualisation_dir = self.run_dir / "annotated_videos"
        if self.visualise:
            visualisation_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  [visualise] Annotated videos → {visualisation_dir}")
            print(f"  [visualise] Cap: {self.n_visualise} per activity class")

        all_distortions = df["Distortion"].unique()
        dist_stats: Dict[str, Dict] = {}
        for dist in all_distortions:
            dkey = str(dist)
            dist_stats[dkey] = {
                "total": 0, "successful": 0, "failed": 0,
                "failure_detection": 0, "failure_tracking": 0, "failure_feature": 0,
                "failure_track_too_short": 0, "failure_unknown": 0,
                "n_tracks_total": 0, "n_frames_total": 0, "track_lengths": [],
            }

        # NEW: accumulate unknown failures by activity class
        unknown_by_class = defaultdict(list)

        for (activity, distortion), video_names in groups.items():
            for vname in video_names:
                vpath = Path(self.config.paths.video_dir) / vname
                processed += 1
                dkey = str(distortion)
                if dkey not in dist_stats:
                    dist_stats[dkey] = {"total": 0, "successful": 0, "failed": 0,
                                        "failure_detection": 0, "failure_tracking": 0,
                                        "failure_feature": 0, "failure_track_too_short": 0,
                                        "failure_unknown": 0,
                                        "n_tracks_total": 0, "n_frames_total": 0, "track_lengths": []}
                dist_stats[dkey]["total"] += 1

                if not vpath.exists():
                    print(f"  [{processed}/{total}] MISSING  {vname}")
                    failure_records.append({"video": vname, "activity": activity, "distortion": distortion, "reason": "file_missing"})
                    dist_stats[dkey]["failed"] += 1
                    continue

                print(f"  [{processed}/{total}] {activity}/{distortion}  {vname}", end="", flush=True)

                # ── Decide whether to generate an annotated video ─────────────
                _vis_path: Optional[Path] = None
                if (self.visualise
                        and visualised_counts.get(activity, 0) < self.n_visualise):
                    _class_dir = visualisation_dir / str(activity)
                    _class_dir.mkdir(parents=True, exist_ok=True)
                    _vis_path = _class_dir / f"{Path(vname).stem}_annotated.mp4"

                result = runner.process_video(
                    vpath,
                    collect_diagnostics=True,
                    frame_sampling=frame_sampling,
                    visualise_path=_vis_path,
                )

                if result.track_lengths:
                    all_track_lengths.extend(result.track_lengths)
                    dist_stats[dkey]["track_lengths"].extend(result.track_lengths)
                dist_stats[dkey]["n_frames_total"] += result.n_frames_processed
                dist_stats[dkey]["n_tracks_total"] += len(result.tracks)

                if result.tracker_metrics:
                    result.tracker_metrics["video"] = vname
                    result.tracker_metrics["activity"] = activity
                    result.tracker_metrics["distortion"] = distortion
                    all_tracker_metrics.append(result.tracker_metrics)

                if result.video_feature:
                    dist_stats[dkey]["successful"] += 1
                    # Increment annotated-video counter now we know extraction succeeded
                    if _vis_path is not None:
                        visualised_counts[activity] = visualised_counts.get(activity, 0) + 1
                        print(f"\n  [visualise] Saved {_vis_path.name}"
                              f"  ({visualised_counts[activity]}/{self.n_visualise} for '{activity}')",
                              end="")
                    fv = result.video_feature
                    fv["activity"] = activity
                    fv["distortion"] = distortion
                    fv["video_name"] = vname
                    fv["n_tracks"] = len(result.tracks)
                    fv["frames_used"] = result.n_frames_processed
                    fv["track_length"] = float(np.mean(result.track_lengths)) if result.track_lengths else 0.0
                    fv["feature_count"] = result.feature_count
                    min_len = self.config.tracking.min_trajectory_len
                    if result.track_lengths:
                        norm_lengths = [min(t / min_len, 1.0) for t in result.track_lengths]
                        fv["track_confidence"] = float(np.mean(norm_lengths))
                    else:
                        fv["track_confidence"] = 0.0

                    # ── Temporal metadata — ALL groups A–I (mentor v6.0) ──────
                    # Read Nframes, Duration, FPS from Excel so the model knows
                    # clip length; core trajectory features are already
                    # normalised so these are pure contextual cues.
                    row_mask = df["Name of Video Series"] == vname
                    if row_mask.any():
                        row        = df[row_mask].iloc[0]
                        v_nframes  = int(row.get("Nframes",          result.n_frames_processed))
                        v_fps      = float(row.get("Frame Rate (FPS)", 10.0))
                        v_duration = float(row.get("Duration (s)",    v_nframes / max(v_fps, 1.0)))
                    else:
                        v_nframes  = result.n_frames_processed
                        v_fps      = 10.0
                        v_duration = v_nframes / 10.0
                    fv.update(extract_video_metadata_features(v_nframes, v_fps, v_duration))

                    video_features.append(fv)
                    _adapt_suffix = ""
                    if hasattr(runner.detector, "get_diagnostics"):
                        _ad = runner.detector.get_diagnostics()
                        all_adaptive_stats.append({"video": vname, "activity": activity,
                                                   "distortion": distortion, **_ad})
                        _adapt_suffix = (f"  trig={_ad['adaptive_trigger_rate']:.0%}"
                                         f" thr={_ad['avg_threshold_used']:.1f}"
                                         f" acc={_ad['rerun_accept_rate']:.0%}"
                                         f" mq={_ad['avg_mask_quality']:.2f}"
                                         f" tq={_ad['avg_track_quality']:.2f}")
                    print(f"  →  tracks={len(result.tracks)} frames={result.n_frames_processed} features={result.feature_count}{_adapt_suffix}")
                else:
                    dist_stats[dkey]["failed"] += 1
                    reason_str = "unknown"
                    if result.failure:
                        if result.failure.detection_failure:
                            reason_str = "detection_failure"
                            dist_stats[dkey]["failure_detection"] += 1
                        elif result.failure.track_too_short:
                            reason_str = "track_too_short"
                            dist_stats[dkey]["failure_track_too_short"] += 1
                        elif result.failure.tracking_failure:
                            reason_str = "tracking_failure"
                            dist_stats[dkey]["failure_tracking"] += 1
                        elif result.failure.feature_failure:
                            reason_str = "feature_failure"
                            dist_stats[dkey]["failure_feature"] += 1
                        elif result.failure.unknown_error:
                            reason_str = f"unknown: {result.failure.error_message[:50]}"
                            dist_stats[dkey]["failure_unknown"] += 1
                            # Record for grouping
                            unknown_by_class[activity].append((vname, result.failure.error_message))
                    failure_records.append({"video": vname, "activity": activity, "distortion": distortion, "reason": reason_str})
                    print(f"  →  FAILED ({reason_str})")

        # NEW: Log unknown failures grouped by activity class
        if unknown_by_class:
            log.warning("Unknown failures by activity class:")
            for cls, failures in unknown_by_class.items():
                log.warning("  Class %s: %d unknown failures", cls, len(failures))
                for name, msg in failures[:3]:
                    log.warning("    %s — %s", name, msg)

        self._log_detector_statistics(dist_stats, total, len(video_features))
        with open(self.run_dir / "failure_taxonomy.json", "w") as f:
            json.dump(failure_records, f, indent=2, default=str)
        if all_tracker_metrics:
            with open(self.run_dir / "tracker_metrics.json", "w") as f:
                json.dump(all_tracker_metrics, f, indent=2, default=str)
        if all_adaptive_stats:
            with open(self.run_dir / "adaptive_mog2_diagnostics.json", "w") as f:
                json.dump(all_adaptive_stats, f, indent=2, default=str)
            # Print a per-video summary to make ablation easy to read
            _agg = {k: float(np.mean([r[k] for r in all_adaptive_stats]))
                    for k in ("adaptive_trigger_rate", "avg_threshold_used",
                              "rerun_accept_rate", "avg_mask_quality", "avg_track_quality")}
            print(f"\n  [AdaptiveMOG2] across {len(all_adaptive_stats)} videos:"
                  f"  trigger={_agg['adaptive_trigger_rate']:.0%}"
                  f"  threshold={_agg['avg_threshold_used']:.2f}"
                  f"  rerun_accept={_agg['rerun_accept_rate']:.0%}"
                  f"  mask_q={_agg['avg_mask_quality']:.3f}"
                  f"  track_q={_agg['avg_track_quality']:.3f}")
        if all_track_lengths:
            self.plots.track_length_histogram(all_track_lengths,
                                              f"Track length distribution — {self.experiment_name}",
                                              "track_length_histogram.png")
            justification = TrackerAnalyzer.justify_min_trajectory_length(all_track_lengths)
            with open(self.run_dir / "trajectory_length_justification.json", "w") as f:
                json.dump(justification, f, indent=2)
        plot_stats = {dkey: {"success_rate": s["successful"]/s["total"] if s["total"]>0 else 0.0}
                      for dkey, s in dist_stats.items()}
        self.plots.detector_stats_plot(plot_stats, "detector_stats_by_distortion.png")

        return video_features

    def _log_detector_statistics(self, dist_stats: Dict, total: int, n_success: int):
        print("\n" + "=" * 70)
        print("  DETECTOR STATISTICS")
        print("=" * 70)
        print(f"  {'Distortion':<20}  {'Videos':>7}  {'Valid':>7}  {'Failed':>7}  {'Rate':>6}  "
              f"  {'Det.Fail':>9}  {'Trk.Fail':>9}  {'TooShort':>9}  {'Feat.Fail':>10}  {'Unknown':>8}")
        print("  " + "-" * 115)
        summary_stats = {}
        for dkey, s in dist_stats.items():
            rate = s["successful"] / s["total"] if s["total"] > 0 else 0.0
            mean_track_len = float(np.mean(s["track_lengths"])) if s["track_lengths"] else 0.0
            print(f"  {dkey:<20}  {s['total']:>7}  {s['successful']:>7}  {s['failed']:>7}  {rate:>6.2f}  "
                  f"  {s['failure_detection']:>9}  {s['failure_tracking']:>9}  {s['failure_track_too_short']:>9}  "
                  f"{s['failure_feature']:>10}  {s['failure_unknown']:>8}")
            summary_stats[dkey] = {
                "total_videos": s["total"], "successful_videos": s["successful"], "failed_videos": s["failed"],
                "success_rate": rate, "failure_detection": s["failure_detection"],
                "failure_tracking": s["failure_tracking"], "failure_track_too_short": s["failure_track_too_short"],
                "failure_feature": s["failure_feature"], "failure_unknown": s["failure_unknown"],
                "n_tracks_total": s["n_tracks_total"], "n_frames_total": s["n_frames_total"],
                "mean_track_length": mean_track_len,
            }
        overall_rate = n_success / total * 100 if total > 0 else 0.0
        print(f"\n  Overall: {n_success}/{total} successful ({overall_rate:.1f}%)")
        print("=" * 70)
        with open(self.run_dir / "detector_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=2)

    def train_and_evaluate(self, video_features: List[Dict], target_col: str = "activity") -> Dict:
        if not video_features:
            print(f"\nNo video features for target '{target_col}'.")
            return {}

        meta_keys = {"activity", "distortion", "video_name", "n_tracks", "frames_used", "track_length", "feature_count", "track_confidence"}
        feature_keys = sorted(set().union(*(fv.keys() for fv in video_features)) - meta_keys)
        X = np.array([[fv.get(k, 0.0) for k in feature_keys] for fv in video_features])
        y = np.array([fv[target_col] for fv in video_features])

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        print(f"\nFeature matrix: {X.shape[0]} videos × {X.shape[1]} features")
        print(f"Target: {target_col}")
        print(f"Class distribution:\n{pd.Series(y).value_counts().to_string()}")

        # ── NEW: Training viability check ──────────────────────────────────
        min_per_class = self.config.training.get("min_per_class", 3)
        n_folds = self.config.training.get("cv_folds", 5)
        viable, msg = check_training_viability(y_enc, n_folds=n_folds, min_per_class=min_per_class)
        if not viable:
            log.warning("Training skipped: %s", msg)
            return {"error": msg, "viable": False}

        # ── NEW: Use cross‑validation if enabled ──────────────────────────
        use_cv = self.config.training.get("use_cv", True)

        results = {}
        all_predictions = {}
        all_accuracies = {}
        importance_accumulator = {}

        for model_def in self.config.models.active:
            if isinstance(model_def, (dict, Config)):
                model_name = model_def.get("name", "")
                params = model_def.get("params", {}) or {}
            else:
                model_name = model_def
                params = {}

            print(f"\n{'='*60}\n  Model: {model_name}  |  Target: {target_col}" +
                  ("  [COMPARISON]" if self.is_comparison else ""))
            try:
                model = build("model", model_name, random_state=self.seed, **params)

                if use_cv:
                    # Create a consistent CV splitter
                    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

                    # 1. Run cross_validate for detailed metrics (mean ± std)
                    cv_metrics = evaluate_with_cv(
                        model, X, y_enc, n_splits=n_folds,
                        label_encoder=le
                    )
                    print(f"  CV Accuracy: {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
                    print(f"  CV F1:       {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
                    print(f"  Overfit gap: {cv_metrics['overfit_gap']:.4f}")

                    # 2. Get out‑of‑fold predictions (each sample predicted by models that did NOT see it)
                    y_pred_cv = cross_val_predict(model, X, y_enc, cv=cv_splitter, n_jobs=-1)

                    # 3. Compute metrics and confusion matrix from these genuine CV predictions
                    acc_cv = accuracy_score(y_enc, y_pred_cv)
                    f1_cv = f1_score(y_enc, y_pred_cv, average="weighted", zero_division=0)
                    cm_cv = confusion_matrix(y_enc, y_pred_cv)

                    print(f"  CV Accuracy (from cross_val_predict): {acc_cv:.4f}")
                    print(f"  CV F1 (from cross_val_predict):       {f1_cv:.4f}")

                    # 4. Fit on full training data (for feature importances, model saving)
                    model.fit(X, y_enc)
                    # (Optional) You may still compute training‑set predictions if you want to
                    # inspect overfitting, but do NOT use them for reporting.
                    y_pred_train = model.predict(X)
                    acc_train = accuracy_score(y_enc, y_pred_train)
                    f1_train = f1_score(y_enc, y_pred_train, average="weighted", zero_division=0)

                    # 5. Store results – use the CV confusion matrix for the plot
                    results[model_name] = {
                        "cv_accuracy_mean": cv_metrics["cv_accuracy_mean"],
                        "cv_accuracy_std": cv_metrics["cv_accuracy_std"],
                        "cv_f1_mean": cv_metrics["cv_f1_mean"],
                        "cv_f1_std": cv_metrics["cv_f1_std"],
                        "overfit_gap": cv_metrics["overfit_gap"],
                        "cv_scores": cv_metrics["cv_scores"],
                        # These are the true CV metrics from out‑of‑fold predictions:
                        "cv_accuracy_from_predict": acc_cv,
                        "cv_f1_from_predict": f1_cv,
                        "cv_confusion_matrix": cm_cv.tolist(),
                        # Training‑set metrics (for reference only – do NOT report as final):
                        "accuracy_train_full": acc_train,
                        "f1_train_full": f1_train,
                        "confusion_matrix_train_full": confusion_matrix(y_enc, y_pred_train).tolist(),
                        "classes": le.classes_.tolist(),
                        "feature_keys": feature_keys,
                        "feature_count": len(feature_keys),
                        "is_comparison": self.is_comparison,
                    }

                    # Use the CV confusion matrix for the plot
                    cm_for_plot = cm_cv
                    # Keep predictions for ensemble smoothing (if any) – use CV predictions
                    all_predictions[model_name] = y_pred_cv
                    all_accuracies[model_name] = acc_cv

                else:
                    # Fallback to old train/test split if CV disabled
                    test_size = self.config.training.test_size
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size,
                                                                            stratify=y_enc, random_state=self.seed)
                    except ValueError:
                        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size,
                                                                            random_state=self.seed)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)
                    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
                    results[model_name] = {
                        "accuracy": acc, "f1_weighted": f1, "precision": prec, "recall": rec,
                        "confusion_matrix": cm.tolist(), "y_true": y_test.tolist(), "y_pred": y_pred.tolist(),
                        "classes": le.classes_.tolist(), "feature_keys": feature_keys, "feature_count": len(feature_keys),
                        "is_comparison": self.is_comparison,
                    }
                    all_predictions[model_name] = y_pred
                    all_accuracies[model_name] = acc
                    cm_for_plot = cm

                # Feature importances
                if model.feature_importances_ is not None:
                    importances = model.feature_importances_
                    if len(importances) == len(feature_keys):
                        top_idx = np.argsort(importances)[-10:][::-1]
                        print("  Top 10 features:")
                        for i in top_idx:
                            print(f"    {feature_keys[i]:40s}  {importances[i]:.4f}")
                        imp_record = {feature_keys[i]: float(importances[i]) for i in range(len(feature_keys))}
                        importance_accumulator[model_name] = imp_record
                        self.plots.feature_importance_plot(feature_names=feature_keys, importances=importances,
                                                           title=f"{model_name} — feature importance ({target_col})",
                                                           filename=f"importance_{model_name}_{target_col}.png")

                # Use cm_for_plot (CV out‑of‑fold or test set)
                self.plots.confusion_matrix_plot(
                    cm_for_plot,
                    le.classes_.tolist(),
                    f"{model_name} — {target_col} (CV out‑of‑fold)" if use_cv else f"{model_name} — {target_col} (test set)",
                    f"cm_{model_name}_{target_col}.png"
                )

                with open(self.run_dir / f"{model_name}_{target_col}.pkl", "wb") as f:
                    pickle.dump({"model": model, "label_encoder": le, "feature_keys": feature_keys}, f)

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                results[model_name] = {"error": str(e)}

        smoothing_cfg = self.config.get("smoothing", None)
        if smoothing_cfg and smoothing_cfg.get("enabled", False) and len(all_predictions) > 1:
            method = smoothing_cfg.get("method", "majority_vote")
            alpha = smoothing_cfg.get("ema_alpha", 0.7)
            smoother = TemporalSmoother(method=method, ema_alpha=alpha)
            smoothed = smoother.smooth(all_predictions, all_accuracies)
            acc_s = accuracy_score(le.inverse_transform(y_enc), smoothed)  # using full set
            f1_s = f1_score(le.inverse_transform(y_enc), smoothed, average="weighted", zero_division=0)
            print(f"\n  [{method}] Smoothed ensemble — Accuracy: {acc_s:.4f}  F1: {f1_s:.4f}")
            results["__smoothed__"] = {"method": method, "accuracy": acc_s, "f1_weighted": f1_s, "y_pred": smoothed.tolist()}

        self.plots.model_comparison_plot(results, "accuracy", target_col, f"model_accuracy_{target_col}.png")

        if importance_accumulator:
            with open(self.run_dir / f"feature_importance_{target_col}.json", "w") as f:
                json.dump(importance_accumulator, f, indent=2)

        with open(self.run_dir / f"results_{target_col}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(self.run_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

        meta = {
            "run_id": self.run_id, "experiment": self.experiment_name,
            "git_commit": get_git_commit(), "config": self.config.experiment.name,
            "target": target_col, "feature_count": len(feature_keys), "n_videos": len(video_features),
            "is_comparison": self.is_comparison, "timestamp": datetime.now().isoformat(),
            "detection_source": self.detection_source.value if self.detection_source else "config",
        }
        with open(self.run_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nResults saved to: {self.run_dir}")
        return results


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                    SECTION 22: CLI MAIN ENTRY POINT                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def main():
    create_default_configs()

    parser = argparse.ArgumentParser(
        description="Distortion-Aware Suspicious Activity Detection Framework (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Group A: Classical baseline
  python run.py --config group_a_classical --target activity

  # Group B: Distortion-aware with full features
  python run.py --config group_b_distortion_aware --target activity --sample

  # Group E: GT upper bound (pure — no MOG2 fallback)
  python run.py --config group_e_gt_upper_bound --target activity --detection-source ground_truth

  # Run with custom detection source override
  python run.py --config group_a_classical --detection-source ground_truth

  # Diagnostic mode (feature extraction only, no training)
  python run.py --config group_a_classical --diagnose

  # Run all groups sequentially
  python run.py --all-groups --target activity

  # List available configs and components
  python run.py --list-configs
  python run.py --list-components
        """)

    parser.add_argument("--config", "-c", type=str, default=None, help="Group config name")
    parser.add_argument("--all-groups", action="store_true", help="Run all groups A–I sequentially")
    parser.add_argument("--target", "-t", type=str, default="activity", choices=["activity", "distortion", "both"],
                        help="Classification target")
    parser.add_argument("--sample", action="store_true", help="Use a random subset of videos per class")
    parser.add_argument("--n", type=int, default=5, help="Videos per group when --sample is set")
    parser.add_argument("--detection-source", "-ds", type=str, default=None, choices=["mog2", "ground_truth"],
                        help="Override detection source (mog2 or ground_truth)")
    parser.add_argument("--diagnose", action="store_true", help="Run feature extraction only (no training)")
    parser.add_argument("--visualise", action="store_true", help="Generate FiftyOne visualisations for a few videos")
    parser.add_argument("--n-visualise", type=int, default=3,
                        help="Max annotated diagnostic videos to generate per activity class (default: 3)")
    parser.add_argument("--list-configs", action="store_true", help="List available group configs")
    parser.add_argument("--list-components", action="store_true", help="List registered components")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (placeholder)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    if project_root.name == "src":
        project_root = project_root.parent
    global PROJECT_ROOT
    PROJECT_ROOT = project_root
    global CONFIGS_DIR, EXPERIMENTS_DIR, OUTPUTS_DIR, DATA_DIR, MODELS_DIR, PLOTS_DIR
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    for _d in [CONFIGS_DIR, EXPERIMENTS_DIR, OUTPUTS_DIR, DATA_DIR, MODELS_DIR, PLOTS_DIR]:
        _d.mkdir(parents=True, exist_ok=True)

    if args.list_configs:
        configs = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml") if p.stem != "base")
        print("Available group configs:")
        for c in configs:
            print(f"  {c}")
        return 0
    if args.list_components:
        for cat in ["detector", "tracker", "preprocessor", "feature", "model"]:
            print(f"\n{cat}:")
            for name in available(cat):
                print(f"  {name}")
        return 0
    if args.gui:
        print("GUI mode: launch videoActivityClassification.py for the full GUI.")
        return 0

    detection_source = None
    if args.detection_source:
        detection_source = DetectionSource(args.detection_source)

    groups = []
    if args.all_groups:
        # Include the new groups as well
        groups = sorted(p.stem for p in CONFIGS_DIR.glob("group_*.yaml"))
    elif args.config:
        groups = [args.config]
    else:
        parser.print_help()
        return 1

    for group_name in groups:
        print(f"\n{'#'*70}\n# GROUP: {group_name}\n{'#'*70}")
        try:
            cfg = load_config(group_name, project_root=PROJECT_ROOT,
                              overrides={"dataset": {"n_random_videos_per_group": args.n}})
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue

        data_path = cfg.get("data_path", "tabular")
        if data_path in ("clip", "hybrid") and not TORCH_AVAILABLE:
            print(f"  SKIPPING: {group_name} requires PyTorch (not installed)")
            continue

        runner = ExperimentRunner(cfg, detection_source=detection_source,
                                  diagnose_only=args.diagnose,
                                  visualise=args.visualise,
                                  n_visualise=args.n_visualise)
        df = runner.load_dataset()

        targets = (["activity", "distortion"] if args.target == "both" else [args.target])

        for target in targets:
            print(f"\n  --- Target: {target} ---")
            if data_path == "clip":
                print("  Starting neural pipeline...")
                clips = runner.prepare_clip_dataset(df, use_sample=args.sample)
                if not clips:
                    print("  No clips generated.")
                    continue
                if args.diagnose:
                    print("  Diagnose mode: skipping neural training.")
                    continue
                runner.train_neural_model(clips, target_col=target)
            else:
                video_features = runner.collect_video_features(df, use_sample=args.sample)
                if video_features:
                    if args.diagnose:
                        print(f"  Diagnose mode: collected {len(video_features)} feature vectors. Skipping training.")
                    else:
                        runner.train_and_evaluate(video_features, target_col=target)
                else:
                    print(f"  No video features collected for target '{target}'.")

    print("\n" + "=" * 70)
    print("DONE. Results saved to: experiments/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())