#!/usr/bin/env python3
"""
=============================================================================
Distortion-Aware Suspicious Activity Detection in Authentically Distorted
Surveillance Videos: Complete Production Framework

Author : Tshephang Phuti-A-Nkutu Matlala | 223004635
UJ Department of Computer Science and Software Engineering

Implements all experiment groups A–E per PROJECT REDESIGN NOTES — VERSION 2:
  Group A: Classical baseline (MOG2 → IoU+Hungarian → Trajectory → RF/SVM/LightGBM)
  Group B: Distortion-aware (Adaptive enhancement → Full features → RF/LightGBM)
  Group C: Neural baselines — comparison_only=True (CNN/CNN+LSTM/3D-CNN)
  Group D: Hybrid fusion (Handcrafted + CNN embeddings)
  Group E: GT upper bound (Ground-truth bboxes → IoU tracker → Full features)

Core contribution: Distortion-aware trajectory pipeline (Groups A/B/E).
CNN models (Group C) are comparison baselines, NOT the primary contribution.

Feature roadmap:
  - Trajectory      (14 features)
  - Circular motion  (4 features)
  - Kinematics       (8 features)
  - Shape            (5 features)
  - Wavelet/DWT      (6 features)
  - Interaction      (7 features)

Tracking: IoU + Hungarian assignment (scipy.optimize.linear_sum_assignment)
Preprocessing: Standard OR distortion-aware — toggled per experiment via YAML
GT mode: Pure ground-truth bboxes, no MOG2 fallback (upper-bound is kept clean)
=============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[warn] LightGBM not installed — LightGBM models unavailable")

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    print("[warn] PyWavelets not installed — DWT wavelet features unavailable")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[warn] PyTorch not installed — neural comparison models unavailable")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — writes files, no display needed
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[warn] Matplotlib not installed — plots will be skipped")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[warn] XGBoost not installed — XGBoost models unavailable")

# This section Includes the main change of the Implementation - to Extend for Experimentation.

# TODO: Let's include different tracking approach that can be able to and let's make
# Arguements for the research considering Adding one to the pipeline of (SORT/ DeepSORT/ ByteTrack)
# NOTE AND TODO My Novelty -:- Lies upon ensuring that all the samples gets used.
# The problem is : Some videos cannot be detected which is likely caused by the tracker loosing Objects from frame to frame, 
# where if we cannot track the Object over-30-frames (We should also have the ground for why ? 30 -Let's at-least changes it and say so Over 80% of the frames for each video)
# This likely causes our model to loose the objects => Detection fails which => Tracking Fails and Trajectory features fails automatically, In the initial-version leading to 
# dropping the video, as we cannot train on a corrupted dataset; Another approach to keep into cosideration is that we do not drop anything and that means we do not put constrains
#  towards any videos (such as we should be able to track features for these frames, We just get whatever we can get from the video dataset and Based on that we model)
# Approach 3 : Given that to classify an activity we do not need to know excatly the motions of the activity but the things,  Tracking local features instead of Blob detectors
# like speed of change of interest, arrangements and so on. can give enough idea about what is the activity:
#  Use Dense Trajectories; where we sample points which can correspond to each pixel at most and track them. Feature to Extract are HOG, HOF, MBH, Texture, LBP, cOLOUR Histogram, Edge density, 
# sparse Optimal Flow, Dense Optimal Flow, Feature Point Tracking (shi-Tomasi corners, Harris corners, and FAST Keypoints)


# Justification for the Change: 
# 1. Blob detection - Can be affected by distortions
# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                      SECTION 1: PROJECT CONFIGURATION                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
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

# Type aliases — kept as plain dicts for simplicity
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
    """Records exactly why a video produced no usable features.
    
    Used for the failure taxonomy output table.
    Only one flag will typically be True per video.
    """
    file_missing: bool = False
    detection_failure: bool = False   # video opened but zero detections
    tracking_failure: bool = False    # detections exist but all tracks too short
    feature_failure: bool = False     # tracks long enough but no valid feature vector


@dataclass
class VideoProcessingResult:
    """Everything collected after processing one video."""
    tracks: List[Track]
    mean_flow_magnitude: float
    mean_flow_angle_deg: float
    n_frames_processed: int
    used_gt: bool = False
    preprocessing_diagnostics: List[dict] = field(default_factory=list)
    track_features: List[FeatureDict] = field(default_factory=list)
    video_feature: Optional[FeatureDict] = None
    # New: richer per-video metadata for paper tables
    track_lengths: List[int] = field(default_factory=list)   # length of each track
    feature_count: int = 0                                   # number of features in video_feature
    failure: Optional[FailureReason] = None                  # set only when video_feature is None


# ── Abstract base classes ─────────────────────────────────────────────────────

class Detector:
    def detect(self, frame: np.ndarray) -> List[BBox]:
        raise NotImplementedError
    def reset(self) -> None:
        raise NotImplementedError


class Tracker:
    def update(self, detections: List[BBox]) -> List[Track]:
        raise NotImplementedError
    def get_active_tracks(self) -> List[Track]:
        raise NotImplementedError
    def reset(self) -> None:
        raise NotImplementedError


class Preprocessor:
    def process(self, frame: np.ndarray) -> PreprocessResult:
        raise NotImplementedError
    def reset(self) -> None:
        pass


class FeatureExtractor:
    def feature_names(self) -> List[str]:
        raise NotImplementedError
    def extract_track(self, track: Track, **context: Any) -> Optional[FeatureDict]:
        raise NotImplementedError


class Model:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        raise NotImplementedError
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
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
    """Class decorator that registers a class under a given category and name."""
    if category not in _REGISTRIES:
        raise KeyError(f"Unknown category '{category}'. Valid: {list(_REGISTRIES.keys())}")
    def decorator(cls: type) -> type:
        if name in _REGISTRIES[category]:
            raise ValueError(f"'{name}' already registered under '{category}'")
        _REGISTRIES[category][name] = cls
        return cls
    return decorator


def build(category: str, name: str, **kwargs) -> Any:
    """Instantiate a registered class by category and name."""
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
# TODO: Does the configurations also keep in mind the dataset information, such as 
# video duration, number of frames, rates and etc, that are provided in the 
# datasetInfo.xlsx file.

class Config:
    """Wraps a nested dict so keys can be accessed as attributes.
    
    Example:
        cfg.preprocessing.type
        cfg.detection.ground_truth.fallback_to_mog2
    """

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
    """Recursively merge override into base, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_paths(data: dict, root: Path) -> dict:
    """Convert relative paths under data['paths'] to absolute paths."""
    data = copy.deepcopy(data)
    if "paths" in data:
        for key, rel in data["paths"].items():
            if isinstance(rel, str) and not Path(rel).is_absolute():
                data["paths"][key] = str((root / rel).resolve())
    return data


def load_config(group_name: str, project_root: Path = PROJECT_ROOT,
                overrides: dict = None) -> Config:
    """Load a YAML config, merge with its base if it has 'extends', apply overrides."""
    configs_dir = project_root / "configs"
    path = Path(group_name)
    if not path.is_absolute():
        if not path.suffix:
            path = path.with_suffix(".yaml")
        path = configs_dir / path
    if not path.exists():
        available_list = [p.stem for p in configs_dir.glob("*.yaml") if p.stem != "base"]
        raise FileNotFoundError(f"Config '{group_name}' not found. Available: {available_list}")

    with open(path) as f:
        group_data = yaml.safe_load(f) or {}

    extends = group_data.pop("extends", None)
    base_data = {}
    if extends:
        base_path = configs_dir / extends
        if base_path.exists():
            with open(base_path) as f:
                base_data = yaml.safe_load(f) or {}

    merged = _deep_merge(base_data, group_data)
    merged = _resolve_paths(merged, project_root)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config(merged)


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
    },
    "preprocessing": {
        "type": "standard",
        # ── Ablation controls ───────────────────────────────────────────────
        # Set enabled=False to bypass preprocessing entirely (pass raw frame).
        # Set distortion_aware=False to use standard pipeline even when type
        # is distortion_aware (useful for within-group ablation).
        "enabled": True,
        "distortion_aware": True,
        # ── Standard params ─────────────────────────────────────────────────
        "frame_resize": [640, 480],
        "blur_kernel":  [5, 5],
        "blur_sigma":   0,
        # ── Distortion-aware params ──────────────────────────────────────────
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
        },
    },
    "detection": {
        "type": "mog2",
        "mog2": {
            "history":        20,
            "var_threshold":  25,
            "detect_shadows": False,
        },
        "morphology": {
            "close_iter": 2,
            "open_iter":  1,
        },
        "min_contour_area": 500,
        "ground_truth": {
            # False keeps the upper-bound experiment scientifically pure.
            # The GT detector will simply return [] for frames with no GT file
            # rather than falling back to MOG2, which would mix two sources.
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
    },
    "smoothing": {
        # Optional temporal smoothing applied after all model predictions.
        # method: "none" | "majority_vote" | "ema"
        # majority_vote: for each test sample, the most common prediction
        #   across all trained models wins.
        # ema: exponential moving average weights models by their accuracy.
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

# TODO : All the choices of the parameters should be supported.

def create_default_configs():
    """Write all group YAML config files to disk if they do not already exist."""
    configs = {

        "base.yaml": BASE_CONFIG_DICT,

        "legacy_centroid_reference.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name":        "legacy_centroid_reference",
                "description": "Legacy centroid tracker for regression testing",
                "group":       "REF",
            },
            "tracking":  {"type": "centroid"},
            "features":  {"modules": ["trajectory"]},
            "models":    {"active": [
                {"name": "random_forest_sklearn",
                 "params": {"n_estimators": 200, "max_depth": None,
                            "class_weight": "balanced"}},
            ]},
        },

        "group_a_classical.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name":        "group_a_classical_baseline",
                "description": "MOG2 + IoU+Hungarian + Trajectory features + Classical models",
                "group":       "A",
            },
            "preprocessing": {
                "type":             "standard",
                "enabled":          True,
                "distortion_aware": False,
            },
            "detection":  {"type": "mog2"},
            "tracking":   {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features":   {"modules": ["trajectory"]},
            "models":     {"active": [
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
            ]},
            "targets": ["activity", "distortion"],
        },

        "group_b_distortion_aware.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name":        "group_b_distortion_aware_classical",
                "description": "Distortion-aware + Full features + RF/LightGBM",
                "group":       "B",
            },
            # ── Both flags must be True to activate distortion-aware preprocessing
            "preprocessing": {
                "type":             "distortion_aware",
                "enabled":          True,
                "distortion_aware": True,
            },
            "detection":  {"type": "mog2"},
            "tracking":   {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features":   {"modules": ["trajectory", "circular_motion", "kinematics",
                                       "shape", "wavelet", "interaction"]},
            "models":     {"active": [
                {"name": "random_forest_sklearn",
                 "params": {"n_estimators": 200, "max_depth": None,
                            "class_weight": "balanced"}},
                {"name": "lightgbm",
                 "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}},
            ]},
            "targets": ["activity", "distortion"],
        },

        "group_c_neural.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name":        "group_c_neural_baselines",
                "description": "CNN/CNN+LSTM/3D-CNN — comparison baselines only",
                "group":       "C",
            },
            # ── CNN is a COMPARISON baseline, not the primary contribution.
            # ── Set comparison_only=True so the runner labels outputs clearly
            # ── and does not mix CNN results into the classical pipeline tables.
            "comparison_only": True,
            "preprocessing":   {"type": "standard", "enabled": True, "distortion_aware": False},
            "data_path":       "clip",
            "features":        {"modules": []},
            "models":          {"active": [
                {"name": "cnn",
                 "params": {"backbone": "resnet18_2d", "pretrained": False,
                            "lr": 0.001, "epochs": 15, "batch_size": 8}},
                {"name": "cnn_lstm",
                 "params": {"backbone": "resnet18_2d", "hidden_size": 256,
                            "num_layers": 1, "lr": 0.0005, "epochs": 20, "batch_size": 8}},
                {"name": "cnn3d",
                 "params": {"backbone": "r3d_18", "pretrained": False,
                            "lr": 0.0005, "epochs": 20, "batch_size": 4}},
            ]},
            "targets": ["activity"],
        },

        "group_d_hybrid.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name":        "group_d_hybrid_fusion",
                "description": "Handcrafted features + CNN embeddings hybrid",
                "group":       "D",
            },
            "preprocessing": {
                "type":             "distortion_aware",
                "enabled":          True,
                "distortion_aware": True,
            },
            "detection":  {"type": "mog2"},
            "tracking":   {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "data_path":  "hybrid",
            "features":   {"modules": ["trajectory", "circular_motion", "kinematics",
                                       "shape", "wavelet", "interaction"]},
            "models":     {"active": [
                {"name": "feature_fusion",
                 "params": {"classifier": "lightgbm", "n_estimators": 200, "max_depth": 6}},
            ]},
            "targets": ["activity", "distortion"],
        },

        "group_e_gt_upper_bound.yaml": {
            "extends": "base.yaml",
            "experiment": {
                "name":        "group_e_ground_truth_upper_bound",
                "description": "GT bounding boxes + Full features — pure upper bound",
                "group":       "E",
            },
            "preprocessing": {
                "type":             "standard",
                "enabled":          True,
                "distortion_aware": False,
            },
            "detection": {
                "type": "ground_truth",
                "ground_truth": {
                    # Must stay False to keep the upper bound scientifically pure.
                    # If a GT file is missing, the video is skipped — not fixed with MOG2.
                    "fallback_to_mog2": False,
                },
            },
            "tracking":  {"type": "iou_hungarian", "iou_hungarian": {"iou_threshold": 0.3}},
            "features":  {"modules": ["trajectory", "circular_motion", "kinematics",
                                      "shape", "wavelet", "interaction"]},
            "models":    {"active": [
                {"name": "random_forest_sklearn",
                 "params": {"n_estimators": 200, "max_depth": None,
                            "class_weight": "balanced"}},
                {"name": "lightgbm",
                 "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}},
            ]},
            "targets": ["activity", "distortion"],
        },
    }

    for name, data in configs.items():
        path = CONFIGS_DIR / name
        if not path.exists():
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"  Created: {path}")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║              SECTION 6: PREPROCESSORS (Standard + Distortion-Aware)        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

@register("preprocessor", "passthrough")
class PassthroughPreprocessor(Preprocessor):
    """Returns the raw frame with no changes.
    
    Used when preprocessing.enabled=False so we can study the pipeline
    without any preprocessing at all.
    """

    def process(self, frame: np.ndarray) -> PreprocessResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return PreprocessResult(
            bgr=frame,
            gray=gray,
            diagnostics={"enabled": False},
            enhanced=False,
        )

# TODO: How do we handle different video durations, number of frames and etc. as specified in the excel file (datasetInfo.xlsx)
# Also why do we process it that way ?

@register("preprocessor", "standard")
class StandardPreprocessor(Preprocessor):
    """Resize → grayscale → Gaussian blur (unchanged from legacy)."""
    # Suppport for why the framse resize size of (640, 480) ? :
    # TODO: I want to first check if all the videos in the dataset have the same Dimension of (1920 × 956) 
    # if YES THEN  KEEP THE SIZE else resize all to (640, 480) given it applies to all the videos.


    def __init__(self, frame_resize=(640, 480), blur_kernel=(5, 5), blur_sigma=0):
        self.frame_resize = tuple(frame_resize)
        self.blur_kernel  = tuple(blur_kernel)
        self.blur_sigma   = blur_sigma

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr  = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)
        return PreprocessResult(
            bgr=bgr,
            gray=gray,
            diagnostics={
                "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "brightness": float(gray.mean()),
            },
        )


@register("preprocessor", "distortion_aware")
class DistortionAwarePreprocessor(Preprocessor):
    # TODO: Why those defaults values ? Let's support it by a Paper or Reference etc.
    # TODO: Is the distortion estimation aware of all the types of the distortions in the dataset: 
    # : Prestine, Exposure, Focus, Focus and Exposure 

    """Distortion estimation → adaptive enhancement.
    
    Detection logic:
      Blur:     Laplacian variance < blur_threshold  → unsharp mask or wiener
      Exposure: brightness < exposure_low            → CLAHE or bilateral
    
    The distortion_type field in diagnostics can be used as an additional
    label when training distortion classifiers.
    """

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
    ):
        self.frame_resize           = tuple(frame_resize)
        self.blur_kernel            = tuple(blur_kernel)
        self.blur_sigma             = blur_sigma
        self.blur_threshold         = blur_threshold
        self.exposure_low           = exposure_low
        self.exposure_high          = exposure_high
        self.enhancement_for_blur   = enhancement_for_blur
        self.enhancement_for_exposure = enhancement_for_exposure
        self.unsharp_sigma          = unsharp_sigma
        self.unsharp_strength       = unsharp_strength
        self.clahe                  = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=tuple(clahe_tile_grid))
        self.bilateral_d            = bilateral_d
        self.bilateral_sigma_color  = bilateral_sigma_color
        self.bilateral_sigma_space  = bilateral_sigma_space

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr  = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ── Distortion estimation ────────────────────────────────────────
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        is_blurry       = blur_score < self.blur_threshold
        is_underexposed = brightness  < self.exposure_low
        is_overexposed  = brightness  > self.exposure_high

        # Build a human-readable distortion label for this frame
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

        # ── Adaptive enhancement ─────────────────────────────────────────
        enhanced = False

        if is_blurry:
            if self.enhancement_for_blur == "unsharp_mask":
                gray     = self._unsharp_mask(gray)
                enhanced = True
            elif self.enhancement_for_blur == "wiener":
                gray     = self._wiener_filter(gray)
                enhanced = True

        if is_underexposed or is_overexposed:
            if self.enhancement_for_exposure == "clahe":
                gray     = self.clahe.apply(gray)
                enhanced = True
            elif self.enhancement_for_exposure == "bilateral":
                gray     = cv2.bilateralFilter(
                    gray, self.bilateral_d,
                    self.bilateral_sigma_color, self.bilateral_sigma_space)
                enhanced = True

        # Final Gaussian blur (legacy step kept for continuity)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)

        return PreprocessResult(
            bgr=bgr,
            gray=gray,
            enhanced=enhanced,
            diagnostics={
                "blur_score":      blur_score,
                "brightness":      brightness,
                "distortion_type": distortion_type,
                "enhanced":        enhanced,
                "is_blurry":       is_blurry,
                "is_underexposed": is_underexposed,
                "is_overexposed":  is_overexposed,
            },
        )

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        """original + strength * (original − blurred)."""
        blurred = cv2.GaussianBlur(gray, (0, 0), self.unsharp_sigma)
        return cv2.addWeighted(
            gray, 1.0 + self.unsharp_strength,
            blurred, -self.unsharp_strength, 0)

    def _wiener_filter(self, gray: np.ndarray) -> np.ndarray:
        """Simplified Wiener filter using local mean/variance."""
        kernel   = (5, 5)
        f64      = gray.astype(np.float64)
        mean     = cv2.blur(f64, kernel)
        mean_sq  = cv2.blur(f64 ** 2, kernel)
        variance = mean_sq - mean ** 2
        noise_var = np.mean(variance) * 0.1
        ratio    = np.maximum(0, variance - noise_var) / np.maximum(variance, noise_var)
        result   = mean + ratio * (f64 - mean)
        return np.clip(result, 0, 255).astype(np.uint8)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║            SECTION 7: DETECTORS (MOG2 + Ground-Truth)                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

STRUCT = generate_binary_structure(2, 2)


def make_bbox(x: int, y: int, w: int, h: int) -> BBox:
    """Build a standard bounding-box dict matching the legacy schema."""
    cx = int(x + w // 2)
    cy = int(y + h // 2)
    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "cx": cx, "cy": cy,
        "centroid": (cx, cy),
        "area": int(w * h),
    }


@register("detector", "mog2")
class MOG2Detector(Detector):
    """MOG2 background subtractor with scipy morphological cleanup.
    Unchanged detection logic from legacy ForegroundDetector.
    """

    def __init__(self, history=20, var_threshold=25, detect_shadows=False,
                morph_close_iter=2, morph_open_iter=1, min_contour_area=500):
        self.history          = history
        self.var_threshold    = var_threshold
        self.detect_shadows   = detect_shadows
        self.morph_close_iter = morph_close_iter
        self.morph_open_iter  = morph_open_iter
        self.min_contour_area = min_contour_area
        self._make_mog2()

    def _make_mog2(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows)

    def clean_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        binary = raw_mask > 0
        closed = binary_closing(binary, structure=STRUCT, iterations=self.morph_close_iter)
        opened = binary_opening(closed, structure=STRUCT, iterations=self.morph_open_iter)
        return (opened.astype(np.uint8) * 255)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        raw_mask  = self.mog2.apply(frame)
        mask      = self.clean_mask(raw_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(make_bbox(x, y, w, h))
        return detections

    def reset(self):
        self._make_mog2()


@register("detector", "ground_truth")
class GroundTruthDetector(Detector):
    """Ground-truth bounding box detector using AD-SVD annotations.
    
    Bypasses MOG2 entirely so the GT experiment remains a pure upper bound.
    
    fallback_to_mog2 is intentionally set to False by default.
    Setting it to True would contaminate the GT condition with two different
    detection sources, making results scientifically ambiguous.
    
    GT file format: one line per frame with space-separated values:
        x y w h
    Lines that are empty or start with '#' mean no object in that frame.
    """

    def __init__(self, gt_dir=None, frame_resize=(640, 480), fallback_to_mog2=False):
        self.gt_dir       = (Path(gt_dir) if gt_dir
                             else DATA_DIR / "surveillanceVideosDataset" / "surveillanceVideosGT")
        self.frame_resize = tuple(frame_resize)
        # NOTE: Keep this False. Fallback pollutes the upper-bound measurement.
        self.fallback_to_mog2 = fallback_to_mog2
        self._loaded_bboxes: List[Optional[BBox]] = []
        self._frame_idx   = 0
        self.current_video = ""

    def load_for_video(self, video_name: str, orig_w: int, orig_h: int) -> bool:
        """Load GT bboxes for a video. Returns True if a GT file was found."""
        self.current_video = video_name
        stem = Path(video_name).stem

        gt_files = []
        if self.gt_dir.exists():
            for pattern in [f"*{stem}*", f"{stem}*", f"*{stem}"]:
                gt_files = list(self.gt_dir.glob(pattern))
                if gt_files:
                    break

        if not gt_files:
            self._loaded_bboxes = []
            return False

        try:
            bboxes = []
            with open(gt_files[0]) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        bboxes.append(None)
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x, y, w, h = map(float, parts[:4])
                            scale_x = self.frame_resize[0] / orig_w
                            scale_y = self.frame_resize[1] / orig_h
                            bboxes.append(make_bbox(
                                int(x * scale_x), int(y * scale_y),
                                int(w * scale_x), int(h * scale_y)))
                        except ValueError:
                            bboxes.append(None)
                    else:
                        bboxes.append(None)

            self._loaded_bboxes = bboxes
            self._frame_idx = 0
            return True

        except Exception as e:
            print(f"  [warn] Failed to parse GT file {gt_files[0]}: {e}")
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
# ║            SECTION 8: TRACKERS (Centroid + IoU+Hungarian)                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union between two bbox dicts {x, y, w, h}."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    inter_w   = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h   = max(0, min(ay2, by2) - max(ay1, by1))
    inter     = inter_w * inter_h
    union     = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


@register("tracker", "centroid")
class CentroidTracker(Tracker):
    """Greedy nearest-centroid matching (legacy, preserved for ablation)."""

    def __init__(self, max_disappeared=30, max_trajectory_len=300):
        self.max_disappeared  = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}

    def _new_track(self, centroid):
        self.tracks[self.next_id] = {
            "id": self.next_id, "centroid": centroid,
            "trajectory": [centroid], "disappeared": 0,
        }
        self.next_id += 1

    def _update_track(self, track: Track, centroid):
        track["centroid"]    = centroid
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

        track_ids  = list(self.tracks.keys())
        old_cents  = np.array([self.tracks[tid]["centroid"] for tid in track_ids], dtype=float)
        new_cents  = np.array(centroids, dtype=float)
        dist_matrix = cdist(old_cents, new_cents, metric="euclidean")
        rows, cols  = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)

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
        expired = [tid for tid, t in self.tracks.items()
                   if t["disappeared"] > self.max_disappeared]
        for tid in expired:
            del self.tracks[tid]

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def reset(self):
        self.tracks.clear()
        self.next_id = 0


@register("tracker", "iou_hungarian")
class IoUHungarianTracker(Tracker):
    """IoU + Hungarian assignment tracker.
    
    Cost matrix: -IoU between each track's last bbox and each detection.
    Solved via Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    Pairs with IoU ≤ iou_threshold are rejected even if assigned by the solver.
    """

    def __init__(self, max_disappeared=30, max_trajectory_len=300, iou_threshold=0.3):
        self.max_disappeared  = max_disappeared
        self.max_trajectory_len = max_trajectory_len
        self.iou_threshold    = iou_threshold
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}

    def _new_track(self, bbox: BBox):
        self.tracks[self.next_id] = {
            "id": self.next_id, "centroid": bbox["centroid"],
            "bbox": bbox, "trajectory": [bbox["centroid"]], "disappeared": 0,
        }
        self.next_id += 1

    def _update_track(self, track: Track, bbox: BBox):
        track["centroid"]    = bbox["centroid"]
        track["bbox"]        = bbox
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
        n_tracks  = len(track_ids)
        n_dets    = len(detections)

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
        expired = [tid for tid, t in self.tracks.items()
                   if t["disappeared"] > self.max_disappeared]
        for tid in expired:
            del self.tracks[tid]

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def reset(self):
        self.tracks.clear()
        self.next_id = 0


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                       SECTION 9: OPTICAL FLOW                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class OpticalFlowEstimator:
    """Sparse Lucas-Kanade optical flow (unchanged from legacy)."""

    def __init__(self, lk_win_size=(15, 15), lk_max_level=2, lk_criteria_eps=0.03,
                 lk_criteria_count=10, max_corners=200, corner_quality=0.3, corner_min_dist=7):
        self.lk_win_size   = tuple(lk_win_size)
        self.lk_max_level  = lk_max_level
        self.lk_criteria   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               lk_criteria_count, lk_criteria_eps)
        self.corner_params = dict(
            maxCorners=max_corners, qualityLevel=corner_quality,
            minDistance=corner_min_dist, blockSize=7)

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> OpticalFlowResult:
        corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.corner_params)
        if corners is None:
            return OpticalFlowResult(0.0, 0.0)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None,
            winSize=self.lk_win_size, maxLevel=self.lk_max_level,
            criteria=self.lk_criteria)

        good_prev = corners[status == 1]
        good_next = next_pts[status == 1]
        if len(good_prev) == 0:
            return OpticalFlowResult(0.0, 0.0)

        flow_vecs  = good_next - good_prev
        magnitudes = np.linalg.norm(flow_vecs, axis=1)
        angles_deg = np.degrees(np.arctan2(flow_vecs[:, 1], flow_vecs[:, 0])) % 360

        return OpticalFlowResult(
            mean_magnitude=float(magnitudes.mean()),
            mean_angle_deg=float(angles_deg.mean()),
            flow_vectors=flow_vecs,
            prev_corners=good_prev)

    @classmethod
    def from_config(cls, cfg) -> "OpticalFlowEstimator":
        of = cfg.optical_flow
        return cls(
            lk_win_size=tuple(of.lk_win_size), lk_max_level=of.lk_max_level,
            lk_criteria_eps=of.lk_criteria_eps, lk_criteria_count=of.lk_criteria_count,
            max_corners=of.max_corners, corner_quality=of.corner_quality,
            corner_min_dist=of.corner_min_dist)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║           SECTION 10: FEATURE EXTRACTORS (Complete Roadmap)                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

@register("feature", "trajectory")
class TrajectoryFeatures(FeatureExtractor):
    """Legacy 14 per-track trajectory/motion features (unchanged formulas)."""

    def __init__(self, min_trajectory_len=30):
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return [
            "trajectory_length", "total_distance", "net_displacement",
            "straightness", "mean_speed", "max_speed", "std_speed",
            "skew_speed", "kurt_speed", "mean_direction_deg",
            "direction_std_deg", "bbox_area",
            "mean_flow_magnitude", "mean_flow_angle_deg",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None

        deltas      = np.diff(traj, axis=0)
        speeds      = np.linalg.norm(deltas, axis=1)
        angles      = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360
        total_dist  = float(speeds.sum())
        net_disp    = float(euclidean(traj[0], traj[-1]))
        straightness = net_disp / total_dist if total_dist > 0 else 0.0
        bbox_wh     = traj.max(axis=0) - traj.min(axis=0)
        bbox_area   = float(bbox_wh[0] * bbox_wh[1])

        return {
            "trajectory_length":   float(len(traj)),
            "total_distance":      total_dist,
            "net_displacement":    net_disp,
            "straightness":        straightness,
            "mean_speed":          float(speeds.mean()),
            "max_speed":           float(speeds.max()),
            "std_speed":           float(speeds.std()),
            "skew_speed":          float(skew(speeds))     if len(speeds) > 2 else 0.0,
            "kurt_speed":          float(kurtosis(speeds)) if len(speeds) > 2 else 0.0,
            "mean_direction_deg":  float(angles.mean()),
            "direction_std_deg":   float(angles.std()),
            "bbox_area":           bbox_area,
            "mean_flow_magnitude": context.get("mean_flow_magnitude", 0.0),
            "mean_flow_angle_deg": context.get("mean_flow_angle_deg", 0.0),
        }


@register("feature", "circular_motion")
class CircularMotionFeatures(FeatureExtractor):
    """Circular direction statistics using unit-vector summation."""

    def feature_names(self) -> List[str]:
        return [
            "circular_mean_rad", "circular_variance",
            "direction_concentration", "angular_dispersion",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None

        deltas     = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        mean_cos   = np.mean(np.cos(angles_rad))
        mean_sin   = np.mean(np.sin(angles_rad))
        circ_mean  = np.arctan2(mean_sin, mean_cos)
        R          = np.sqrt(mean_cos**2 + mean_sin**2)  # resultant length [0,1]
        circ_var   = 1.0 - R                             # 0=all same dir, 1=uniform

        ang_diff   = np.arctan2(
            np.sin(angles_rad - circ_mean),
            np.cos(angles_rad - circ_mean))

        return {
            "circular_mean_rad":      float(circ_mean),
            "circular_variance":      float(circ_var),
            "direction_concentration": float(R),
            "angular_dispersion":     float(np.std(ang_diff)),
        }


@register("feature", "kinematics")
class KinematicsFeatures(FeatureExtractor):
    """Acceleration and jerk statistics (4+4 features)."""

    def feature_names(self) -> List[str]:
        return [
            "mean_accel", "std_accel", "max_accel", "skew_accel",
            "mean_jerk",  "std_jerk",  "max_jerk",  "skew_jerk",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None

        deltas = np.diff(traj, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        accel  = np.diff(speeds)
        jerk   = np.diff(accel)

        if len(accel) < 2 or len(jerk) < 2:
            return None

        return {
            "mean_accel": float(np.mean(accel)),
            "std_accel":  float(np.std(accel)),
            "max_accel":  float(np.max(np.abs(accel))),
            "skew_accel": float(skew(accel)) if len(accel) > 2 else 0.0,
            "mean_jerk":  float(np.mean(jerk)),
            "std_jerk":   float(np.std(jerk)),
            "max_jerk":   float(np.max(np.abs(jerk))),
            "skew_jerk":  float(skew(jerk)) if len(jerk) > 2 else 0.0,
        }


@register("feature", "shape")
class ShapeFeatures(FeatureExtractor):
    # TODO : Why these feature ? 

    """Trajectory shape descriptors (5 features)."""

    def feature_names(self) -> List[str]:
        return [
            "mean_curvature", "std_curvature", "convex_hull_area",
            "compactness", "path_efficiency",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None

        deltas     = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        curvature  = np.abs(np.diff(angles_rad))
        curvature  = np.minimum(curvature, 2 * np.pi - curvature)

        hull_area   = 0.0
        compactness = 0.0
        try:
            if len(traj) >= 3:
                hull       = ConvexHull(traj)
                hull_area  = float(hull.volume)  # .volume = area in 2D
                hull_perim = float(hull.area)     # .area = perimeter in 2D
                if hull_perim > 0:
                    compactness = 4 * np.pi * hull_area / (hull_perim ** 2)
        except Exception:
            pass

        total_dist    = float(np.sum(np.linalg.norm(deltas, axis=1)))
        net_disp      = float(euclidean(traj[0], traj[-1]))
        path_eff      = net_disp / total_dist if total_dist > 0 else 0.0

        return {
            "mean_curvature":  float(np.mean(curvature)),
            "std_curvature":   float(np.std(curvature)),
            "convex_hull_area": hull_area,
            "compactness":     compactness,
            "path_efficiency": path_eff,
        }


@register("feature", "wavelet")
class WaveletFeatures(FeatureExtractor):
    # TODO: Include why these feature? Justification
    """DWT energy bands on per-track speed signal (6 features)."""

    def __init__(self, wavelet="db4", level=3, min_trajectory_len=30):
        self.wavelet           = wavelet
        self.level             = level
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return [
            "dwt_energy_approx",   "dwt_energy_detail1",
            "dwt_energy_detail2",  "dwt_energy_detail3",
            "dwt_energy_total",    "dwt_entropy",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        if not PYWAVELETS_AVAILABLE:
            return None

        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None

        deltas = np.diff(traj, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)

        # Pad to next power of 2 for wavedec
        n       = len(speeds)
        pad_len = 2 ** int(np.ceil(np.log2(max(n, 8)))) - n
        if pad_len > 0:
            speeds = np.pad(speeds, (0, pad_len), mode="reflect")

        try:
            coeffs = pywt.wavedec(speeds, self.wavelet, level=self.level)
        except Exception:
            return None

        energies     = [float(np.sum(np.array(c) ** 2)) for c in coeffs]
        total_energy = sum(energies)
        eps          = 1e-10
        energy_dist  = np.array(energies) / (total_energy + eps)
        entropy      = float(-np.sum(energy_dist * np.log(energy_dist + eps)))

        result = {"dwt_energy_approx": energies[0] if energies else 0.0}
        for i in range(1, 4):
            result[f"dwt_energy_detail{i}"] = energies[i] if i < len(energies) else 0.0
        result["dwt_energy_total"] = total_energy
        result["dwt_entropy"]      = entropy
        return result


@register("feature", "interaction")
class InteractionFeatures(FeatureExtractor):
    """Pairwise inter-track distance and relative velocity (7 features)."""

    def __init__(self, proximity_threshold=100.0, min_trajectory_len=30):
        self.proximity_threshold = proximity_threshold
        self.min_trajectory_len  = min_trajectory_len

    def feature_names(self) -> List[str]:
        return [
            "mean_pairwise_distance", "min_pairwise_distance",
            "std_pairwise_distance",  "mean_relative_speed",
            "max_relative_speed",     "proximity_duration_ratio",
            "n_proximity_events",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        all_tracks = context.get("all_tracks", [])
        if len(all_tracks) < 2:
            return None

        my_traj = np.array(track["trajectory"], dtype=float)
        if len(my_traj) < self.min_trajectory_len:
            return None

        pairwise_dists  = []
        rel_speeds      = []
        proximity_frames = 0
        total_frames    = 0
        n_events        = 0

        for other in all_tracks:
            if other["id"] == track["id"]:
                continue
            other_traj = np.array(other["trajectory"], dtype=float)
            min_len    = min(len(my_traj), len(other_traj))
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
                        n_events    += 1
                        in_proximity = True
                else:
                    in_proximity = False

            my_speeds    = np.linalg.norm(np.diff(my_traj[:min_len], axis=0), axis=1)
            other_speeds = np.linalg.norm(np.diff(other_traj[:min_len], axis=0), axis=1)
            rel_speeds.extend(np.abs(my_speeds - other_speeds))

        if not pairwise_dists:
            return None

        return {
            "mean_pairwise_distance":  float(np.mean(pairwise_dists)),
            "min_pairwise_distance":   float(np.min(pairwise_dists)),
            "std_pairwise_distance":   float(np.std(pairwise_dists)),
            "mean_relative_speed":     float(np.mean(rel_speeds)) if rel_speeds else 0.0,
            "max_relative_speed":      float(np.max(rel_speeds))  if rel_speeds else 0.0,
            "proximity_duration_ratio": proximity_frames / total_frames if total_frames > 0 else 0.0,
            "n_proximity_events":      n_events,
        }


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                SECTION 11: FEATURE AGGREGATION                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class FeatureAggregator:
    """Aggregates per-track features into a single per-video feature vector."""

    def __init__(self, aggregation_fns=None):
        self.aggregation_fns = aggregation_fns or ["mean", "max", "std"]

    def aggregate(self, track_features: List[FeatureDict]) -> Optional[FeatureDict]:
        if not track_features:
            return None

        # Collect all unique feature keys across all tracks
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
                video_fv[f"max_{key}"]  = float(np.max(arr))
            if "std" in self.aggregation_fns:
                video_fv[f"std_{key}"]  = float(np.std(arr)) if len(arr) > 1 else 0.0

        return video_fv


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                   SECTION 12: PIPELINE BUILDER                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class PipelineBuilder:
    """Builds detector/tracker/preprocessor/optical-flow from a Config."""

    @staticmethod
    def build_preprocessor(cfg) -> Preprocessor:
        pre = cfg.preprocessing

        # Honor the enabled flag — if False, bypass all preprocessing
        enabled = pre.get("enabled", True)
        if not enabled:
            return build("preprocessor", "passthrough")

        ptype = pre.type

        # Honor the distortion_aware flag — even if type says distortion_aware,
        # we fall back to standard when distortion_aware=False (ablation mode)
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
                         bilateral_sigma_space=da.bilateral_sigma_space)

        raise ValueError(f"Unknown preprocessor type: {ptype}")

    @staticmethod
    def build_detector(cfg) -> Detector:
        dtype = cfg.detection.type
        if dtype == "mog2":
            return build("detector", "mog2",
                         history=cfg.detection.mog2.history,
                         var_threshold=cfg.detection.mog2.var_threshold,
                         detect_shadows=cfg.detection.mog2.detect_shadows,
                         morph_close_iter=cfg.detection.morphology.close_iter,
                         morph_open_iter=cfg.detection.morphology.open_iter,
                         min_contour_area=cfg.detection.min_contour_area)
        if dtype == "ground_truth":
            gt = cfg.detection.ground_truth
            return build("detector", "ground_truth",
                         gt_dir=cfg.paths.gt_dir,
                         frame_resize=tuple(cfg.preprocessing.frame_resize),
                         fallback_to_mog2=gt.fallback_to_mog2)
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
        raise ValueError(f"Unknown tracker type: {ttype}")

    @staticmethod
    def build_optical_flow(cfg) -> OpticalFlowEstimator:
        return OpticalFlowEstimator.from_config(cfg)

    @classmethod
    def build_all(cls, cfg):
        return (cls.build_preprocessor(cfg), cls.build_detector(cfg),
                cls.build_tracker(cfg),      cls.build_optical_flow(cfg))


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                    SECTION 13: PIPELINE RUNNER                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class PipelineRunner:
    """Runs the detection → tracking → feature extraction pipeline on one video."""

    def __init__(self, preprocessor: Preprocessor, detector: Detector,
                 tracker: Tracker, optical_flow: Optional[OpticalFlowEstimator] = None,
                 feature_extractors: Optional[List[FeatureExtractor]] = None,
                 aggregator: Optional[FeatureAggregator] = None,
                 min_trajectory_len: int = 30):
        self.preprocessor      = preprocessor
        self.detector          = detector
        self.tracker           = tracker
        self.optical_flow      = optical_flow
        self.feature_extractors = feature_extractors or []
        self.aggregator        = aggregator or FeatureAggregator()
        self.min_trajectory_len = min_trajectory_len

    def process_video(self, video_path: Path,
                      collect_diagnostics: bool = False) -> VideoProcessingResult:
        """Run the full pipeline on one video. Returns VideoProcessingResult."""
        video_path = Path(video_path)
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return VideoProcessingResult(
                    tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                    n_frames_processed=0,
                    failure=FailureReason(detection_failure=True))
        except Exception:
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=0,
                failure=FailureReason(detection_failure=True))

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load GT annotations if this is the GT detector
        used_gt = False
        if isinstance(self.detector, GroundTruthDetector):
            used_gt = self.detector.load_for_video(video_path.name, orig_w, orig_h)

        self.detector.reset()
        self.tracker.reset()
        self.preprocessor.reset()

        prev_gray    = None
        flow_mags    = []
        flow_angles  = []
        diagnostics  = []
        frame_no     = 0
        n_detections = 0

        while True:
            ret, raw = cap.read()
            if not ret:
                break

            pre_result = self.preprocessor.process(raw)
            if collect_diagnostics and pre_result.diagnostics:
                diagnostics.append(pre_result.diagnostics)

            dets = self.detector.detect(pre_result.gray)
            n_detections += len(dets)
            self.tracker.update(dets)

            if self.optical_flow and prev_gray is not None:
                flow_result = self.optical_flow.compute(prev_gray, pre_result.gray)
                flow_mags.append(flow_result.mean_magnitude)
                flow_angles.append(flow_result.mean_angle_deg)

            prev_gray = pre_result.gray
            frame_no  += 1

        cap.release()

        # ── Failure detection ────────────────────────────────────────────────
        if n_detections == 0 and frame_no > 0:
            return VideoProcessingResult(
                tracks=[], mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0,
                n_frames_processed=frame_no, used_gt=used_gt,
                failure=FailureReason(detection_failure=True))

        tracks    = self.tracker.get_active_tracks()
        mean_mag  = float(np.mean(flow_mags))  if flow_mags  else 0.0
        mean_ang  = float(np.mean(flow_angles)) if flow_angles else 0.0

        # ── Feature extraction ───────────────────────────────────────────────
        context = {
            "mean_flow_magnitude": mean_mag,
            "mean_flow_angle_deg": mean_ang,
            "all_tracks":          tracks,
        }

        track_feats   = []
        track_lengths = []
        has_short_tracks = False

        for track in tracks:
            tlen = len(track["trajectory"])
            track_lengths.append(tlen)
            if tlen < self.min_trajectory_len:
                has_short_tracks = True
                continue
            combined = {}
            for extractor in self.feature_extractors:
                try:
                    fv = extractor.extract_track(track, **context)
                    if fv:
                        combined.update(fv)
                except Exception:
                    pass
            if combined:
                track_feats.append(combined)

        # Detect tracking failure (tracks exist but all too short)
        if tracks and not track_feats and has_short_tracks:
            return VideoProcessingResult(
                tracks=tracks, mean_flow_magnitude=mean_mag,
                mean_flow_angle_deg=mean_ang, n_frames_processed=frame_no,
                used_gt=used_gt, preprocessing_diagnostics=diagnostics,
                track_lengths=track_lengths,
                failure=FailureReason(tracking_failure=True))

        video_fv = self.aggregator.aggregate(track_feats)

        # Detect feature failure (tracks long enough but aggregation produced nothing)
        if track_feats and video_fv is None:
            return VideoProcessingResult(
                tracks=tracks, mean_flow_magnitude=mean_mag,
                mean_flow_angle_deg=mean_ang, n_frames_processed=frame_no,
                used_gt=used_gt, preprocessing_diagnostics=diagnostics,
                track_lengths=track_lengths,
                failure=FailureReason(feature_failure=True))

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
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║              SECTION 14: DECISION TREE + RANDOM FOREST (Scratch)           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class DecisionTree(BaseEstimator):
    """CART decision tree built from scratch (unchanged from legacy)."""

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_state=42):
        self.min_samples_split = min_samples_split
        self.max_depth         = max_depth
        self.n_features        = n_features
        self.random_state      = random_state
        self.root              = None
        self.n_total_features  = None

    def fit(self, X, y):
        self.n_total_features   = X.shape[1]
        self.n_split_features   = (X.shape[1] if not self.n_features
                                   else min(X.shape[1], self.n_features))
        rng       = np.random.default_rng(self.random_state)
        self.root = self._grow(X, y, depth=0, rng=rng)
        return self

    def _grow(self, X, y, depth, rng):
        if X.shape[0] == 0:
            return {"leaf": True, "value": 0}
        if (depth >= self.max_depth or len(np.unique(y)) == 1
                or X.shape[0] < self.min_samples_split):
            return {"leaf": True, "value": self._majority(y)}

        feat_idxs = rng.choice(self.n_total_features, self.n_split_features, replace=False)
        best_feat, best_thr = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return {"leaf": True, "value": self._majority(y)}

        left_mask = X[:, best_feat] <= best_thr
        return {
            "leaf":      False,
            "feature":   best_feat,
            "threshold": best_thr,
            "left":  self._grow(X[left_mask],  y[left_mask],  depth + 1, rng),
            "right": self._grow(X[~left_mask], y[~left_mask], depth + 1, rng),
        }

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
        n   = len(y)
        n_l = left_mask.sum()
        return (self._entropy(y)
                - (n_l / n) * self._entropy(y[left_mask])
                - ((n - n_l) / n) * self._entropy(y[~left_mask]))

    def _entropy(self, y):
        counts = np.bincount(y)
        ps     = counts / len(y)
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
        self._accumulate(node["left"],  importances)
        self._accumulate(node["right"], importances)


class RandomForestCustom(BaseEstimator):
    """Bagging ensemble of DecisionTree learners (from scratch)."""

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2,
                 n_features=None, class_weight=None, random_state=42):
        self.n_trees           = n_trees
        self.max_depth         = max_depth or 1_000_000
        self.min_samples_split = min_samples_split
        self.n_features        = n_features
        self.class_weight      = class_weight
        self.random_state      = random_state
        self.trees             = []
        self.feature_importances_ = None
        self.classes_          = None

    def fit(self, X, y):
        rng    = np.random.default_rng(self.random_state)
        n_feat = self.n_features or int(np.sqrt(X.shape[1]))
        self.classes_ = np.unique(y)
        self.trees    = []

        sample_weights = None
        if self.class_weight == "balanced":
            class_counts   = np.bincount(y)
            class_w        = 1.0 / class_counts
            sample_weights = class_w[y]
            sample_weights /= sample_weights.sum()

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_feat,
                random_state=int(rng.integers(0, 1_000_000)))

            if sample_weights is not None:
                idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True, p=sample_weights)
            else:
                idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True)

            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

        raw   = np.sum([t.compute_importances(X.shape[1]) for t in self.trees], axis=0)
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw
        return self

    def predict(self, X):
        # Collect all tree predictions then take majority vote per sample
        all_preds = np.array([t.predict(X) for t in self.trees])  # (n_trees, n_samples)
        result    = []
        for i in range(all_preds.shape[1]):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            result.append(vals[np.argmax(counts)])
        return np.array(result)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║             SECTION 15: MODEL WRAPPERS (Classical + Boosting)              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def _build_pipeline(estimator) -> Pipeline:
    """StandardScaler + estimator pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


class SklearnModelAdapter(Model):
    # TODO: Need to work on why do we specifically choose these models ?
    # Let's answer this using a Paper experiments or Valid arguements.

    """Adapts an sklearn Pipeline to the Model interface."""

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
class RandomForestSklearnModel(SklearnModelAdapter):
    def __init__(self, n_estimators=200, max_depth=None, class_weight="balanced",
                 random_state=42, n_jobs=-1):
        super().__init__(_build_pipeline(RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight=class_weight, random_state=random_state, n_jobs=n_jobs)))


@register("model", "random_forest_custom")
class RandomForestCustomModel(SklearnModelAdapter):
    def __init__(self, n_trees=200, max_depth=None, min_samples_split=2,
                 class_weight="balanced", random_state=42):
        super().__init__(_build_pipeline(RandomForestCustom(
            n_trees=n_trees, max_depth=max_depth,
            min_samples_split=min_samples_split, class_weight=class_weight,
            random_state=random_state)))


@register("model", "svm_rbf")
class SVMRBFModel(SklearnModelAdapter):
    # TODO : We need a support for the parameters, why choose these specific parameter ? 
    # Let's answer this using a paper or Valid arguements

    def __init__(self, C=10, gamma="scale", class_weight="balanced", random_state=42):
        super().__init__(_build_pipeline(SVC(
            kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
            random_state=random_state, probability=True)))


@register("model", "knn")
class KNNModel(SklearnModelAdapter):
    def __init__(self, n_neighbors=5, metric="euclidean"):
        super().__init__(_build_pipeline(
            KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)))


@register("model", "lightgbm")
class LightGBMModel(Model):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1,
                 num_leaves=31, class_weight="balanced", random_state=42):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. pip install lightgbm")
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.num_leaves    = num_leaves
        self.class_weight  = class_weight
        self.random_state  = random_state
        self._model        = None
        self._scaler       = StandardScaler()
        self._le           = LabelEncoder()
        self._classes      = None

    def fit(self, X, y):
        self._le.fit(y)
        y_enc       = self._le.transform(y)
        self._classes = self._le.classes_
        X_scaled    = self._scaler.fit_transform(X)
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            class_weight=self.class_weight, random_state=self.random_state,
            verbose=-1)
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
class XGBoostModel(Model):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. pip install xgboost")
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self._model        = None
        self._scaler       = StandardScaler()
        self._le           = LabelEncoder()
        self._classes      = None

    def fit(self, X, y):
        self._le.fit(y)
        y_enc       = self._le.transform(y)
        self._classes = self._le.classes_
        X_scaled    = self._scaler.fit_transform(X)
        class_counts = np.bincount(y_enc)
        sample_w    = np.array([max(class_counts) / class_counts[yi] for yi in y_enc])
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
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
# NOTE: These models run ONLY when comparison_only=True in the config (Group C).
# They are NOT part of the distortion-aware trajectory pipeline. They serve
# as neural comparison points so the paper can benchmark classical vs. CNN.

if TORCH_AVAILABLE:

    class VideoClipDataset(TorchDataset):
        """Extracts frame clips from video files for neural model input."""

        def __init__(self, video_paths, labels, clip_len=16, frame_size=(112, 112),
                     sampling="uniform", transform=None):
            self.video_paths = video_paths
            self.labels      = labels
            self.clip_len    = clip_len
            self.frame_size  = frame_size
            self.sampling    = sampling
            self.transform   = transform or T.Compose([
                T.ToPILImage(),
                T.Resize(frame_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.video_paths)

        def __getitem__(self, idx):
            path   = self.video_paths[idx]
            frames = self._extract_frames(path)

            if len(frames) >= self.clip_len:
                if self.sampling == "uniform":
                    indices = np.linspace(0, len(frames) - 1, self.clip_len, dtype=int)
                else:
                    start   = random.randint(0, len(frames) - self.clip_len)
                    indices = range(start, start + self.clip_len)
            else:
                # Pad by repeating the last frame
                indices = list(range(len(frames)))
                while len(indices) < self.clip_len:
                    indices.append(len(frames) - 1)

            clip = torch.stack([self.transform(frames[i]) for i in indices[:self.clip_len]])
            return clip, torch.tensor(self.labels[idx])

        def _extract_frames(self, path):
            cap    = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            return frames if frames else [np.zeros((112, 112, 3), dtype=np.uint8)]

    @register("model", "cnn")
    class CNNModel(Model):
        """2D CNN (ResNet18) — Group C comparison baseline."""

        def __init__(self, backbone="resnet18_2d", pretrained=False, lr=0.001,
                     epochs=15, batch_size=8, random_state=42):
            self.backbone     = backbone
            self.pretrained   = pretrained
            self.lr           = lr
            self.epochs       = epochs
            self.batch_size   = batch_size
            self.random_state = random_state
            self._model       = None
            self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._le          = LabelEncoder()
            self._classes     = None

        def _build_backbone(self, n_classes):
            model    = tv_models.resnet18(pretrained=self.pretrained)
            model.fc = nn.Linear(512, n_classes)
            return model.to(self._device)

        def fit(self, X, y):
            self._le.fit(y)
            y_enc      = self._le.transform(y)
            self._classes = self._le.classes_
            self._model   = self._build_backbone(len(self._classes))
            optimizer  = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            criterion  = nn.CrossEntropyLoss()
            dataset    = VideoClipDataset(X, y_enc)
            loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self._model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for clips, labels in loader:
                    clips  = clips.mean(dim=1).to(self._device)  # avg over time → 2D
                    labels = labels.to(self._device)
                    optimizer.zero_grad()
                    loss   = criterion(self._model(clips), labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"    Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")
            return self

        def predict(self, X):
            if self._model is None:
                raise ValueError("Model not fitted")
            self._model.eval()
            dataset    = VideoClipDataset(X, [0] * len(X))
            loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            preds      = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips  = clips.mean(dim=1).to(self._device)
                    preds.extend(self._model(clips).argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        def predict_proba(self, X):
            if self._model is None:
                return None
            self._model.eval()
            dataset = VideoClipDataset(X, [0] * len(X))
            loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            probas  = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips  = clips.mean(dim=1).to(self._device)
                    logits = self._model(clips)
                    probas.extend(torch.softmax(logits, dim=1).cpu().numpy())
            return np.array(probas)

        @property
        def classes_(self):
            return self._classes

    @register("model", "cnn_lstm")
    class CNNLSTMModel(Model):
        """CNN + LSTM — Group C comparison baseline."""

        def __init__(self, backbone="resnet18_2d", hidden_size=256, num_layers=1,
                     lr=0.0005, epochs=20, batch_size=8, random_state=42):
            self.hidden_size  = hidden_size
            self.num_layers   = num_layers
            self.lr           = lr
            self.epochs       = epochs
            self.batch_size   = batch_size
            self.random_state = random_state
            self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._le          = LabelEncoder()
            self._classes     = None
            self._cnn         = None
            self._lstm        = None
            self._classifier  = None

        def _build(self, n_classes):
            cnn     = tv_models.resnet18(pretrained=False)
            cnn.fc  = nn.Identity()
            self._cnn = cnn.to(self._device)
            self._lstm = nn.LSTM(512, self.hidden_size, self.num_layers,
                                  batch_first=True).to(self._device)
            self._classifier = nn.Linear(self.hidden_size, n_classes).to(self._device)

        def fit(self, X, y):
            self._le.fit(y)
            y_enc       = self._le.transform(y)
            self._classes = self._le.classes_
            self._build(len(self._classes))
            params     = (list(self._cnn.parameters())
                          + list(self._lstm.parameters())
                          + list(self._classifier.parameters()))
            optimizer  = torch.optim.Adam(params, lr=self.lr)
            criterion  = nn.CrossEntropyLoss()
            dataset    = VideoClipDataset(X, y_enc)
            loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self._cnn.train(); self._lstm.train(); self._classifier.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for clips, labels in loader:
                    clips  = clips.to(self._device)
                    labels = labels.to(self._device)
                    B, T, C, H, W = clips.shape
                    with torch.no_grad():
                        feats = self._cnn(clips.view(B * T, C, H, W)).view(B, T, -1)
                    out, _ = self._lstm(feats)
                    loss   = criterion(self._classifier(out[:, -1, :]), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"    Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")
            return self

        def predict(self, X):
            if self._classifier is None:
                raise ValueError("Model not fitted")
            self._cnn.eval(); self._lstm.eval(); self._classifier.eval()
            dataset = VideoClipDataset(X, [0] * len(X))
            loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            preds   = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips  = clips.to(self._device)
                    B, T, C, H, W = clips.shape
                    feats  = self._cnn(clips.view(B * T, C, H, W)).view(B, T, -1)
                    out, _ = self._lstm(feats)
                    preds.extend(self._classifier(out[:, -1, :]).argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        @property
        def classes_(self):
            return self._classes

    @register("model", "cnn3d")
    class CNN3DModel(Model):
        """3D ResNet — Group C comparison baseline."""

        def __init__(self, backbone="r3d_18", pretrained=False, lr=0.0005,
                     epochs=20, batch_size=4, random_state=42):
            self.backbone     = backbone
            self.pretrained   = pretrained
            self.lr           = lr
            self.epochs       = epochs
            self.batch_size   = batch_size
            self.random_state = random_state
            self._model       = None
            self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._le          = LabelEncoder()
            self._classes     = None

        def _build(self, n_classes):
            model    = tv_models.video.r3d_18(pretrained=self.pretrained)
            model.fc = nn.Linear(512, n_classes)
            return model.to(self._device)

        def fit(self, X, y):
            self._le.fit(y)
            y_enc       = self._le.transform(y)
            self._classes = self._le.classes_
            self._model   = self._build(len(self._classes))
            optimizer  = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            criterion  = nn.CrossEntropyLoss()
            dataset    = VideoClipDataset(X, y_enc)
            loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self._model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for clips, labels in loader:
                    clips  = clips.permute(0, 2, 1, 3, 4).to(self._device)
                    labels = labels.to(self._device)
                    loss   = criterion(self._model(clips), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"    Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")
            return self

        def predict(self, X):
            if self._model is None:
                raise ValueError("Model not fitted")
            self._model.eval()
            dataset = VideoClipDataset(X, [0] * len(X))
            loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            preds   = []
            with torch.no_grad():
                for clips, _ in loader:
                    clips  = clips.permute(0, 2, 1, 3, 4).to(self._device)
                    preds.extend(self._model(clips).argmax(dim=1).cpu().numpy())
            return self._le.inverse_transform(np.array(preds))

        @property
        def classes_(self):
            return self._classes


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 17: TEMPORAL SMOOTHER                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TemporalSmoother:
    """Optional post-prediction ensemble across models.
    
    After all models in a run have predicted, the smoother can combine their
    predictions using majority vote or EMA-weighted voting.
    
    majority_vote : For each test sample the most common label wins.
    ema           : Weight models by their individual accuracy using
                    exponential weighting — better models contribute more.
    """

    def __init__(self, method="majority_vote", ema_alpha=0.7):
        self.method    = method
        self.ema_alpha = ema_alpha  # not used for majority_vote

    def smooth(self, predictions_by_model: Dict[str, np.ndarray],
               accuracy_by_model: Dict[str, float] = None) -> np.ndarray:
        """Combine predictions from multiple models into one final prediction.
        
        predictions_by_model: {model_name: array of string labels}
        accuracy_by_model:    {model_name: float accuracy} (used for ema)
        Returns: array of final string labels
        """
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
        result    = []
        for i in range(n_samples):
            votes = [predictions_by_model[m][i] for m in model_names]
            labels, counts = np.unique(votes, return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.array(result)

    def _ema_vote(self, predictions_by_model, model_names, accuracy_by_model):
        if not accuracy_by_model:
            return self._majority_vote(predictions_by_model, model_names)

        # Sort models by accuracy so EMA weights favour better models
        sorted_names = sorted(model_names,
                            key=lambda m: accuracy_by_model.get(m, 0.0))

        # Compute EMA weights: last model (best) has highest weight
        n      = len(sorted_names)
        weight = 1.0
        weights = {}
        for name in sorted_names:
            weights[name] = weight
            weight *= self.ema_alpha

        # Weighted vote per sample
        n_samples = len(predictions_by_model[sorted_names[0]])
        result    = []
        for i in range(n_samples):
            vote_count = {}
            for name in sorted_names:
                label = predictions_by_model[name][i]
                vote_count[label] = vote_count.get(label, 0) + weights[name]
            result.append(max(vote_count, key=vote_count.get))
        return np.array(result)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 18: PLOT MANAGER                                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class PlotManager:
    """Generates and saves plots to support paper writing.
    
    All plots are written directly to disk (in-memory figure, then saved).
    No GUI windows are opened. All methods are safe to call even when
    matplotlib is unavailable (they will print a warning and return).
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check(self) -> bool:
        if not MATPLOTLIB_AVAILABLE:
            print("  [plot] Matplotlib not available — skipping plot")
            return False
        return True

    def confusion_matrix_plot(self, cm: np.ndarray, class_names: List[str],
                              title: str, filename: str):
        """Heatmap of a confusion matrix."""
        if not self._check():
            return
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ticks = range(len(class_names))
        ax.set_xticks(list(ticks))
        ax.set_yticks(list(ticks))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(title)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def feature_importance_plot(self, feature_names: List[str],
                                importances: np.ndarray,
                                title: str, filename: str, top_n: int = 20):
        """Horizontal bar chart of feature importances (top N)."""
        if not self._check():
            return
        if len(importances) != len(feature_names):
            return

        # Sort and take top N
        order      = np.argsort(importances)[-top_n:]
        top_names  = [feature_names[i] for i in order]
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

    def model_comparison_plot(self, results: Dict[str, Dict],
                              metric: str, target: str, filename: str):
        """Bar chart comparing a metric across models."""
        if not self._check():
            return
        model_names = []
        metric_vals = []
        for mname, res in results.items():
            if metric in res:
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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def detector_stats_plot(self, stats_by_distortion: Dict[str, Dict], filename: str):
        """Bar chart of detection success rate per distortion type."""
        if not self._check():
            return
        dist_names = list(stats_by_distortion.keys())
        rates      = [stats_by_distortion[d].get("success_rate", 0.0) for d in dist_names]
        if not dist_names:
            return

        fig, ax = plt.subplots(figsize=(max(5, len(dist_names) * 1.5), 5))
        bars = ax.bar(dist_names, rates, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
        ax.set_ylabel("Detection success rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Detection success rate by distortion type")
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{rate:.2f}", ha="center", va="bottom")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def track_length_histogram(self, all_track_lengths: List[int],
                               title: str, filename: str):
        """Histogram of track lengths across all processed videos."""
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
# ║               SECTION 19: EXPERIMENT RUNNER UTILITIES                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def get_next_run_id(experiment_dir: Path) -> str:
    """Return the next available run ID in run_001, run_002, ... format."""
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
    """Return the short git commit hash of the current HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║               SECTION 20: EXPERIMENT RUNNER                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class ExperimentRunner:
    """Orchestrates one full experiment: load → process → extract → train → evaluate.
    
    Supports Groups A–E via YAML config.
    Each run gets a stable ID (run_001, run_002, …) and saves:
      - run_meta.json        : git commit, config name, feature count
      - results_<target>.json: per-model metrics + confidence outputs
      - detector_stats.json  : detection stats broken down by distortion
      - failure_taxonomy.json: per-video failure reasons
      - feature_importance_<model>.json: importance scores for paper
      - config_snapshot.yaml : exact config used
      - plots/               : confusion matrices, importance charts, etc.
    """

    def __init__(self, config: Config, experiment_name: str = None):
        self.config          = config
        self.experiment_name = experiment_name or config.experiment.name
        self.seed            = config.experiment.get("seed", 42)
        self.is_comparison   = config.get("comparison_only", False)

        exp_dir    = EXPERIMENTS_DIR / self.experiment_name
        self.run_id  = get_next_run_id(exp_dir)
        self.run_dir = exp_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.plots  = PlotManager(self.run_dir / "plots")

        random.seed(self.seed)
        np.random.seed(self.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)

        if self.is_comparison:
            print(f"\n  [NOTE] Group {config.experiment.group} is a comparison baseline.")
            print(f"         CNN results will be stored separately and NOT mixed")
            print(f"         with the classical pipeline tables.")

    # ── Dataset loading ───────────────────────────────────────────────────────

    def load_dataset(self) -> pd.DataFrame:
        """Load the AD-SVD dataset Excel file."""
        excel_path = self.config.paths.dataset_excel
        if not Path(excel_path).exists():
            raise FileNotFoundError(f"Dataset Excel not found: {excel_path}")
        df = pd.read_excel(excel_path)
        required = {"Activity", "Distortion", "Name of Video Series"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Excel missing columns: {missing}")
        return df

    # ── Video feature collection ──────────────────────────────────────────────

    def collect_video_features(self, df: pd.DataFrame,
                               use_sample: bool = False) -> List[Dict]:
        """Process all videos and return a list of per-video feature dicts."""
        # Group videos by (activity, distortion)
        groups = {}
        for (act, dist), grp in df.groupby(["Activity", "Distortion"]):
            pool = grp["Name of Video Series"].tolist()
            if use_sample:
                n    = self.config.dataset.get("n_random_videos_per_group", 5)
                pool = random.sample(pool, min(n, len(pool)))
            groups[(act, dist)] = pool

        total = sum(len(v) for v in groups.values())
        print(f"\n  Videos to process: {total}")

        preprocessor, detector, tracker, optical_flow = PipelineBuilder.build_all(self.config)

        # Build feature extractors from config
        extractors    = []
        feature_mods  = self.config.features.get("modules", [])
        for mod_name in feature_mods:
            if is_registered("feature", mod_name):
                ext = build("feature", mod_name,
                            min_trajectory_len=self.config.tracking.min_trajectory_len)
                extractors.append(ext)
            else:
                print(f"  [warn] Feature module '{mod_name}' not registered — skipping")

        aggregator = FeatureAggregator(
            aggregation_fns=self.config.features.get("aggregation", ["mean", "max", "std"]))

        runner = PipelineRunner(
            preprocessor, detector, tracker, optical_flow,
            extractors, aggregator, self.config.tracking.min_trajectory_len)

        video_features   = []
        failure_records  = []       # for failure_taxonomy.json
        all_track_lengths = []      # for histogram plot
        processed        = 0

        # Per-distortion counters for the detector stats table
        dist_stats: Dict[str, Dict] = {}
        for dist in self.config.dataset.distortion_types:
            dist_stats[dist] = {
                "total": 0, "successful": 0, "failed": 0,
                "failure_detection": 0, "failure_tracking": 0, "failure_feature": 0,
                "n_tracks_total": 0, "n_frames_total": 0, "track_lengths": [],
            }

        for (activity, distortion), video_names in groups.items():
            for vname in video_names:
                vpath    = Path(self.config.paths.video_dir) / vname
                processed += 1

                # Track per-distortion stats (use the distortion label from the df)
                dkey = str(distortion)
                if dkey not in dist_stats:
                    dist_stats[dkey] = {
                        "total": 0, "successful": 0, "failed": 0,
                        "failure_detection": 0, "failure_tracking": 0,
                        "failure_feature": 0,
                        "n_tracks_total": 0, "n_frames_total": 0, "track_lengths": [],
                    }
                dist_stats[dkey]["total"] += 1

                # File missing
                if not vpath.exists():
                    print(f"  [{processed}/{total}] MISSING  {vname}")
                    failure_records.append({
                        "video": vname, "activity": activity,
                        "distortion": distortion, "reason": "file_missing",
                    })
                    dist_stats[dkey]["failed"] += 1
                    continue

                print(f"  [{processed}/{total}] {activity}/{distortion}  {vname}",
                      end="", flush=True)

                result = runner.process_video(vpath, collect_diagnostics=True)

                # Accumulate track lengths for histogram
                if result.track_lengths:
                    all_track_lengths.extend(result.track_lengths)
                    dist_stats[dkey]["track_lengths"].extend(result.track_lengths)

                dist_stats[dkey]["n_frames_total"] += result.n_frames_processed
                dist_stats[dkey]["n_tracks_total"] += len(result.tracks)

                # Success
                if result.video_feature:
                    dist_stats[dkey]["successful"] += 1
                    fv = result.video_feature
                    # Store rich metadata alongside the feature vector
                    fv["activity"]      = activity
                    fv["distortion"]    = distortion
                    fv["video_name"]    = vname
                    fv["n_tracks"]      = len(result.tracks)
                    fv["frames_used"]   = result.n_frames_processed
                    fv["track_length"]  = (float(np.mean(result.track_lengths))
                                           if result.track_lengths else 0.0)
                    fv["feature_count"] = result.feature_count
                    # Track confidence: how long tracks are relative to minimum
                    min_len = self.config.tracking.min_trajectory_len
                    if result.track_lengths:
                        norm_lengths = [min(t / min_len, 1.0) for t in result.track_lengths]
                        fv["track_confidence"] = float(np.mean(norm_lengths))
                    else:
                        fv["track_confidence"] = 0.0
                    video_features.append(fv)
                    print(f"  →  tracks={len(result.tracks)}"
                          f"  frames={result.n_frames_processed}"
                          f"  features={result.feature_count}")

                # Failure
                else:
                    dist_stats[dkey]["failed"] += 1
                    reason_str = "unknown"
                    if result.failure:
                        f = result.failure
                        if f.detection_failure:
                            reason_str = "detection_failure"
                            dist_stats[dkey]["failure_detection"] += 1
                        elif f.tracking_failure:
                            reason_str = "tracking_failure"
                            dist_stats[dkey]["failure_tracking"] += 1
                        elif f.feature_failure:
                            reason_str = "feature_failure"
                            dist_stats[dkey]["failure_feature"] += 1
                    failure_records.append({
                        "video": vname, "activity": activity,
                        "distortion": distortion, "reason": reason_str,
                    })
                    print(f"  →  FAILED ({reason_str})")

        # ── Compute and log detector statistics ──────────────────────────────
        self._log_detector_statistics(dist_stats, total, len(video_features))

        # ── Save failure taxonomy ────────────────────────────────────────────
        failure_path = self.run_dir / "failure_taxonomy.json"
        with open(failure_path, "w") as f:
            json.dump(failure_records, f, indent=2, default=str)
        print(f"\n  Failure taxonomy saved: {failure_path}")

        # ── Save track length histogram ───────────────────────────────────────
        if all_track_lengths:
            self.plots.track_length_histogram(
                all_track_lengths,
                title=f"Track length distribution — {self.experiment_name}",
                filename="track_length_histogram.png")

        # ── Plot detector stats ───────────────────────────────────────────────
        plot_stats = {}
        for dkey, s in dist_stats.items():
            plot_stats[dkey] = {
                "success_rate": s["successful"] / s["total"] if s["total"] > 0 else 0.0,
            }
        self.plots.detector_stats_plot(plot_stats, "detector_stats_by_distortion.png")

        return video_features

    def _log_detector_statistics(self, dist_stats: Dict, total: int, n_success: int):
        """Print and save the detector statistics table broken down by distortion."""
        print("\n" + "=" * 70)
        print("  DETECTOR STATISTICS")
        print("=" * 70)
        print(f"  {'Distortion':<12}  {'Videos':>7}  {'Valid':>7}  {'Failed':>7}"
              f"  {'Rate':>6}  {'Det.Fail':>9}  {'Trk.Fail':>9}  {'Feat.Fail':>10}"
              f"  {'Tracks':>7}  {'Frames':>8}")
        print("  " + "-" * 95)

        summary_stats = {}
        for dkey, s in dist_stats.items():
            rate = s["successful"] / s["total"] if s["total"] > 0 else 0.0
            mean_track_len = (float(np.mean(s["track_lengths"]))
                              if s["track_lengths"] else 0.0)
            print(f"  {dkey:<12}  {s['total']:>7}  {s['successful']:>7}"
                  f"  {s['failed']:>7}  {rate:>6.2f}"
                  f"  {s['failure_detection']:>9}  {s['failure_tracking']:>9}"
                  f"  {s['failure_feature']:>10}"
                  f"  {s['n_tracks_total']:>7}  {s['n_frames_total']:>8}")
            summary_stats[dkey] = {
                "total_videos":     s["total"],
                "successful_videos": s["successful"],
                "failed_videos":    s["failed"],
                "success_rate":     rate,
                "failure_detection": s["failure_detection"],
                "failure_tracking": s["failure_tracking"],
                "failure_feature":  s["failure_feature"],
                "n_tracks_total":   s["n_tracks_total"],
                "n_frames_total":   s["n_frames_total"],
                "mean_track_length": mean_track_len,
            }

        overall_rate = n_success / total * 100 if total > 0 else 0.0
        print(f"\n  Overall: {n_success}/{total} successful ({overall_rate:.1f}%)")
        print("=" * 70)

        stats_path = self.run_dir / "detector_stats.json"
        with open(stats_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"  Detector stats saved: {stats_path}")

    # ── Training and evaluation ───────────────────────────────────────────────

    def train_and_evaluate(self, video_features: List[Dict],
                           target_col: str = "activity") -> Dict:
        """Train all configured models, evaluate, and save all outputs."""
        if not video_features:
            print(f"\nNo video features for target '{target_col}'.")
            return {}

        # Metadata keys that are not model input features
        meta_keys = {"activity", "distortion", "video_name", "n_tracks",
                     "frames_used", "track_length", "feature_count", "track_confidence"}

        feature_keys = sorted(set().union(*(fv.keys() for fv in video_features)) - meta_keys)
        X = np.array([[fv.get(k, 0.0) for k in feature_keys] for fv in video_features])
        y = np.array([fv[target_col] for fv in video_features])

        le    = LabelEncoder()
        y_enc = le.fit_transform(y)

        print(f"\nFeature matrix: {X.shape[0]} videos × {X.shape[1]} features")
        print(f"Target: {target_col}")
        print(f"Class distribution:\n{pd.Series(y).value_counts().to_string()}")

        use_stratify = self.config.training.get("stratify", True)
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=self.config.training.test_size,
                stratify=y_enc, random_state=self.seed)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=self.config.training.test_size,
                random_state=self.seed)

        results = {}
        all_predictions = {}   # for temporal smoothing
        all_accuracies  = {}   # for EMA weighting

        # ── Accumulated feature importance across runs ────────────────────────
        importance_accumulator = {}

        for model_def in self.config.models.active:
            if isinstance(model_def, dict):
                model_name = model_def.get("name", "")
                params     = model_def.get("params", {})
            else:
                model_name = model_def
                params     = {}

            print(f"\n{'='*60}")
            print(f"  Model: {model_name}  |  Target: {target_col}"
                + ("  [COMPARISON]" if self.is_comparison else ""))

            try:
                model = build("model", model_name, random_state=self.seed, **params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc  = accuracy_score(y_test, y_pred)
                f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

                print(f"  Accuracy:  {acc:.4f}")
                print(f"  F1 (w):    {f1:.4f}")
                print(f"  Precision: {prec:.4f}")
                print(f"  Recall:    {rec:.4f}")

                cm = confusion_matrix(y_test, y_pred)
                print(f"  Confusion Matrix:\n{cm}")

                # Confidence outputs — max predicted class probability per sample
                class_probas = model.predict_proba(X_test)
                if class_probas is not None:
                    max_probas = class_probas.max(axis=1).tolist()
                    mean_conf  = float(np.mean(max_probas))
                    print(f"  Mean prediction confidence: {mean_conf:.4f}")
                else:
                    max_probas = []
                    mean_conf  = None

                # Track confidence (mean track_confidence from video metadata)
                track_conf_vals = [fv.get("track_confidence", 0.0) for fv in video_features]
                detector_conf   = (sum(1 for v in track_conf_vals if v > 0)
                                   / len(track_conf_vals)) if track_conf_vals else 0.0

                results[model_name] = {
                    "accuracy":          acc,
                    "f1_weighted":       f1,
                    "precision":         prec,
                    "recall":            rec,
                    "confusion_matrix":  cm.tolist(),
                    "y_true":            y_test.tolist(),
                    "y_pred":            y_pred.tolist(),
                    "classes":           le.classes_.tolist(),
                    "feature_keys":      feature_keys,
                    "feature_count":     len(feature_keys),
                    "is_comparison":     self.is_comparison,
                    # Confidence outputs
                    "class_probability":  max_probas,
                    "mean_confidence":    mean_conf,
                    "track_confidence":   float(np.mean(track_conf_vals))
                                          if track_conf_vals else None,
                    "detector_confidence": detector_conf,
                }

                all_predictions[model_name] = y_pred
                all_accuracies[model_name]  = acc

                # Save model to disk
                model_path = self.run_dir / f"{model_name}_{target_col}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump({
                        "model":         model,
                        "label_encoder": le,
                        "feature_keys":  feature_keys,
                    }, f)

                # Feature importance — log and accumulate across runs
                if model.feature_importances_ is not None:
                    importances = model.feature_importances_
                    if len(importances) == len(feature_keys):
                        top_idx = np.argsort(importances)[-10:][::-1]
                        print("  Top 10 features:")
                        for i in top_idx:
                            print(f"    {feature_keys[i]:40s}  {importances[i]:.4f}")

                        # Store importance for this model
                        imp_record = {
                            feature_keys[i]: float(importances[i])
                            for i in range(len(feature_keys))
                        }
                        importance_accumulator[model_name] = imp_record

                        # Save importance chart
                        self.plots.feature_importance_plot(
                            feature_names=feature_keys,
                            importances=importances,
                            title=f"{model_name} — feature importance ({target_col})",
                            filename=f"importance_{model_name}_{target_col}.png")

                # Save confusion matrix plot
                self.plots.confusion_matrix_plot(
                    cm=cm,
                    class_names=le.classes_.tolist(),
                    title=f"{model_name} — {target_col}",
                    filename=f"cm_{model_name}_{target_col}.png")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                results[model_name] = {"error": str(e)}

        # ── Temporal smoothing (optional) ─────────────────────────────────────
        smoothing_cfg = self.config.get("smoothing", None)
        if smoothing_cfg and smoothing_cfg.get("enabled", False) and len(all_predictions) > 1:
            method  = smoothing_cfg.get("method", "majority_vote")
            alpha   = smoothing_cfg.get("ema_alpha", 0.7)
            smoother = TemporalSmoother(method=method, ema_alpha=alpha)
            smoothed = smoother.smooth(all_predictions, all_accuracies)
            # Decode smoothed labels (predictions are already string labels)
            acc_s  = accuracy_score(le.inverse_transform(y_test), smoothed)
            f1_s   = f1_score(le.inverse_transform(y_test), smoothed,
                              average="weighted", zero_division=0)
            print(f"\n  [{method}] Smoothed ensemble — Accuracy: {acc_s:.4f}  F1: {f1_s:.4f}")
            results["__smoothed__"] = {
                "method":       method,
                "accuracy":     acc_s,
                "f1_weighted":  f1_s,
                "y_pred":       smoothed.tolist(),
            }

        # ── Model comparison plot ─────────────────────────────────────────────
        self.plots.model_comparison_plot(results, "accuracy", target_col,
                                         f"model_accuracy_{target_col}.png")

        # ── Save feature importance accumulation ──────────────────────────────
        if importance_accumulator:
            imp_path = self.run_dir / f"feature_importance_{target_col}.json"
            with open(imp_path, "w") as f:
                json.dump(importance_accumulator, f, indent=2)
            print(f"\n  Feature importance saved: {imp_path}")

        # ── Save results ──────────────────────────────────────────────────────
        results_path = self.run_dir / f"results_{target_col}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # ── Save config snapshot ──────────────────────────────────────────────
        snap_path = self.run_dir / "config_snapshot.yaml"
        with open(snap_path, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

        # ── Save run metadata ─────────────────────────────────────────────────
        meta = {
            "run_id":        self.run_id,
            "experiment":    self.experiment_name,
            "git_commit":    get_git_commit(),
            "config":        self.config.experiment.name,
            "target":        target_col,
            "feature_count": len(feature_keys),
            "n_videos":      len(video_features),
            "is_comparison": self.is_comparison,
            "timestamp":     datetime.now().isoformat(),
        }
        meta_path = self.run_dir / "run_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nResults saved to: {self.run_dir}")
        return results


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                    SECTION 21: CLI MAIN ENTRY POINT                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def main():
    """Main CLI entry point."""
    create_default_configs()

    parser = argparse.ArgumentParser(
        description="Distortion-Aware Suspicious Activity Detection Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Group A: Classical baseline (primary contribution)
  python run.py --config group_a_classical --target activity

  # Group B: Distortion-aware with full features (primary contribution)
  python run.py --config group_b_distortion_aware --target activity --sample

  # Group C: Neural comparisons only (NOT primary — comparison_only=True)
  python run.py --config group_c_neural --target activity --n 10

  # Group D: Hybrid fusion
  python run.py --config group_d_hybrid --target both

  # Group E: GT upper bound (pure — no MOG2 fallback)
  python run.py --config group_e_gt_upper_bound --target activity

  # Run all groups sequentially
  python run.py --all-groups --target activity

  # List available configs and registered components
  python run.py --list-configs
  python run.py --list-components
        """)

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Group config name (e.g., group_a_classical)")
    parser.add_argument("--all-groups", action="store_true",
                        help="Run all groups A–E sequentially")
    parser.add_argument("--target", "-t", type=str, default="activity",
                        choices=["activity", "distortion", "both"],
                        help="Classification target (both = run activity then distortion)")
    parser.add_argument("--sample", action="store_true",
                        help="Use a random subset of videos per class")
    parser.add_argument("--n", type=int, default=5,
                        help="Videos per group when --sample is set")
    parser.add_argument("--list-configs", action="store_true",
                        help="List available group configs")
    parser.add_argument("--list-components", action="store_true",
                        help="List registered components")
    parser.add_argument("--gui", action="store_true",
                        help="Launch GUI (placeholder)")
    parser.add_argument("--regression-test", action="store_true",
                        help="Run regression test against legacy baseline")

    args = parser.parse_args()

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
        print("This CLI provides batch processing only.")
        return 0

    if args.regression_test:
        print("Regression test requires a legacy baseline JSON.")
        print("See docs/regression/README.md for setup instructions.")
        return 0

    # Determine which groups to run
    if args.all_groups:
        groups = sorted(p.stem for p in CONFIGS_DIR.glob("group_*.yaml"))
    elif args.config:
        groups = [args.config]
    else:
        parser.print_help()
        return 1

    for group_name in groups:
        print(f"\n{'#'*70}")
        print(f"# GROUP: {group_name}")
        print(f"{'#'*70}")

        try:
            cfg = load_config(group_name, PROJECT_ROOT, overrides={
                "dataset": {"n_random_videos_per_group": args.n}})
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue

        # Skip neural groups if PyTorch is not available
        data_path = cfg.get("data_path", "tabular")
        if data_path in ("clip", "hybrid") and not TORCH_AVAILABLE:
            print(f"  SKIPPING: {group_name} requires PyTorch (not installed)")
            print("  Install: pip install torch torchvision")
            continue

        runner = ExperimentRunner(cfg)
        df     = runner.load_dataset()

        targets = (["activity", "distortion"] if args.target == "both"
                   else [args.target])

        for target in targets:
            print(f"\n  --- Target: {target} ---")

            if data_path == "clip":
                # Neural pipeline — process videos for clip extraction
                video_paths = []
                labels      = []
                for _, row in df.iterrows():
                    vpath = Path(cfg.paths.video_dir) / row["Name of Video Series"]
                    if vpath.exists():
                        video_paths.append(str(vpath))
                        lab = row["Activity"] if target == "activity" else row["Distortion"]
                        labels.append(lab)

                if video_paths:
                    le    = LabelEncoder()
                    y_enc = le.fit_transform(labels)
                    X_train, X_test, y_train, y_test = train_test_split(
                        video_paths, y_enc, test_size=0.15,
                        stratify=y_enc, random_state=cfg.experiment.seed)

                    for model_def in cfg.models.active:
                        model_name = (model_def.get("name") if isinstance(model_def, dict)
                                    else model_def)
                        params     = (model_def.get("params", {}) if isinstance(model_def, dict)
                                    else {})
                        print(f"\n  Model: {model_name}  [COMPARISON BASELINE]")
                        try:
                            model  = build("model", model_name,
                                           random_state=cfg.experiment.seed, **params)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc    = accuracy_score(y_test, y_pred)
                            print(f"  Accuracy: {acc:.4f}")
                        except Exception as e:
                            print(f"  FAILED: {e}")
            else:
                # Classical / hybrid pipeline (primary contribution path)
                video_features = runner.collect_video_features(df, use_sample=args.sample)
                if video_features:
                    runner.train_and_evaluate(video_features, target_col=target)
                else:
                    print(f"  No video features collected for target '{target}'.")

    print("\n" + "=" * 70)
    print("DONE. Results saved to: experiments/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())