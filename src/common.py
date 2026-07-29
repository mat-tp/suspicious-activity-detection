from __future__ import annotations

import abc
import argparse
import copy
import json
import logging
import multiprocessing
import os
import pickle
import random
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# NOTE: DatasetInfoHandler / plotting / etc. below optionally use these
# Re-exported so 'from .common import *' gives every module the same third-party surface
