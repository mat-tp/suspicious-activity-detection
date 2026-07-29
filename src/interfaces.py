from __future__ import annotations
from .common import *
from .paths import *

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
    # Per-frame active-track bounding boxes, one list per processed frame.
    # Only populated when `collect_frame_detections=True` is passed to
    # `PipelineRunner.process_video` (off by default — it holds one bbox
    # list per frame in memory, which isn't needed for normal runs and is
    # only worth the cost when actually building a FiftyOne dataset).
    frame_detections: List[List[BBox]] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None  # (width, height)


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
