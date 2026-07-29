from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from ..processing.preprocessing import *
from ..processing.detectors import *
from ..processing.trackers import *
from ..processing.optical_flow import *
from ..processing.features import *
from ..processing.pipeline import *
from ..models.classic import *
from ..models.neural import *
from ..processing.smoothing import *
from .plotting import *

class FiftyOneVisualizer:
    """
    Populates a FiftyOne video dataset with real per-frame detections so the
    results can be browsed interactively after a run.

    Previous version (bug): `create_video_dataset()` created a brand-new,
    *empty*, non-persistent `fo.Dataset` for every single video and never
    added the video file, frames, or detections to it — and nothing in the
    rest of the codebase ever called it. The result was silently no-op
    visualisation regardless of any flag.

    This version creates ONE persistent dataset per experiment run (the
    normal FiftyOne pattern — one dataset with many video samples, not one
    dataset per video), and actually adds each video as a sample with
    frame-level `fo.Detections` built from the real bounding boxes recorded
    during `PipelineRunner.process_video(..., collect_frame_detections=True)`.
    """

    def __init__(self, output_dir: Path, dataset_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        self.dataset_name = dataset_name or f"ad_svd_{self.output_dir.parent.name}"
        self._n_samples = 0

    def _get_or_create_dataset(self):
        if not FIFTYONE_AVAILABLE:
            return None
        if self.dataset is not None:
            return self.dataset
        if fo.dataset_exists(self.dataset_name):
            fo.delete_dataset(self.dataset_name)
        self.dataset = fo.Dataset(name=self.dataset_name, persistent=True)
        return self.dataset

    def add_video_sample(
        self,
        video_path: Path,
        frame_detections: List[List[BBox]],
        frame_size: Tuple[int, int],
        activity: str = "",
        distortion: str = "",
        video_name: str = "",
        tracks: Optional[List[Track]] = None,
    ):
        """
        Add one video, with per-frame bounding boxes, as a sample in the
        run's FiftyOne dataset. `frame_detections[i]` is the list of BBox
        dicts (from `make_bbox`) active on the i-th *processed* frame;
        `frame_size` is (width, height) of the source video, needed because
        FiftyOne expects bounding boxes normalised to [0, 1].
        """
        if not FIFTYONE_AVAILABLE:
            print("  [warn] FiftyOne not installed — skipping visualisation")
            return None
        dataset = self._get_or_create_dataset()
        w, h = frame_size
        if w <= 0 or h <= 0:
            log.warning("FiftyOneVisualizer: invalid frame_size %s for %s — skipping", frame_size, video_name)
            return None

        sample = fo.Sample(filepath=str(video_path))
        sample["activity"] = activity
        sample["distortion"] = distortion
        sample["video_name"] = video_name or Path(video_path).stem
        sample["n_tracks"] = len(tracks) if tracks is not None else None

        for i, dets in enumerate(frame_detections, start=1):
            detections = []
            for bb in dets:
                # FiftyOne bounding_box format: [x, y, w, h], normalised, top-left origin.
                rel_box = [bb["x"] / w, bb["y"] / h, bb["w"] / w, bb["h"] / h]
                detections.append(fo.Detection(label=activity or "object", bounding_box=rel_box))
            if detections:
                sample.frames[i] = fo.Frame(detections=fo.Detections(detections=detections))

        dataset.add_sample(sample)
        self._n_samples += 1
        print(f"  [fiftyone] Added sample for {sample['video_name']} "
              f"({len(frame_detections)} frames, dataset='{self.dataset_name}')")
        return sample

    def finalize(self):
        """Call once at the end of a run. Prints how to actually view the
        dataset — FiftyOne's viewer is interactive (opens a browser session),
        so it's launched by the user afterward rather than mid-batch-run."""
        if not FIFTYONE_AVAILABLE or self.dataset is None or self._n_samples == 0:
            return
        print(f"\n  [fiftyone] Dataset '{self.dataset_name}' ready "
              f"({self._n_samples} videos). To view it, run:")
        print(f"      python -c \"import fiftyone as fo; "
              f"session = fo.launch_app(fo.load_dataset('{self.dataset_name}')); session.wait()\"")

    def launch_session(self, dataset=None):
        ds = dataset if dataset is not None else self.dataset
        if FIFTYONE_AVAILABLE and ds is not None:
            return fo.launch_app(ds)
        return None
