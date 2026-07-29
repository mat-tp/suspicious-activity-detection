from __future__ import annotations
from . import paths
from .common import *
from .paths import *
from .interfaces import *
from .registry import *
from .config import *
from .processing.preprocessing import *
from .processing.detectors import *
from .processing.trackers import *
from .processing.optical_flow import *
from .processing.features import *
from .processing.pipeline import *
from .models.classic import *
from .models.neural import *
from .processing.smoothing import *
from .viz.plotting import *
from .viz.fiftyone_viz import *

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



# ── Parallel video processing ────────────────────────────────────────────────
#
# The sequential path builds ONE PipelineRunner (preprocessor/detector/
# tracker/optical_flow) per group and calls `.reset()` between videos,
# reusing the same stateful objects for every video in that group. That
# statefulness (background models, track IDs, adaptive-detector EMAs) is
# exactly what has to change to parallelize safely: each concurrent worker
# needs its own fully independent set of components, not a shared, reset-
# between-calls one. Since this is CPU-bound OpenCV/NumPy/SciPy work,
# process-based parallelism (ProcessPoolExecutor) is used rather than
# threads, which would not help with GIL-bound work.
#
# `_process_one_video` is therefore a module-level, picklable function: it
# receives a plain dict snapshot of the config (Config.to_dict()) and a few
# primitives, and reconstructs everything it needs from scratch inside the
# worker process. Nothing stateful is shared across workers.
def _process_one_video(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = Config(task["config_dict"])
    detection_source = DetectionSource(task["detection_source"]) if task["detection_source"] else None
    vpath = Path(task["vpath"])
    activity = task["activity"]
    distortion = task["distortion"]
    vname = task["vname"]
    frame_sampling = task["frame_sampling"]
    vis_path = Path(task["vis_path"]) if task["vis_path"] else None
    min_trajectory_len = task["min_trajectory_len"]
    enable_mf = task["enable_mf"]

    preprocessor, detector, tracker, optical_flow = PipelineBuilder.build_all(cfg, detection_source)

    extractors = []
    feature_mods = cfg.features.get("modules", [])
    _ACCEPTS_MIN_LEN = {"trajectory", "wavelet", "interaction", "dense_trajectories"}
    for mod_name in feature_mods:
        if is_registered("feature", mod_name):
            kwargs = {}
            if mod_name in _ACCEPTS_MIN_LEN:
                kwargs["min_trajectory_len"] = min_trajectory_len
            extractors.append(build("feature", mod_name, **kwargs))

    use_dense = cfg.features.get("dense_trajectories", {}).get("enabled", False)
    if use_dense:
        dt_cfg = cfg.features.dense_trajectories
        extractors.append(build(
            "feature", "dense_trajectories",
            min_trajectory_len=dt_cfg.get("step_size", 5),
            step_size=dt_cfg.get("step_size", 5),
            block_size=dt_cfg.get("block_size", 32),
            feature_types=dt_cfg.get("feature_types", ["trajectory", "hog", "hof", "mbh"]),
        ))

    aggregator = FeatureAggregator(aggregation_fns=cfg.features.get("aggregation", ["mean", "max", "std"]))

    runner = PipelineRunner(preprocessor, detector, tracker, optical_flow,
                            extractors, aggregator, min_trajectory_len,
                            use_dense_trajectories=use_dense,
                            enable_multifamily=enable_mf)

    result = runner.process_video(
        vpath,
        collect_diagnostics=True,
        frame_sampling=frame_sampling,
        visualise_path=vis_path,
        collect_frame_detections=vis_path is not None,
    )

    adaptive_diag = None
    if hasattr(runner.detector, "get_diagnostics"):
        adaptive_diag = runner.detector.get_diagnostics()

    return {
        "activity": activity,
        "distortion": distortion,
        "vname": vname,
        "result": result,
        "adaptive_diagnostics": adaptive_diag,
        "vis_path": str(vis_path) if vis_path else None,
    }


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

        exp_dir = paths.EXPERIMENTS_DIR / self.experiment_name
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
            excel_path = paths.PROJECT_ROOT / excel_path
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

        def _ensure_dist_key(dkey):
            if dkey not in dist_stats:
                dist_stats[dkey] = {"total": 0, "successful": 0, "failed": 0,
                                    "failure_detection": 0, "failure_tracking": 0,
                                    "failure_feature": 0, "failure_track_too_short": 0,
                                    "failure_unknown": 0,
                                    "n_tracks_total": 0, "n_frames_total": 0, "track_lengths": []}

        def _fold_video_result(activity, distortion, vname, idx, result, adaptive_diag, vis_path, vpath=None):
            """Apply one video's VideoProcessingResult to the shared
            accumulators. Identical logic regardless of whether `result`
            came from the sequential runner or a parallel worker process."""
            dkey = str(distortion)
            _ensure_dist_key(dkey)

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
                if vis_path is not None:
                    visualised_counts[activity] = visualised_counts.get(activity, 0) + 1
                    print(f"\n  [visualise] Saved {Path(vis_path).name}"
                          f"  ({visualised_counts[activity]}/{self.n_visualise} for '{activity}')",
                          end="")
                    if result.frame_detections and result.frame_size and vpath is not None:
                        self.fo_visualizer.add_video_sample(
                            video_path=vpath,
                            frame_detections=result.frame_detections,
                            frame_size=result.frame_size,
                            activity=activity, distortion=distortion, video_name=vname,
                            tracks=result.tracks,
                        )
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
                if adaptive_diag is not None:
                    all_adaptive_stats.append({"video": vname, "activity": activity,
                                               "distortion": distortion, **adaptive_diag})
                    _adapt_suffix = (f"  trig={adaptive_diag['adaptive_trigger_rate']:.0%}"
                                     f" thr={adaptive_diag['avg_threshold_used']:.1f}"
                                     f" acc={adaptive_diag['rerun_accept_rate']:.0%}"
                                     f" mq={adaptive_diag['avg_mask_quality']:.2f}"
                                     f" tq={adaptive_diag['avg_track_quality']:.2f}")
                print(f"  [{idx}/{total}] {activity}/{distortion}  {vname}"
                      f"  →  tracks={len(result.tracks)} frames={result.n_frames_processed}"
                      f" features={result.feature_count}{_adapt_suffix}")
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
                        unknown_by_class[activity].append((vname, result.failure.error_message))
                failure_records.append({"video": vname, "activity": activity, "distortion": distortion, "reason": reason_str})
                print(f"  [{idx}/{total}] {activity}/{distortion}  {vname}  →  FAILED ({reason_str})")

        # ── Build the flat task list up front (needed by both paths, and
        # required for process-pool dispatch either way) ────────────────────
        tasks = []
        for (activity, distortion), video_names in groups.items():
            for vname in video_names:
                vpath = Path(self.config.paths.video_dir) / vname
                dkey = str(distortion)
                _ensure_dist_key(dkey)
                dist_stats[dkey]["total"] += 1
                if not vpath.exists():
                    processed += 1
                    print(f"  [{processed}/{total}] MISSING  {vname}")
                    failure_records.append({"video": vname, "activity": activity, "distortion": distortion, "reason": "file_missing"})
                    dist_stats[dkey]["failed"] += 1
                    continue
                _vis_path: Optional[Path] = None
                if (self.visualise
                        and visualised_counts.get(activity, 0) < self.n_visualise):
                    _class_dir = visualisation_dir / str(activity)
                    _class_dir.mkdir(parents=True, exist_ok=True)
                    _vis_path = _class_dir / f"{Path(vname).stem}_annotated.mp4"
                tasks.append((activity, distortion, vname, vpath, _vis_path))

        # ── n_workers: how many videos to process concurrently. Defaults to
        # 1 (legacy sequential behaviour, single reused stateful runner).
        # >1 switches to process-based parallelism — each worker builds its
        # own detector/tracker/preprocessor from scratch (see
        # `_process_one_video`), so this is safe even for stateful adaptive
        # detectors, at the cost of losing the "warm" background model reuse
        # across a group (each video always starts from a cold model either
        # way, since `.reset()` was called between videos in the old path
        # too — so there is no behavioural difference in what each video
        # sees, only in how many run at once). ────────────────────────────
        n_workers = int(self.config.experiment.get("n_workers", 1) or 1)
        n_workers = max(1, min(n_workers, len(tasks) or 1))

        min_trajectory_len = self.config.tracking.min_trajectory_len

        if n_workers <= 1:
            for idx, (activity, distortion, vname, vpath, _vis_path) in enumerate(tasks, start=1):
                processed += 1
                result = runner.process_video(
                    vpath, collect_diagnostics=True,
                    frame_sampling=frame_sampling, visualise_path=_vis_path,
                    collect_frame_detections=_vis_path is not None,
                )
                adaptive_diag = runner.detector.get_diagnostics() if hasattr(runner.detector, "get_diagnostics") else None
                _fold_video_result(activity, distortion, vname, processed, result, adaptive_diag, _vis_path, vpath=vpath)
        else:
            print(f"\n  [parallel] Processing {len(tasks)} videos across {n_workers} worker processes")
            config_dict = self.config.to_dict()
            detection_source_value = self.detection_source.value if self.detection_source else None
            task_payloads = [
                {
                    "config_dict": config_dict,
                    "detection_source": detection_source_value,
                    "vpath": str(vpath),
                    "activity": activity,
                    "distortion": distortion,
                    "vname": vname,
                    "frame_sampling": frame_sampling,
                    "vis_path": str(vis_path) if vis_path else None,
                    "min_trajectory_len": min_trajectory_len,
                    "enable_mf": enable_mf,
                }
                for (activity, distortion, vname, vpath, vis_path) in tasks
            ]
            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
                futures = {pool.submit(_process_one_video, payload): payload for payload in task_payloads}
                for future in as_completed(futures):
                    processed += 1
                    payload = futures[future]
                    try:
                        out = future.result()
                    except Exception as exc:
                        log.error("Worker failed for %s: %s", payload["vname"], exc)
                        failure_records.append({"video": payload["vname"], "activity": payload["activity"],
                                                "distortion": payload["distortion"], "reason": f"worker_error: {exc}"})
                        dkey = str(payload["distortion"])
                        _ensure_dist_key(dkey)
                        dist_stats[dkey]["failed"] += 1
                        continue
                    _fold_video_result(out["activity"], out["distortion"], out["vname"], processed,
                                      out["result"], out["adaptive_diagnostics"], out["vis_path"],
                                      vpath=payload["vpath"])

        # NEW: Log unknown failures grouped by activity class
        if unknown_by_class:
            log.warning("Unknown failures by activity class:")
            for cls, failures in unknown_by_class.items():
                log.warning("  Class %s: %d unknown failures", cls, len(failures))
                for name, msg in failures[:3]:
                    log.warning("    %s — %s", name, msg)

        self._log_detector_statistics(dist_stats, total, len(video_features))
        self.fo_visualizer.finalize()
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
