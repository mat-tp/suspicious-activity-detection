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
from .experiment import *

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

    # activity_detection/cli.py -> activity_detection/ -> project root
    # (mirrors the original single-file layout where run.py lived in a "src"
    # folder one level below the project root).
    project_root = Path(__file__).resolve().parent.parent
    if project_root.name == "src":
        project_root = project_root.parent
    paths.PROJECT_ROOT = project_root
    paths.CONFIGS_DIR = project_root / "configs"
    paths.EXPERIMENTS_DIR = project_root / "experiments"
    paths.OUTPUTS_DIR = project_root / "outputs"
    paths.DATA_DIR = project_root / "data"
    paths.MODELS_DIR = project_root / "models"
    paths.PLOTS_DIR = project_root / "plots"
    for _d in [paths.CONFIGS_DIR, paths.EXPERIMENTS_DIR, paths.OUTPUTS_DIR,
               paths.DATA_DIR, paths.MODELS_DIR, paths.PLOTS_DIR]:
        _d.mkdir(parents=True, exist_ok=True)

    if args.list_configs:
        configs = sorted(p.stem for p in paths.CONFIGS_DIR.glob("*.yaml") if p.stem != "base")
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
        groups = sorted(p.stem for p in paths.CONFIGS_DIR.glob("group_*.yaml"))
    elif args.config:
        groups = [args.config]
    else:
        parser.print_help()
        return 1

    for group_name in groups:
        print(f"\n{'#'*70}\n# GROUP: {group_name}\n{'#'*70}")
        try:
            cfg = load_config(group_name, project_root=paths.PROJECT_ROOT,
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


