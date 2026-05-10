"""
CLI entry point: extract features and train activity / distortion classifiers.

Usage:
  python train_and_evaluate.py [--sample] [--n N] [--use-gt] [--save-model DIR] [--sklearn-only]
"""

import argparse
import random
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import config as cfg
from src.vision.pipeline import process_video
from src.ml.features     import extract_all_features, aggregate_video_features, to_dataframe
from src.ml.classifier   import ActivityClassifier, build_sklearn_pipelines, build_all_pipelines


def load_dataset(excel_path=None):
    path = Path(excel_path or cfg.DATASET_EXCEL)
    df   = pd.read_excel(path)
    required = {"Activity", "Distortion", "Name of Video Series"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing columns: {missing}")
    return df


def collect_video_features(df, use_sample, n_per_group, use_gt):
    """Process videos and return a list of video‑level feature dicts."""
    random.seed(cfg.RANDOM_SEED)

    groups = {}
    for (act, dist), grp in df.groupby(["Activity", "Distortion"]):
        pool = grp["Name of Video Series"].tolist()
        groups[(act, dist)] = (
            random.sample(pool, min(n_per_group, len(pool)))
            if use_sample else pool
        )

    total    = sum(len(v) for v in groups.values())
    mode_str = "GT bboxes" if use_gt else "MOG2"
    print(f"  Videos to process: {total}  |  Detection mode: {mode_str}")

    video_features = []
    processed = 0

    for (activity, distortion), video_names in groups.items():
        for vname in video_names:
            vpath = cfg.VIDEO_DIR / vname
            processed += 1

            if not vpath.exists():
                print(f"  [{processed}/{total}] MISSING  {vname}")
                continue

            print(f"  [{processed}/{total}] {activity}/{distortion}  {vname}",
                  end="", flush=True)

            tracks, mean_mag, mean_ang = process_video(
                vpath,
                activity     = activity,
                distortion   = distortion,
                show_preview = False,
                use_gt       = use_gt,
            )

            track_feats = extract_all_features(
                tracks,
                video_name          = vname,
                activity            = activity,
                distortion          = distortion,
                mean_flow_magnitude = mean_mag,
                mean_flow_angle_deg = mean_ang,
            )

            print(f"  →  {len(tracks)} tracks, {len(track_feats)} valid",
                  flush=True)

            video_fv = aggregate_video_features(track_feats)
            if video_fv is not None:
                video_features.append(video_fv)

    return video_features


def main():

    parser = argparse.ArgumentParser(
        description="Extract features and train activity classifiers.")
    parser.add_argument("--sample", action="store_true",
                        help="Use a random subset of videos per group.")
    parser.add_argument("--n", type=int, default=cfg.N_RANDOM_VIDEOS_PER_GROUP,
                        metavar="N",
                        help="Videos per group when --sample is set.")
    parser.add_argument("--use-gt", action="store_true",
                        help="Use AD‑SVD GT bboxes to drive the tracker.")
    parser.add_argument("--save-model", type=str, default=None, metavar="DIR",
                        help="Save trained models into DIR.")
    parser.add_argument("--sklearn-only", action="store_true",
                        help="Skip custom Random Forest for speed.")
    args = parser.parse_args()

    df = load_dataset()

    print(f"Dataset loaded: {len(df)} videos, "
          f"{df['Activity'].nunique()} activities, "
          f"{df['Distortion'].nunique()} distortions.\n")

    print("Extracting features...")
    video_features = collect_video_features(df, use_sample = args.sample, n_per_group = args.n, use_gt = args.use_gt)

    if not video_features:
        print("\nNo video features extracted — check video paths.")
        sys.exit(1)

    feat_df = to_dataframe(video_features)

    print(f"\nVideo‑level feature matrix: {feat_df.shape[0]} videos × {feat_df.shape[1]} columns")
    print(f"Activity distribution:\n{feat_df['activity'].value_counts().to_string()}")
    print(f"\nDistortion distribution:\n{feat_df['distortion'].value_counts().to_string()}")

    pipelines = (build_sklearn_pipelines() if args.sklearn_only else build_all_pipelines())

    save_plots = cfg.PLOTS_DIR if args.save_model else None

    # Activity classifier
    clf_activity = ActivityClassifier()
    print("\n" + "=" * 60)
    print("  Classifying by ACTIVITY")
    clf_activity.fit_and_evaluate(video_features, target_col="activity", pipelines=pipelines, save_dir=save_plots)

    # Distortion classifier
    clf_distortion = ActivityClassifier()
    print("\n" + "=" * 60)
    print("  Classifying by DISTORTION")
    clf_distortion.fit_and_evaluate(video_features, target_col="distortion", pipelines=pipelines, save_dir=save_plots)

    if args.save_model:

        models_dir = cfg.MODEL_DIR
        plots_dir  = cfg.PLOTS_DIR

        models_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        clf_activity.save(models_dir / "activity.pkl",  target_col="activity")
        clf_distortion.save(models_dir / "distortion.pkl", target_col="distortion")
        
        print(f"\nModels saved to : {models_dir.resolve()}")
        print(f"Plots saved to  : {plots_dir.resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()