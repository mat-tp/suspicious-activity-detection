import argparse
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import settings as cfg
from vision.pipeline import process_video
from ml.features     import extract_all_features, aggregate_video_features, to_dataframe
from ml.classifier   import ActivityClassifier, build_sklearn_pipelines, build_all_pipelines

def load_dataset(excel_path=None):
    path = Path(excel_path or cfg.DATASET_EXCEL)
    df   = pd.read_excel(path)
    required = {"Activity", "Distortion", "Name of Video Series"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing columns: {missing}")
    return df

def collect_video_features(df, use_sample, n_per_group):
    random.seed(cfg.RANDOM_SEED)

    groups = {}
    for (act, dist), grp in df.groupby(["Activity", "Distortion"]):
        pool = grp["Name of Video Series"].tolist()
        groups[(act, dist)] = (
            random.sample(pool, min(n_per_group, len(pool)))
            if use_sample else pool
        )

    total_videos = sum(len(v) for v in groups.values())
    print(f"  Videos to process: {total_videos}")

    video_features = []
    processed = 0

    for (activity, distortion), video_names in groups.items():
        for vname in video_names:
            vpath = Path(cfg.VIDEO_DIR) / vname
            processed += 1

            if not vpath.exists():
                print(f"  [{processed}/{total_videos}] MISSING  {vname}")
                continue

            print(f"  [{processed}/{total_videos}] {activity}/{distortion}  {vname}",
                  end="", flush=True)

            tracks, mean_mag, mean_ang = process_video(
                vpath, activity=activity, distortion=distortion,
                show_preview=False,
            )

            track_feats = extract_all_features(
                tracks,
                video_name=vname, activity=activity, distortion=distortion,
                mean_flow_magnitude=mean_mag, mean_flow_angle_deg=mean_ang,
            )

            print(f"  ->  {len(tracks)} tracks, {len(track_feats)} valid",
                  flush=True)

            video_fv = aggregate_video_features(track_feats)
            if video_fv is not None:
                video_features.append(video_fv)

    return video_features

def main():
    parser = argparse.ArgumentParser(
        description="Extract features and train activity classifiers.")
    parser.add_argument(
        "--sample", action="store_true",
        help="Use a random subset of videos per (Activity, Distortion) group.")
    parser.add_argument(
        "--n", type=int, default=cfg.N_RANDOM_VIDEOS_PER_GROUP, metavar="N",
        help=f"Videos per group when --sample is set "
             f"(default: {cfg.N_RANDOM_VIDEOS_PER_GROUP}).")
    parser.add_argument(
        "--save-model", type=str, default=None, metavar="DIR",
        help="Save trained models to DIR as activity.pkl and distortion.pkl. "
             "Example: --save-model models/")
    parser.add_argument(
        "--sklearn-only", action="store_true",
        help="Skip custom classifiers (faster, good for quick iterations).")
    args = parser.parse_args()

    df = load_dataset()
    print(f"Dataset loaded: {len(df)} videos, "
          f"{df['Activity'].nunique()} activities, "
          f"{df['Distortion'].nunique()} distortions.\n")

    print("Extracting features...")
    video_features = collect_video_features(
        df, use_sample=args.sample, n_per_group=args.n)

    if not video_features:
        print("\nNo video features extracted - check that video paths exist.")
        sys.exit(1)

    feat_df = to_dataframe(video_features)
    print(f"\nVideo-level feature matrix: "
          f"{feat_df.shape[0]} videos x {feat_df.shape[1]} columns")
    print(f"Activity distribution:\n"
          f"{feat_df['activity'].value_counts().to_string()}")
    print(f"\nDistortion distribution:\n"
          f"{feat_df['distortion'].value_counts().to_string()}")

    pipelines = build_sklearn_pipelines() if args.sklearn_only \
                else build_all_pipelines()

    clf_activity = ActivityClassifier()
    print("\n" + "=" * 60)
    print("  Classifying by ACTIVITY")
    clf_activity.fit_and_evaluate(
        video_features, target_col="activity", pipelines=pipelines)

    clf_distortion = ActivityClassifier()
    print("\n" + "=" * 60)
    print("  Classifying by DISTORTION")
    clf_distortion.fit_and_evaluate(
        video_features, target_col="distortion", pipelines=pipelines)

    if args.save_model:
        save_dir = Path(args.save_model)
        clf_activity.save(save_dir / "activity.pkl",   target_col="activity")
        clf_distortion.save(save_dir / "distortion.pkl", target_col="distortion")
        print(f"\nModels saved to {save_dir.resolve()}")

    print("\nDone.")

if __name__ == "__main__":
    main()