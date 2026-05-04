# """
# debug_single_video.py
# ======================
# Quick script — runs the full pipeline on ONE video with live preview.
# Edit VIDEO_PATH / ACTIVITY / DISTORTION below and run:

#     python debug_single_video.py
# """

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# # ── edit these ────────────────────────────────────────────────────────────────
# VIDEO_PATH = "suspicious-activity-detection/surveillanceVideosDataset/surveillanceVideos/3982ExFo_IndPO_HQ_C3.mp4"
# ACTIVITY   = "PO"
# DISTORTION = "ExFo"
# # ─────────────────────────────────────────────────────────────────────────────

# from src.vision.pipeline import process_video
# from src.ml.features     import extract_all_features


# def main():
#     tracks, mean_mag, mean_ang = process_video(
#         VIDEO_PATH, activity=ACTIVITY, distortion=DISTORTION,
#         show_preview=True,
#     )

#     feature_dicts = extract_all_features(
#         tracks,
#         video_name=Path(VIDEO_PATH).name,
#         activity=ACTIVITY, distortion=DISTORTION,
#         mean_flow_magnitude=mean_mag, mean_flow_angle_deg=mean_ang,
#     )

#     print(f"\n{'═' * 60}")
#     print(f"  Extracted {len(feature_dicts)} feature dict(s) from "
#           f"{len(tracks)} tracks.")

#     for fv in feature_dicts:
#         print(f"\n  Track {fv['track_id']}")
#         for k, v in fv.items():
#             if k in ("track_id", "video_name", "activity", "distortion"):
#                 continue
#             print(f"    {k:<25} {v:.4f}" if isinstance(v, float)
#                   else f"    {k:<25} {v}")


# if __name__ == "__main__":
#     main()