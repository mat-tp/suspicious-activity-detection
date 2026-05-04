import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import skew, kurtosis

import settings as cfg

TRACK_FEATURE_NAMES = [
    "trajectory_length",
    "total_distance",
    "net_displacement",
    "straightness",
    "mean_speed", "max_speed", "std_speed", "skew_speed", "kurt_speed",
    "mean_direction_deg", "direction_std_deg",
    "bbox_area",
    "mean_flow_magnitude",
    "mean_flow_angle_deg",
]

FEATURE_NAMES = (
    [f"mean_{f}" for f in TRACK_FEATURE_NAMES] +
    [f"max_{f}"  for f in TRACK_FEATURE_NAMES] +
    [f"std_{f}"  for f in TRACK_FEATURE_NAMES] +
    ["n_valid_tracks"]
)

def extract_features(track, video_name="", activity="", distortion="",
                     mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0):
    traj = np.array(track["trajectory"], dtype=float)
    if len(traj) < cfg.MIN_TRAJECTORY_LEN:
        return None

    deltas = np.diff(traj, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360

    total_dist   = float(speeds.sum())
    net_disp     = float(euclidean(traj[0], traj[-1]))
    straightness = net_disp / total_dist if total_dist > 0 else 0.0

    bbox_wh   = traj.max(axis=0) - traj.min(axis=0)
    bbox_area = float(bbox_wh[0] * bbox_wh[1])

    speed_skew = float(skew(speeds)) if len(speeds) > 2 else 0.0
    speed_kurt = float(kurtosis(speeds)) if len(speeds) > 2 else 0.0

    return {
        "track_id":   track["id"],
        "video_name": video_name,
        "activity":   activity,
        "distortion": distortion,
        "trajectory_length":   float(len(traj)),
        "total_distance":      total_dist,
        "net_displacement":    net_disp,
        "straightness":        straightness,
        "mean_speed":          float(speeds.mean()),
        "max_speed":           float(speeds.max()),
        "std_speed":           float(speeds.std()),
        "skew_speed":          speed_skew,
        "kurt_speed":          speed_kurt,
        "mean_direction_deg":  float(angles.mean()),
        "direction_std_deg":   float(angles.std()),
        "bbox_area":           bbox_area,
        "mean_flow_magnitude": mean_flow_magnitude,
        "mean_flow_angle_deg": mean_flow_angle_deg,
    }

def extract_all_features(tracks, video_name="", activity="", distortion="",
                         mean_flow_magnitude=0.0, mean_flow_angle_deg=0.0):
    results = []
    for track in tracks:
        fv = extract_features(
            track,
            video_name=video_name, activity=activity, distortion=distortion,
            mean_flow_magnitude=mean_flow_magnitude,
            mean_flow_angle_deg=mean_flow_angle_deg,
        )
        if fv is not None:
            results.append(fv)
    return results

def aggregate_video_features(track_feature_dicts):
    if not track_feature_dicts:
        return None

    first = track_feature_dicts[0]
    video_name  = first["video_name"]
    activity    = first["activity"]
    distortion  = first["distortion"]

    df = pd.DataFrame(track_feature_dicts)[TRACK_FEATURE_NAMES]
    agg = {}
    for col in TRACK_FEATURE_NAMES:
        vals = df[col].values
        agg[f"mean_{col}"] = float(np.mean(vals))
        agg[f"max_{col}"]  = float(np.max(vals))
        agg[f"std_{col}"]  = float(np.std(vals))

    agg["n_valid_tracks"] = float(len(track_feature_dicts))
    agg["video_name"] = video_name
    agg["activity"]   = activity
    agg["distortion"] = distortion

    return agg

def to_dataframe(feature_dicts):
    return pd.DataFrame(feature_dicts) if feature_dicts else pd.DataFrame()

def get_XY(df, target_col="activity"):
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing expected feature columns: {missing}\n"
                         "Did you forget to call aggregate_video_features()?")
    X = df[FEATURE_NAMES].values.astype(float)
    y = df[target_col].values
    return X, y, FEATURE_NAMES