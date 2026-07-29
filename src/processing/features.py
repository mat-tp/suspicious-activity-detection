from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from .preprocessing import *
from .detectors import *
from .trackers import *
from .optical_flow import *

@register("feature", "trajectory")
class TrajectoryFeatures(FeatureExtractor):
    """
    Trajectory-level motion descriptors, fully normalised to physical units.

    Normalisation (mentor v6.0 — applied identically to ALL groups A–I):
      speeds    [px/frame]  × fps   → px/s
      path_len, net_disp             ÷ duration_sec → px/s (rates)
      track_duration               = n_frames / fps (seconds)
      dimensionless ratios (straightness, skewness, kurtosis,
        direction_std_deg) — unchanged.

    fps is read from context["fps"] (injected by PipelineRunner from
    CAP_PROP_FPS; default 10.0 for backward compatibility).
    Per-track duration uses the track's own frame count, NOT the total
    video duration, so a short track inside a long video is correctly
    normalised.  Research: mentor review v6.0; Sargano et al. (2017).
    """

    def __init__(self, min_trajectory_len=30):
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return [
            "track_duration_sec",           # track's own duration (s)
            "path_speed_px_s",              # path_length / duration (px/s)
            "net_displacement_speed_px_s",  # net_disp / duration  (px/s)
            "straightness",                 # net_disp / path_len  ∈ [0,1]
            "mean_speed_px_s",              # mean instantaneous speed (px/s)
            "max_speed_px_s",               # max  instantaneous speed (px/s)
            "std_speed_px_s",               # std  of speed (px/s)
            "skew_speed",                   # dimensionless skewness
            "kurt_speed",                   # dimensionless kurtosis
            "mean_direction_deg",           # mean heading (°)
            "direction_std_deg",            # std of heading (°)
            "bbox_area",                    # spatial extent (px²)
            "mean_flow_magnitude",          # video-level LK flow magnitude
            "mean_flow_angle_deg",          # video-level LK flow angle (°)
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None

        fps = float(context.get("fps", 10.0))
        n_frames_tracked = len(traj)
        duration_sec = n_frames_tracked / fps      # this track's own duration

        deltas      = np.diff(traj, axis=0)
        speeds_pf   = np.linalg.norm(deltas, axis=1)   # px / frame
        speeds_ps   = speeds_pf * fps                   # px / s
        angles_deg  = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360

        total_dist_px = float(speeds_pf.sum())
        net_disp_px   = float(euclidean(traj[0], traj[-1]))
        straightness  = net_disp_px / total_dist_px if total_dist_px > 0 else 0.0

        bbox_wh   = traj.max(axis=0) - traj.min(axis=0)
        bbox_area = float(bbox_wh[0] * bbox_wh[1])

        return {
            "track_duration_sec":           duration_sec,
            "path_speed_px_s":              total_dist_px / duration_sec if duration_sec > 0 else 0.0,
            "net_displacement_speed_px_s":  net_disp_px   / duration_sec if duration_sec > 0 else 0.0,
            "straightness":                 straightness,
            "mean_speed_px_s":              float(speeds_ps.mean()),
            "max_speed_px_s":               float(speeds_ps.max()),
            "std_speed_px_s":               float(speeds_ps.std()),
            "skew_speed":                   float(skew(speeds_ps)) if len(speeds_ps) > 2 else 0.0,
            "kurt_speed":                   float(kurtosis(speeds_ps)) if len(speeds_ps) > 2 else 0.0,
            "mean_direction_deg":           float(angles_deg.mean()),
            "direction_std_deg":            float(angles_deg.std()),
            "bbox_area":                    bbox_area,
            "mean_flow_magnitude":          context.get("mean_flow_magnitude", 0.0),
            "mean_flow_angle_deg":          context.get("mean_flow_angle_deg", 0.0),
        }


@register("feature", "circular_motion")
class CircularMotionFeatures(FeatureExtractor):
    def feature_names(self) -> List[str]:
        return ["circular_mean_rad", "circular_variance", "direction_concentration", "angular_dispersion"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 2:
            return None
        deltas = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))
        circ_mean = np.arctan2(mean_sin, mean_cos)
        R = np.sqrt(mean_cos**2 + mean_sin**2)
        circ_var = 1.0 - R
        ang_diff = np.arctan2(np.sin(angles_rad - circ_mean), np.cos(angles_rad - circ_mean))
        return {
            "circular_mean_rad": float(circ_mean),
            "circular_variance": float(circ_var),
            "direction_concentration": float(R),
            "angular_dispersion": float(np.std(ang_diff)),
        }


@register("feature", "kinematics")
class KinematicsFeatures(FeatureExtractor):
    """
    Acceleration and jerk in physical units (mentor v6.0).

      speed     [px/frame]  × fps   = speed  [px/s]
      accel     Δspeed/Δt   × fps   = accel  [px/s²]  (Δt = 1/fps s)
      jerk      Δaccel/Δt   × fps   = jerk   [px/s³]

    Skewness statistics are dimensionless and unchanged.
    fps from context["fps"]; default 10.0.
    """

    def feature_names(self) -> List[str]:
        return [
            "mean_accel_px_s2", "std_accel_px_s2", "max_accel_px_s2", "skew_accel",
            "mean_jerk_px_s3",  "std_jerk_px_s3",  "max_jerk_px_s3",  "skew_jerk",
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None
        fps = float(context.get("fps", 10.0))

        deltas     = np.diff(traj, axis=0)
        speeds_pf  = np.linalg.norm(deltas, axis=1)   # px/frame
        accel_ps2  = np.diff(speeds_pf) * fps          # px/s²  (×fps converts Δ/frame to Δ/s)
        jerk_ps3   = np.diff(accel_ps2) * fps          # px/s³

        if len(accel_ps2) < 2 or len(jerk_ps3) < 2:
            return None
        return {
            "mean_accel_px_s2": float(np.mean(accel_ps2)),
            "std_accel_px_s2":  float(np.std(accel_ps2)),
            "max_accel_px_s2":  float(np.max(np.abs(accel_ps2))),
            "skew_accel":       float(skew(accel_ps2)) if len(accel_ps2) > 2 else 0.0,
            "mean_jerk_px_s3":  float(np.mean(jerk_ps3)),
            "std_jerk_px_s3":   float(np.std(jerk_ps3)),
            "max_jerk_px_s3":   float(np.max(np.abs(jerk_ps3))),
            "skew_jerk":        float(skew(jerk_ps3))  if len(jerk_ps3)  > 2 else 0.0,
        }


@register("feature", "shape")
class ShapeFeatures(FeatureExtractor):
    """
    Trajectory shape descriptors (mentor v6.0 normalisation).

      mean/std curvature: |Δangle| per step (rad) — already dimensionless.
      turning_rate_rad_s: total angle turned ÷ duration_sec → rad/s.
        Captures direction-change rate independently of video length.
      convex_hull_area:   spatial extent in px² — length-independent within
        a video; kept raw.
      compactness:        4π·area/perimeter² ∈ [0,1] — dimensionless.
      path_efficiency:    net_disp / path_len ∈ [0,1] — dimensionless.

    fps from context["fps"]; default 10.0.
    """

    def feature_names(self) -> List[str]:
        return [
            "mean_curvature",      # mean |Δangle| per step (rad)
            "std_curvature",       # std  |Δangle| per step (rad)
            "turning_rate_rad_s",  # total angle turned / duration (rad/s)
            "convex_hull_area",    # spatial extent (px²)
            "compactness",         # 4π·area/perimeter²
            "path_efficiency",     # net_disp / path_len ∈ [0,1]
        ]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < 4:
            return None

        fps          = float(context.get("fps", 10.0))
        duration_sec = len(traj) / fps

        deltas     = np.diff(traj, axis=0)
        angles_rad = np.arctan2(deltas[:, 1], deltas[:, 0])
        curvature  = np.abs(np.diff(angles_rad))
        curvature  = np.minimum(curvature, 2 * np.pi - curvature)

        total_angle   = float(curvature.sum())
        turning_rate  = total_angle / duration_sec if duration_sec > 0 else 0.0

        hull_area   = 0.0
        compactness = 0.0
        try:
            if len(traj) >= 3:
                hull = ConvexHull(traj)
                hull_area  = float(hull.volume)
                hull_perim = float(hull.area)
                if hull_perim > 0:
                    compactness = 4 * np.pi * hull_area / (hull_perim ** 2)
        except Exception:
            pass

        total_dist = float(np.sum(np.linalg.norm(deltas, axis=1)))
        net_disp   = float(euclidean(traj[0], traj[-1]))
        path_eff   = net_disp / total_dist if total_dist > 0 else 0.0

        return {
            "mean_curvature":     float(np.mean(curvature)) if len(curvature) > 0 else 0.0,
            "std_curvature":      float(np.std(curvature))  if len(curvature) > 0 else 0.0,
            "turning_rate_rad_s": turning_rate,
            "convex_hull_area":   hull_area,
            "compactness":        compactness,
            "path_efficiency":    path_eff,
        }


@register("feature", "wavelet")
class WaveletFeatures(FeatureExtractor):
    def __init__(self, wavelet="db4", level=3, min_trajectory_len=30):
        self.wavelet = wavelet
        self.level = level
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return ["dwt_energy_approx", "dwt_energy_detail1", "dwt_energy_detail2",
                "dwt_energy_detail3", "dwt_energy_total", "dwt_entropy"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        if not PYWAVELETS_AVAILABLE:
            return None
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None
        fps    = float(context.get("fps", 10.0))
        deltas = np.diff(traj, axis=0)
        # Normalise to px/s so wavelet energies are in (px/s)², comparable
        # across videos with different frame rates (mentor v6.0).
        # Research: Zhang et al. (2022) — DWT on speed signals.
        speeds = np.linalg.norm(deltas, axis=1) * fps   # px/s
        n = len(speeds)
        pad_len = 2 ** int(np.ceil(np.log2(max(n, 8)))) - n
        if pad_len > 0:
            speeds = np.pad(speeds, (0, pad_len), mode="reflect")
        try:
            coeffs = pywt.wavedec(speeds, self.wavelet, level=self.level)
        except Exception:
            return None
        energies = [float(np.sum(np.array(c) ** 2)) for c in coeffs]
        total_energy = sum(energies)
        eps = 1e-10
        energy_dist = np.array(energies) / (total_energy + eps)
        entropy = float(-np.sum(energy_dist * np.log(energy_dist + eps)))
        result = {"dwt_energy_approx": energies[0] if energies else 0.0}
        for i in range(1, 4):
            result[f"dwt_energy_detail{i}"] = energies[i] if i < len(energies) else 0.0
        result["dwt_energy_total"] = total_energy
        result["dwt_entropy"] = entropy
        return result


@register("feature", "interaction")
class InteractionFeatures(FeatureExtractor):
    def __init__(self, proximity_threshold=100.0, min_trajectory_len=30):
        self.proximity_threshold = proximity_threshold
        self.min_trajectory_len = min_trajectory_len

    def feature_names(self) -> List[str]:
        return ["mean_pairwise_distance", "min_pairwise_distance", "std_pairwise_distance",
                "mean_relative_speed", "max_relative_speed", "proximity_duration_ratio", "n_proximity_events"]

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        """
        Interaction features with normalised speed and event rate (v6.0).

          relative speed:  px/frame → px/s  (× fps)
          pairwise_dists:  spatial (px) — unchanged (not temporal)
          proximity_duration_ratio: already dimensionless fraction
          proximity_event_rate_s:  n_events / overlap_duration_sec (events/s)

        fps from context["fps"]; default 10.0.
        """
        all_tracks = context.get("all_tracks", [])
        if len(all_tracks) < 2:
            return None
        my_traj = np.array(track["trajectory"], dtype=float)
        if len(my_traj) < self.min_trajectory_len:
            return None

        fps = float(context.get("fps", 10.0))

        pairwise_dists: List[float] = []
        rel_speeds_ps:  List[float] = []
        proximity_frames = 0
        total_frames     = 0
        n_events         = 0

        for other in all_tracks:
            if other["id"] == track["id"]:
                continue
            other_traj = np.array(other["trajectory"], dtype=float)
            min_len = min(len(my_traj), len(other_traj))
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
                        n_events += 1
                        in_proximity = True
                else:
                    in_proximity = False
            my_spd_pf    = np.linalg.norm(np.diff(my_traj[:min_len],    axis=0), axis=1)
            other_spd_pf = np.linalg.norm(np.diff(other_traj[:min_len], axis=0), axis=1)
            # Convert differential speed from px/frame to px/s
            rel_speeds_ps.extend(np.abs(my_spd_pf - other_spd_pf) * fps)

        if not pairwise_dists:
            return None

        overlap_sec = total_frames / fps if fps > 0 else 1.0
        return {
            "mean_pairwise_distance":   float(np.mean(pairwise_dists)),
            "min_pairwise_distance":    float(np.min(pairwise_dists)),
            "std_pairwise_distance":    float(np.std(pairwise_dists)),
            "mean_relative_speed_px_s": float(np.mean(rel_speeds_ps)) if rel_speeds_ps else 0.0,
            "max_relative_speed_px_s":  float(np.max(rel_speeds_ps))  if rel_speeds_ps else 0.0,
            "proximity_duration_ratio": proximity_frames / total_frames if total_frames > 0 else 0.0,
            "proximity_event_rate_s":   n_events / overlap_sec if overlap_sec > 0 else 0.0,
        }


@register("feature", "dense_trajectories")
class DenseTrajectoryFeatures(FeatureExtractor):
    def __init__(self, min_trajectory_len=15, step_size=5, block_size=32, feature_types=None):
        self.min_trajectory_len = min_trajectory_len
        self.step_size = step_size
        self.block_size = block_size
        self.feature_types = feature_types or ["trajectory", "hog", "hof", "mbh"]
        self.hog_descriptor = cv2.HOGDescriptor(_winSize=(block_size, block_size), _blockSize=(16, 16),
                                                _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
        self.fast_detector = cv2.FastFeatureDetector_create(threshold=20)

    def feature_names(self) -> List[str]:
        names = []
        if "trajectory" in self.feature_types:
            names.extend(["dense_traj_length", "dense_traj_displacement", "dense_traj_mean_speed",
                          "dense_traj_std_speed", "dense_traj_direction_mean", "dense_traj_direction_std"])
        if "hog" in self.feature_types:
            names.extend([f"dense_hog_bin_{i}" for i in range(9)])
        if "hof" in self.feature_types:
            names.extend([f"dense_hof_bin_{i}" for i in range(9)])
        if "mbh" in self.feature_types:
            names.extend([f"dense_mbhx_bin_{i}" for i in range(9)])
            names.extend([f"dense_mbhy_bin_{i}" for i in range(9)])
        names.extend(["dense_keypoint_density", "dense_flow_magnitude_mean", "dense_flow_magnitude_std",
                      "dense_texture_variance", "dense_edge_density"])
        return names

    def extract_keypoints(self, gray_frame: np.ndarray) -> np.ndarray:
        h, w = gray_frame.shape[:2]
        points = []
        shi_tomasi = cv2.goodFeaturesToTrack(gray_frame, maxCorners=100, qualityLevel=0.01,
                                             minDistance=self.step_size, blockSize=3)
        if shi_tomasi is not None:
            points.extend(shi_tomasi.reshape(-1, 2))
        kp = self.fast_detector.detect(gray_frame, None)
        points.extend([kp_item.pt for kp_item in kp[:50]])
        for y in range(self.step_size, h - self.step_size, self.step_size):
            for x in range(self.step_size, w - self.step_size, self.step_size):
                patch = gray_frame[y-2:y+3, x-2:x+3]
                if np.std(patch) > 10:
                    points.append([float(x), float(y)])
        if points:
            points = np.array(points)
            if len(points) > 1:
                # Grid-based NMS
                cell = self.step_size
                seen: Dict[Tuple[int, int], bool] = {}
                keep = np.zeros(len(points), dtype=bool)
                for i, (px, py) in enumerate(points):
                    key = (int(px) // cell, int(py) // cell)
                    if key not in seen:
                        seen[key] = True
                        keep[i] = True
                points = points[keep]
        return points if len(points) > 0 else np.array([])

    def compute_local_descriptors(self, gray_patch: np.ndarray, flow_patch: np.ndarray) -> Dict[str, float]:
        features = {}
        if "hog" in self.feature_types and gray_patch.shape[0] >= self.block_size:
            try:
                resized = cv2.resize(gray_patch, (self.block_size, self.block_size))
                hog = self.hog_descriptor.compute(resized)
                if hog is not None:
                    hog = hog.reshape(-1, 9).mean(axis=0)
                    for i in range(min(9, len(hog))):
                        features[f"dense_hog_bin_{i}"] = float(hog[i])
            except Exception:
                for i in range(9):
                    features[f"dense_hog_bin_{i}"] = 0.0
        if "hof" in self.feature_types and flow_patch.size >= 2:
            try:
                mag, ang = cv2.cartToPolar(flow_patch[..., 0], flow_patch[..., 1])
                mask = mag > 1.0
                if mask.sum() > 0:
                    hof, _ = np.histogram(ang[mask], bins=9, range=(0, 2*np.pi))
                    hof = hof.astype(float) / (hof.sum() + 1e-7)
                else:
                    hof = np.zeros(9)
                for i in range(9):
                    features[f"dense_hof_bin_{i}"] = float(hof[i])
            except Exception:
                for i in range(9):
                    features[f"dense_hof_bin_{i}"] = 0.0
        if "mbh" in self.feature_types and flow_patch.size >= 4:
            try:
                grad_x = cv2.Sobel(flow_patch[..., 0], cv2.CV_32F, 1, 0)
                grad_y = cv2.Sobel(flow_patch[..., 1], cv2.CV_32F, 0, 1)
                mbhx, _ = np.histogram(grad_x.flatten(), bins=9, range=(-10, 10))
                mbhy, _ = np.histogram(grad_y.flatten(), bins=9, range=(-10, 10))
                mbhx = mbhx.astype(float) / (mbhx.sum() + 1e-7)
                mbhy = mbhy.astype(float) / (mbhy.sum() + 1e-7)
                for i in range(9):
                    features[f"dense_mbhx_bin_{i}"] = float(mbhx[i])
                    features[f"dense_mbhy_bin_{i}"] = float(mbhy[i])
            except Exception:
                for i in range(9):
                    features[f"dense_mbhx_bin_{i}"] = 0.0
                    features[f"dense_mbhy_bin_{i}"] = 0.0
        return features

    def extract_track(self, track: Track, **context) -> Optional[FeatureDict]:
        traj = np.array(track["trajectory"], dtype=float)
        if len(traj) < self.min_trajectory_len:
            return None
        features = {}
        if "trajectory" in self.feature_types:
            deltas = np.diff(traj, axis=0)
            speeds = np.linalg.norm(deltas, axis=1)
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            features.update({
                "dense_traj_length": float(len(traj)),
                "dense_traj_displacement": float(np.linalg.norm(traj[-1] - traj[0])),
                "dense_traj_mean_speed": float(np.mean(speeds)),
                "dense_traj_std_speed": float(np.std(speeds)),
                "dense_traj_direction_mean": float(np.mean(angles)),
                "dense_traj_direction_std": float(np.std(angles))
            })
        prev_gray = context.get("prev_gray")
        curr_gray = context.get("curr_gray")
        if prev_gray is not None and curr_gray is not None:
            h, w = curr_gray.shape[:2]
            keypoints = self.extract_keypoints(curr_gray)
            if len(keypoints) > 0:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_mags = np.linalg.norm(flow, axis=2)
                edges = cv2.Canny(curr_gray, 50, 150)
                features.update({
                    "dense_keypoint_density": float(len(keypoints)),
                    "dense_flow_magnitude_mean": float(np.mean(flow_mags)),
                    "dense_flow_magnitude_std": float(np.std(flow_mags)),
                    "dense_texture_variance": float(np.var(curr_gray)),
                    "dense_edge_density": float(np.sum(edges > 0) / edges.size)
                })
                sampled_kps = keypoints[:20]
                for pt in sampled_kps:
                    x, y = int(pt[0]), int(pt[1])
                    x1 = max(0, x - self.block_size//2)
                    y1 = max(0, y - self.block_size//2)
                    x2 = min(w, x + self.block_size//2)
                    y2 = min(h, y + self.block_size//2)
                    if x2 > x1 and y2 > y1:
                        gray_patch = curr_gray[y1:y2, x1:x2]
                        flow_patch = flow[y1:y2, x1:x2]
                        patch_feats = self.compute_local_descriptors(gray_patch, flow_patch)
                        for key, val in patch_feats.items():
                            features[key] = features.get(key, 0.0) + val
                if sampled_kps and len(sampled_kps) > 0:
                    for key in list(features.keys()):
                        if any(key.startswith(prefix) for prefix in ["dense_hog_", "dense_hof_", "dense_mbh"]):
                            features[key] /= len(sampled_kps)
        return features if features else None



class FeatureAggregator:
    def __init__(self, aggregation_fns=None):
        self.aggregation_fns = aggregation_fns or ["mean", "max", "std"]

    def aggregate(self, track_features: List[FeatureDict]) -> Optional[FeatureDict]:
        if not track_features:
            return None
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
                video_fv[f"max_{key}"] = float(np.max(arr))
            if "std" in self.aggregation_fns:
                video_fv[f"std_{key}"] = float(np.std(arr)) if len(arr) > 1 else 0.0
        return video_fv
