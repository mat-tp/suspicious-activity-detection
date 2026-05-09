import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np

import settings as cfg
from src.vision.detector import ForegroundDetector
from src.vision.tracker  import CentroidTracker

def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap

def preprocess_frame(frame):
    bgr  = cv2.resize(frame, cfg.FRAME_RESIZE)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, cfg.BLUR_KERNEL, cfg.BLUR_SIGMA)
    return bgr, gray

LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    cfg.LK_CRITERIA[1],
    cfg.LK_CRITERIA[2],
)
CORNER_PARAMS = dict(
    maxCorners   = cfg.MAX_CORNERS,
    qualityLevel = cfg.CORNER_QUALITY,
    minDistance  = cfg.CORNER_MIN_DIST,
    blockSize    = 7,
)

def compute_optical_flow(prev_gray, curr_gray):
    corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **CORNER_PARAMS)
    if corners is None:
        return 0.0, 0.0, None, None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, corners, None,
        winSize=cfg.LK_WIN_SIZE, maxLevel=cfg.LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )

    good_prev = corners[status == 1]
    good_next = next_pts[status == 1]
    if len(good_prev) == 0:
        return 0.0, 0.0, None, None

    flow_vecs  = good_next - good_prev
    magnitudes = np.linalg.norm(flow_vecs, axis=1)
    angles_deg = np.degrees(np.arctan2(flow_vecs[:, 1], flow_vecs[:, 0])) % 360

    return float(magnitudes.mean()), float(angles_deg.mean()), flow_vecs, good_prev

def draw_frame(bgr, mask, detections, tracks, frame_no,
               activity, distortion, flow_mag, flow_vecs, prev_corners):
    vis = bgr.copy()

    for det in detections:
        cv2.rectangle(vis, (det["x"], det["y"]),
                      (det["x"] + det["w"], det["y"] + det["h"]),
                      (180, 180, 180), 1)

    tail = cfg.TAIL_LENGTH
    for track in tracks:
        col  = cfg.TRACK_PALETTE[track["id"] % len(cfg.TRACK_PALETTE)]
        traj = track["trajectory"]
        for i in range(1, min(len(traj), tail)):
            alpha = i / tail
            faded = tuple(int(c * alpha) for c in col)
            cv2.line(vis, traj[-i], traj[-(i + 1)], faded, 1)
        cx, cy = track["centroid"]
        cv2.circle(vis, (cx, cy), 6, col, -1)
        cv2.putText(vis, f"ID{track['id']}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    if flow_vecs is not None and prev_corners is not None:
        for i, (dx, dy) in enumerate(flow_vecs):
            x0, y0 = int(prev_corners[i, 0]), int(prev_corners[i, 1])
            cv2.arrowedLine(vis, (x0, y0), (x0 + int(dx), y0 + int(dy)),
                            (0, 230, 0), 1, tipLength=0.3)

    for i, text in enumerate([
        f"Frame {frame_no:04d}  |  Tracks: {len(tracks)}",
        f"Activity: {activity}  Distortion: {distortion}",
        f"Opt-flow mag: {flow_mag:.2f}",
        "ESC/q=quit   s=skip   f=features",
    ]):
        cv2.putText(vis, text, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return vis

def process_video(video_path, activity="", distortion="", show_preview=True):
    video_path = Path(video_path)
    try:
        cap = open_video(video_path)
    except IOError:
        return [], 0.0, 0.0

    detector        = ForegroundDetector()
    tracker         = CentroidTracker()
    prev_gray       = None
    flow_magnitudes = []
    flow_angles     = []
    frame_no        = 0
    skip            = False

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        bgr, gray    = preprocess_frame(raw)
        mask, dets   = detector.apply(gray)
        active       = tracker.update(dets)

        flow_mag, flow_ang, flow_vecs, prev_corners = 0.0, 0.0, None, None
        if prev_gray is not None:
            flow_mag, flow_ang, flow_vecs, prev_corners = compute_optical_flow(
                prev_gray, gray)
            flow_magnitudes.append(flow_mag)
            flow_angles.append(flow_ang)
        prev_gray = gray

        if show_preview and frame_no % cfg.DISPLAY_EVERY_N_FRAMES == 0:
            vis = draw_frame(bgr, mask, dets, active, frame_no,
                             activity, distortion, flow_mag,
                             flow_vecs, prev_corners)
            cv2.imshow(f"[{activity}/{distortion}] {video_path.name}", vis)
            cv2.imshow("MOG2 mask", mask)
            key = cv2.waitKey(cfg.WAIT_MS) & 0xFF
            if key in (27, ord('q')):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit(0)
            if key == ord('s'):
                skip = True
                break

        frame_no += 1

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    if skip:
        return [], 0.0, 0.0

    mean_mag = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
    mean_ang = float(np.mean(flow_angles)) if flow_angles else 0.0

    return tracker.get_active_tracks(), mean_mag, mean_ang