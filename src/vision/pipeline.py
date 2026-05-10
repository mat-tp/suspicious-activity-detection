"""
Per‑video processing loop.
Detection mode controlled by `use_gt`:
  - False → MOG2 drives the tracker.
  - True  → AD‑SVD ground‑truth bounding boxes drive the tracker;
             MOG2 still runs in parallel for IoU overlay.
"""

import cv2
import numpy as np
from pathlib import Path

import config as cfg
from vision.detector import ForegroundDetector, load_gt_bboxes
from vision.tracker  import CentroidTracker


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def preprocess_frame(frame):
    """Resize → greyscale → Gaussian blur. Returns (bgr, gray)."""
    bgr  = cv2.resize(frame, cfg.FRAME_RESIZE)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, cfg.BLUR_KERNEL, cfg.BLUR_SIGMA)
    return bgr, gray


# ── optical flow ──────────────────────────────────────────────────────────────

_LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    cfg.LK_CRITERIA[1],
    cfg.LK_CRITERIA[2],
)
_CORNER_PARAMS = dict(
    maxCorners   = cfg.MAX_CORNERS,
    qualityLevel = cfg.CORNER_QUALITY,
    minDistance  = cfg.CORNER_MIN_DIST,
    blockSize    = 7,
)


def compute_optical_flow(prev_gray, curr_gray):
    """Lucas‑Kanade sparse optical flow.
    Returns (mean_magnitude, mean_angle_deg, flow_vectors, prev_corners).
    """
    corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **_CORNER_PARAMS)
    if corners is None:
        return 0.0, 0.0, None, None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, corners, None,
        winSize=cfg.LK_WIN_SIZE, maxLevel=cfg.LK_MAX_LEVEL,
        criteria=_LK_CRITERIA,
    )
    good_prev = corners[status == 1]
    good_next = next_pts[status == 1]
    if len(good_prev) == 0:
        return 0.0, 0.0, None, None

    flow_vecs  = good_next - good_prev
    magnitudes = np.linalg.norm(flow_vecs, axis=1)
    angles_deg = np.degrees(np.arctan2(flow_vecs[:, 1], flow_vecs[:, 0])) % 360
    return float(magnitudes.mean()), float(angles_deg.mean()), flow_vecs, good_prev


# ── IoU helper ────────────────────────────────────────────────────────────────

def bbox_iou(a, b):
    """Intersection‑over‑Union between two bbox dicts {x, y, w, h}."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


# ── display ───────────────────────────────────────────────────────────────────

def draw_frame(bgr, detections, gt_bbox, tracks,
               frame_no, activity, distortion, flow_mag,
               flow_vecs, prev_corners, using_gt):
    vis = bgr.copy()

    # MOG2 detection boxes — grey
    for det in detections:
        cv2.rectangle(vis, (det["x"], det["y"]),
                      (det["x"] + det["w"], det["y"] + det["h"]),
                      (160, 160, 160), 1)

    # GT bbox — yellow, thicker; show IoU vs best MOG2 detection
    if gt_bbox is not None:
        g = gt_bbox
        cv2.rectangle(vis, (g["x"], g["y"]),
                      (g["x"] + g["w"], g["y"] + g["h"]),
                      (0, 220, 255), 2)
        cv2.putText(vis, "GT", (g["x"], g["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        if detections:
            best_iou = max(bbox_iou(g, d) for d in detections)
            cv2.putText(vis, f"IoU:{best_iou:.2f}",
                        (g["x"], g["y"] + g["h"] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

    # Track trails and centroids
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

    # Optical flow arrows
    if flow_vecs is not None and prev_corners is not None:
        for i, (dx, dy) in enumerate(flow_vecs):
            x0, y0 = int(prev_corners[i, 0]), int(prev_corners[i, 1])
            cv2.arrowedLine(vis, (x0, y0), (x0 + int(dx), y0 + int(dy)),
                            (0, 230, 0), 1, tipLength=0.3)

    mode = "GT" if using_gt else "MOG2"
    hud = [
        f"Frame {frame_no:04d}  |  Tracks: {len(tracks)}  |  Mode: {mode}",
        f"Activity: {activity}  Distortion: {distortion}",
        f"Opt-flow mag: {flow_mag:.2f}",
        "ESC/q=quit   s=skip",
    ]
    for i, text in enumerate(hud):
        cv2.putText(vis, text, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis

def process_video(video_path, activity="", distortion="", show_preview=True, use_gt=False):
    """ Runs full vision pipeline on one video file.

        Returns (tracks, mean_flow_mag, mean_flow_ang).
    """

    video_path = Path(video_path)

    try:
        cap = open_video(video_path)
    except IOError:
        return [], 0.0, 0.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gt_bboxes = load_gt_bboxes(video_path.name, orig_w, orig_h) if use_gt else []
    using_gt  = bool(gt_bboxes)
    if use_gt and not using_gt:
        print(f"  [warn] No GT file for {video_path.name} — falling back to MOG2.")

    detector = ForegroundDetector()
    tracker = CentroidTracker()

    prev_gray = None
    flow_magnitudes = []
    flow_angles = []
    frame_no = 0
    skip = False

    while True:

        ret, raw = cap.read()
        
        if not ret:
            break

        bgr, gray  = preprocess_frame(raw)
        mask, dets = detector.apply(gray)

        # Choose what drives the tracker
        gt_frame_bbox = gt_bboxes[frame_no] if using_gt and frame_no < len(gt_bboxes) else None
        tracking_input = [gt_frame_bbox] if gt_frame_bbox is not None else dets

        active = tracker.update(tracking_input)

        flow_mag, flow_ang, flow_vecs, prev_corners = 0.0, 0.0, None, None
        if prev_gray is not None:
            flow_mag, flow_ang, flow_vecs, prev_corners = compute_optical_flow(
                prev_gray, gray)
            flow_magnitudes.append(flow_mag)
            flow_angles.append(flow_ang)
        prev_gray = gray

        if show_preview and frame_no % cfg.DISPLAY_EVERY_N_FRAMES == 0:
            vis = draw_frame(bgr, dets, gt_frame_bbox, active,
                             frame_no, activity, distortion,
                             flow_mag, flow_vecs, prev_corners, using_gt)
            cv2.imshow(f"[{activity}/{distortion}] {video_path.name}", vis)
            if not using_gt:
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
    mean_ang = float(np.mean(flow_angles))     if flow_angles     else 0.0

    return tracker.get_active_tracks(), mean_mag, mean_ang