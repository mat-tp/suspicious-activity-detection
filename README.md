# Detection of Suspicious Activities in Public Spaces

### Tshephang P-A-N Matlala

#### University of Johannesburg — ACSSE

---

Evaluating classical (non-deep-learning) computer vision for suspicious activity detection specifically under real-world visual distortions.
Breaking that down into its three parts:
Most existing work either uses deep learning for activity recognition, or evaluates classical methods only on clean/pristine video. Your contribution sits in the gap between those two. Specifically:

1. The "classical only" constraint is deliberate. You're not trying to compete with deep learning on accuracy — you're investigating how interpretable, computationally efficient, traditional CV methods (MOG2, optical flow, SVM/KNN/RF) hold up when video quality degrades. This is a legitimate research question because deep learning is not always available in resource-constrained deployments.
2. The distortion axis is your experimental variable. You're using the AD-SVD dataset which has four conditions — pristine (Pri), exposure distortion (Exp), focus blur (Fo), and combined (ExFo) — and you're measuring how much classification performance degrades across those conditions. Nobody has systematically done this comparison for classical methods on this dataset.
3. The activity classes are genuinely suspicious/dangerous. LPP, PO, PW, PPP, RK, FG — these are not trivial gesture recognition tasks. Detecting them reliably under distortion has real public safety relevance.

---

## Project overview

Classical computer vision pipeline for detecting and classifying suspicious
human activities in surveillance video under different visual distortions
(pristine, exposure, focus blur, combined).

**No deep learning is used.** The system relies entirely on:

- MOG2 background subtraction
- Morphological mask cleanup
- Centroid-based object tracking
- Lucas-Kanade sparse optical flow
- Hand-crafted trajectory and motion features
- SVM, K-NN, and Random Forest classifiers

---

## Directory structure

```
suspicious_activity_detection/
│
├── config/
│   └── settings.py          ← ALL tunable parameters live here
│
├── src/
│   ├── preprocessing/
│   │   └── frame_processor.py   ← resize, grayscale, blur
│   │
│   ├── detection/
│   │   └── background_subtractor.py  ← MOG2 + morphology + contours
│   │
│   ├── tracking/
│   │   ├── centroid_tracker.py  ← greedy nearest-neighbour tracker
│   │   └── optical_flow.py      ← Lucas-Kanade sparse flow
│   │
│   ├── features/
│   │   └── extractor.py         ← trajectory + flow feature vectors
│   │
│   ├── classification/
│   │   └── classifier.py        ← SVM / KNN / RF training + evaluation
│   │
│   ├── utils/
│   │   ├── logger.py            ← shared logger
│   │   ├── dataset.py           ← Excel loader + video sampler
│   │   └── display.py           ← all OpenCV drawing (no logic here)
│   │
│   └── pipeline.py              ← wires everything together per-video
│
├── explore_dataset.py       ← Entry point 1: stats + raw video preview
├── run_pipeline.py          ← Entry point 2: full pipeline with preview
├── train_and_evaluate.py    ← Entry point 3: headless batch + classifiers
├── debug_single_video.py    ← Dev tool: run pipeline on one video
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Make sure your dataset is laid out as:

```
suspicious_activity_detection/
├── datasetInfo.xlsx
└── surveillanceVideosDataset/
    └── surveillanceVideos/
        ├── 3982ExFo_IndPO_HQ_C3.mp4
        └── ...
```

Or update `VIDEO_DIR` and `DATASET_EXCEL` in `config/settings.py`.

---

## Usage

### 1. Explore the dataset (statistics + video preview)

```bash
python explore_dataset.py
```

Prints activity/distortion counts and plays sampled videos live.

### 2. Run the full pipeline with live preview

```bash
python run_pipeline.py
```

Shows annotated tracking + MOG2 mask windows. Press `f` to print
feature vectors for active tracks. Press `s` to skip a video.

### 3. Train and evaluate classifiers (headless, all videos)

```bash
python train_and_evaluate.py
```

Runs on ALL videos with no preview windows. Prints accuracy,
classification report, confusion matrix, and CV scores for SVM,
KNN, and Random Forest.

To run on a small sample first:

```bash
python train_and_evaluate.py --sample
```

### 4. Debug a single video

Edit `VIDEO_PATH`, `ACTIVITY`, and `DISTORTION` at the top of
`debug_single_video.py`, then:

```bash
python debug_single_video.py
```

---

## Controls (all preview scripts)

| Key          | Action                                       |
| ------------ | -------------------------------------------- |
| `ESC` or `q` | Quit the program                             |
| `s`          | Skip to the next video                       |
| `f`          | Print current track features to the terminal |

---

## Pipeline stages

```
Video frame
    │
    ▼
[1] Preprocessing          resize → grayscale → Gaussian blur
    │
    ▼
[2] Foreground Detection   MOG2 → CLOSE → OPEN → findContours
    │
    ▼
[3] Object Tracking        centroid matching → trajectory history
    │
    ▼
[3b] Optical Flow          Lucas-Kanade sparse flow (parallel)
    │
    ▼
[4] Feature Extraction     trajectory stats + flow magnitude/direction
    │
    ▼
[5] Classification         SVM / KNN / Random Forest
```

---

## Tuning

All parameters are in `config/settings.py`:

| Parameter                   | Default | Effect                                  |
| --------------------------- | ------- | --------------------------------------- |
| `MOG2_HISTORY`              | 20      | Shorter = faster adaptation to lighting |
| `MOG2_VAR_THRESHOLD`        | 25      | Lower = more sensitive detection        |
| `MIN_CONTOUR_AREA`          | 500     | Higher = ignore smaller blobs           |
| `MAX_DISAPPEARED`           | 30      | Higher = tracks survive longer gaps     |
| `MIN_TRAJECTORY_LEN`        | 10      | Higher = discard very short tracks      |
| `N_RANDOM_VIDEOS_PER_GROUP` | 2       | Videos sampled per group                |

---

## Activity labels

| Code | Activity                      |
| ---- | ----------------------------- |
| LPP  | Leaving a package unattended  |
| PO   | Passing out                   |
| PW   | Prowling                      |
| PPP  | Person pushing another person |
| RK   | Robbery with a knife          |
| FG   | Group fighting                |

## Distortion types

| Code | Type                      |
| ---- | ------------------------- |
| Pri  | Pristine (no distortion)  |
| Exp  | Exposure distortion       |
| Fo   | Focus blur                |
| ExFo | Combined exposure + focus |

### Training Phase

Step 1: Load dataset
datasetInfo.xlsx
Columns: Activity, Distortion, Name of Video Series
Distortion values: Pri, Exp, Fo, ExFo

Step 2: Process each video
For each video:
→ Background subtraction (MOG2)
→ Morphological cleanup
→ Contour extraction
→ Centroid tracking
→ Lucas-Kanade optical flow

Step 3: Extract per-track features (14 numeric features)
trajectory_length, total_distance, net_displacement,
straightness, mean_speed, max_speed, std_speed,
skew_speed, kurt_speed, mean_direction_deg,
direction_std_deg, bbox_area,
mean_flow_magnitude, mean_flow_angle_deg

Step 4: Aggregate to video-level features (43 features)
mean*{feature} × 14
max*{feature} × 14
std\_{feature} × 14
n_valid_tracks × 1

Step 5: Label encoding
y = ["Pri", "Exp", "Fo", "ExFo", ...]
encoder.fit(y) → [0, 1, 2, 3]

Step 6: Train classifiers
X = (n_videos × 43) feature matrix
y = encoded distortion labels

        For each model (SVM, KNN, Random Forest, Custom RF):
            fit(X_train, y_train)
            evaluate on validation set
            cross-validate
            evaluate on test set
