# Detection of Suspicious Activities in Public Spaces

### Tshephang P-A-N Matlala

#### University of Johannesburg — ACSSE

---

# Introduction

Public surveillance systems are increasingly used to improve safety and monitor suspicious human activities in crowded or sensitive environments. Most modern approaches rely heavily on deep learning models, which often require large datasets, powerful hardware, and significant computational resources.

This project investigates whether classical computer vision techniques can still perform effectively for suspicious activity recognition under difficult real-world conditions such as focus blur and exposure distortions.

The system evaluates suspicious activity recognition using:

- Background subtraction
- Object tracking
- Optical flow analysis
- Trajectory-based motion analysis
- Classical machine learning classifiers

The project uses the **AD-SVD (Activity Detection under Surveillance Video Distortions)** dataset, which contains surveillance videos under four distortion conditions:

- Pristine
- Exposure distortion
- Focus blur
- Combined exposure and focus distortion

The research focuses on understanding how visual distortions affect classification performance while maintaining a lightweight and interpretable computer vision pipeline.

---

# Project Overview

This project implements a classical computer vision pipeline for detecting and classifying suspicious human activities in surveillance videos.

### Core techniques used

1. **MOG2 Background Subtraction**  
   Separates moving foreground objects from the background.

2. **Centroid Tracking**  
   Tracks detected moving objects across frames.

3. **Lucas–Kanade Optical Flow**  
   Estimates motion direction and movement dynamics.

4. **Trajectory Analysis**  
   Tracks object movement patterns over time.

5. **Feature Extraction**  
   Extracts handcrafted trajectory and motion features.

6. **Machine Learning Classification**  
   Uses:
   - SVM
   - K-NN
   - Random Forest

---

# Activity Labels

| Code | Activity                      |
| ---- | ----------------------------- |
| LPP  | Leaving a package unattended  |
| PO   | Passing out                   |
| PW   | Prowling                      |
| PPP  | Person pushing another person |
| RK   | Robbery with a knife          |
| FG   | Group fighting                |

---

# Distortion Types

| Code | Distortion                  |
| ---- | --------------------------- |
| Pri  | Pristine                    |
| Exp  | Exposure distortion         |
| Fo   | Focus blur                  |
| ExFo | Exposure + Focus distortion |

---

# Project Setup

```bash
pip install -r requirements.txt
```

# Usage

## Train and evaluate classifiers

```bash
python train_and_evaluate.py
```

## Feature Extraction

- Per-track features
- Trajectory length
- Total distance
- Net displacement
- Straightness
- Mean speed
- Maximum speed
- Speed variance
- Direction statistics
- Bounding-box area
- Optical flow magnitude
- Optical flow direction
- Video-level aggregation

### The extracted trajectory features are aggregated into video-level descriptors using:

- Mean
- Maximum
- Standard deviation

These features are then used for classification.

## Research Contribution

This work investigates:

- The effectiveness of classical computer vision methods for suspicious activity recognition.
- The impact of visual distortions on recognition performance.
- The practicality of lightweight surveillance systems in environments where deep learning may not be feasible.

Unlike many existing studies, this project specifically evaluates traditional approaches under degraded surveillance conditions using the AD-SVD dataset

## Acknowledgements

1. University of Johannesburg — Academy of Computer Science and Software Engineering (ACSSE)
2. Supervision and academic guidance provided by Dr. Moodley
3. Dataset used: AD-SVD (Activity Detection under Surveillance Video Distortions)
