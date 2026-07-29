# Distortion-Aware Suspicious Activity Detection — Modular Layout

Refactored from a single ~6,200-line `run.py` into a package, now further
split into `processing/`, `models/`, and `viz/` sub-packages. **Behavior is
unchanged** — same CLI, same configs, same results — only the file layout
changed, plus the Group C (neural) GPU/throughput fixes (DataLoader workers,
pinned memory, non_blocking transfers, mixed precision) are included.

## Running it

Nothing about usage changes. From the project root:

```bash
python run.py --config group_a_classical --target activity
python run.py --config group_c_neural --target activity --sample
python run.py --all-groups --target activity
python run.py --list-configs
python run.py --list-components
```

`run.py` at the project root is a thin entry point — all logic lives in `src/`.

## Layout

```
project/
├── run.py                      thin entry point (`from src.cli import main`)
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── common.py                all third-party imports, optional-dependency
    │                            flags (TORCH_AVAILABLE, etc.), logging
    ├── paths.py                  PROJECT_ROOT, CONFIGS_DIR, etc. (mutable —
    │                            reassigned at runtime by cli.main())
    ├── interfaces.py             core interfaces / shared type aliases
    ├── registry.py                register/build/available component registry
    ├── config.py                  Config, load_config, DatasetInfoHandler,
    │                             base config dict, create_default_configs
    ├── experiment.py               experiment runner utilities + ExperimentRunner
    ├── cli.py                      main() / argparse CLI
    │
    ├── processing/                 detection, tracking, feature & pipeline code
    │   ├── preprocessing.py         preprocessors (standard / distortion-aware)
    │   ├── detectors.py             MOG2, adaptive MOG2, ground-truth,
    │   │                           optical-flow/frame-diff/cascade detectors
    │   ├── trackers.py              centroid, IoU+Hungarian, SORT, TFCR
    │   ├── optical_flow.py          dense optical flow helpers
    │   ├── features.py              feature extractors + aggregation
    │   ├── pipeline.py              pipeline builder/runner, multi-family
    │   │                           features, explainability visualizer
    │   └── smoothing.py             temporal smoother (post-processing)
    │
    ├── models/                     every trainable model
    │   ├── classic.py               from-scratch decision tree/RF + classical
    │   │                           wrappers (SVM, KNN, RF, LightGBM, XGBoost)
    │   └── neural.py                Group C: CNN / CNN+LSTM / 3D-CNN — includes
    │                               the GPU throughput fixes (DataLoader workers,
    │                               pinned memory, non_blocking, mixed precision)
    │
    └── viz/                         everything that produces a plot/visual
        ├── plotting.py               plot manager (matplotlib/seaborn)
        └── fiftyone_viz.py           FiftyOne visualisation
```

Each module imports everything it needs via `from <path> import *`, chained in
the same dependency order as the original section numbering, so nothing had
to be manually re-traced symbol-by-symbol. Cross-package imports use the
expected relative dots, e.g. `src/models/neural.py` reaches
`src/processing/pipeline.py` via `from ..processing.pipeline import *`.

### The one subtlety: mutable path globals

`cli.main()` resolves the real project root at runtime and overwrites
`PROJECT_ROOT`, `CONFIGS_DIR`, etc. (exactly like the original script did with
`global`). Because those are read from other modules too (`config.py`,
`processing/detectors.py`, `experiment.py`), those files import the `paths`
module itself (`from . import paths` / `from .. import paths`) and reference
`paths.CONFIGS_DIR` etc. instead of a plain name, so they always see the live
value after `main()` updates it.

## Group C / GPU notes

`src/models/neural.py` includes:
- `_make_loader()` — `DataLoader` with `num_workers`, `pin_memory=True`,
  `persistent_workers=True`, so CPU-side video decoding overlaps with GPU
  compute instead of blocking it.
- `_to_device()` — `.to(device, non_blocking=True)`.
- `_AmpHelper` — mixed precision (`autocast` + `GradScaler`), which helps
  most on lower-VRAM GPUs by cutting memory pressure.

These only affect speed, not results — same architectures, same optimizer
config, same train/val split logic as before.

## Installing dependencies

```bash
pip install -r requirements.txt
```

Everything below `torch`/`torchvision` in `requirements.txt` is optional —
the script detects missing packages and simply disables the corresponding
models/features (you'll see a `WARNING ... not installed` log line, not a
crash).
