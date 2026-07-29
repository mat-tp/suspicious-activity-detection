#!/usr/bin/env python3
"""
Distortion-Aware Suspicious Activity Detection in Authentically Distorted
Surveillance Videos: Complete Production Framework (Enhanced)

Thin entry point. All logic lives in the src/ package (further split into
src/processing, src/models, src/viz — see README.md for the full map).
Run exactly as before, e.g.:

    python run.py --config group_a_classical --target activity
    python run.py --all-groups --target activity
    python run.py --list-configs
"""
import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
