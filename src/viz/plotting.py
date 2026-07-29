from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from ..processing.preprocessing import *
from ..processing.detectors import *
from ..processing.trackers import *
from ..processing.optical_flow import *
from ..processing.features import *
from ..processing.pipeline import *
from ..models.classic import *
from ..models.neural import *
from ..processing.smoothing import *

class PlotManager:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check(self) -> bool:
        if not MATPLOTLIB_AVAILABLE:
            print("  [plot] Matplotlib not available — skipping plot")
            return False
        return True

    def confusion_matrix_plot(self, cm: np.ndarray, class_names: List[str], title: str, filename: str):
        if not self._check():
            return
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues', ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def combined_confusion_matrices(self, cm_activity: np.ndarray, classes_activity: List[str],
                                    cm_distortion: np.ndarray, classes_distortion: List[str],
                                    title: str, filename: str):
        if not self._check():
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(cm_activity, annot=True, fmt='d', xticklabels=classes_activity, yticklabels=classes_activity,
                    cmap='Blues', ax=axes[0], cbar=True)
        axes[0].set_title("Activity Confusion Matrix")
        axes[0].set_ylabel("True")
        axes[0].set_xlabel("Predicted")
        sns.heatmap(cm_distortion, annot=True, fmt='d', xticklabels=classes_distortion, yticklabels=classes_distortion,
                    cmap='Oranges', ax=axes[1], cbar=True)
        axes[1].set_title("Distortion Confusion Matrix")
        axes[1].set_ylabel("True")
        axes[1].set_xlabel("Predicted")
        plt.suptitle(title)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def feature_importance_plot(self, feature_names: List[str], importances: np.ndarray, title: str, filename: str, top_n: int = 20):
        if not self._check() or len(importances) != len(feature_names):
            return
        order = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in order]
        top_values = importances[order]
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        ax.barh(range(len(top_names)), top_values, color="steelblue")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def model_comparison_plot(self, results: Dict[str, Dict], metric: str, target: str, filename: str):
        if not self._check():
            return
        model_names = []
        metric_vals = []
        for mname, res in results.items():
            if metric in res and isinstance(res[metric], (int, float)):
                model_names.append(mname)
                metric_vals.append(res[metric])
        if not model_names:
            return
        fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.2), 5))
        bars = ax.bar(range(len(model_names)), metric_vals, color="steelblue")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric} by model — target: {target}")
        for bar, val in zip(bars, metric_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def detector_stats_plot(self, stats_by_distortion: Dict[str, Dict], filename: str):
        if not self._check() or not stats_by_distortion:
            return
        dist_names = list(stats_by_distortion.keys())
        rates = [stats_by_distortion[d].get("success_rate", 0.0) for d in dist_names]
        fig, ax = plt.subplots(figsize=(max(5, len(dist_names) * 1.5), 5))
        bars = ax.bar(dist_names, rates, color=plt.cm.tab10(np.linspace(0, 1, len(dist_names))))
        ax.set_ylabel("Detection success rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Detection success rate by distortion type")
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{rate:.2f}", ha="center", va="bottom")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")

    def track_length_histogram(self, all_track_lengths: List[int], title: str, filename: str):
        if not self._check() or not all_track_lengths:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(all_track_lengths, bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel("Track length (frames)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved: {path}")
