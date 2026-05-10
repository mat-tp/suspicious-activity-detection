"""
    Trains and evaluates classifiers on video level feature dicts.
    Pipelines: StandardScaler + estimator.
    Split: stratified 70% train / 15% val / 15% test.
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib # added
matplotlib.use('Agg') # added
import matplotlib.pyplot as plt

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, classification_report,
                                     ConfusionMatrixDisplay, confusion_matrix)

import config as cfg
from ml.features     import to_dataframe, get_XY
from ml.random_forest import RandomForest
from ml.features import FEATURE_NAMES

def build_sklearn_pipelines():
    """Standard sklearn baselines."""
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                          class_weight="balanced",
                          random_state=cfg.RANDOM_SEED)),
        ]),
        # k=5 is a common small‑k baseline; odd to avoid ties
        "KNN (k=10)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=None,
                                              class_weight="balanced",
                                              random_state=cfg.RANDOM_SEED,
                                              n_jobs=-1)),
        ]),
    }


def build_custom_pipelines():
    """Custom implementations (from scratch)."""
    return {
        "Custom Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForest(n_trees=200, max_depth=10,
                                   min_samples_split=2,
                                   random_state=cfg.RANDOM_SEED)),
        ]),
    }


def build_all_pipelines():
    return {**build_sklearn_pipelines(), **build_custom_pipelines()}


def safe_cv_folds(y_train, requested):
    """Clamp CV folds so every class has at least k training samples."""
    _, counts = np.unique(y_train, return_counts=True)
    return max(2, min(requested, int(counts.min())))


def print_results(name, split_name, y_true, y_pred, label_names):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  {name}  [{split_name}]")
    print(f"{'-'*60}")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=label_names,
        labels=range(len(label_names)),
        zero_division=0,
    ))

    return acc



def plot_confusion(name, split_name, y_true, y_pred, label_names, save_dir=None):

    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))

    fig, ax = plt.subplots(figsize=(max(8, len(label_names)*0.7), 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")

    ax.set_title(f"{name} [{split_name}] — Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        fname = f"cm_{split_name.lower()}_{safe_name}.png"
        plt.savefig(Path(save_dir) / fname, dpi=200, bbox_inches='tight')
        print(f"  Saved: {fname}")

    plt.close(fig)   # Close only after saving

    return fig, conf_matrix


def plot_accuracy_comparison(test_results, show_plot=True, save_dir=None, target_col="activity"):
    """Bar chart comparing test accuracy of all classifiers."""

    names = list(test_results.keys())
    accs = [test_results[n] for n in names]

    colors = ["#2196F3" if "Custom" not in n else "#FF9800" for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Classifier Comparison — {target_col.capitalize()}")
    plt.xticks(rotation=30, ha="right")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{acc:.3f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = f"accuracy_comparison_{target_col}.png"
        plt.savefig(Path(save_dir) / fname, dpi=200, bbox_inches='tight')
        print(f"  Saved: {fname}")

    plt.close(fig)

    return fig

def plot_feature_importance(rf_pipeline, feature_names, show_plot=True, save_dir=None):
    """Top‑15 feature importances from a Random‑Forest pipeline."""
    
    importances = rf_pipeline.named_steps["clf"].feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(range(len(indices)), importances[indices])
    ax.set_title("Random Forest — Top 15 Feature Importances")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=60, ha="right")
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = "feature_importance.png"
        plt.savefig(Path(save_dir) / fname, dpi=200, bbox_inches='tight')
        print(f"  Saved: {fname}")

    plt.close(fig)
    return fig


class ActivityClassifier:
    """Orchestrates data preparation, splitting, training, and evaluation."""

    def __init__(self):
        self.encoder = LabelEncoder()
        self.pipelines = {}
        self.label_names = []
        self.feature_names = []
        self.test_results = {}   # {name: accuracy}
        self.target_col = ""
        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None

    def prepare_data(self, video_feature_dicts, target_col):
        """Encode labels, drop singleton classes, build X and y."""
        df = to_dataframe(video_feature_dicts)

        counts = df[target_col].value_counts()
        singletons = counts[counts < 2].index.tolist()
        if singletons:
            print(f"  [info] Dropping {len(singletons)} singleton class(es): {singletons}")
            df = df[~df[target_col].isin(singletons)]

        if len(df) < 6:
            raise ValueError(
                f"Only {len(df)} usable samples for '{target_col}'. "
                "Run with more videos (increase --n)."
            )

        X, y_raw, feature_names = get_XY(df, target_col=target_col)

        y = self.encoder.fit_transform(y_raw)
        label_names = list(self.encoder.classes_)

        return X, y, label_names, feature_names

    def split_data(self, X, y):
        """ Splitting data into 70/15/15 (Train/Val/Test). """
        n_classes = len(np.unique(y))
        
        # Separating the Test set (15%)
        n_test = int(len(y) * cfg.TEST_SIZE)
        can_stratify_test = n_test >= n_classes

        if not can_stratify_test:
            print(f"  [warn] Test set too small ({n_test} samples) for {n_classes} classes. Using random split.")

        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y,
            test_size    = cfg.TEST_SIZE,
            random_state = cfg.RANDOM_SEED,
            stratify     = y if can_stratify_test else None
        )

        # Separating Train and Val from the remainder by calculate val_frac relative to the remaining 85% of data
        val_frac = cfg.VAL_SIZE / (cfg.TRAIN_SIZE + cfg.VAL_SIZE)
        
        # Calculate expected number of validation samples to check stratification
        n_val = int(len(y_tv) * val_frac)
        n_tv_classes = len(np.unique(y_tv))
        can_stratify_val = n_val >= n_tv_classes

        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size    = val_frac,
            random_state = cfg.RANDOM_SEED,
            stratify     = y_tv if can_stratify_val else None
        )

        # Store for internal use
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test

        print(f"  Split complete >  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_one(self, name, pipe, X_train, y_train):
        pipe.fit(X_train, y_train)
        self.pipelines[name] = pipe

    def evaluate_one(self, name, pipe, X, y_true, split_name, show_plot=True, save_dir=None):
        """Predict, print metrics, plot confusion matrix. Returns (y_pred, accuracy, fig)."""
        y_pred = pipe.predict(X)
        acc = print_results(name, split_name, y_true, y_pred, self.label_names)
        plot_confusion(name, split_name, y_true, y_pred, self.label_names, save_dir= save_dir)

        return y_pred, acc, None

    def cross_validate_one(self, name, pipe, X_train, y_train):
        """Stratified K fold CV on train split only."""

        k = safe_cv_folds(y_train, cfg.CV_FOLDS)
        skf = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=cfg.RANDOM_SEED)
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"  {k}-fold Stratified CV (train)  "
              f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"per-fold={np.round(scores, 3).tolist()}")

    def fit_and_evaluate(self, video_feature_dicts, target_col="activity", run_cv=True, pipelines=None, show_plot=True, save_dir=None):
        """prepares -> split -> train ->  val -> CV -> test -> plots."""

        if not video_feature_dicts:
            print("No video feature dicts — nothing to classify.")
            return

        self.target_col = target_col

        try:
            X, y, label_names, feature_names = self.prepare_data(
                video_feature_dicts, target_col
            )
        except ValueError as e:
            print(f"  [skip] {e}")
            return

        self.label_names   = label_names
        self.feature_names = feature_names

        print(f"\n{'='*60}")
        print(f"  CLASSIFICATION — target: '{target_col}'")
        print(f"  Labels ({len(label_names)}): {label_names}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        active = pipelines if pipelines is not None else build_all_pipelines()
        self.test_results = {}

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for name, pipe in active.items():
            print(f"\n{'-'*60}")
            print(f"  {name}")

            self.train_one(name, pipe, X_train, y_train)
            self.evaluate_one(name, pipe, X_val, y_val, "VAL", save_dir=save_dir)

            if run_cv:
                self.cross_validate_one(name, pipe, X_train, y_train)

            _, test_acc, _ = self.evaluate_one(name, pipe, X_test, y_test, "TEST", save_dir=save_dir)
            self.test_results[name] = test_acc

        plot_accuracy_comparison(self.test_results, show_plot=show_plot, save_dir=save_dir, target_col=target_col)

        for rf_name in ("Random Forest", "Custom Random Forest"):
            if rf_name in self.pipelines:
                plot_feature_importance(self.pipelines[rf_name],
                                        feature_names,
                                        show_plot=show_plot, save_dir=save_dir)
                break

    def predict(self, video_feature_dict, model_name="Random Forest"):
        """Predict the label for a single video‑level feature dict."""
        if model_name not in self.pipelines:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        X     = np.array([video_feature_dict[f] for f in FEATURE_NAMES]).reshape(1, -1)
        y_enc = self.pipelines[model_name].predict(X)
        return self.encoder.inverse_transform(y_enc)[0]

    def save(self, path, target_col=None):
        """Pickle the classifier and associated metadata."""
        from ml.features import FEATURE_NAMES
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classifier":    self,
            "target_col":    target_col or self.target_col,
            "test_results":  self.test_results,
            "label_names":   self.label_names,
            "feature_names": FEATURE_NAMES,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [saved] {path}  ({path.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load(cls, path):
        """Load a saved classifier payload."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "classifier" not in payload:
            raise ValueError(f"{path} is not a valid saved ActivityClassifier.")
        return payload