import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, classification_report,
                                     ConfusionMatrixDisplay, confusion_matrix)

import settings as cfg
from ml.features     import to_dataframe, get_XY
from ml.random_forest import RandomForest

def build_sklearn_pipelines():
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                          class_weight="balanced",
                          random_state=cfg.RANDOM_SEED)),
        ]),
        "KNN (k=5)": Pipeline([
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
    return {
        "Custom Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForest(n_trees=100, max_depth=10,
                                   min_samples_split=2,
                                   random_state=cfg.RANDOM_SEED)),
        ]),
    }

def build_all_pipelines():
    return {**build_sklearn_pipelines(), **build_custom_pipelines()}

def safe_cv_folds(y_train, requested):
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

def plot_confusion(name, split_name, y_true, y_pred, label_names, plot_show=True):
    cm   = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(max(7, len(label_names)), 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{name} [{split_name}] - Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if plot_show:
        plt.show()
    return fig, cm

def plot_accuracy_comparison(results, plot_show=True):
    names  = list(results.keys())
    accs   = [results[n] for n in names]
    colors = ["#2196F3" if "Custom" not in n else "#FF9800" for n in names]

    fig = plt.figure(figsize=(12, 5))
    bars = plt.bar(names, accs, color=colors)
    plt.ylim(0, 1.05)
    plt.ylabel("Test Accuracy")
    plt.title("Classifier Comparison - Test Accuracy\n(blue = sklearn, orange = custom)")
    plt.xticks(rotation=25, ha="right")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if plot_show:
        plt.show()
    return fig

def plot_feature_importance(rf_pipeline, feature_names, plot_show=True):
    importances = rf_pipeline.named_steps["clf"].feature_importances_
    if importances is None:
        return None
    indices = np.argsort(importances)[::-1][:15]
    fig = plt.figure(figsize=(12, 6))
    plt.title("Random Forest - Top 15 Feature Importances")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices], rotation=60, ha="right")
    plt.tight_layout()
    if plot_show:
        plt.show()
    return fig

class ActivityClassifier:
    def __init__(self):
        self.encoder      = LabelEncoder()
        self.pipelines    = {}
        self.label_names  = []
        self.feature_names = []
        self.test_results = {}
        self.target_col   = ""

        self.X_train = self.y_train = None
        self.X_val   = self.y_val   = None
        self.X_test  = self.y_test  = None

    def prepare_data(self, video_feature_dicts, target_col):
        df = to_dataframe(video_feature_dicts)

        counts     = df[target_col].value_counts()
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
        y           = self.encoder.fit_transform(y_raw)
        label_names = list(self.encoder.classes_)
        return X, y, label_names, feature_names

    def split_data(self, X, y):
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size    = cfg.TEST_SIZE,
            random_state = cfg.RANDOM_SEED,
            stratify     = y,
        )
        val_fraction = cfg.VAL_SIZE / (cfg.TRAIN_SIZE + cfg.VAL_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size    = val_fraction,
            random_state = cfg.RANDOM_SEED,
            stratify     = y_trainval,
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test
        print(f"  Split -> train: {len(y_train)}  val: {len(y_val)}  test: {len(y_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_one(self, name, pipe, X_train, y_train):
        pipe.fit(X_train, y_train)
        self.pipelines[name] = pipe

    def evaluate_one(self, name, pipe, X, y_true, split_name, plot_show=True):
        y_pred     = pipe.predict(X)
        acc        = print_results(name, split_name, y_true, y_pred, self.label_names)
        fig, _     = plot_confusion(name, split_name, y_true, y_pred,
                                    self.label_names, plot_show=plot_show)
        return y_pred, acc, fig

    def cross_validate_one(self, name, pipe, X_train, y_train):
        k = safe_cv_folds(y_train, cfg.CV_FOLDS)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.RANDOM_SEED)
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"  {k}-fold Stratified CV (train)  "
              f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"per-fold={np.round(scores, 3).tolist()}")

    def fit_and_evaluate(self, video_feature_dicts, target_col="activity",
                         run_cv=True, pipelines=None, plot_show=True):
        if not video_feature_dicts:
            print("No video feature dicts - nothing to classify.")
            return

        self.target_col = target_col

        try:
            X, y, label_names, feature_names = self.prepare_data(
                video_feature_dicts, target_col)
        except ValueError as e:
            print(f"  [skip] {e}")
            return

        self.label_names   = label_names
        self.feature_names = feature_names

        print(f"\n{'='*60}")
        print(f"  CLASSIFICATION - target: '{target_col}'")
        print(f"  Labels ({len(label_names)}): {label_names}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        active = pipelines if pipelines is not None else build_all_pipelines()
        self.test_results = {}

        for name, pipe in active.items():
            print(f"\n{'-'*60}")
            print(f"  {name}")

            self.train_one(name, pipe, X_train, y_train)
            self.evaluate_one(name, pipe, X_val, y_val, "VAL", plot_show=plot_show)

            if run_cv:
                self.cross_validate_one(name, pipe, X_train, y_train)

            _, test_acc, _ = self.evaluate_one(
                name, pipe, X_test, y_test, "TEST", plot_show=plot_show)
            self.test_results[name] = test_acc

        plot_accuracy_comparison(self.test_results, plot_show=plot_show)

        for rf_name in ("Random Forest", "Custom Random Forest"):
            if rf_name in self.pipelines:
                plot_feature_importance(
                    self.pipelines[rf_name], feature_names, plot_show=plot_show)
                break

    def predict(self, video_feature_dict, model_name="Random Forest"):
        if model_name not in self.pipelines:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        from ml.features import FEATURE_NAMES
        X     = np.array([video_feature_dict[f] for f in FEATURE_NAMES]).reshape(1, -1)
        y_enc = self.pipelines[model_name].predict(X)
        return self.encoder.inverse_transform(y_enc)[0]

    def save(self, path, target_col=None):
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
        print(f"  [saved] {path}  "
              f"({path.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "classifier" not in payload:
            raise ValueError(
                f"{path} is not a valid saved ActivityClassifier. "
                "Expected a dict with a 'classifier' key."
            )
        return payload