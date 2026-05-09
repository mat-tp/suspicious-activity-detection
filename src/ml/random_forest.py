"""
src/ml/random_forest.py
========================
Random Forest classifier implemented from scratch.

  - Bootstrap ensemble of CART decision trees (bagging)
  - Each tree sees a random subset of features per split (feature randomness)
  - Splits maximise entropy-based information gain
  - Prediction by majority vote across all trees
  - feature_importances_ matches sklearn's interface so the existing
    plot_feature_importance() in classifier.py works identically

Inherits BaseEstimator for sklearn Pipeline / cross_val_score compatibility.
"""

import numpy as np
from sklearn.base import BaseEstimator
from ml.decision_tree import DecisionTree


class RandomForest(BaseEstimator):
    """
    Random Forest — from scratch.

    Parameters
    ----------
    n_trees           : number of trees in the ensemble
    max_depth         : maximum depth per tree
    min_samples_split : minimum samples needed to attempt a split
    n_features        : features considered per split (None → sqrt heuristic)
    random_state      : seed for reproducibility

    Attributes (set after fit)
    --------------------------
    feature_importances_ : (n_features,) normalised split-count array.
    classes_             : sorted unique integer class labels.
    """

    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2,
                 n_features=None, random_state=42):
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_features        = n_features
        self.random_state      = random_state

        self.trees                = []
        self.feature_importances_ = None
        self.classes_             = None

    def fit(self, X, y):
        rng    = np.random.default_rng(self.random_state)
        n_feat = self.n_features or int(np.sqrt(X.shape[1]))

        self.classes_ = np.unique(y)
        self.trees    = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                n_features        = n_feat,
                random_state      = int(rng.integers(0, 1_000_000)),
            )
            X_b, y_b = self.bootstrap(X, y, rng)
            tree.fit(X_b, y_b)
            self.trees.append(tree)

        # Aggregate and normalise feature importances across all trees
        raw   = np.sum([t.compute_importances(X.shape[1]) for t in self.trees],
                       axis=0)
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw

        return self

    def bootstrap(self, X, y, rng):
        idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """Majority vote across all trees."""
        preds = np.array([t.predict(X) for t in self.trees])
        return np.apply_along_axis(
            lambda col: int(np.bincount(col).argmax()), axis=0, arr=preds
        )