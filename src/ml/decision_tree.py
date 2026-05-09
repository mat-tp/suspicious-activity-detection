"""
src/ml/decision_tree.py
========================
CART decision tree implemented from scratch.
Used as the base learner inside RandomForest.

Splits nodes on information gain (entropy reduction).
Pruned by max_depth and min_samples_split.
Each split considers only a random subset of n_features features.

Inherits BaseEstimator so it is compatible with sklearn Pipeline
and cross_val_score without manual __sklearn_tags__ boilerplate.
"""

import numpy as np
from sklearn.base import BaseEstimator


class DecisionTree(BaseEstimator):

    def __init__(self, min_samples_split=2, max_depth=100,
                 n_features=None, random_state=42):
        self.min_samples_split = min_samples_split
        self.max_depth         = max_depth
        self.n_features        = n_features
        self.random_state      = random_state
        self.root              = None
        self.n_total_features  = None

    def fit(self, X, y):
        self.n_total_features = X.shape[1]
        self.n_split_features = (
            X.shape[1] if not self.n_features
            else min(X.shape[1], self.n_features)
        )
        rng = np.random.default_rng(self.random_state)
        self.root = self.grow(X, y, depth=0, rng=rng)
        return self

    def grow(self, X, y, depth, rng):
        # Guard: never recurse into an empty partition
        if X.shape[0] == 0:
            return {"leaf": True, "value": 0}
        if (depth >= self.max_depth or
                len(np.unique(y)) == 1 or
                X.shape[0] < self.min_samples_split):
            return {"leaf": True, "value": self.majority(y)}

        feat_idxs = rng.choice(self.n_total_features,
                               self.n_split_features, replace=False)
        best_feat, best_thr = self.best_split(X, y, feat_idxs)

        if best_feat is None:
            return {"leaf": True, "value": self.majority(y)}

        left_mask  = X[:, best_feat] <= best_thr
        right_mask = ~left_mask

        return {
            "leaf":      False,
            "feature":   best_feat,
            "threshold": best_thr,
            "left":  self.grow(X[left_mask],  y[left_mask],  depth + 1, rng),
            "right": self.grow(X[right_mask], y[right_mask], depth + 1, rng),
        }

    def best_split(self, X, y, feat_idxs):
        best_gain, best_feat, best_thr = -1.0, None, None
        for feat in feat_idxs:
            col = X[:, feat]
            for thr in np.unique(col):
                gain = self.info_gain(y, col, thr)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr

    def info_gain(self, y, col, thr):
        left_mask  = col <= thr
        right_mask = ~left_mask
        # Both sides must have at least one sample to be a valid split
        if left_mask.sum() < 1 or right_mask.sum() < 1:
            return 0.0
        n   = len(y)
        n_l = left_mask.sum()
        n_r = right_mask.sum()
        return (self.entropy(y) -
                (n_l / n) * self.entropy(y[left_mask]) -
                (n_r / n) * self.entropy(y[right_mask]))

    def entropy(self, y):
        counts = np.bincount(y)
        ps     = counts / len(y)
        return -np.sum(ps[ps > 0] * np.log(ps[ps > 0]))

    def majority(self, y):
        if len(y) == 0:
            return 0   # fallback — should not occur with a guarded grow()
        return int(np.bincount(y).argmax())

    def predict(self, X):
        return np.array([self.traverse(x, self.root) for x in X])

    def traverse(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self.traverse(x, node["left"])
        return self.traverse(x, node["right"])

    def compute_importances(self, n_features):
        """Count how many times each feature index is used as a split."""
        importances = np.zeros(n_features)
        self.accumulate(self.root, importances)
        return importances

    def accumulate(self, node, importances):
        if node["leaf"]:
            return
        importances[node["feature"]] += 1.0
        self.accumulate(node["left"],  importances)
        self.accumulate(node["right"], importances)