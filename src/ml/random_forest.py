import numpy as np
from sklearn.base import BaseEstimator
from src.ml.decision_tree import DecisionTree

class RandomForest(BaseEstimator):
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2,
                 n_features=None, random_state=42):
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_features        = n_features
        self.random_state      = random_state

        self.trees              = []
        self.feature_importances_ = None
        self.classes_             = None

    def fit(self, X, y):
        rng        = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        n_feat     = self.n_features or int(np.sqrt(n_features))

        self.classes_ = np.unique(y)
        self.trees    = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                n_features        = n_feat,
                random_state      = int(rng.integers(0, 1_000_000)),
            )
            X_boot, y_boot = self.bootstrap(X, y, rng)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        raw   = np.sum([t.compute_importances(n_features) for t in self.trees],
                       axis=0)
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw

        return self

    def bootstrap(self, X, y, rng):
        idxs = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([t.predict(X) for t in self.trees])
        return np.apply_along_axis(
            lambda col: int(np.bincount(col).argmax()), axis=0, arr=tree_preds
        )