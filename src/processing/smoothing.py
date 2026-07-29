from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *
from .preprocessing import *
from .detectors import *
from .trackers import *
from .optical_flow import *
from .features import *
from .pipeline import *
from ..models.classic import *
from ..models.neural import *

class TemporalSmoother:
    def __init__(self, method="majority_vote", ema_alpha=0.7):
        self.method = method
        self.ema_alpha = ema_alpha

    def smooth(self, predictions_by_model: Dict[str, np.ndarray], accuracy_by_model: Dict[str, float] = None) -> np.ndarray:
        model_names = list(predictions_by_model.keys())
        if not model_names:
            return np.array([])
        if len(model_names) == 1:
            return predictions_by_model[model_names[0]]
        if self.method == "majority_vote":
            return self._majority_vote(predictions_by_model, model_names)
        elif self.method == "ema":
            return self._ema_vote(predictions_by_model, model_names, accuracy_by_model)
        return predictions_by_model[model_names[0]]

    def _majority_vote(self, predictions_by_model, model_names):
        n_samples = len(predictions_by_model[model_names[0]])
        result = []
        for i in range(n_samples):
            votes = [predictions_by_model[m][i] for m in model_names]
            labels, counts = np.unique(votes, return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.array(result)

    def _ema_vote(self, predictions_by_model, model_names, accuracy_by_model):
        if not accuracy_by_model:
            return self._majority_vote(predictions_by_model, model_names)
        sorted_names = sorted(model_names, key=lambda m: accuracy_by_model.get(m, 0.0))
        weight = 1.0
        weights = {}
        for name in sorted_names:
            weights[name] = weight
            weight *= self.ema_alpha
        n_samples = len(predictions_by_model[sorted_names[0]])
        result = []
        for i in range(n_samples):
            vote_count = {}
            for name in sorted_names:
                label = predictions_by_model[name][i]
                vote_count[label] = vote_count.get(label, 0) + weights[name]
            result.append(max(vote_count, key=vote_count.get))
        return np.array(result)
