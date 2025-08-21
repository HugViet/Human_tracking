import numpy as np
from collections import deque

def cosine_distance(a, b, data_is_normalized=False):
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]))
    if not data_is_normalized:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    return 1.0 - np.dot(a, b.T)

class NearestNeighborDistanceMetric:
    def __init__(self, matching_threshold=0.2, budget=100):
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}  # track_id -> deque of features

    def partial_fit(self, features, targets):
        for f, t in zip(features, targets):
            self.samples.setdefault(t, deque(maxlen=self.budget)).append(np.asarray(f, dtype=np.float32))

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(features), len(targets)), dtype=np.float32)
        for i, f in enumerate(features):
            if not targets:
                break
            for j, t in enumerate(targets):
                feats = self.samples.get(t, None)
                if feats is None or len(feats) == 0:
                    cost_matrix[i, j] = 1.0  # no appearance info yet
                else:
                    feats_mat = np.vstack(feats)
                    d = cosine_distance(f[None, :], feats_mat, data_is_normalized=True).min()
                    cost_matrix[i, j] = d
        return cost_matrix
