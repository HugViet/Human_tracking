import numpy as np
from enum import IntEnum
from .utils.box_ops import tlwh_to_xyah, tlwh_to_xyxy

class TrackState(IntEnum):
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age
        self._tlwh = None  # cache

    def to_tlwh(self):
        # Convert current mean to tlwh
        ret = self.mean.copy()
        cx, cy, a, h = ret[0], ret[1], ret[2], ret[3]
        w = a * h
        x = cx - w / 2.0
        y = cy - h / 2.0
        self._tlwh = np.array([x, y, w, h], dtype=np.float32)
        return self._tlwh

    def to_xyxy(self):
        tlwh = self.to_tlwh()
        x, y, w, h = tlwh
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.is_tentative() and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.is_tentative():
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
