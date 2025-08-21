import numpy as np
from .kalman_filter import KalmanFilter
from .track import Track, TrackState
from .linear_assignment import min_cost_matching, gate_cost_matrix, iou_cost, INFTY_COST

class Tracker:
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for t in self.tracks:
            t.predict(self.kf)

    def update(self, detections):
        # Step 1: match confirmed tracks using appearance + motion gating
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_dets = self._match(confirmed, detections)

        # Step 2: iou match for remaining confirmed + unconfirmed
        iou_track_candidates = unconfirmed + unmatched_tracks_a
        matches_b, unmatched_tracks_b, unmatched_dets = self._match_iou(iou_track_candidates, detections, unmatched_dets)

        matches = matches_a + matches_b
        unmatched_tracks = [t for t in unmatched_tracks_b if self.tracks[t].is_confirmed()] +                            [t for t in unmatched_tracks_b if not self.tracks[t].is_confirmed()]

        # Update matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(self.kf, detections[d_idx])

        # Mark unmatched as missed
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # Initiate new tracks for unmatched detections
        for d_idx in unmatched_dets:
            self._initiate_track(detections[d_idx])

        # Remove deleted
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update appearance model
        active_targets, features = [], []
        for t in self.tracks:
            if t.is_confirmed() and t.time_since_update == 0 and len(t.features) > 0:
                active_targets.append(t.track_id)
                features.append(t.features[-1])
        if len(features) > 0:
            self.metric.partial_fit(features, active_targets)

    def _match(self, track_indices, detections):
        if len(track_indices) == 0 or len(detections) == 0:
            return [], list(track_indices), list(range(len(detections)))

        # Build cost from appearance distances (cosine), then gate with Mahalanobis distance
        features = [detections[i].feature for i in range(len(detections))]
        targets = [self.tracks[i].track_id for i in track_indices]
        cost_matrix = self.metric.distance(features, targets).T  # shape: tracks x detections

        # Gate by motion (Mahalanobis)
        cost_matrix = gate_cost_matrix(self.kf, self.tracks, detections, track_indices, list(range(len(detections))), cost_matrix)

        matches, unmatched_tracks, unmatched_dets = min_cost_matching(
            lambda f, t: cost_matrix, self.metric.matching_threshold, self.tracks, detections,
            track_indices=track_indices, detection_indices=list(range(len(detections)))
        )
        return matches, unmatched_tracks, unmatched_dets

    def _match_iou(self, track_indices, detections, detection_indices):
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], list(track_indices), list(detection_indices)

        cost_matrix = iou_cost(self.tracks, detections, track_indices, detection_indices)
        matches, unmatched_tracks, unmatched_dets = min_cost_matching(
            lambda f, t: cost_matrix, 1. - self.max_iou_distance, self.tracks, detections,
            track_indices=track_indices, detection_indices=detection_indices
        )
        return matches, unmatched_tracks, unmatched_dets

    def _initiate_track(self, detection):
        mean, cov = self.kf.initiate(detection.to_xyah())
        t = Track(mean, cov, self._next_id, self.n_init, self.max_age, feature=detection.feature)
        self.tracks.append(t)
        self._next_id += 1
