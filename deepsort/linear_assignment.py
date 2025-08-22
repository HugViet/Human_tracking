import numpy as np
from scipy.optimize import linear_sum_assignment

INFTY_COST = 1e5

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], list(track_indices), list(detection_indices)

    features = np.array([detections[i].feature for i in detection_indices], dtype=np.float32)
    targets = [tracks[i].track_id for i in track_indices]
    cost_matrix = distance_metric(features, targets)

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_dets = [], [], []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_distance:
            unmatched_tracks.append(track_indices[r])
            unmatched_dets.append(detection_indices[c])
        else:
            matches.append((track_indices[r], detection_indices[c]))

    unmatched_tracks += [k for k in track_indices if k not in [m[0] for m in matches] and k not in unmatched_tracks]
    unmatched_dets += [k for k in detection_indices if k not in [m[1] for m in matches] and k not in unmatched_dets]

    return matches, unmatched_tracks, unmatched_dets

def gate_cost_matrix(kf, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False, mahalanobis_threshold=9.4877):
    # chi2inv(0.95) for 4 dof ≈ 9.4877; for 2 dof ≈ 5.9915
    if only_position:
        mahalanobis_threshold = 5.9915

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        mean, cov = track.mean, track.covariance
        measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
        mdist = kf.gating_distance(mean, cov, measurements, only_position=only_position)
        for col, d in enumerate(detection_indices):
            if mdist[col] > mahalanobis_threshold:
                gated_cost[row, col] = INFTY_COST
    return gated_cost

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)
    for i, t_idx in enumerate(track_indices):
        trk = tracks[t_idx]
        t_tlwh = trk.to_tlwh()
        cands = np.array([detections[j].tlwh for j in detection_indices], dtype=np.float32)
        cost_matrix[i, :] = 1. - iou_single_vs_many(t_tlwh, cands)
    return cost_matrix

def iou_single_vs_many(bbox, candidates):
    tl = np.maximum(bbox[:2], candidates[:, :2])
    br = np.minimum(bbox[:2] + bbox[2:4], candidates[:, :2] + candidates[:, 2:4])
    wh = np.maximum(0., br - tl)
    inter = wh[:, 0] * wh[:, 1]
    union = bbox[2] * bbox[3] + candidates[:, 2] * candidates[:, 3] - inter
    return inter / (union + 1e-6)
