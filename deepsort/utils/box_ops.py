import numpy as np

def tlwh_to_xyah(tlwh):
    x, y, w, h = tlwh
    cx = x + w / 2.0
    cy = y + h / 2.0
    a = w / float(h + 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)

def xyxy_to_tlwh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return np.array([x1, y1, w, h], dtype=np.float32)

def tlwh_to_xyxy(tlwh):
    x, y, w, h = tlwh
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def clip_box(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou(bbox, candidates):
    # bbox: [x, y, w, h]  candidates: Nx4
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:4]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:4]

    tl = np.maximum(bbox_tl, candidates_tl)
    br = np.minimum(bbox_br, candidates_br)
    wh = np.maximum(0., br - tl)
    inter_area = wh[:, 0] * wh[:, 1]
    bbox_area = bbox[2] * bbox[3]
    candidates_area = candidates[:, 2] * candidates[:, 3]
    union = bbox_area + candidates_area - inter_area
    return inter_area / (union + 1e-6)
