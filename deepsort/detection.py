import numpy as np
from .utils.box_ops import tlwh_to_xyah

class Detection:
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_xyah(self):
        return tlwh_to_xyah(self.tlwh)
