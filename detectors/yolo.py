import numpy as np
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, weights='yolov8n.pt', device=None, imgsz=640, conf=0.3):
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def __call__(self, frame):
        # returns: (xyxy Nx4), (conf N), (cls N)
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        xyxy_list, conf_list, cls_list = [], [], []
        for r in res:
            if r.boxes is None: 
                continue
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy.numpy()
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf.numpy()
            cls  = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls.numpy()
            # filter person (class 0 in COCO)
            mask = (cls == 0)
            xyxy_list.append(xyxy[mask])
            conf_list.append(conf[mask])
            cls_list.append(cls[mask])
        if len(xyxy_list) == 0:
            return np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
        return np.concatenate(xyxy_list, axis=0).astype(np.float32),                np.concatenate(conf_list, axis=0).astype(np.float32),                np.concatenate(cls_list, axis=0).astype(np.int32)
