import argparse
import os
import cv2
import glob
import re
from deepsort.tracker import Tracker
from deepsort.detection import Detection
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.utils.box_ops import xyxy_to_tlwh
from embedder.reid_model import FeatureExtractor
from detectors.yolo import YOLOv8Detector

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def process_sequence(img_dir, detector, extractor, tracker, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    f = open(out_path, 'w')
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')), key=natural_key)
    for i, img_path in enumerate(img_files, start=1):
        frame = cv2.imread(img_path)
        det_xyxy, det_conf, det_cls = detector(frame)
        features = extractor(frame, det_xyxy)

        detections = []
        for j in range(len(det_xyxy)):
            tlwh = xyxy_to_tlwh(det_xyxy[j])
            detections.append(Detection(tlwh, float(det_conf[j]), features[j]))

        tracker.predict()
        tracker.update(detections)

        for t in tracker.tracks:
            if t.is_confirmed() and t.time_since_update == 0:
                x, y, w, h = t.to_tlwh()
                f.write(f"{i},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
    f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True, help='Path to MOT17 split folder containing sequences (e.g., MOT17/train)')
    ap.add_argument('--split', default='train', help='train/test (only used for naming)')
    ap.add_argument('--yolo-weights', default='yolov8n.pt')
    ap.add_argument('--device', default=None)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--out-dir', default='out/mot17_results')
    ap.add_argument('--max-age', type=int, default=30)
    ap.add_argument('--n-init', type=int, default=3)
    ap.add_argument('--max-cosine-distance', type=float, default=0.2)
    ap.add_argument('--budget', type=int, default=100)
    args = ap.parse_args()

    detector = YOLOv8Detector(weights=args.yolo_weights, device=args.device, imgsz=args.imgsz, conf=args.conf)
    extractor = FeatureExtractor(device=('cuda:' + str(args.device) if str(args.device).isdigit() else args.device) if args.device is not None else 'cpu')
    metric = NearestNeighborDistanceMetric(matching_threshold=args.max_cosine_distance, budget=args.budget)
    tracker = Tracker(metric, max_age=args.max_age, n_init=args.n_init)

    seq_dirs = [d for d in sorted(os.listdir(args.data_dir)) if os.path.isdir(os.path.join(args.data_dir, d))]
    for seq in seq_dirs:
        img_dir = os.path.join(args.data_dir, seq, 'img1')
        if not os.path.isdir(img_dir):
            continue
        out_path = os.path.join(args.out_dir, f'{seq}.txt')
        print(f'Processing {seq} ...')
        process_sequence(img_dir, detector, extractor, tracker, out_path)

if __name__ == '__main__':
    main()
