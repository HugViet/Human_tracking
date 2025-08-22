import argparse
import os
import cv2
from deepsort.tracker import Tracker
from deepsort.detection import Detection
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.utils.box_ops import xyxy_to_tlwh
from embedder.reid_model import FeatureExtractor
from detectors.yolo import YOLOv8Detector

def draw_tracks(frame, tracks, show_conf=False):
    for t in tracks:
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        x1, y1, x2, y2 = t.to_xyxy().astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {t.track_id}', (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def save_mot_line(f, frame_idx, track):
    x, y, w, h = track.to_tlwh()
    line = f"{frame_idx},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
    f.write(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='Path to video file or webcam index')
    ap.add_argument('--yolo-weights', default='yolov8n.pt')
    ap.add_argument('--device', default=None, help='e.g., 0 for cuda:0')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--save-vid', default=None, help='Output video path')
    ap.add_argument('--save-mot', default=None, help='Directory prefix to save MOT txt (without .txt)')
    ap.add_argument('--max-age', type=int, default=30)
    ap.add_argument('--n-init', type=int, default=3)
    ap.add_argument('--max-cosine-distance', type=float, default=0.2)
    ap.add_argument('--budget', type=int, default=100)
    args = ap.parse_args()

    # Detector, embedder, tracker
    detector = YOLOv8Detector(weights=args.yolo_weights, device=args.device, imgsz=args.imgsz, conf=args.conf)
    extractor = FeatureExtractor(device=('cuda:' + str(args.device) if str(args.device).isdigit() else args.device) if args.device is not None else 'cpu')
    metric = NearestNeighborDistanceMetric(matching_threshold=args.max_cosine_distance, budget=args.budget)
    tracker = Tracker(metric, max_age=args.max_age, n_init=args.n_init)

    cap_source = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(cap_source)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save_vid is not None:
        os.makedirs(os.path.dirname(args.save_vid), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_vid, fourcc, fps, (width, height))

    mot_f = None
    if args.save_mot is not None:
        os.makedirs(os.path.dirname(args.save_mot), exist_ok=True)
        mot_f = open(args.save_mot + '.txt', 'w')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1) Detect persons
        det_xyxy, det_conf, det_cls = detector(frame)
        # 2) Compute appearance features
        features = extractor(frame, det_xyxy)
        # 3) Build detections list
        detections = []
        for i in range(len(det_xyxy)):
            tlwh = xyxy_to_tlwh(det_xyxy[i])
            detections.append(Detection(tlwh, det_conf[i], features[i]))

        # 4) Predict & update
        tracker.predict()
        tracker.update(detections)

        # 5) Draw and save
        draw_tracks(frame, tracker.tracks)
        if writer is not None:
            writer.write(frame)
        if mot_f is not None:
            for t in tracker.tracks:
                if t.is_confirmed() and t.time_since_update == 0:
                    save_mot_line(mot_f, frame_idx, t)

        cv2.imshow('DeepSORT + YOLOv8', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer is not None: 
        writer.release()
    if mot_f is not None: 
        mot_f.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
