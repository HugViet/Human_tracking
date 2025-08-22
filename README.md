# DeepSORT (from scratch) + YOLOv8 for Multi-Human Tracking

This repo gives you a clean, readable DeepSORT implementation (Kalman filter, appearance embeddings, matching cascade) wired to a YOLOv8 detector. It can track people in any video and export MOTChallenge-format results for MOT17 sequences.

## 1) Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r deepsort_yolov8/requirements.txt
```

> Optional: if you have a GPU, install a CUDA build of PyTorch first.

## 2) Quick start on a video

```bash
python deepsort_yolov8/run_tracker.py   --source your_video.mp4   --yolo-weights yolov8n.pt   --device 0   --imgsz 640   --conf 0.35   --save-vid out/tracked.mp4   --save-mot out/mot_results/seq
```

- `--save-vid` writes an annotated video.
- `--save-mot` writes a MOTChallenge result txt: `frame, id, x, y, w, h, conf, -1, -1, -1`.

## 3) Run on MOT17 (images)

Assuming `MOT17` is like:
```
/path/to/MOT17/
  train/
    MOT17-02-FRCNN/img1/000001.jpg ...
    ...
```

```bash
python deepsort_yolov8/tools/run_mot17.py   --data-dir /path/to/MOT17/train   --yolo-weights yolov8n.pt   --split train   --device 0   --out-dir out/mot17_results
```

This writes one results file per sequence under `out/mot17_results/<SEQ_NAME>.txt`.

## 4) Notes

- Detector is YOLOv8 (people class only). You can swap weights (n/s/m/l/x) or trained ones.
- Appearance embeddings use a lightweight ResNet18 head (ImageNet backbone) → 128-D L2-normalized features.
- Matching uses cascade (age-based), Mahalanobis gating on motion, and cosine distance on appearance.
- Kalman state: `[cx, cy, a, h, vx, vy, va, vh]` with model & noise scalars close to the DeepSORT paper.

## 5) Parameters to tweak

- `max_age` (missed frames before deleting), `n_init` (hits before confirm), `max_cosine_distance`, `budget` (feature memory), detector `conf`/`imgsz`.
- For crowded scenes, try stronger YOLO weights and lower `max_cosine_distance` (e.g. 0.2–0.3).
