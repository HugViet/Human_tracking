# DeepSORT (from scratch) + YOLOv8 for Multi-Human Tracking

This repo gives you a clean, readable DeepSORT implementation (Kalman filter, appearance embeddings, matching cascade) wired to a YOLOv8 detector. It can track people in any video and export MOTChallenge-format results for MOT17 sequences.

## 1) Setup
With Python virtual environment:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
With Conda:
```bash
conda create -n your_venv_name python=3.9
conda activate your_venv_name
pip install -r requirements.txt
```
> Note: Use Python 3.9. if you have a GPU, install a CUDA build of PyTorch first.

Then, download pretrained weight from YOLO and MOT17 dataset

## 2) Combine photos into video on MOT17

```bash
python -m tools.make_mot17_videos \
  --data-dir datasets/MOT17/train \
  --out-dir out/videos
```

Default FPS is 30. Add --fps your_fps to config another value.

## 3) Run with MOT17 images

```bash
python -m tools.run_mot17 \
  --data-dir datasets/MOT17/train \
  --yolo-weights yolov8n.pt \
  --split train \
  --device 0 \
  --out-dir out/mot17_results
```

## 4) Evaluation

```bash
python -m tools.eval_mot17 \
  --gt-dir datasets/MOT17/train \
  --res-dir out/mot17_results \
  --dist-thresh 0.5 \
  --save-csv out/mot17_eval_summary.csv
```

## 5) Evaluation

```bash
python -m tools.viz_mot17_from_results \
  --data-dir datasets/MOT17/train \
  --res-dir out/mot17_results \
  --out-dir out/vis
```

## 6) Run on local video

```bash
python -m run_tracker \
  --source /path/to/your_video.mp4 \
  --yolo-weights yolov8n.pt \
  --device 0 \
  --imgsz 640 \
  --conf 0.35 \
  --save-vid out/tracked.mp4 \
  --save-mot out/mot_results/your_video
```

--source points to your video file (absolute path is better).

--device 0 uses GPU 0; if CUDA is not available, omit this parameter or set --device cpu.

--save-vid outputs video with box + ID.

--save-mot outputs out/mot_results/your_video.txt in MOT format (optional).

A preview will be displayed; press Esc to stop.

## 7) Run on webcam (if your device has at least 1 webcam)

```bash
python -m run_tracker --source 0 --yolo-weights yolov8n.pt --device 0 --save-vid out/webcam.mp4
```

This command activates your webcam (index 0 - change if you have more than 1). The result is saved in `out/webcam.mp4`.

## 8) Notes

- Detector is YOLOv8 (people class only). You can swap weights (n/s/m/l/x) or trained ones.
- Appearance embeddings use a lightweight ResNet18 head (ImageNet backbone) → 128-D L2-normalized features.
- Matching uses cascade (age-based), Mahalanobis gating on motion, and cosine distance on appearance.
- Kalman state: `[cx, cy, a, h, vx, vy, va, vh]` with model & noise scalars close to the DeepSORT paper.

## 9) Parameters to tweak

- `max_age` (missed frames before deleting), `n_init` (hits before confirm), `max_cosine_distance`, `budget` (feature memory), detector `conf`/`imgsz`.
- For crowded scenes, try stronger YOLO weights and lower `max_cosine_distance` (e.g. 0.2–0.3).
