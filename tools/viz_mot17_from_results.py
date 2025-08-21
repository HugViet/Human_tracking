
import os
import argparse
import cv2
import glob
import re
import numpy as np
import configparser

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def get_fps_from_seqinfo(seq_dir, default_fps=30):
    ini = os.path.join(seq_dir, 'seqinfo.ini')
    if not os.path.isfile(ini):
        return default_fps
    config = configparser.ConfigParser()
    config.read(ini)
    try:
        return config.getint('Sequence', 'frameRate', fallback=default_fps)
    except Exception:
        return default_fps

def load_results(res_file):
    # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
    tracks = {}
    with open(res_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7: 
                continue
            fr = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            tracks.setdefault(fr, []).append((tid, x, y, w, h, conf))
    return tracks

def visualize_sequence(seq_dir, res_file, out_path, fps=None, conf_th=0.0):
    img_dir = os.path.join(seq_dir, 'img1')
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')), key=natural_key)
    if not imgs:
        print(f'[WARN] No images in {img_dir}'); return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    first = cv2.imread(imgs[0])
    h, w = first.shape[:2]
    if fps is None:
        fps = get_fps_from_seqinfo(seq_dir, default_fps=30)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    tracks = load_results(res_file)

    for idx, img_path in enumerate(imgs, 1):
        frame = cv2.imread(img_path)
        if frame is None: continue

        dets = tracks.get(idx, [])
        for (tid, x, y, bw, bh, conf) in dets:
            if conf < conf_th: 
                continue
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {tid}', (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f'Frame {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        writer.write(frame)

    writer.release()
    print(f'[OK] Wrote {out_path} with tracking overlays at {fps} FPS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True, help='Path to MOT17 split, e.g. datasets/MOT17/train')
    ap.add_argument('--res-dir', required=True, help='Directory containing <SEQ>.txt results from tracker')
    ap.add_argument('--out-dir', default='out/vis', help='Directory to save visualization videos')
    ap.add_argument('--fps', type=int, default=None, help='Override FPS (else read from seqinfo.ini or 30)')
    ap.add_argument('--conf-th', type=float, default=0.0, help='Filter result lines by confidence')
    args = ap.parse_args()

    seqs = [d for d in sorted(os.listdir(args.data_dir)) if os.path.isdir(os.path.join(args.data_dir, d))]
    for seq in seqs:
        seq_dir = os.path.join(args.data_dir, seq)
        res_file = os.path.join(args.res_dir, f'{seq}.txt')
        if not os.path.isfile(res_file):
            print(f'[SKIP] No results for {seq} at {res_file}')
            continue
        out_path = os.path.join(args.out_dir, f'{seq}.mp4')
        visualize_sequence(seq_dir, res_file, out_path, fps=args.fps, conf_th=args.conf_th)

if __name__ == '__main__':
    main()
