
import os
import argparse
import cv2
import glob
import re
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

def sequence_to_video(seq_dir, out_path, fps=None):
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

    for i, img_path in enumerate(imgs, 1):
        frame = cv2.imread(img_path)
        if frame is None: continue
        writer.write(frame)
    writer.release()
    print(f'[OK] Wrote {out_path} at {fps} FPS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True, help='Path to MOT17 split, e.g. datasets/MOT17/train')
    ap.add_argument('--out-dir', default='out/videos', help='Directory to save videos')
    ap.add_argument('--fps', type=int, default=None, help='Override FPS (otherwise read from seqinfo.ini or 30)')
    args = ap.parse_args()

    seqs = [d for d in sorted(os.listdir(args.data_dir)) if os.path.isdir(os.path.join(args.data_dir, d))]
    for seq in seqs:
        seq_dir = os.path.join(args.data_dir, seq)
        out_path = os.path.join(args.out_dir, f'{seq}.mp4')
        sequence_to_video(seq_dir, out_path, fps=args.fps)

if __name__ == '__main__':
    main()
