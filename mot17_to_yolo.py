import os
import shutil

MOT_PATH = "MOT17/train"
OUT_PATH = "datasets/MOT17_YOLO"
CLASSES = {1: 0}

def convert():
    for seq in os.listdir(MOT_PATH):
        seq_path = os.path.join(MOT_PATH, seq)
        img_dir = os.path.join(seq_path, "img1")
        label_file = os.path.join(seq_path, "gt/gt.txt")

        out_img_dir = os.path.join(OUT_PATH, "images/train", seq)
        out_lbl_dir = os.path.join(OUT_PATH, "labels/train", seq)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        for img_file in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, img_file), out_img_dir)

        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                frame_id, obj_id, x, y, w, h, conf, cls, vis = line.strip().split(",")
                if int(cls) != 1:
                    continue
                frame_file = f"{int(frame_id):06d}.jpg"
                img_path = os.path.join(out_img_dir, frame_file)
                if not os.path.exists(img_path):
                    continue
                img_label_file = os.path.join(out_lbl_dir, frame_file.replace(".jpg", ".txt"))

                x, y, w, h = map(float, [x, y, w, h])
                xc = x + w / 2
                yc = y + h / 2

                img_w, img_h = 1920, 1080
                norm = lambda val, max_val: val / max_val
                xc, yc, w, h = map(norm, [xc, yc, w, h], [img_w, img_h, img_w, img_h])

                with open(img_label_file, "a") as f_lbl:
                    f_lbl.write(f"0 {xc} {yc} {w} {h}\n")

if __name__ == "__main__":
    convert()
