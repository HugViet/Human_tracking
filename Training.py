from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")
    model.train(data="mot17.yaml", epochs=25, imgsz=640, batch=16)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
