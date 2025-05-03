from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"./ultralytics/cfg/models/v8/yolov8.yaml")
    # with torch.autograd.detect_anomaly():
    # torch.autograd.set_detect_anomaly(True)
    model.train(data=r'C:\Users\Mr.Wang\Desktop\ultralytics-main\ultralytics-main\debug.yaml',
                epochs=300,
                batch=32,
                optimizer='SGD',
                # imgsz=1024,
                close_mosaic=300,
                # device='6,7',
                # amp=False,
                # box=7.5 * 0.1,  # (float) box loss gain
                # cls=0.5 * 0.1,  # (float) cls loss gain (scale with pixels)
                # dfl=1.5 * 0.1,  # (float) dfl loss gain
                seed=0)

