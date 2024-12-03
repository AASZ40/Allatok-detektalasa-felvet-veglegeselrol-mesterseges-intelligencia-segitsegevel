from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "data.yaml", batch=8, imgsz=640, save=True, epochs=100, workers=4)


