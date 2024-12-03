from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source="/example_vids/chickens.mp4", show=True, save=True, save_txt=True, conf=0.5)




