from ultralytics import YOLO

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# results = model(source="../data/individual_01.jpg", show=True, conf=0.4, save=True)
# results = model(source="../data/gymnasts_2.mp4", show=True, conf=0.4, save=True)
# results = model(source=0, show=True, conf=0.4, save=True)
model("https://ultralytics.com/images/bus.jpg", show=True, save=True)
