from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

# Run the inference on the source
video_path = "../data/gymnasts_2.mp4"

# results = model(source=video_path, show=True, conf=0.3, save=True)
results = model(source=0, show=True, conf=0.3, save=True)
