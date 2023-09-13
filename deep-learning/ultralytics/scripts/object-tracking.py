from ultralytics import YOLO
import cv2 as cv

model = YOLO("yolov8m.pt")

video_path = "../data/gymnasts_2.mp4"

# model.track(video_path, show=True, tracker="bytetrack.yaml")

model.track(source=0, show=True, tracker="bytetrack.yaml")
