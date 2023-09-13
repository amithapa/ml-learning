from ultralytics import YOLO
import cv2 as cv
import time

# load a pretrained YOLOv8n model
model = YOLO("yolov8n-seg.pt")

# results = model(source="../data/individual_01.jpg", show=True, conf=0.4, save=True)
# results = model(source="../data/gymnasts_2.mp4", show=True, conf=0.4, save=True)
# results = model(source=0, show=True, conf=0.4, save=True)
# model("https://ultralytics.com/images/bus.jpg", show=True, save=True)

video_path = "../data/gymnasts_2.mp4"

# cap = cv.VideoCapture(video_path)
cap = cv.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        start = time.perf_counter()
        result = model(frame)

        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        annotated_frame = result[0].plot()

        # Display the annotated frame
        cv.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv.imshow("YOLOv8 Inference", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break
