import torch
import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, capture_index: int | str):
        self.capture_index = capture_index

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = self.load_model()

    def load_model(self):
        model = YOLO("yolov8m.pt")  # Load a pretrained YOLOv*n model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_boxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()

            # xyxys = boxes.xyxy

            for xyxy in xyxys:
                # cv.rectangle(
                #     frame,
                #     (int(xyxy[0]), int(xyxy[1])),
                #     (int(xyxy[2]), int(xyxy[3])),
                #     (0, 255, 0),
                #     2,
                # )
                xyxys.append(boxes.xyxy)
                confidences.append(boxes.conf)
                class_ids.append(boxes.cls)

        return results[0].plot(), xyxys, confidences, class_ids

    def __call__(self):
        cap = cv.VideoCapture(self.capture_index)
        assert cap.isOpened()

        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0

        while cap.isOpened():
            start_time = time.perf_counter()

            success, frame = cap.read()

            assert success

            results = self.predict(frame)

            frame, _, _, _ = self.plot_boxes(results, frame)

            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 2)

            cv.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            cv.imshow("YOLOv8 Detection", frame)
            frame_count += 1

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    od = ObjectDetection(0)
    od()
