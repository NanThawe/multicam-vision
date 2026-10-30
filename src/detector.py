"""Detect people in one synchronised batch of camera frames."""

import cv2
from ultralytics import YOLO

from src.framesource import iter_frames  # Match the filename used to run File 1.


class PersonDetector:
    def __init__(self, weights="yolo26n.pt", device="cpu"):
        self.model = YOLO(weights)  # Downloads pretrained weights on first use.
        self.device = device

    def detect(self, frames):
        """Return {camera: [{xyxy, confidence, class_id}, ...]}.

        xyxy coordinates refer to the original 1920x1080 images.
        """
        cameras = list(frames)
        results = self.model.predict(
            source=[frames[camera] for camera in cameras],
            classes=[0],  # Person class in the pretrained COCO labels.
            conf=0.30,
            imgsz=640,
            device=self.device,
            verbose=False,
        )

        detections = {}
        for camera, result in zip(cameras, results):
            boxes = result.boxes
            detections[camera] = [
                {"xyxy": box, "confidence": score, "class_id": int(cls)}
                for box, score, cls in zip(
                    boxes.xyxy.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                )
            ]
        return detections


if __name__ == "__main__":
    reader = iter_frames()
    try:
        frame_id, frames = next(reader)  # Test one batch, four cameras.
    finally:
        reader.close()

    detections = PersonDetector().detect(frames)
    for camera, people in detections.items():
        print(f"Frame {frame_id}, {camera}: {len(people)} people")

    # Save one annotated image to check whether bounding boxes make sense.
    preview = frames["C1"].copy()
    for person in detections["C1"]:
        x1, y1, x2, y2 = map(int, person["xyxy"])
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, f"{person['confidence']:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if not cv2.imwrite("preview_C1.jpg", preview):
        raise OSError("Could not write preview_C1.jpg")
    print("Saved preview_C1.jpg")
