import os
import cv2
import torch
from ultralytics import YOLO


class PlateDetector:
    def __init__(
        self,
        model_path="plate_yolo.pt",
        conf=0.30,
        iou=0.45,
        imgsz=960,
    ):
        # Resolve model path relative to this file
        if not os.path.isabs(model_path):
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), model_path
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Plate model not found: {model_path}")

        self.model = YOLO(model_path)

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []

        # Make sure image is valid
        if len(frame.shape) != 3:
            return []

        h, w = frame.shape[:2]

        # Slight enhancement for difficult CCTV frames
        enhanced = frame.copy()

        # Convert to LAB and improve local contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        l = clahe.apply(l)

        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        results = self.model.predict(
            source=enhanced,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=20,
            verbose=False,
            device=self.device,
        )

        plates = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])

                # Clamp coordinates
                x1 = max(0, min(int(x1), w - 1))
                y1 = max(0, min(int(y1), h - 1))
                x2 = max(0, min(int(x2), w))
                y2 = max(0, min(int(y2), h))

                if x2 <= x1 or y2 <= y1:
                    continue

                plate_w = x2 - x1
                plate_h = y2 - y1

                # Ignore extremely tiny detections
                if plate_w < 12 or plate_h < 6:
                    continue

                crop = frame[y1:y2, x1:x2].copy()

                if crop.size == 0:
                    continue

                # Add a little padding around plate
                pad_x = int(plate_w * 0.10)
                pad_y = int(plate_h * 0.15)

                px1 = max(0, x1 - pad_x)
                py1 = max(0, y1 - pad_y)
                px2 = min(w, x2 + pad_x)
                py2 = min(h, y2 + pad_y)

                padded_crop = frame[py1:py2, px1:px2].copy()

                plates.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "crop": padded_crop,
                        "width": plate_w,
                        "height": plate_h,
                    }
                )

        # Highest confidence first
        plates.sort(key=lambda x: x["confidence"], reverse=True)

        return plates
