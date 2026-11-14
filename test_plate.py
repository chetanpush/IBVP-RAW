import os
import re
import csv
import cv2
import pytesseract
from collections import Counter
from datetime import datetime
from plate_detector import PlateDetector


class PlateANPR:
    def __init__(self, model_path="plate_yolo.pt", conf=0.20):
        self.detector = PlateDetector(model_path, conf=conf)
        self.plate_history = {}
        self.plate_votes = {}
        self.confirmed_vehicles = set()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.plates_dir = os.path.join(self.base_dir, "plates")
        os.makedirs(self.plates_dir, exist_ok=True)
        self.csv_file = os.path.join(self.base_dir, "plates.csv")
        tesseract_paths = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        )
        tesseract_path = next(
            (path for path in tesseract_paths if os.path.exists(path)), None
        )
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.ocr_available = bool(tesseract_path)

    def clean_text(self, text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def plate_key(self, plate):
        return self.clean_text(plate)

    def similar_plate(self, first, second):
        first = self.plate_key(first)
        second = self.plate_key(second)
        if len(first) != 8 or len(second) != 8:
            return False
        differences = sum(left != right for left, right in zip(first, second))
        return differences <= max(1, min(len(first), len(second)) // 4)

    def find_similar_plate(self, plate, plates):
        return next((item for item in plates if self.similar_plate(plate, item)), None)

    def format_plate(self, text):
        text = self.clean_text(text)
        if len(text) != 8:
            return ""
        return text

    def read_plate(self, crop):
        if not self.ocr_available or crop is None or crop.size == 0:
            return ""
        h, w = crop.shape[:2]
        if w < 20 or h < 8:
            return ""
        border_x = max(2, int(w * 0.04))
        border_y = max(2, int(h * 0.12))
        crop = cv2.copyMakeBorder(
            crop,
            border_y,
            border_y,
            border_x,
            border_x,
            cv2.BORDER_REPLICATE,
        )
        crop = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        original_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(original_gray, 5, 50, 50)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        sharpened = cv2.addWeighted(
            gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 2), -0.6, 0
        )
        images = [sharpened, original_gray]
        readings = []
        for image in images:
            for psm in (7, 6):
                text = pytesseract.image_to_string(
                    image,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    ),
                )
                plate = self.format_plate(text)
                if plate:
                    readings.append(plate)
                    break
        if not readings:
            return ""
        return Counter(readings).most_common(1)[0][0]

    def center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def plate_inside_vehicle(self, plate_box, vehicle_box):
        px, py = self.center(plate_box)
        x1, y1, x2, y2 = vehicle_box
        padding_x = max(10, int((x2 - x1) * 0.08))
        padding_y = max(10, int((y2 - y1) * 0.08))
        return (
            x1 - padding_x <= px <= x2 + padding_x
            and y1 - padding_y <= py <= y2 + padding_y
        )

    def process_frame(self, frame, vehicles):
        results = []
        try:
            plates = self.detector.detect(frame)
            for plate in plates:
                plate_box = plate["box"]
                plate_crop = plate["crop"]
                confidence = plate["confidence"]
                vehicle_id = None
                vehicle_box = None

                for vehicle in vehicles:
                    object_id, box = vehicle
                    if self.plate_inside_vehicle(plate_box, box):
                        vehicle_id = object_id
                        vehicle_box = box
                        break

                if vehicle_id is None:
                    continue

                text = self.read_plate(plate_crop)

                if not text:
                    continue

                if vehicle_id not in self.plate_history:
                    self.plate_history[vehicle_id] = []

                self.plate_history[vehicle_id].append(text)
                self.plate_history[vehicle_id] = self.plate_history[vehicle_id][-5:]

                vehicle_votes = self.plate_votes.setdefault(vehicle_id, Counter())
                vote_key = self.find_similar_plate(text, vehicle_votes.keys())
                if vote_key is None:
                    vote_key = text
                vehicle_votes[vote_key] += 1
                best_plate, count = vehicle_votes.most_common(1)[0]

                print(f"Vehicle {vehicle_id}: {text}")

                self.confirmed_vehicles.add(vehicle_id)

                if count < (1 if confidence >= 0.75 else 2):
                    continue

                results.append(
                    {
                        "vehicle_id": vehicle_id,
                        "plate": best_plate,
                        "crop": plate_crop,
                        "box": plate_box,
                        "vehicle_box": vehicle_box,
                        "confidence": confidence,
                    }
                )

        except Exception as e:
            print("ANPR ERROR:", e)

        return results

    def save_plate(self, vehicle_id, plate, crop, confidence):
        headers = [
            "date",
            "time",
            "plate_number",
            "vehicle_id",
            "plate_confidence",
            "image_path",
        ]
        rows = []

        if os.path.exists(self.csv_file):
            with open(self.csv_file, "r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    if len(row) >= len(headers):
                        rows.append(row[: len(headers)])

        for row in rows:
            existing_path = row[5]
            if not os.path.exists(existing_path) or crop is None or crop.size == 0:
                continue
            existing_crop = cv2.imread(existing_path)
            if existing_crop is None or existing_crop.size == 0:
                continue
            first = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 24))
            second = cv2.resize(
                cv2.cvtColor(existing_crop, cv2.COLOR_BGR2GRAY), (64, 24)
            )
            if cv2.absdiff(first, second).mean() < 22:
                return existing_path

        duplicate_plate = next(
            (row[2] for row in rows if self.similar_plate(plate, row[2])), None
        )

        if duplicate_plate is None:
            filename = plate + ".jpg"
            image_path = os.path.join(self.plates_dir, filename)
            if os.path.exists(image_path):
                filename = f"{plate}_{datetime.now().strftime('%H%M%S%f')}.jpg"
                image_path = os.path.join(self.plates_dir, filename)
            cv2.imwrite(image_path, crop)
            now = datetime.now()
            rows.append(
                [
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    plate,
                    vehicle_id,
                    f"{confidence:.2f}",
                    image_path,
                ]
            )
        else:
            image_path = next(row[5] for row in rows if row[2] == duplicate_plate)

        with open(self.csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

        return image_path
