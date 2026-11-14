from PySide6.QtCore import QThread, Signal
from test_plate import PlateANPR


class ANPRWorker(QThread):
    result_ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.running = True
        self.frame = None
        self.vehicle_boxes = []
        self.frame_id = 0
        self.last_process = -1
        self.busy = False
        self.anpr = PlateANPR("plate_yolo.pt", conf=0.10)

    def submit(self, frame, vehicle_boxes):
        if self.busy:
            return False
        self.frame = frame.copy()
        self.vehicle_boxes = list(vehicle_boxes)
        self.frame_id += 1
        return True

    def run(self):
        while self.running:
            if self.frame is None:
                self.msleep(20)
                continue

            if self.busy:
                self.msleep(20)
                continue

            if self.frame_id == self.last_process:
                self.msleep(20)
                continue

            self.busy = True

            frame = self.frame
            vehicles = self.vehicle_boxes
            frame_id = self.frame_id
            self.last_process = frame_id

            try:
                results = self.anpr.process_frame(frame, vehicles)

                for result in results:
                    self.result_ready.emit(result)

            except Exception as e:
                print("ANPR ERROR:", e)

            self.busy = False

    def save_plate(self, vehicle_id, plate, crop, confidence):
        return self.anpr.save_plate(vehicle_id, plate, crop, confidence)

    def stop(self):
        self.running = False
        self.wait()
