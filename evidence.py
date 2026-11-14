import os
import cv2


class EvidenceManager:
    def __init__(self, folder="evidence"):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def save_frame(self, event_id, frame):
        path = os.path.join(self.folder, f"{event_id}.jpg")
        cv2.imwrite(path, frame)
        return path
