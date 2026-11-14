import cv2


class NightDetector:
    def __init__(self, threshold=70):
        self.threshold = threshold

    def get_brightness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.mean()

    def get_mode(self, frame):
        brightness = self.get_brightness(frame)

        if brightness <= self.threshold:
            return "NIGHT"

        return "DAY"
