import math
import time


class BehaviorAnalyzer:
    def __init__(self, loiter_time=10, movement_threshold=30):
        self.loiter_time = loiter_time
        self.movement_threshold = movement_threshold
        self.start_positions = {}
        self.start_times = {}
        self.last_positions = {}

    def analyze(self, tracks):
        results = {}
        now = time.time()

        for object_id, center in tracks.items():
            if object_id not in self.start_positions:
                self.start_positions[object_id] = center
                self.start_times[object_id] = now
                self.last_positions[object_id] = center
                results[object_id] = "NEW"
                continue

            old = self.last_positions[object_id]
            distance = math.dist(old, center)

            elapsed = now - self.start_times[object_id]
            total_distance = math.dist(self.start_positions[object_id], center)

            if distance > 100:
                results[object_id] = "RAPID MOVEMENT"
            elif (
                total_distance < self.movement_threshold and elapsed >= self.loiter_time
            ):
                results[object_id] = "LOITERING"
            elif total_distance < self.movement_threshold:
                results[object_id] = "STATIONARY"
            else:
                results[object_id] = "MOVING"
            self.last_positions[object_id] = center
            
        return results
