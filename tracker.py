import math
import time
from collections import deque

class CentroidTracker:
    def __init__(self, max_distance=100, max_missing=10):
        self.next_id = 1
        self.objects = {}
        self.missing = {}
        self.history = {}
        self.labels = {}
        self.max_distance = max_distance
        self.max_missing = max_missing

    def register(self, center, label):
        object_id = self.next_id
        self.objects[object_id] = center
        self.missing[object_id] = 0
        self.labels[object_id] = label
        self.history[object_id] = deque(maxlen=30)
        self.history[object_id].append((time.time(), center))
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.missing[object_id]
        del self.labels[object_id]
        del self.history[object_id]

    def update(self, detections):
        current_time = time.time()

        if len(detections) == 0:
            for object_id in list(self.missing.keys()):
                self.missing[object_id] += 1

                if self.missing[object_id] > self.max_missing:
                    self.deregister(object_id)

            return self.objects

        if len(self.objects) == 0:
            for center, label in detections:
                self.register(center, label)

            return self.objects

        used_objects = set()

        for center, label in detections:
            closest_id = None
            closest_distance = float("inf")

            for object_id, old_center in self.objects.items():
                if object_id in used_objects:
                    continue

                if self.labels[object_id] != label:
                    continue

                distance = math.dist(center, old_center)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_id = object_id

            if closest_id is not None and closest_distance < self.max_distance:
                self.objects[closest_id] = center
                self.missing[closest_id] = 0
                self.history[closest_id].append((current_time, center))
                used_objects.add(closest_id)
            else:
                self.register(center, label)
                used_objects.add(self.next_id - 1)

        for object_id in list(self.objects.keys()):
            if object_id not in used_objects:
                self.missing[object_id] += 1

                if self.missing[object_id] > self.max_missing:
                    self.deregister(object_id)

        return self.objects

    def get_label(self, object_id):
        return self.labels.get(object_id, "UNKNOWN")

    def get_speed(self, object_id):
        history = self.history.get(object_id)

        if history is None or len(history) < 2:
            return 0

        old_time, old_position = history[0]
        new_time, new_position = history[-1]

        distance = math.dist(old_position, new_position)
        time_difference = new_time - old_time

        if time_difference <= 0:
            return 0

        return distance / time_difference