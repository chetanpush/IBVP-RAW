import time


class EventEngine:
    def __init__(self):
        self.next_event_id = 1
        self.active_events = {}

    def get_severity(self, reason):
        if reason == "INTRUSION":
            return "HIGH"

        if reason == "RAPID MOVEMENT":
            return "MEDIUM"

        if reason == "LOITERING":
            return "MEDIUM"

        return "LOW"

    def create_event(self, object_id, reason):
        key = (object_id, reason)

        if key in self.active_events:
            return None

        event = {
            "event_id": f"EVT-{self.next_event_id:03d}",
            "track_id": object_id,
            "event_type": reason,
            "severity": self.get_severity(reason),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "ACTIVE",
        }

        self.active_events[key] = event
        self.next_event_id += 1

        return event

    def resolve_event(self, object_id, reason):
        key = (object_id, reason)

        if key not in self.active_events:
            return None

        event = self.active_events[key]
        event["status"] = "RESOLVED"

        del self.active_events[key]

        return event
