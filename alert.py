class AlertManager:
    def send_alert(self, event):
        if event is None:
            return

        severity = event["severity"]
        event_type = event["event_type"]

        if severity == "HIGH":
            message = f"CRITICAL: {event_type} detected"
        elif severity == "MEDIUM":
            message = f"WARNING: {event_type} detected"
        else:
            message = f"INFO: {event_type} detected"

        print()
        print("========== ALERT ==========")
        print("Alert ID:", event["event_id"])
        print("Track ID:", event["track_id"])
        print("Message:", message)
        print("Severity:", severity)
        print("Time:", event["timestamp"])
        print("Status:", event["status"])
        print("===========================")
