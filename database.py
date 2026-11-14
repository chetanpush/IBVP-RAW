import sqlite3
import time


class Database:
    def __init__(self, db_name="ibvap.db"):
        self.db_name = db_name
        self.create_tables()

    def connect(self):
        return sqlite3.connect(self.db_name)

    def create_tables(self):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            track_id INTEGER,
            event_type TEXT,
            severity TEXT,
            timestamp TEXT,
            status TEXT,
            evidence_path TEXT
        )""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS face_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            name TEXT,
            timestamp TEXT
        )""")

        cursor.execute("PRAGMA table_info(events)")
        columns = [row[1] for row in cursor.fetchall()]

        if "evidence_path" not in columns:
            cursor.execute("ALTER TABLE events ADD COLUMN evidence_path TEXT")

        conn.commit()
        conn.close()

    def save_event(self, event):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO events (
            event_id, track_id, event_type, severity, timestamp, status, evidence_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event["event_id"],
                event["track_id"],
                event["event_type"],
                event["severity"],
                event["timestamp"],
                event["status"],
                event.get("evidence_path", ""),
            ),
        )

        conn.commit()
        conn.close()

    def resolve_event(self, event):
        conn = self.connect()

        cursor = conn.cursor()
        cursor.execute(
            "UPDATE events SET status = ? WHERE event_id = ?",
            ("RESOLVED", event["event_id"]),
        )

        conn.commit()
        conn.close()

    def get_events(self):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM events ORDER BY id DESC")
        rows = cursor.fetchall()

        conn.close()

        events = []

        for row in rows:
            events.append(
                {
                    "id": row[0],
                    "event_id": row[1],
                    "track_id": row[2],
                    "event_type": row[3],
                    "severity": row[4],
                    "timestamp": row[5],
                    "status": row[6],
                    "evidence_path": row[7],
                }
            )

        return events

    def log_face_sighting(self, track_id, name):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO face_sightings (
            track_id, name, timestamp
        ) VALUES (?, ?, ?)""",
            (
                track_id,
                name,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

        conn.commit()
        conn.close()

    def get_face_sightings(self):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM face_sightings ORDER BY id DESC")
        rows = cursor.fetchall()

        conn.close()

        sightings = []

        for row in rows:
            sightings.append(
                {
                    "id": row[0],
                    "track_id": row[1],
                    "name": row[2],
                    "timestamp": row[3],
                }
            )

        return sightings
