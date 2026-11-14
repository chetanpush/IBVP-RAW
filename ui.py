import sys
import time
import math
import json
import os
import glob
import csv

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QSlider,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QGridLayout,
    QGraphicsDropShadowEffect,
    QScrollArea,
)

from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QConicalGradient,
    QDesktopServices,
)

from PySide6.QtCore import (
    QTimer,
    Qt,
    Signal,
    QPointF,
    QRectF,
    QUrl,
    QPropertyAnimation,
)

from ultralytics import YOLO

from tracker import CentroidTracker
from detection_worker import AsyncDetectionWorker
from face_id import FaceIdentifier, AsyncFaceWorker
from database import Database
from video_capture import LatestFrameCapture
from behavior import BehaviorAnalyzer
from suspicious import SuspiciousActivity
from event import EventEngine
from alert import AlertManager
from evidence import EvidenceManager
from night import NightDetector
from anpr_worker import ANPRWorker

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

DISPLAY_FPS = 30

# YOLO does not need to run on every displayed frame.
# The tracker keeps objects alive between detections.
YOLO_FPS = 8

# Face recognition is expensive, so run it periodically.
FACE_FPS = 3

# ANPR is even more expensive. Only submit stable vehicles.
ANPR_FPS = 2

YOLO_INTERVAL = max(1, round(DISPLAY_FPS / YOLO_FPS))

FACE_INTERVAL = max(1, round(DISPLAY_FPS / FACE_FPS))

ANPR_INTERVAL = max(1, round(DISPLAY_FPS / ANPR_FPS))


# ============================================================
# VIDEO LABEL
# ============================================================


class VideoLabel(QLabel):

    zone_changed = Signal(tuple)
    zoom_wheel = Signal(int)
    recenter_clicked = Signal(int, int)

    def __init__(self):

        super().__init__()

        self.drawing = False

        self.start_point = None
        self.end_point = None

        self.configure_zone = False
        self.zone = None

        self.setAlignment(Qt.AlignCenter)

    def set_zone(self, zone):

        self.zone = zone

        self.update()

    def mousePressEvent(self, event):

        if not self.configure_zone:

            if event.button() == Qt.LeftButton:

                point = event.position().toPoint()

                self.recenter_clicked.emit(point.x(), point.y())

            return

        if event.button() == Qt.LeftButton:

            self.drawing = True

            self.start_point = event.position().toPoint()

            self.end_point = self.start_point

            self.update()

    def mouseMoveEvent(self, event):

        if self.drawing and self.configure_zone:

            self.end_point = event.position().toPoint()

            self.update()

    def mouseReleaseEvent(self, event):

        if not self.drawing:
            return

        if not self.configure_zone:
            return

        if event.button() != Qt.LeftButton:
            return

        self.drawing = False

        self.end_point = event.position().toPoint()

        x1 = self.start_point.x()
        y1 = self.start_point.y()

        x2 = self.end_point.x()
        y2 = self.end_point.y()

        x1, x2 = sorted((x1, x2))

        y1, y2 = sorted((y1, y2))

        if x2 - x1 >= 20 and y2 - y1 >= 20:

            self.zone_changed.emit((x1, y1, x2, y2))

        self.update()

    def wheelEvent(self, event):

        if self.configure_zone:
            return

        direction = 1 if event.angleDelta().y() > 0 else -1

        self.zoom_wheel.emit(direction)

        event.accept()

    def paintEvent(self, event):

        super().paintEvent(event)

        if self.drawing and self.start_point and self.end_point:

            painter = QPainter(self)

            painter.setPen(QPen(Qt.yellow, 2))

            x1 = min(self.start_point.x(), self.end_point.x())

            y1 = min(self.start_point.y(), self.end_point.y())

            x2 = max(self.start_point.x(), self.end_point.x())

            y2 = max(self.start_point.y(), self.end_point.y())

            painter.drawRect(x1, y1, x2 - x1, y2 - y1)


# ============================================================
# PERIMETER RADAR (live tactical minimap)
# ============================================================


class RadarWidget(QWidget):
    """A small radar-style minimap showing tracked objects relative
    to the camera frame, with a sweeping-scan animation. Purely a
    visual/situational-awareness aid — it reads state pushed to it
    from the main window each frame."""

    def __init__(self):

        super().__init__()

        self.setObjectName("radarWidget")

        self.setMinimumHeight(150)

        self.tracks = {}

        self.zone = None

        self.frame_w = 0

        self.frame_h = 0

        self.sweep_angle = 0.0

        self._sweep_timer = QTimer(self)

        self._sweep_timer.timeout.connect(self._advance_sweep)

        self._sweep_timer.start(35)

    def _advance_sweep(self):

        self.sweep_angle = (self.sweep_angle + 3.2) % 360

        self.update()

    def update_data(self, tracks, zone, frame_w, frame_h):

        self.tracks = dict(tracks) if tracks else {}

        self.zone = zone

        self.frame_w = frame_w

        self.frame_h = frame_h

        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()

        h = self.height()

        cx, cy = w / 2.0, h / 2.0

        radius = max(10.0, min(w, h) / 2.0 - 8.0)

        painter.setPen(Qt.NoPen)

        painter.setBrush(QColor("#070d18"))

        painter.drawRoundedRect(0, 0, w, h, 10, 10)

        ring_pen = QPen(QColor("#17304a"))

        ring_pen.setWidth(1)

        painter.setPen(ring_pen)

        painter.setBrush(Qt.NoBrush)

        for fraction in (0.34, 0.67, 1.0):

            r = radius * fraction

            painter.drawEllipse(QPointF(cx, cy), r, r)

        painter.drawLine(int(cx - radius), int(cy), int(cx + radius), int(cy))

        painter.drawLine(int(cx), int(cy - radius), int(cx), int(cy + radius))

        painter.save()

        gradient = QConicalGradient(cx, cy, -self.sweep_angle)

        gradient.setColorAt(0.0, QColor(80, 220, 255, 110))

        gradient.setColorAt(0.14, QColor(80, 220, 255, 0))

        gradient.setColorAt(1.0, QColor(80, 220, 255, 0))

        painter.setPen(Qt.NoPen)

        painter.setBrush(QBrush(gradient))

        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        painter.restore()

        fw, fh = self.frame_w, self.frame_h

        if self.zone and fw and fh:

            zx1, zy1, zx2, zy2 = self.zone

            rx1 = cx - radius + (zx1 / fw) * (2 * radius)

            ry1 = cy - radius + (zy1 / fh) * (2 * radius)

            rx2 = cx - radius + (zx2 / fw) * (2 * radius)

            ry2 = cy - radius + (zy2 / fh) * (2 * radius)

            zone_pen = QPen(QColor("#f87171"))

            zone_pen.setWidth(1)

            painter.setPen(zone_pen)

            painter.setBrush(QColor(248, 113, 113, 30))

            painter.drawRect(QRectF(QPointF(rx1, ry1), QPointF(rx2, ry2)))

        if fw and fh:

            for object_id, (tx, ty) in self.tracks.items():

                px = cx - radius + (tx / fw) * (2 * radius)

                py = cy - radius + (ty / fh) * (2 * radius)

                inside = False

                if self.zone:

                    inside = (
                        self.zone[0] <= tx <= self.zone[2]
                        and self.zone[1] <= ty <= self.zone[3]
                    )

                color = QColor("#fb5b5b") if inside else QColor("#38e58b")

                painter.setPen(Qt.NoPen)

                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 60))

                painter.drawEllipse(QPointF(px, py), 9, 9)

                painter.setBrush(color)

                painter.drawEllipse(QPointF(px, py), 3.4, 3.4)

        painter.end()


# ============================================================
# EVIDENCE GALLERY
# ============================================================


class EvidenceGalleryWindow(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("IBVAP — Evidence Gallery")

        self.resize(980, 640)

        self.setStyleSheet("""
            QDialog {
                background:#070c14;
            }

            QLabel#galleryTitle {
                color:#f8fafc;
                font-size:22px;
                font-weight:800;
                padding:8px;
            }

            QLabel#galleryMuted {
                color:#64748b;
                font-size:10px;
                padding:0 8px 8px 8px;
            }

            QScrollArea {
                background:transparent;
                border:none;
            }

            QFrame#thumbCard {
                background:#0d1522;
                border:1px solid #243247;
                border-radius:8px;
            }

            QLabel#thumbCaption {
                color:#93a4bb;
                font-size:9px;
                font-weight:700;
                padding:4px 2px 0 2px;
            }
            """)

        layout = QVBoxLayout(self)

        title = QLabel("CAPTURED EVIDENCE")

        title.setObjectName("galleryTitle")

        layout.addWidget(title)

        self.subtitle = QLabel("")

        self.subtitle.setObjectName("galleryMuted")

        layout.addWidget(self.subtitle)

        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        self.grid_host = QWidget()

        self.grid = QGridLayout(self.grid_host)

        self.grid.setSpacing(10)

        scroll.setWidget(self.grid_host)

        layout.addWidget(scroll, 1)

        self.load_evidence()

    def _collect_images(self):

        paths = []

        for folder in ("evidence", "snapshots"):

            if os.path.isdir(folder):

                for pattern in ("*.jpg", "*.jpeg", "*.png"):

                    paths.extend(glob.glob(os.path.join(folder, pattern)))

        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        return paths

    def load_evidence(self):

        while self.grid.count():

            item = self.grid.takeAt(0)

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        images = self._collect_images()[:60]

        self.subtitle.setText(
            f"{len(images)} captured frame(s) across evidence/ and snapshots/"
        )

        columns = 5

        for index, path in enumerate(images):

            card = QFrame()

            card.setObjectName("thumbCard")

            card_layout = QVBoxLayout(card)

            card_layout.setContentsMargins(6, 6, 6, 6)

            thumb = QLabel()

            thumb.setAlignment(Qt.AlignCenter)

            pixmap = QPixmap(path)

            if not pixmap.isNull():

                thumb.setPixmap(
                    pixmap.scaled(170, 105, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

            card_layout.addWidget(thumb)

            caption = QLabel(os.path.basename(path))

            caption.setObjectName("thumbCaption")

            caption.setWordWrap(True)

            card_layout.addWidget(caption)

            self.grid.addWidget(card, index // columns, index % columns)

        if not images:

            empty = QLabel(
                "No evidence captured yet. Frames are saved automatically "
                "when a high-risk event or snapshot occurs."
            )

            empty.setObjectName("galleryMuted")

            empty.setWordWrap(True)

            self.grid.addWidget(empty, 0, 0, 1, columns)


# ============================================================
# NUMBER PLATE WINDOW
# ============================================================


class PlatesWindow(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setWindowTitle("IBVAP — Captured Number Plates")

        self.resize(1000, 600)

        self.setStyleSheet("""
            QDialog {
                background:#070c14;
            }

            QTableWidget {
                background:#0d1522;
                color:#e5e7eb;
                gridline-color:#1f2b3d;
                border:1px solid #243247;
            }

            QHeaderView::section {
                background:#121d2e;
                color:#93a4bb;
                padding:9px;
                border:0;
                font-weight:700;
            }
            """)

        layout = QVBoxLayout(self)

        title = QLabel("CAPTURED NUMBER PLATES")

        title.setStyleSheet("""
            color:#f8fafc;
            font-size:22px;
            font-weight:800;
            padding:8px;
            """)

        layout.addWidget(title)

        self.table = QTableWidget(0, 6)

        self.table.setHorizontalHeaderLabels(
            [
                "Vehicle ID",
                "Plate Number",
                "Preview",
                "Image Path",
                "Date",
                "Time",
            ]
        )

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.table)

        self.load_plates()

    def load_plates(self):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "plates.csv")

        if not os.path.exists(csv_path):

            self.table.setRowCount(0)

            return

        try:

            with open(csv_path, "r", newline="", encoding="utf-8") as file:

                reader = csv.DictReader(file)

                rows = list(reader)

        except Exception as exc:

            print("Could not read plates.csv:", exc)

            return

        self.table.setRowCount(len(rows))
        self.table.verticalHeader().setDefaultSectionSize(86)

        for row_index, row in enumerate(rows):

            vehicle_id = row.get("vehicle_id", "")

            plate_number = row.get("plate_number", "")

            image_path = row.get("image_path", "")
            if image_path and not os.path.isabs(image_path):
                image_path = os.path.join(base_dir, image_path)

            date = row.get("date", "")

            capture_time = row.get("time", "")

            self.table.setItem(row_index, 0, QTableWidgetItem(str(vehicle_id)))

            self.table.setItem(row_index, 1, QTableWidgetItem(str(plate_number)))

            image_label = QLabel()

            image_label.setAlignment(Qt.AlignCenter)

            if image_path and os.path.exists(image_path):

                pixmap = QPixmap(image_path)

                if not pixmap.isNull():

                    image_label.setPixmap(
                        pixmap.scaled(
                            150, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )

            self.table.setCellWidget(row_index, 2, image_label)

            self.table.setItem(row_index, 3, QTableWidgetItem(str(image_path)))

            self.table.setItem(row_index, 4, QTableWidgetItem(str(date)))

            self.table.setItem(row_index, 5, QTableWidgetItem(str(capture_time)))


# ============================================================
# MAIN WINDOW
# ============================================================


class IBVAPWindow(QMainWindow):

    def __init__(self, video_source=0, is_live_source=True):

        super().__init__()

        self.setWindowTitle("IBVAP 2.0 — Intelligent Border Video Analytics")

        self.setMinimumSize(1280, 800)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = YOLO("yolo11n.pt")

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        self.is_live_source = is_live_source

        self.current_source = video_source

        if is_live_source and isinstance(video_source, int):

            self.current_source_kind = "webcam"

        elif is_live_source:

            self.current_source_kind = "stream"

        else:

            self.current_source_kind = "file"

        self.playback_start_time = None

        self.cap = self._open_capture(video_source, is_live_source)

        if self.cap is None:

            raise RuntimeError(f"Could not open video source: " f"{video_source}")

        self.video_fps = self._get_fps(self.cap)

        # ----------------------------------------------------
        # TRACKER
        # ----------------------------------------------------

        self.tracker = CentroidTracker(max_distance=100, max_missing=10)

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        allowed = {
            0: "PERSON",
            2: "CAR",
            3: "MOTORCYCLE",
            5: "BUS",
            7: "TRUCK",
        }

        self.detector = AsyncDetectionWorker(self.model, allowed, conf_threshold=0.45)

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        self.face_id = FaceIdentifier()

        self.face_worker = AsyncFaceWorker(self.face_id)

        # ----------------------------------------------------
        # DATABASE / INTELLIGENCE
        # ----------------------------------------------------

        self.db = Database()

        self.behavior = BehaviorAnalyzer(loiter_time=10, movement_threshold=30)

        self.suspicious = SuspiciousActivity()

        self.events = EventEngine()

        self.alerts = AlertManager()

        self.evidence = EvidenceManager()

        self.night = NightDetector(threshold=70)

        # ----------------------------------------------------
        # ANPR
        # ----------------------------------------------------

        self.anpr = ANPRWorker()

        self.anpr.result_ready.connect(self.handle_anpr_result)

        self.anpr.start()

        # ----------------------------------------------------
        # OBJECT STATE
        # ----------------------------------------------------

        self.vehicle_ids = {}

        self.next_vehicle_id = 1

        self.vehicle_plates = {}

        self.saved_plates = set()

        self.identified_names = set()

        # ----------------------------------------------------
        # ZONE
        # ----------------------------------------------------

        self.zone = None

        self.configure_zone = False

        self.load_zone()

        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        self.event_count = 0

        self.alert_count = 0

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        self.frame_number = 0

        self.last_detection_frame = -999999

        self.last_detection_info = []

        self.last_detection_time = 0

        self.ai_latency_ms = 0

        self.ai_fps = 0

        self.render_fps = 0

        self._fps_last_tick = time.perf_counter()

        # IMPORTANT:
        # This is the source of truth for active tracks.
        # CentroidTracker.update() returns the dictionary.
        self._active_tracks = {}

        # ----------------------------------------------------
        # TRACK HISTORY
        # ----------------------------------------------------

        self.track_history = {}

        self.track_last_seen = {}

        # ----------------------------------------------------
        # FACE / ANPR COOLDOWNS
        # ----------------------------------------------------

        self.last_face_frame = {}

        self.last_anpr_frame = {}

        # ----------------------------------------------------
        # INCIDENTS
        # ----------------------------------------------------

        self.incident_history = []

        self.incident_cooldown = {}

        # ----------------------------------------------------
        # FRAME CACHE
        # ----------------------------------------------------

        self._last_annotated_frame = None

        self._last_frame_wh = None

        self._last_zoom_crop = None

        # ----------------------------------------------------
        # ZOOM
        # ----------------------------------------------------

        self.zoom_level = 1.0

        self.max_zoom = 6.0

        self.zoom_cx = 0.5

        self.zoom_cy = 0.5

        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        self.is_paused = False

        self._paused_at = None

        # ----------------------------------------------------
        # PLATES WINDOW
        # ----------------------------------------------------

        self.plates_window = None

        self.evidence_window = None

        # ----------------------------------------------------
        # MOTION HEATMAP
        # ----------------------------------------------------

        self.heatmap_enabled = False

        self._heatmap_accum = None

        self._pulse_active = False

        os.makedirs("snapshots", exist_ok=True)

        os.makedirs("reports", exist_ok=True)

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.create_ui()

        # ----------------------------------------------------
        # DISPLAY TIMER
        # ----------------------------------------------------

        self.timer = QTimer(self)

        self.timer.timeout.connect(self.update_frame)

        self.timer.start(max(15, int(1000 / DISPLAY_FPS)))

        # ----------------------------------------------------
        # METRICS TIMER
        # ----------------------------------------------------

        self.metrics_timer = QTimer(self)

        self.metrics_timer.timeout.connect(self.refresh_metrics)

        self.metrics_timer.start(500)

    # ========================================================
    # CAPTURE
    # ========================================================

    def _open_capture(self, video_source, is_live_source):

        try:

            if is_live_source:

                cap = LatestFrameCapture(
                    video_source, width=640, height=480, target_fps=30
                )

            else:

                cap = cv2.VideoCapture(video_source)

            if not cap.isOpened():

                return None

            return cap

        except Exception as exc:

            print(f"ERROR opening video source " f"{video_source}: {exc}")

            return None

    def _get_fps(self, cap):

        try:

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps and fps > 0:

                return fps

        except Exception:
            pass

        return 30.0

    def discover_sample_videos(self):

        found = []

        for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):

            found.extend(glob.glob(os.path.join("samples", pattern)))

        return sorted(found)

    # ========================================================
    # UI
    # ========================================================

    def create_ui(self):

        root = QWidget()

        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)

        root_layout.setContentsMargins(0, 0, 0, 0)

        root_layout.setSpacing(0)

        # ====================================================
        # SIDEBAR
        # ====================================================

        sidebar = QFrame()

        sidebar.setObjectName("sidebar")

        sidebar.setFixedWidth(245)

        sidebar_layout = QVBoxLayout(sidebar)

        sidebar_layout.setContentsMargins(18, 22, 18, 18)

        sidebar_layout.setSpacing(6)

        brand_row = QHBoxLayout()

        brand = QLabel("IBVAP")

        brand.setObjectName("brand")

        brand_row.addWidget(brand)

        brand_row.addStretch()

        self.live_dot = QLabel("●")

        self.live_dot.setObjectName("liveDot")

        brand_row.addWidget(self.live_dot)

        sidebar_layout.addLayout(brand_row)

        side_subtitle = QLabel("INTELLIGENT BORDER\n" "VIDEO ANALYTICS PLATFORM")

        side_subtitle.setObjectName("sideSubtitle")

        sidebar_layout.addWidget(side_subtitle)

        sidebar_layout.addSpacing(20)

        self.nav_buttons = {}

        navigation = [
            ("dashboard", "⌂   DASHBOARD"),
            ("cameras", "◉   CAMERAS"),
            ("events", "⚠   EVENTS"),
            ("alerts", "●   ALERTS"),
            ("intelligence", "✦   INTELLIGENCE"),
            ("evidence", "▣   EVIDENCE"),
        ]

        for key, text in navigation:

            button = QPushButton(text)

            button.setObjectName("navButton")

            button.setCheckable(True)

            button.clicked.connect(lambda checked=False, k=key: self.navigate_to(k))

            self.nav_buttons[key] = button

            sidebar_layout.addWidget(button)

        plates_button = QPushButton("▤   NUMBER PLATES")

        plates_button.setObjectName("navButton")

        plates_button.clicked.connect(self.show_captured_plates)

        sidebar_layout.addWidget(plates_button)

        sidebar_layout.addSpacing(18)

        perimeter_label = QLabel("PERIMETER CONTROL")

        perimeter_label.setObjectName("sideSection")

        sidebar_layout.addWidget(perimeter_label)

        self.zone_button = QPushButton("＋  CONFIGURE FENCE")

        self.zone_button.setObjectName("controlButton")

        self.zone_button.clicked.connect(self.start_zone_config)

        sidebar_layout.addWidget(self.zone_button)

        self.done_zone_button = QPushButton("✓  FINISH FENCE")

        self.done_zone_button.setObjectName("successButton")

        self.done_zone_button.clicked.connect(self.finish_zone_config)

        self.done_zone_button.hide()

        sidebar_layout.addWidget(self.done_zone_button)

        reset_button = QPushButton("↺  RESET FENCE")

        reset_button.setObjectName("controlButton")

        reset_button.clicked.connect(self.reset_zone)

        sidebar_layout.addWidget(reset_button)

        sidebar_layout.addSpacing(18)

        reports_label = QLabel("REPORTING")

        reports_label.setObjectName("sideSection")

        sidebar_layout.addWidget(reports_label)

        export_button = QPushButton("⤓  EXPORT SESSION REPORT")

        export_button.setObjectName("exportButton")

        export_button.clicked.connect(self.export_session_report)

        sidebar_layout.addWidget(export_button)

        sidebar_layout.addStretch()

        self.system_status = QLabel("●  SYSTEM ONLINE")

        self.system_status.setObjectName("systemStatus")

        sidebar_layout.addWidget(self.system_status)

        version_label = QLabel("COMMAND CENTER  •  v2.0")

        version_label.setObjectName("versionLabel")

        sidebar_layout.addWidget(version_label)

        # ====================================================
        # CONTENT
        # ====================================================

        content = QWidget()

        content.setObjectName("content")

        content_layout = QVBoxLayout(content)

        content_layout.setContentsMargins(24, 20, 24, 18)

        content_layout.setSpacing(12)

        # ====================================================
        # HEADER
        # ====================================================

        header = QHBoxLayout()

        title_box = QVBoxLayout()

        self.page_title = QLabel("COMMAND CENTER")

        self.page_title.setObjectName("pageTitle")

        title_box.addWidget(self.page_title)

        self.page_subtitle = QLabel(
            "Real-time border surveillance " "with multi-signal intelligence"
        )

        self.page_subtitle.setObjectName("pageSubtitle")

        title_box.addWidget(self.page_subtitle)

        header.addLayout(title_box)

        header.addStretch()

        self.camera_label = QLabel("● SYSTEM LIVE")

        self.camera_label.setObjectName("cameraStatus")

        header.addWidget(self.camera_label)

        content_layout.addLayout(header)

        # ====================================================
        # SOURCE TOOLBAR
        # ====================================================

        toolbar = QFrame()

        toolbar.setObjectName("toolbar")

        toolbar_layout = QHBoxLayout(toolbar)

        toolbar_layout.setContentsMargins(10, 8, 10, 8)

        source_label = QLabel("SOURCE")

        source_label.setObjectName("toolbarLabel")

        toolbar_layout.addWidget(source_label)

        self.source_combo = QComboBox()

        self.source_combo.addItem("Webcam", {"kind": "webcam"})

        self.source_combo.addItem("HTTP / RTSP", {"kind": "stream"})

        for path in self.discover_sample_videos():

            self.source_combo.addItem(
                "Sample: " + os.path.basename(path), {"kind": "file", "path": path}
            )

        self.source_combo.currentIndexChanged.connect(self.on_source_combo_changed)

        toolbar_layout.addWidget(self.source_combo, 1)

        self.source_input = QLineEdit("0")

        self.source_input.setPlaceholderText("Camera index / RTSP URL")

        toolbar_layout.addWidget(self.source_input, 2)

        switch_button = QPushButton("SWITCH")

        switch_button.setObjectName("toolButton")

        switch_button.clicked.connect(self.on_source_apply)

        toolbar_layout.addWidget(switch_button)

        self.pause_button = QPushButton("PAUSE")

        self.pause_button.setObjectName("toolButton")

        self.pause_button.clicked.connect(self.toggle_pause)

        toolbar_layout.addWidget(self.pause_button)

        snapshot_button = QPushButton("SNAPSHOT")

        snapshot_button.setObjectName("toolButton")

        snapshot_button.clicked.connect(self.take_snapshot)

        toolbar_layout.addWidget(snapshot_button)

        self.heatmap_button = QPushButton("HEATMAP")

        self.heatmap_button.setObjectName("toolButton")

        self.heatmap_button.setCheckable(True)

        self.heatmap_button.clicked.connect(self.toggle_heatmap)

        toolbar_layout.addWidget(self.heatmap_button)

        zoom_out = QPushButton("−")

        zoom_out.setObjectName("toolButton")

        zoom_out.clicked.connect(lambda: self.adjust_zoom(-1))

        toolbar_layout.addWidget(zoom_out)

        self.zoom_label = QLabel("1.0x")

        self.zoom_label.setObjectName("zoomLabel")

        toolbar_layout.addWidget(self.zoom_label)

        zoom_in = QPushButton("+")

        zoom_in.setObjectName("toolButton")

        zoom_in.clicked.connect(lambda: self.adjust_zoom(1))

        toolbar_layout.addWidget(zoom_in)

        reset_zoom = QPushButton("RESET")

        reset_zoom.setObjectName("toolButton")

        reset_zoom.clicked.connect(self.reset_zoom)

        toolbar_layout.addWidget(reset_zoom)

        content_layout.addWidget(toolbar)

        # ====================================================
        # MAIN BODY
        # ====================================================

        body = QHBoxLayout()

        body.setSpacing(12)

        # ====================================================
        # VIDEO
        # ====================================================

        video_card = QFrame()

        video_card.setObjectName("videoCard")

        video_layout = QVBoxLayout(video_card)

        video_layout.setContentsMargins(8, 8, 8, 8)

        video_header = QHBoxLayout()

        feed_title = QLabel("LIVE SURVEILLANCE FEED")

        feed_title.setObjectName("sectionTitle")

        video_header.addWidget(feed_title)

        video_header.addStretch()

        self.feed_badge = QLabel("● ANALYTICS ACTIVE")

        self.feed_badge.setObjectName("good")

        video_header.addWidget(self.feed_badge)

        video_layout.addLayout(video_header)

        self.video = VideoLabel()

        self.video.setObjectName("videoFeed")

        self.video.setMinimumSize(760, 460)

        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.video.setText("WAITING FOR SURVEILLANCE FEED...")

        self.video.zone_changed.connect(self.set_zone)

        self.video.zoom_wheel.connect(self.adjust_zoom)

        self.video.recenter_clicked.connect(self.on_video_clicked)

        video_layout.addWidget(self.video, 1)

        self.fence_hint = QLabel("Virtual fence: NOT CONFIGURED")

        self.fence_hint.setObjectName("fenceHint")

        video_layout.addWidget(self.fence_hint)

        body.addWidget(video_card, 4)

        # ====================================================
        # INTELLIGENCE RAIL
        # ====================================================

        intelligence_card = QFrame()

        intelligence_card.setObjectName("intelligenceCard")

        intelligence_card.setMinimumWidth(285)

        intelligence_card.setMaximumWidth(330)

        intelligence_layout = QVBoxLayout(intelligence_card)

        intelligence_layout.setContentsMargins(14, 14, 14, 14)

        intelligence_layout.setSpacing(9)

        intelligence_header = QHBoxLayout()

        intelligence_title = QLabel("LIVE INTELLIGENCE")

        intelligence_title.setObjectName("sectionTitle")

        intelligence_header.addWidget(intelligence_title)

        intelligence_header.addStretch()

        self.threat_badge = QLabel("NOMINAL")

        self.threat_badge.setObjectName("nominal")

        intelligence_header.addWidget(self.threat_badge)

        self.threat_glow = QGraphicsDropShadowEffect(self.threat_badge)

        self.threat_glow.setOffset(0, 0)

        self.threat_glow.setColor(QColor(255, 70, 70, 230))

        self.threat_glow.setBlurRadius(0)

        self.threat_badge.setGraphicsEffect(self.threat_glow)

        self.threat_pulse = QPropertyAnimation(self.threat_glow, b"blurRadius", self)

        self.threat_pulse.setDuration(750)

        self.threat_pulse.setStartValue(6)

        self.threat_pulse.setKeyValueAt(0.5, 30)

        self.threat_pulse.setEndValue(6)

        self.threat_pulse.setLoopCount(-1)

        intelligence_layout.addLayout(intelligence_header)

        radar_title = QLabel("PERIMETER RADAR")

        radar_title.setObjectName("miniTitle")

        intelligence_layout.addWidget(radar_title)

        self.radar = RadarWidget()

        intelligence_layout.addWidget(self.radar)

        self.incident_list = QListWidget()

        self.incident_list.setObjectName("incidentList")

        intelligence_layout.addWidget(self.incident_list, 1)

        signal_title = QLabel("SIGNAL STATUS")

        signal_title.setObjectName("miniTitle")

        intelligence_layout.addWidget(signal_title)

        self.signal_label = QLabel("YOLO 0   •   TRACKS 0")

        self.signal_label.setObjectName("muted")

        intelligence_layout.addWidget(self.signal_label)

        reason_title = QLabel("EXPLAINABLE EVENT")

        reason_title.setObjectName("miniTitle")

        intelligence_layout.addWidget(reason_title)

        self.reason_label = QLabel(
            "No active anomaly.\n" "System is observing the perimeter."
        )

        self.reason_label.setWordWrap(True)

        self.reason_label.setObjectName("reasonBox")

        intelligence_layout.addWidget(self.reason_label)

        performance_title = QLabel("PERFORMANCE")

        performance_title.setObjectName("miniTitle")

        intelligence_layout.addWidget(performance_title)

        self.performance_label = QLabel(
            "RENDER      0 FPS\n"
            "AI          0 FPS\n"
            "INFERENCE   0 ms\n"
            "TRACKS      0"
        )

        self.performance_label.setObjectName("performance")

        intelligence_layout.addWidget(self.performance_label)

        body.addWidget(intelligence_card, 1)

        content_layout.addLayout(body, 1)

        # ====================================================
        # KPI BAR
        # ====================================================

        kpi_layout = QHBoxLayout()

        self.track_value = QLabel("0")

        self.event_value = QLabel("0")

        self.alert_value = QLabel("0")

        self.identity_value = QLabel("0")

        self.mode_value = QLabel("DAY")

        self.fps_value = QLabel("0")

        self.ai_value = QLabel("0")

        cards = [
            ("ACTIVE TRACKS", self.track_value),
            ("EVENTS", self.event_value),
            ("ALERTS", self.alert_value),
            ("IDENTIFIED", self.identity_value),
            ("ENVIRONMENT", self.mode_value),
            ("RENDER FPS", self.fps_value),
            ("AI FPS", self.ai_value),
        ]

        for title, value in cards:

            kpi_layout.addWidget(self.create_card(title, value), 1)

        content_layout.addLayout(kpi_layout)

        root_layout.addWidget(sidebar)

        root_layout.addWidget(content, 1)

        self.setStyleSheet(self.stylesheet())

        self.nav_buttons["dashboard"].setChecked(True)

        self.update_camera_label()

        if self.zone:

            self.video.set_zone(self.zone)

            self.update_fence_hint()

    # ========================================================
    # KPI CARD
    # ========================================================

    def create_card(self, title, value):

        card = QFrame()

        card.setObjectName("kpiCard")

        layout = QVBoxLayout(card)

        layout.setContentsMargins(12, 9, 12, 9)

        title_label = QLabel(title)

        title_label.setObjectName("kpiTitle")

        value.setObjectName("kpiValue")

        layout.addWidget(title_label)

        layout.addWidget(value)

        return card

    # ========================================================
    # STYLE
    # ========================================================

    def stylesheet(self):

        return """

        QMainWindow,
        QWidget#content {
            background:#070b13;
        }

        QFrame#sidebar {
            background:#0b111c;
            border-right:1px solid #1c2839;
        }

        QLabel#brand {
            color:#f8fafc;
            font-size:30px;
            font-weight:900;
            letter-spacing:2px;
        }

        QLabel#liveDot {
            color:#38e58b;
            font-size:18px;
        }

        QLabel#sideSubtitle {
            color:#526176;
            font-size:9px;
            font-weight:800;
            letter-spacing:1.2px;
        }

        QLabel#sideSection {
            color:#45546a;
            font-size:9px;
            font-weight:900;
            padding:8px 4px;
        }

        QPushButton#navButton {
            color:#8391a6;
            background:transparent;
            border:1px solid transparent;
            border-radius:8px;
            text-align:left;
            padding:11px;
            font-size:11px;
            font-weight:700;
        }

        QPushButton#navButton:hover {
            color:#e2e8f0;
            background:#111b2a;
            border-color:#1d2b3e;
        }

        QPushButton#navButton:checked {
            color:#ffffff;
            background:#152236;
            border-color:#263954;
        }

        QPushButton#controlButton,
        QPushButton#successButton {
            color:#b9c6d7;
            background:#101a29;
            border:1px solid #26364d;
            border-radius:7px;
            padding:9px;
            font-size:10px;
            font-weight:700;
        }

        QPushButton#successButton {
            color:#70efaa;
            background:#0d2418;
            border-color:#17623b;
        }

        QLabel#systemStatus {
            color:#54e697;
            background:#0b2117;
            border:1px solid #145c37;
            border-radius:7px;
            padding:8px;
            font-size:10px;
            font-weight:800;
        }

        QLabel#versionLabel {
            color:#45546a;
            font-size:8px;
            font-weight:700;
        }

        QLabel#pageTitle {
            color:#f8fafc;
            font-size:26px;
            font-weight:850;
        }

        QLabel#pageSubtitle {
            color:#617087;
            font-size:11px;
        }

        QLabel#cameraStatus {
            color:#55e79b;
            background:#0b2017;
            border:1px solid #145c37;
            border-radius:16px;
            padding:8px 13px;
            font-size:10px;
            font-weight:800;
        }

        QFrame#toolbar {
            background:#0d1521;
            border:1px solid #1e2b3e;
            border-radius:9px;
        }

        QLabel#toolbarLabel {
            color:#5e6f87;
            font-size:9px;
            font-weight:900;
        }

        QComboBox,
        QLineEdit {
            color:#d9e3ef;
            background:#101a29;
            border:1px solid #26364d;
            border-radius:6px;
            padding:7px;
        }

        QComboBox QAbstractItemView {
            background:#101a29;
            color:#d9e3ef;
            selection-background-color:#1a304b;
        }

        QPushButton#toolButton {
            color:#b9c6d7;
            background:#101a29;
            border:1px solid #26364d;
            border-radius:6px;
            padding:7px 10px;
            font-size:9px;
            font-weight:800;
        }

        QPushButton#toolButton:hover {
            background:#17263a;
            border-color:#38506d;
        }

        QFrame#videoCard {
            background:#0a111b;
            border:1px solid #1e2b3e;
            border-radius:10px;
        }

        QLabel#videoFeed {
            background:#02050a;
            border:1px solid #203047;
            border-radius:7px;
            color:#45546a;
            font-size:12px;
        }

        QLabel#sectionTitle {
            color:#cbd6e5;
            font-size:10px;
            font-weight:900;
            letter-spacing:1px;
        }

        QLabel#good {
            color:#4ade80;
            font-size:9px;
            font-weight:900;
        }

        QLabel#fenceHint {
            color:#64758b;
            font-size:9px;
            padding:3px;
        }

        QFrame#intelligenceCard {
            background:#0d1521;
            border:1px solid #1e2b3e;
            border-radius:10px;
        }

        QListWidget#incidentList {
            background:#09111c;
            border:1px solid #1c2a3c;
            border-radius:7px;
            color:#cbd5e1;
            outline:0;
        }

        QListWidget#incidentList::item {
            padding:8px 6px;
            border-bottom:1px solid #162335;
        }

        QListWidget#incidentList::item:selected {
            background:#14253a;
        }

        QLabel#miniTitle {
            color:#506077;
            font-size:8px;
            font-weight:900;
            letter-spacing:1px;
        }

        QLabel#muted {
            color:#7b8ba1;
            font-size:9px;
        }

        QLabel#reasonBox {
            color:#b8c6d7;
            background:#09121e;
            border:1px solid #1d2b3e;
            border-radius:6px;
            padding:9px;
            font-size:9px;
        }

        QLabel#performance {
            color:#74869e;
            background:#09111c;
            border:1px solid #1b293b;
            border-radius:6px;
            padding:8px;
            font-size:9px;
        }

        QLabel#nominal {
            color:#62e99e;
            background:#0b2017;
            border:1px solid #165d39;
            border-radius:10px;
            padding:4px 8px;
            font-size:8px;
            font-weight:900;
        }

        QFrame#kpiCard {
            background:#0d1521;
            border:1px solid #1e2b3e;
            border-radius:8px;
        }

        QLabel#kpiTitle {
            color:#53637a;
            font-size:8px;
            font-weight:900;
            letter-spacing:.7px;
        }

        QLabel#kpiValue {
            color:#f5f7fa;
            font-size:19px;
            font-weight:850;
        }

        QLabel#zoomLabel {
            color:#8fa0b5;
            font-size:9px;
            font-weight:800;
            padding:0 4px;
        }

        QPushButton#toolButton:checked {
            color:#0b0f16;
            background:#f7b955;
            border-color:#f7b955;
        }

        QPushButton#exportButton {
            color:#7fd4ff;
            background:#0d1e2c;
            border:1px solid #1c4a63;
            border-radius:7px;
            padding:9px;
            font-size:10px;
            font-weight:800;
        }

        QPushButton#exportButton:hover {
            background:#123146;
            border-color:#2c6d8e;
        }

        QWidget#radarWidget {
            background:transparent;
        }

        QScrollBar:vertical {
            background:#0a111c;
            width:9px;
            border-radius:4px;
        }

        QScrollBar::handle:vertical {
            background:#233b52;
            border-radius:4px;
            min-height:24px;
        }

        QScrollBar::handle:vertical:hover {
            background:#345070;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height:0;
        }

        QFrame#kpiCard:hover {
            border-color:#345070;
        }
        """

    # ========================================================
    # NAVIGATION
    # ========================================================

    def navigate_to(self, section):

        for key, button in self.nav_buttons.items():

            button.setChecked(key == section)

        titles = {
            "dashboard": (
                "COMMAND CENTER",
                "Real-time border surveillance " "with multi-signal intelligence",
            ),
            "cameras": ("CAMERA CONTROL", "Switch and inspect surveillance sources"),
            "events": ("EVENT MONITOR", "Current surveillance events"),
            "alerts": ("ALERT MONITOR", "Generated security alerts"),
            "intelligence": (
                "INTELLIGENCE",
                "Detection, tracking and explainable signals",
            ),
            "evidence": ("EVIDENCE", "Captured surveillance evidence"),
        }

        title, subtitle = titles.get(section, titles["dashboard"])

        self.page_title.setText(title)

        self.page_subtitle.setText(subtitle)

        # These panels are intentionally still the
        # live dashboard rather than fake pages.
        if section == "cameras":

            self.source_combo.setFocus()

        elif section == "events":

            self.show_status_dialog("EVENT MONITOR", "events")

        elif section == "alerts":

            self.show_status_dialog("ALERT MONITOR", "alerts")

        elif section == "evidence":

            self.show_evidence_gallery()

    # ========================================================
    # STATUS DIALOG
    # ========================================================

    def show_status_dialog(self, title, kind):

        dialog = QDialog(self)

        dialog.setWindowTitle("IBVAP — " + title)

        dialog.setMinimumSize(540, 350)

        dialog.setStyleSheet("""
            QDialog {
                background:#0b1220;
            }

            QLabel#dialogTitle {
                color:#f8fafc;
                font-size:20px;
                font-weight:800;
            }

            QLabel#dialogMuted {
                color:#64748b;
                font-size:11px;
            }

            QLabel#dialogValue {
                color:#f8fafc;
                font-size:28px;
                font-weight:800;
            }

            QFrame#dialogCard {
                background:#111b2c;
                border:1px solid #26354d;
                border-radius:10px;
            }

            QPushButton {
                background:#172337;
                color:#cbd5e1;
                border:1px solid #2b3b54;
                border-radius:7px;
                padding:8px 16px;
            }

            QPushButton:hover {
                background:#20304a;
            }
            """)

        layout = QVBoxLayout(dialog)

        title_label = QLabel(title)

        title_label.setObjectName("dialogTitle")

        layout.addWidget(title_label)

        description = QLabel(
            "Live IBVAP command-center status. "
            "The underlying event, alert and database "
            "modules remain connected."
        )

        description.setObjectName("dialogMuted")

        description.setWordWrap(True)

        layout.addWidget(description)

        grid = QGridLayout()

        tracks = len(self._active_tracks)

        if kind == "events":

            items = [
                ("Events generated", self.event_count),
                ("Active tracks", tracks),
                ("Virtual fence", ("Configured" if self.zone else "Not configured")),
                ("Environment", self.mode_value.text()),
            ]

        else:

            items = [
                ("Alerts generated", self.alert_count),
                ("Current tracks", tracks),
                ("Security state", ("ATTENTION" if self.alert_count else "NOMINAL")),
                ("Environment", self.mode_value.text()),
            ]

        for index, (label, value) in enumerate(items):

            card = QFrame()

            card.setObjectName("dialogCard")

            card_layout = QVBoxLayout(card)

            label_widget = QLabel(label.upper())

            label_widget.setObjectName("dialogMuted")

            value_widget = QLabel(str(value))

            value_widget.setObjectName("dialogValue")

            card_layout.addWidget(label_widget)

            card_layout.addWidget(value_widget)

            grid.addWidget(card, index // 2, index % 2)

        layout.addLayout(grid)

        layout.addStretch()

        close_button = QPushButton("CLOSE")

        close_button.clicked.connect(dialog.accept)

        layout.addWidget(close_button, alignment=Qt.AlignRight)

        dialog.exec()

    # ========================================================
    # SOURCE
    # ========================================================

    def on_source_combo_changed(self, index):

        data = self.source_combo.currentData()

        if data is None:
            return

        kind = data.get("kind")

        if kind == "webcam":

            self.source_input.show()

            self.source_input.setPlaceholderText("Camera index e.g. 0")

            text = self.source_input.text().strip()

            if not text or text.startswith(("http", "rtsp")):

                self.source_input.setText("0")

        elif kind == "stream":

            self.source_input.show()

            self.source_input.setPlaceholderText("rtsp:// or https:// stream URL")

            text = self.source_input.text().strip()

            if not text or text.isdigit():

                self.source_input.setText("rtsp://")

        else:

            self.source_input.hide()

    def on_source_apply(self):

        data = self.source_combo.currentData()

        if data is None:
            return

        kind = data.get("kind")

        if kind == "webcam":

            text = self.source_input.text().strip()

            try:

                source = int(text)

            except ValueError:

                QMessageBox.warning(
                    self, "Invalid Camera", "Camera index must be a number."
                )

                return

            self.switch_source(source, True)

        elif kind == "stream":

            source = self.source_input.text().strip()

            if not source:

                QMessageBox.warning(self, "Invalid Stream", "Enter a stream URL.")

                return

            self.switch_source(source, True)

        else:

            source = data.get("path")

            self.switch_source(source, False)

    def switch_source(self, source, is_live):

        new_cap = self._open_capture(source, is_live)

        if new_cap is None:

            QMessageBox.warning(self, "Source Error", f"Could not open:\n{source}")

            return

        self.timer.stop()

        try:

            self.cap.release()

        except Exception:
            pass

        self.cap = new_cap

        self.current_source = source

        self.is_live_source = is_live

        if is_live and isinstance(source, int):

            self.current_source_kind = "webcam"

        elif is_live:

            self.current_source_kind = "stream"

        else:

            self.current_source_kind = "file"

        self.video_fps = self._get_fps(self.cap)

        self.reset_runtime_state()

        self.timer.start(max(15, int(1000 / DISPLAY_FPS)))

        self.update_camera_label()

    def reset_runtime_state(self):

        self.tracker = CentroidTracker(max_distance=100, max_missing=10)

        self.behavior = BehaviorAnalyzer(loiter_time=10, movement_threshold=30)

        self.vehicle_ids = {}

        self.next_vehicle_id = 1

        self.vehicle_plates = {}

        self.saved_plates = set()

        self.identified_names = set()

        self.track_history = {}

        self.track_last_seen = {}

        self.last_face_frame = {}

        self.last_anpr_frame = {}

        self.incident_history = []

        self.incident_cooldown = {}

        self._active_tracks = {}

        self.last_detection_info = []

        self.last_detection_frame = -999999

        self.frame_number = 0

        self._last_annotated_frame = None

        self._last_frame_wh = None

        self._last_zoom_crop = None

        self.playback_start_time = None

        self.zoom_level = 1.0

        self.zoom_cx = 0.5

        self.zoom_cy = 0.5

        self._fps_last_tick = time.perf_counter()

        self.render_fps = 0

        self.is_paused = False

        self._paused_at = None

        self.pause_button.setText("PAUSE")

        self.zoom_label.setText("1.0x")

    def update_camera_label(self):

        if self.current_source_kind == "webcam":

            text = f"WEBCAM " f"{self.current_source} " f"● LIVE"

        elif self.current_source_kind == "stream":

            text = "STREAM ● LIVE"

        else:

            text = os.path.basename(str(self.current_source)) + " ● PLAYBACK"

        self.camera_label.setText(text)

        if hasattr(self, "feed_badge") and not self.configure_zone:

            self.feed_badge.setText("● ANALYTICS ACTIVE")

    # ========================================================
    # PAUSE
    # ========================================================

    def toggle_pause(self):

        self.is_paused = not self.is_paused

        if self.is_paused:

            self._paused_at = time.time()

            self.pause_button.setText("RESUME")

        else:

            if self._paused_at is not None and self.playback_start_time is not None:

                self.playback_start_time += time.time() - self._paused_at

            self._paused_at = None

            self.pause_button.setText("PAUSE")

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def take_snapshot(self):

        if self._last_annotated_frame is None:

            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        path = os.path.join("snapshots", f"snapshot_{timestamp}.jpg")

        cv2.imwrite(path, self._last_annotated_frame)

        self.system_status.setText("● SNAPSHOT SAVED")

        QTimer.singleShot(2500, lambda: self.system_status.setText("● SYSTEM ONLINE"))

        print("Snapshot saved:", path)

    # ========================================================
    # ZOOM
    # ========================================================

    def adjust_zoom(self, direction):

        factor = 1.25 if direction > 0 else 1 / 1.25

        self.zoom_level = max(1.0, min(self.max_zoom, self.zoom_level * factor))

        if self.zoom_level <= 1.01:

            self.zoom_level = 1.0

            self.zoom_cx = 0.5

            self.zoom_cy = 0.5

        self.zoom_label.setText(f"{self.zoom_level:.1f}x")

    def reset_zoom(self):

        self.zoom_level = 1.0

        self.zoom_cx = 0.5

        self.zoom_cy = 0.5

        self.zoom_label.setText("1.0x")

    def on_video_clicked(self, label_x, label_y):

        if self.zoom_level <= 1.0 or self._last_frame_wh is None:

            return

        frame_w, frame_h = self._last_frame_wh

        label_w = self.video.width()

        label_h = self.video.height()

        scale = min(label_w / frame_w, label_h / frame_h)

        display_w = frame_w * scale

        display_h = frame_h * scale

        offset_x = (label_w - display_w) / 2

        offset_y = (label_h - display_h) / 2

        x = (label_x - offset_x) / scale

        y = (label_y - offset_y) / scale

        x = max(0, min(frame_w - 1, x))

        y = max(0, min(frame_h - 1, y))

        zx, zy, zw, zh = self._last_zoom_crop or (0, 0, frame_w, frame_h)

        native_x = zx + (x / frame_w) * zw

        native_y = zy + (y / frame_h) * zh

        self.zoom_cx = native_x / frame_w

        self.zoom_cy = native_y / frame_h

    def compute_zoom_crop(self, frame_w, frame_h):

        if self.zoom_level <= 1:

            return (0, 0, frame_w, frame_h)

        crop_w = frame_w / self.zoom_level

        crop_h = frame_h / self.zoom_level

        cx = self.zoom_cx * frame_w

        cy = self.zoom_cy * frame_h

        x1 = cx - crop_w / 2

        y1 = cy - crop_h / 2

        x1 = max(0, min(frame_w - crop_w, x1))

        y1 = max(0, min(frame_h - crop_h, y1))

        return (int(x1), int(y1), max(1, int(crop_w)), max(1, int(crop_h)))

    def render_display_frame(self, frame):

        frame_h, frame_w = frame.shape[:2]

        zx, zy, zw, zh = self.compute_zoom_crop(frame_w, frame_h)

        self._last_frame_wh = (frame_w, frame_h)

        self._last_zoom_crop = (zx, zy, zw, zh)

        if self.zoom_level > 1:

            displayed = frame[zy : zy + zh, zx : zx + zw]

            displayed = cv2.resize(
                displayed, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR
            )

        else:

            displayed = frame

        rgb = cv2.cvtColor(displayed, cv2.COLOR_BGR2RGB)

        image = QImage(
            rgb.data, frame_w, frame_h, 3 * frame_w, QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        transform = (
            Qt.SmoothTransformation
            if (self.is_paused or self.zoom_level > 1)
            else Qt.FastTransformation
        )

        self.video.setPixmap(
            pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, transform)
        )

    # ========================================================
    # ZONE
    # ========================================================

    def load_zone(self):

        try:

            with open("zone.json", "r", encoding="utf-8") as file:

                data = json.load(file)

            self.zone = (data["x1"], data["y1"], data["x2"], data["y2"])

        except Exception:

            self.zone = None

    def save_zone(self):

        if self.zone is None:
            return

        x1, y1, x2, y2 = self.zone

        data = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

        try:

            with open("zone.json", "w", encoding="utf-8") as file:

                json.dump(data, file)

        except Exception as exc:

            print("Could not save zone:", exc)

    def start_zone_config(self):

        self.configure_zone = True

        self.video.configure_zone = True

        self.zone_button.hide()

        self.done_zone_button.show()

        self.feed_badge.setText("● DRAWING VIRTUAL FENCE")

        self.feed_badge.setStyleSheet("""
            color:#fbbf24;
            font-size:10px;
            font-weight:800;
            """)

        self.fence_hint.setText("Draw a rectangle on the feed.")

    def finish_zone_config(self):

        self.configure_zone = False

        self.video.configure_zone = False

        self.done_zone_button.hide()

        self.zone_button.show()

        self.save_zone()

        self.feed_badge.setText("● ANALYTICS ACTIVE")

        self.feed_badge.setStyleSheet("""
            color:#4ade80;
            font-size:9px;
            font-weight:900;
            """)

        self.update_fence_hint()

    def reset_zone(self):

        self.zone = None

        self.video.set_zone(None)

        self.update_fence_hint()

        if os.path.exists("zone.json"):

            try:

                os.remove("zone.json")

            except Exception as exc:

                print("Could not remove zone:", exc)

    def set_zone(self, zone):

        self.zone = zone

        self.video.set_zone(zone)

        self.save_zone()

        self.update_fence_hint()

    def update_fence_hint(self):

        if self.zone:

            x1, y1, x2, y2 = self.zone

            self.fence_hint.setText(
                "Virtual fence: ACTIVE  •  "
                f"({x1}, {y1}) → ({x2}, {y2})  •  "
                "intrusion monitoring enabled"
            )

            self.fence_hint.setStyleSheet("""
                color:#f87171;
                font-size:9px;
                padding:3px;
                """)

        else:

            self.fence_hint.setText("Virtual fence: NOT CONFIGURED")

            self.fence_hint.setStyleSheet("""
                color:#64758b;
                font-size:9px;
                padding:3px;
                """)

    def map_zone_to_frame(self, frame):

        if self.zone is None:
            return None

        x1, y1, x2, y2 = self.zone

        label_w = self.video.width()

        label_h = self.video.height()

        frame_h, frame_w = frame.shape[:2]

        scale = min(label_w / frame_w, label_h / frame_h)

        display_w = frame_w * scale

        display_h = frame_h * scale

        offset_x = (label_w - display_w) / 2

        offset_y = (label_h - display_h) / 2

        fx1 = int((x1 - offset_x) / scale)

        fy1 = int((y1 - offset_y) / scale)

        fx2 = int((x2 - offset_x) / scale)

        fy2 = int((y2 - offset_y) / scale)

        fx1 = max(0, min(frame_w - 1, fx1))

        fy1 = max(0, min(frame_h - 1, fy1))

        fx2 = max(0, min(frame_w - 1, fx2))

        fy2 = max(0, min(frame_h - 1, fy2))

        return (fx1, fy1, fx2, fy2)

    # ========================================================
    # TRACK MATCHING
    # ========================================================

    def match_track(self, center, label, tracks):

        best_id = None

        best_distance = 60

        for object_id, track_center in tracks.items():

            try:

                track_label = self.tracker.get_label(object_id)

            except Exception:

                continue

            if track_label != label:

                continue

            distance = math.dist(center, track_center)

            if distance < best_distance:

                best_distance = distance

                best_id = object_id

        return best_id

    # ========================================================
    # BORDER APPROACH
    # ========================================================

    def detect_boundary_approach(self, object_id, center, zone):

        if zone is None or object_id is None:

            return False

        history = self.track_history.setdefault(
            object_id, __import__("collections").deque(maxlen=12)
        )

        history.append(center)

        if len(history) < 5:

            return False

        zx1, zy1, zx2, zy2 = zone

        closest_x = max(zx1, min(center[0], zx2))

        closest_y = max(zy1, min(center[1], zy2))

        current_distance = math.dist(center, (closest_x, closest_y))

        old = history[0]

        old_x = max(zx1, min(old[0], zx2))

        old_y = max(zy1, min(old[1], zy2))

        old_distance = math.dist(old, (old_x, old_y))

        improvement = old_distance - current_distance

        return improvement > 8 and current_distance > 0

    # ========================================================
    # RISK
    # ========================================================

    def calculate_risk(self, state, inside, speed, approaching):

        score = 0

        reasons = []

        if inside:

            score += 55

            reasons.append("protected-zone intrusion")

        if state == "LOITERING":

            score += 25

            reasons.append("loitering")

        if speed > 80:

            score += 15

            reasons.append("rapid movement")

        if approaching:

            score += 20

            reasons.append("approaching protected boundary")

        return (min(100, score), reasons)

    # ========================================================
    # INCIDENT
    # ========================================================

    def register_incident(self, object_id, reason, score):

        key = (object_id, reason)

        now = time.time()

        previous = self.incident_cooldown.get(key, 0)

        if now - previous < 5:

            return

        self.incident_cooldown[key] = now

        timestamp = time.strftime("%H:%M:%S")

        item = QListWidgetItem(
            f"{timestamp}   "
            f"TRACK-{object_id:03d}\n"
            f"{reason.upper()}   •   "
            f"RISK {score}"
        )

        self.incident_list.insertItem(0, item)

        while self.incident_list.count() > 12:

            self.incident_list.takeItem(self.incident_list.count() - 1)

        self.incident_history.append((now, object_id, reason, score))

        self.reason_label.setText(
            f"TRACK-{object_id:03d}\n" f"{reason}\n" f"Risk score: {score}/100"
        )

        if score >= 75:

            self.threat_badge.setText("CRITICAL")

            self.threat_badge.setStyleSheet("""
                color:#ffb4b4;
                background:#3a1116;
                border:1px solid #8d2732;
                border-radius:10px;
                padding:4px 8px;
                font-size:8px;
                font-weight:900;
                """)

            if not self._pulse_active:

                self.threat_pulse.start()

                self._pulse_active = True

        else:

            self.threat_badge.setText("WATCH")

            self.threat_badge.setStyleSheet("""
                color:#ffd58a;
                background:#35260d;
                border:1px solid #795719;
                border-radius:10px;
                padding:4px 8px;
                font-size:8px;
                font-weight:900;
                """)

            self.threat_pulse.stop()

            self.threat_glow.setBlurRadius(0)

            self._pulse_active = False

    # ========================================================
    # EVENT
    # ========================================================

    def create_behavior_event(self, object_id, reason, frame):

        try:

            event = self.events.create_event(object_id, reason)

        except Exception as exc:

            print("Event engine error:", exc)

            return

        if event is None:

            return

        try:

            evidence_path = self.evidence.save_frame(event["event_id"], frame)

            event["evidence_path"] = evidence_path

            self.db.save_event(event)

            self.alerts.send_alert(event)

        except Exception as exc:

            print("Event persistence error:", exc)

        self.event_count += 1

        self.alert_count += 1

        self.event_value.setText(str(self.event_count))

        self.alert_value.setText(str(self.alert_count))

    # ========================================================
    # MAIN FRAME LOOP
    # ========================================================

    def update_frame(self):

        # ----------------------------------------------------
        # PAUSED
        # ----------------------------------------------------

        if self.is_paused:

            if self._last_annotated_frame is not None:

                self.render_display_frame(self._last_annotated_frame)

            return

        # ----------------------------------------------------
        # FILE PLAYBACK
        # ----------------------------------------------------

        if not self.is_live_source:

            if self.playback_start_time is None:

                self.playback_start_time = time.time()

            elapsed = time.time() - self.playback_start_time

            target_frame = int(elapsed * self.video_fps)

            try:

                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            except Exception:

                current_frame = target_frame

            # Do not repeatedly seek.
            # Only catch up if playback is substantially behind.
            if target_frame > current_frame + max(2, int(self.video_fps)):

                try:

                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                except Exception:
                    pass

        # ----------------------------------------------------
        # READ
        # ----------------------------------------------------

        ret, frame = self.cap.read()

        if not ret:

            if self.is_live_source:

                return

            try:

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            except Exception:
                pass

            self.playback_start_time = time.time()

            self.tracker = CentroidTracker(max_distance=100, max_missing=10)

            self._active_tracks = {}

            return

        self.frame_number += 1

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        try:

            mode = self.night.get_mode(frame)

            brightness = self.night.get_brightness(frame)

        except Exception:

            mode = "DAY"

            brightness = 0

        self.mode_value.setText(str(mode))

        # ----------------------------------------------------
        # YOLO THROTTLING
        # ----------------------------------------------------

        should_detect = self.frame_number - self.last_detection_frame >= YOLO_INTERVAL

        if should_detect:

            self.last_detection_time = time.perf_counter()

            try:

                self.detector.submit(frame)

                self.last_detection_frame = self.frame_number

            except Exception as exc:

                print("Detector submit error:", exc)

        # ----------------------------------------------------
        # LATEST DETECTION RESULT
        # ----------------------------------------------------

        try:

            detection_info = self.detector.get_latest()

        except Exception as exc:

            print("Detector result error:", exc)

            detection_info = []

        if detection_info is not None:

            if detection_info:

                self.last_detection_info = detection_info

                if self.last_detection_time:

                    self.ai_latency_ms = (
                        time.perf_counter() - self.last_detection_time
                    ) * 1000

                    if self.ai_latency_ms > 0:

                        self.ai_fps = 1000 / self.ai_latency_ms

        # ----------------------------------------------------
        # TRACKING
        #
        # IMPORTANT:
        # `tracks` is returned by tracker.update().
        # We DO NOT access tracker.tracks.
        # ----------------------------------------------------

        detections = []

        for item in self.last_detection_info:

            try:

                center, x1, y1, x2, y2, label, confidence = item

                detections.append((center, label))

            except Exception:

                continue

        try:

            tracks = self.tracker.update(detections)

        except Exception as exc:

            print("Tracker error:", exc)

            tracks = {}

        # This is the FIX for the previous error.
        self._active_tracks = tracks

        # ----------------------------------------------------
        # BEHAVIOR
        # ----------------------------------------------------

        try:

            states = self.behavior.analyze(tracks)

        except Exception as exc:

            print("Behavior error:", exc)

            states = {}

        # ----------------------------------------------------
        # VEHICLE IDS
        # ----------------------------------------------------

        for object_id in tracks:

            try:

                label = self.tracker.get_label(object_id)

            except Exception:

                continue

            if label != "PERSON" and object_id not in self.vehicle_ids:

                self.vehicle_ids[object_id] = self.next_vehicle_id

                self.next_vehicle_id += 1

        # ----------------------------------------------------
        # TRACK HISTORY
        # ----------------------------------------------------

        for object_id, center in tracks.items():

            self.track_last_seen[object_id] = time.time()

            if object_id not in (self.track_history):

                from collections import deque

                self.track_history[object_id] = deque(maxlen=12)

            self.track_history[object_id].append(center)

        # ----------------------------------------------------
        # MOTION HEATMAP
        #
        # Accumulated regardless of whether the overlay is
        # currently visible, so toggling it on shows history
        # rather than starting from a blank slate.
        # ----------------------------------------------------

        try:

            self.update_heatmap(frame, tracks)

        except Exception as exc:

            print("Heatmap error:", exc)

        # ----------------------------------------------------
        # ZONE
        # ----------------------------------------------------

        zone = self.map_zone_to_frame(frame)

        if zone:

            zx1, zy1, zx2, zy2 = zone

            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (30, 50, 220), 2)

            cv2.putText(
                frame,
                "PROTECTED BORDER ZONE",
                (zx1, max(20, zy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 50, 220),
                2,
            )

        # ----------------------------------------------------
        # MATCH + DRAW DETECTIONS
        # ----------------------------------------------------

        matched_tracks = set()

        yolo_count = len(self.last_detection_info)

        for item in self.last_detection_info:

            try:

                center, x1, y1, x2, y2, label, confidence = item

            except Exception:

                continue

            object_id = self.match_track(center, label, tracks)

            if object_id is not None:

                matched_tracks.add(object_id)

            # --------------------------------------------
            # ZONE STATE
            # --------------------------------------------

            inside = False

            if zone and object_id is not None:

                tx, ty = tracks[object_id]

                inside = zone[0] <= tx <= zone[2] and zone[1] <= ty <= zone[3]

            # --------------------------------------------
            # BEHAVIOR
            # --------------------------------------------

            state = (
                states.get(object_id, "UNKNOWN")
                if object_id is not None
                else "DETECTION"
            )

            try:

                speed = (
                    self.tracker.get_speed(object_id) if object_id is not None else 0
                )

            except Exception:

                speed = 0

            # --------------------------------------------
            # APPROACH
            # --------------------------------------------

            approaching = (
                self.detect_boundary_approach(object_id, center, zone)
                if object_id is not None
                else False
            )

            # --------------------------------------------
            # RISK
            # --------------------------------------------

            risk, reasons = self.calculate_risk(state, inside, speed, approaching)

            if object_id is not None and risk >= 50:

                reason = " + ".join(reasons)

                self.register_incident(object_id, reason, risk)

            # --------------------------------------------
            # BOX COLOR
            # --------------------------------------------

            if inside:

                box_color = (30, 50, 230)

            elif approaching:

                box_color = (0, 190, 255)

            else:

                box_color = (40, 210, 100)

            # --------------------------------------------
            # OBJECT LABEL
            # --------------------------------------------

            if object_id is not None:

                if label == "PERSON":

                    object_text = "PERSON"

                else:

                    vehicle_number = self.vehicle_ids.get(object_id, 0)

                    object_text = f"VEHICLE-" f"{vehicle_number}"

                track_text = f"TRACK-{object_id:03d}"

            else:

                object_text = label

                track_text = "DETECTION"

            # --------------------------------------------
            # BOX
            # --------------------------------------------

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # --------------------------------------------
            # PRIMARY LABEL
            # --------------------------------------------

            cv2.putText(
                frame,
                (f"{object_text}  " f"{track_text}"),
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                box_color,
                2,
            )

            # --------------------------------------------
            # STATE
            # --------------------------------------------

            cv2.putText(
                frame,
                (f"{state} | " f"{confidence:.2f}"),
                (x1, min(frame.shape[0] - 8, y2 + 17)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (225, 235, 245),
                1,
            )

            # --------------------------------------------
            # APPROACH WARNING
            # --------------------------------------------

            if approaching:

                cv2.putText(
                    frame,
                    "APPROACHING BORDER",
                    (x1, min(frame.shape[0] - 25, y2 + 34)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (0, 190, 255),
                    2,
                )

            # --------------------------------------------
            # PLATE
            # --------------------------------------------

            if (
                object_id is not None
                and label != "PERSON"
                and object_id in self.vehicle_plates
            ):

                plate = self.vehicle_plates[object_id]

                cv2.putText(
                    frame,
                    str(plate),
                    (x1, max(18, y1 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (80, 230, 130),
                    2,
                )

        # ----------------------------------------------------
        # TRACK CONTINUITY
        #
        # This is what lets the UI remain responsive even
        # when YOLO is only running at ~8 FPS.
        # ----------------------------------------------------

        for object_id, center in tracks.items():

            if object_id in matched_tracks:

                continue

            x, y = center

            cv2.circle(frame, center, 5, (255, 180, 40), -1)

            cv2.putText(
                frame,
                (f"TRACK-{object_id:03d}" " • TRACK CONTINUITY"),
                (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 180, 40),
                2,
            )

        # ----------------------------------------------------
        # ANPR
        #
        # IMPORTANT FIX:
        #
        # ANPRWorker expects:
        #
        # (
        #     vehicle_id,
        #     (x1, y1, x2, y2)
        # )
        #
        # NOT:
        #
        # (
        #     vehicle_id,
        #     x1,
        #     y1,
        #     x2,
        #     y2
        # )
        # ----------------------------------------------------

        if self.frame_number % ANPR_INTERVAL == 0:

            vehicle_boxes = []

            for item in self.last_detection_info:

                try:

                    center, x1, y1, x2, y2, label, confidence = item

                except Exception:

                    continue

                if label == "PERSON":

                    continue

                object_id = self.match_track(center, label, tracks)

                if object_id is None:

                    continue

                # Do not run ANPR again after
                # a plate has been captured.
                if object_id in (self.vehicle_plates):

                    continue

                last_frame = self.last_anpr_frame.get(object_id, -999999)

                # Per-track cooldown.
                if self.frame_number - last_frame < 6:

                    continue

                # ----------------------------------------
                # CORRECT ANPR STRUCTURE
                # ----------------------------------------

                vehicle_boxes.append((object_id, (x1, y1, x2, y2)))

            if vehicle_boxes:

                try:

                    submitted = self.anpr.submit(frame, vehicle_boxes)

                    if submitted:
                        for object_id, _ in vehicle_boxes:
                            self.last_anpr_frame[object_id] = self.frame_number

                except Exception as exc:

                    print("ANPR submit error:", exc)

        # ----------------------------------------------------
        # FACE RECOGNITION
        # ----------------------------------------------------

        if self.frame_number % FACE_INTERVAL == 0:

            for item in self.last_detection_info:

                try:

                    center, x1, y1, x2, y2, label, confidence = item

                except Exception:

                    continue

                if label != "PERSON":

                    continue

                object_id = self.match_track(center, label, tracks)

                if object_id is None:

                    continue

                last_frame = self.last_face_frame.get(object_id, -999999)

                if self.frame_number - last_frame < 10:

                    continue

                crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]

                if crop.size == 0:

                    continue

                try:

                    self.face_worker.submit(object_id, crop)

                    self.last_face_frame[object_id] = self.frame_number

                except Exception as exc:

                    print("Face submit error:", exc)

        # ----------------------------------------------------
        # FACE CLEANUP
        # ----------------------------------------------------

        try:

            if hasattr(self.face_worker, "cleanup"):

                self.face_worker.cleanup(set(tracks.keys()))

        except Exception as exc:

            print("Face cleanup error:", exc)

        # ----------------------------------------------------
        # BEHAVIOR / EVENTS
        # ----------------------------------------------------

        for object_id, center in tracks.items():

            x, y = center

            try:

                label = self.tracker.get_label(object_id)

            except Exception:

                label = "UNKNOWN"

            state = states.get(object_id, "UNKNOWN")

            try:

                speed = self.tracker.get_speed(object_id)

            except Exception:

                speed = 0

            inside = False

            if zone:

                inside = zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]

            intrusion = "INTRUSION" if inside else "SAFE"

            try:

                suspicious, reason = self.suspicious.check(state, intrusion)

            except Exception:

                suspicious = False

                reason = ""

            if suspicious:

                self.create_behavior_event(object_id, reason, frame)

            # ------------------------------------------------
            # FACE RESULT
            # ------------------------------------------------

            try:

                name = self.face_worker.get_name(object_id)

            except Exception:

                name = None

            if name:

                if not name.startswith("UNKNOWN"):

                    self.identified_names.add((object_id, name))

                cv2.putText(
                    frame,
                    str(name),
                    (x + 10, y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (70, 230, 130),
                    2,
                )

        # ----------------------------------------------------
        # TOP HUD
        # ----------------------------------------------------

        cv2.rectangle(frame, (0, 0), (270, 78), (5, 12, 22), -1)

        cv2.putText(
            frame,
            "IBVAP 2.0",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 210, 240),
            2,
        )

        cv2.putText(
            frame,
            (f"{mode}  |  " f"VIS {brightness:.0f}"),
            (14, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (210, 220, 230),
            1,
        )

        cv2.putText(
            frame,
            (f"TRACKS {len(tracks)}"),
            (14, 67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (130, 180, 200),
            1,
        )

        # ----------------------------------------------------
        # UI COUNTERS
        # ----------------------------------------------------

        self.track_value.setText(str(len(tracks)))

        known_names = {
            name for _, name in self.identified_names if not name.startswith("UNKNOWN")
        }

        self.identity_value.setText(str(len(known_names)))

        self.event_value.setText(str(self.event_count))

        self.alert_value.setText(str(self.alert_count))

        self.signal_label.setText(
            f"YOLO {yolo_count}" f"   •   " f"TRACKS {len(tracks)}"
        )

        try:

            frame_h, frame_w = frame.shape[:2]

            self.radar.update_data(tracks, zone, frame_w, frame_h)

        except Exception as exc:

            print("Radar update error:", exc)

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        self._last_annotated_frame = frame.copy()

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        self.render_display_frame(frame)

        # ----------------------------------------------------
        # RENDER FPS
        # ----------------------------------------------------

        now = time.perf_counter()

        delta = now - self._fps_last_tick

        self._fps_last_tick = now

        if delta > 0:

            instant_fps = 1 / delta

            if self.render_fps == 0:

                self.render_fps = instant_fps

            else:

                self.render_fps = 0.9 * self.render_fps + 0.1 * instant_fps

    # ========================================================
    # METRICS
    # ========================================================

    def refresh_metrics(self):

        # FIX:
        # Never use self.tracker.tracks.
        # CentroidTracker doesn't have that attribute.
        active_tracks = self._active_tracks

        self.fps_value.setText(f"{self.render_fps:.1f}")

        self.ai_value.setText(f"{self.ai_fps:.1f}")

        self.performance_label.setText(
            f"RENDER      "
            f"{self.render_fps:.1f} FPS\n"
            f"AI          "
            f"{self.ai_fps:.1f} FPS\n"
            f"INFERENCE   "
            f"{self.ai_latency_ms:.0f} ms\n"
            f"TRACKS      "
            f"{len(active_tracks)}"
        )

        # Automatically return the threat
        # indicator to nominal after inactivity.

        if self.incident_history:

            last_time = self.incident_history[-1][0]

            if time.time() - last_time > 6:

                self.threat_badge.setText("NOMINAL")

                self.threat_badge.setStyleSheet("""
                    color:#62e99e;
                    background:#0b2017;
                    border:1px solid #165d39;
                    border-radius:10px;
                    padding:4px 8px;
                    font-size:8px;
                    font-weight:900;
                    """)

                if self._pulse_active:

                    self.threat_pulse.stop()

                    self.threat_glow.setBlurRadius(0)

                    self._pulse_active = False

    # ========================================================
    # ANPR RESULT
    # ========================================================

    def handle_anpr_result(self, result):

        try:

            object_id = result["vehicle_id"]

            plate = result["plate"]

            crop = result["crop"]

            confidence = result["confidence"]

        except Exception as exc:

            print("Invalid ANPR result:", exc)

            return

        if object_id not in (self.vehicle_ids):

            return

        if object_id in (self.vehicle_plates):

            return

        vehicle_id = self.vehicle_ids[object_id]

        if vehicle_id in (self.saved_plates):

            return

        try:

            image_path = self.anpr.save_plate(vehicle_id, plate, crop, confidence)

        except Exception as exc:

            print("Plate save error:", exc)

            return

        self.vehicle_plates[object_id] = plate

        self.saved_plates.add(vehicle_id)

        print()
        print("===================================")
        print("PLATE CAPTURED")
        print("Vehicle ID:", vehicle_id)
        print("Plate:", plate)
        print("Image:", image_path)
        print("===================================")
        print()

        if self.plates_window is not None:

            self.plates_window.load_plates()

    # ========================================================
    # PLATES
    # ========================================================

    def show_captured_plates(self):

        if self.plates_window is None:

            self.plates_window = PlatesWindow(self)

        self.plates_window.load_plates()

        self.plates_window.show()

        self.plates_window.raise_()

        self.plates_window.activateWindow()

    # ========================================================
    # EVIDENCE GALLERY
    # ========================================================

    def show_evidence_gallery(self):

        if self.evidence_window is None:

            self.evidence_window = EvidenceGalleryWindow(self)

        self.evidence_window.load_evidence()

        self.evidence_window.show()

        self.evidence_window.raise_()

        self.evidence_window.activateWindow()

    # ========================================================
    # MOTION HEATMAP
    # ========================================================

    def toggle_heatmap(self):

        self.heatmap_enabled = self.heatmap_button.isChecked()

    def update_heatmap(self, frame, tracks):

        h, w = frame.shape[:2]

        if self._heatmap_accum is None or self._heatmap_accum.shape[:2] != (h, w):

            self._heatmap_accum = np.zeros((h, w), dtype=np.float32)

        self._heatmap_accum *= 0.965

        for x, y in tracks.values():

            if 0 <= x < w and 0 <= y < h:

                cv2.circle(self._heatmap_accum, (int(x), int(y)), 34, 1.0, -1)

        if not self.heatmap_enabled:
            return

        blurred = cv2.GaussianBlur(self._heatmap_accum, (0, 0), 15)

        max_val = float(blurred.max())

        if max_val <= 0.05:
            return

        normalized = np.clip(blurred / max_val * 255, 0, 255).astype(np.uint8)

        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        mask = (normalized > 12).astype(np.uint8)

        mask3 = cv2.merge([mask, mask, mask]).astype(bool)

        blended = cv2.addWeighted(frame, 1.0, colored, 0.45, 0)

        frame[mask3] = blended[mask3]

    # ========================================================
    # SESSION REPORT
    # ========================================================

    def export_session_report(self):

        try:

            events = self.db.get_events()

        except Exception:

            events = []

        try:

            sightings = self.db.get_face_sightings()

        except Exception:

            sightings = []

        plates_rows = []

        if os.path.exists("plates.csv"):

            try:

                with open("plates.csv", "r", newline="", encoding="utf-8") as file:

                    plates_rows = list(csv.DictReader(file))

            except Exception:

                plates_rows = []

        known_names = {
            name for _, name in self.identified_names if not name.startswith("UNKNOWN")
        }

        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

        def rows_html(headers, data_rows):

            head = "".join(f"<th>{h}</th>" for h in headers)

            body_rows = []

            for row in data_rows:

                cells = "".join(f"<td>{value}</td>" for value in row)

                body_rows.append(f"<tr>{cells}</tr>")

            body = (
                "".join(body_rows)
                if body_rows
                else (
                    f'<tr><td colspan="{len(headers)}" class="empty">No records</td></tr>'
                )
            )

            return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

        events_table = rows_html(
            ["Event ID", "Track", "Type", "Severity", "Time", "Status"],
            [
                (
                    e["event_id"],
                    e["track_id"],
                    e["event_type"],
                    e["severity"],
                    e["timestamp"],
                    e["status"],
                )
                for e in events[:200]
            ],
        )

        plates_table = rows_html(
            ["Vehicle ID", "Plate", "Date", "Time"],
            [
                (
                    r.get("vehicle_id", ""),
                    r.get("plate_number", ""),
                    r.get("date", ""),
                    r.get("time", ""),
                )
                for r in plates_rows[-200:]
            ],
        )

        sightings_table = rows_html(
            ["Track", "Name", "Time"],
            [(s["track_id"], s["name"], s["timestamp"]) for s in sightings[:200]],
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>IBVAP Session Report — {generated_at}</title>
<style>
    body {{ background:#070b13; color:#d9e3ef; font-family:Segoe UI,Arial,sans-serif; padding:32px; }}
    h1 {{ color:#f8fafc; font-size:26px; margin-bottom:4px; }}
    .muted {{ color:#7b8ba1; font-size:12px; margin-bottom:24px; }}
    .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:28px; }}
    .card {{ background:#0d1521; border:1px solid #1e2b3e; border-radius:10px; padding:14px 20px; min-width:140px; }}
    .card .label {{ color:#53637a; font-size:10px; font-weight:700; letter-spacing:.5px; }}
    .card .value {{ color:#f5f7fa; font-size:24px; font-weight:800; margin-top:4px; }}
    h2 {{ color:#cbd6e5; font-size:14px; letter-spacing:1px; margin-top:32px; border-bottom:1px solid #1e2b3e; padding-bottom:8px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:12px; }}
    th {{ text-align:left; color:#93a4bb; background:#121d2e; padding:8px; }}
    td {{ padding:7px 8px; border-bottom:1px solid #162335; color:#cbd5e1; }}
    td.empty {{ color:#526176; text-align:center; }}
</style>
</head>
<body>
    <h1>IBVAP — Session Intelligence Report</h1>
    <div class="muted">Generated {generated_at} &nbsp;•&nbsp; Source: {self.current_source} ({self.current_source_kind})</div>
    <div class="cards">
        <div class="card"><div class="label">TOTAL EVENTS</div><div class="value">{self.event_count}</div></div>
        <div class="card"><div class="label">TOTAL ALERTS</div><div class="value">{self.alert_count}</div></div>
        <div class="card"><div class="label">ACTIVE TRACKS</div><div class="value">{len(self._active_tracks)}</div></div>
        <div class="card"><div class="label">IDENTIFIED PERSONS</div><div class="value">{len(known_names)}</div></div>
        <div class="card"><div class="label">PLATES CAPTURED</div><div class="value">{len(plates_rows)}</div></div>
        <div class="card"><div class="label">ENVIRONMENT</div><div class="value">{self.mode_value.text()}</div></div>
    </div>
    <h2>EVENT LOG</h2>
    {events_table}
    <h2>NUMBER PLATE CAPTURES</h2>
    {plates_table}
    <h2>FACE SIGHTINGS</h2>
    {sightings_table}
</body>
</html>"""

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        path = os.path.abspath(
            os.path.join("reports", f"session_report_{timestamp}.html")
        )

        try:

            with open(path, "w", encoding="utf-8") as file:

                file.write(html)

        except Exception as exc:

            QMessageBox.warning(
                self, "Export Failed", f"Could not write report:\n{exc}"
            )

            return

        self.system_status.setText("● REPORT EXPORTED")

        QTimer.singleShot(2500, lambda: self.system_status.setText("● SYSTEM ONLINE"))

        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        try:

            self.timer.stop()

        except Exception:
            pass

        try:

            self.metrics_timer.stop()

        except Exception:
            pass

        try:

            self.detector.stop()

        except Exception as exc:

            print("Detector stop error:", exc)

        try:

            self.face_worker.stop()

        except Exception as exc:

            print("Face worker stop error:", exc)

        try:

            self.anpr.stop()

        except Exception as exc:

            print("ANPR stop error:", exc)

        try:

            self.cap.release()

        except Exception:
            pass

        event.accept()


# ============================================================
# DEFAULT SOURCE
# ============================================================


def default_startup_source():

    samples = sorted(glob.glob(os.path.join("samples", "*.mp4")))

    if samples:

        return (samples[0], False)

    return (0, True)


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    default_source, default_is_live = default_startup_source()

    window = IBVAPWindow(video_source=default_source, is_live_source=default_is_live)

    window.show()

    sys.exit(app.exec())
