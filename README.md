# IBVP-RAW

AI-assisted video surveillance desktop application for monitoring people and
vehicles. It combines YOLO object detection and tracking with face
identification, automatic number plate recognition (ANPR), behavior analysis,
configurable zones, event alerts, and evidence capture.

## Features

- Live camera, stream, and video playback monitoring
- Person and vehicle detection with object tracking
- Face watchlist enrollment and recurring unknown-face identification
- Number plate detection and OCR
- Loitering, rapid-movement, and zone-intrusion event detection
- Event history, snapshots, reports, and evidence management

## Requirements

- Windows with Python 3.10 or later
- Tesseract OCR installed for number-plate text recognition

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

The YOLO model files (`yolo11n.pt` and `plate_yolo.pt`) are expected in the
project root. Start the application from that directory:

```powershell
python ui.py
```

Or run `run.bat`.

## Local data and privacy

Face images and embeddings, captured evidence, plate records, reports, videos,
the event database, and zone settings are local runtime data and are excluded
from version control. Add authorized enrollment images to `known_faces/` on
your own machine. The application creates its runtime folders and records as
needed.
