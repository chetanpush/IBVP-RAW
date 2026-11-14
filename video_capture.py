import platform
import threading

import cv2


class LatestFrameCapture:
    """
    Wraps cv2.VideoCapture for a LIVE source (webcam, or an RTSP/HTTP(S)
    network stream from an IP camera / CCTV feed) with a background thread that continuously reads frames and keeps only the
    newest one.

    Plain cv2.VideoCapture.read() pulls from an internal buffer/queue. If
    your processing loop (YOLO + tracking + drawing) ever takes longer than
    the camera's frame interval, that queue backs up and every read() gives
    you an older and older frame - the feed visibly falls behind and the
    delay keeps growing. This class fixes that: the background thread grabs
    frames as fast as the camera produces them and immediately overwrites
    the stored frame, so read() always returns whatever the camera saw most
    recently, at the cost of silently dropping frames you never got to.

    Do NOT use this for prerecorded video files - for those, read every
    frame in order with a plain cv2.VideoCapture so you don't skip any.
    """

    def __init__(self, source=0, width=640, height=480, target_fps=30):
        is_local_webcam = isinstance(source, int)

        if is_local_webcam and platform.system() == "Windows":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")

        # Forcing resolution/fps only makes sense for a local webcam - a
        # network stream (RTSP/HTTP) already has a fixed resolution set by
        # the camera/encoder, and most IP cameras ignore or reject these.
        if is_local_webcam:
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if target_fps:
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        # Honored by some backends (V4L2, DSHOW) to shrink the internal
        # queue; harmless no-op if the backend ignores it. The background
        # thread below is what actually guarantees "latest frame only".
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._ret = False
        self._frame = None
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()

            with self._lock:
                self._ret = ret
                self._frame = frame

            if not ret:
                break

    def read(self):
        """Non-blocking. Returns (ret, frame) for the most recently
        captured frame - never waits on the camera."""
        with self._lock:
            return self._ret, self._frame

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._stop_event.set()
        self._thread.join(timeout=1)
        self.cap.release()
