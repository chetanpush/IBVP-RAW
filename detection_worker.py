import queue
import threading
import time


class AsyncDetectionWorker:
    """
    Low-latency YOLO inference worker.

    Design goals:
    - Never block the UI/video thread.
    - Keep only the newest frame waiting for inference.
    - Drop stale frames instead of building a backlog.
    - Preserve the existing UI interface:
          submit(frame)
          get_latest()
          stop()
    - Measure inference latency and FPS.
    """

    def __init__(self, model, allowed_classes, conf_threshold=0.45, max_queue=1):

        self.model = model

        self.allowed_classes = allowed_classes

        self.conf_threshold = conf_threshold

        # ----------------------------------------------------
        # LATEST-FRAME QUEUE
        # ----------------------------------------------------
        #
        # We deliberately keep the queue tiny.
        #
        # In surveillance:
        #
        # OLD FRAME
        #     ↓
        # NEW FRAME
        #
        # Processing an old frame is worse than dropping it.
        #
        # This prevents:
        #
        # camera → 30 FPS
        # YOLO   → 8 FPS
        #
        # from creating seconds of latency.
        # ----------------------------------------------------

        self._queue = queue.Queue(maxsize=max_queue)

        # ----------------------------------------------------
        # THREAD STATE
        # ----------------------------------------------------

        self._lock = threading.Lock()

        self._stop_event = threading.Event()

        self._busy = False

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="IBVAP-YOLO"
        )

        # ----------------------------------------------------
        # LATEST RESULT
        # ----------------------------------------------------

        self._latest_detection_info = []

        self._latest_result_frame = -1

        self._result_timestamp = 0.0

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        self._last_inference_ms = 0.0

        self._inference_fps = 0.0

        self._inference_count = 0

        self._dropped_frames = 0

        self._submitted_frames = 0

        # ----------------------------------------------------
        # FRAME AGE
        # ----------------------------------------------------

        self._frame_counter = 0

        self._latest_submitted_id = -1

        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        self._thread.start()

    # ========================================================
    # WORKER LOOP
    # ========================================================

    def _run(self):

        while not self._stop_event.is_set():

            try:

                packet = self._queue.get(timeout=0.1)

            except queue.Empty:

                continue

            try:

                frame_id, frame = packet

                start_time = time.perf_counter()

                detection_info = self._run_inference(frame)

                elapsed = time.perf_counter() - start_time

                inference_ms = elapsed * 1000.0

                # ------------------------------------------------
                # RESULT UPDATE
                # ------------------------------------------------

                with self._lock:

                    self._latest_detection_info = detection_info

                    self._latest_result_frame = frame_id

                    self._result_timestamp = time.perf_counter()

                    self._last_inference_ms = inference_ms

                    self._inference_count += 1

                    # Exponential moving average.
                    if self._inference_fps <= 0:

                        self._inference_fps = 1.0 / max(elapsed, 0.0001)

                    else:

                        current_fps = 1.0 / max(elapsed, 0.0001)

                        self._inference_fps = (
                            self._inference_fps * 0.8 + current_fps * 0.2
                        )

            except Exception as exc:

                print("YOLO WORKER ERROR:", exc)

            finally:

                with self._lock:

                    self._busy = False

                try:

                    self._queue.task_done()

                except Exception:
                    pass

    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    def _run_inference(self, frame):

        detection_info = []

        try:

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------
            #
            # Keep verbose=False so the terminal doesn't
            # constantly receive YOLO output.
            # ------------------------------------------------

            results = self.model(frame, verbose=False, conf=self.conf_threshold)

            for result in results:

                if result.boxes is None:

                    continue

                for box in result.boxes:

                    try:

                        cls_id = int(box.cls[0])

                        confidence = float(box.conf[0])

                    except Exception:

                        continue

                    # ------------------------------------------------
                    # CLASS FILTER
                    # ------------------------------------------------

                    if cls_id not in self.allowed_classes:

                        continue

                    # ------------------------------------------------
                    # CONFIDENCE FILTER
                    # ------------------------------------------------

                    if confidence < self.conf_threshold:

                        continue

                    try:

                        coordinates = box.xyxy[0].tolist()

                        x1 = int(coordinates[0])

                        y1 = int(coordinates[1])

                        x2 = int(coordinates[2])

                        y2 = int(coordinates[3])

                    except Exception:

                        continue

                    # ------------------------------------------------
                    # SANITY CHECK
                    # ------------------------------------------------

                    if x2 <= x1:

                        continue

                    if y2 <= y1:

                        continue

                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    label = self.allowed_classes[cls_id]

                    detection_info.append((center, x1, y1, x2, y2, label, confidence))

            return detection_info

        except Exception as exc:

            print("YOLO INFERENCE ERROR:", exc)

            return []

    # ========================================================
    # SUBMIT
    # ========================================================

    def submit(self, frame):
        """
        Non-blocking frame submission.

        The UI never waits for YOLO.

        If YOLO is already processing a frame,
        the new frame is skipped.

        If a stale frame is waiting in the queue,
        it is removed and replaced with the newest frame.
        """

        if frame is None:

            return

        with self._lock:

            self._submitted_frames += 1

            self._frame_counter += 1

            frame_id = self._frame_counter

            self._latest_submitted_id = frame_id

            # ------------------------------------------------
            # DO NOT BLOCK
            # ------------------------------------------------

            if self._busy:

                self._dropped_frames += 1

                return

            self._busy = True

        # ----------------------------------------------------
        # FRAME COPY
        # ----------------------------------------------------
        #
        # The camera/display thread may reuse or overwrite
        # its numpy array. Therefore inference gets its own
        # safe copy.
        # ----------------------------------------------------

        try:

            frame_copy = frame.copy()

        except Exception as exc:

            with self._lock:

                self._busy = False

            print("Frame copy error:", exc)

            return

        # ----------------------------------------------------
        # QUEUE
        # ----------------------------------------------------

        try:

            self._queue.put_nowait((frame_id, frame_copy))

        except queue.Full:

            # ------------------------------------------------
            # If a frame somehow accumulated in the queue,
            # remove the old one and insert the newest one.
            # ------------------------------------------------

            try:

                old_packet = self._queue.get_nowait()

                try:

                    self._queue.task_done()

                except Exception:
                    pass

                with self._lock:

                    self._dropped_frames += 1

            except queue.Empty:

                pass

            try:

                self._queue.put_nowait((frame_id, frame_copy))

            except queue.Full:

                with self._lock:

                    self._busy = False

                    self._dropped_frames += 1

    # ========================================================
    # GET LATEST
    # ========================================================

    def get_latest(self):
        """
        Returns the latest completed detection list.

        This function NEVER waits for YOLO.
        """

        with self._lock:

            return list(self._latest_detection_info)

    # ========================================================
    # PERFORMANCE INFO
    # ========================================================

    def get_stats(self):
        """
        Optional performance information.

        Existing ui.py does not need this,
        but the command center can use it later.
        """

        with self._lock:

            return {
                "busy": self._busy,
                "inference_ms": self._last_inference_ms,
                "inference_fps": self._inference_fps,
                "inference_count": self._inference_count,
                "submitted_frames": self._submitted_frames,
                "dropped_frames": self._dropped_frames,
                "latest_result_frame": self._latest_result_frame,
                "latest_submitted_frame": self._latest_submitted_id,
            }

    # ========================================================
    # INFERENCE LATENCY
    # ========================================================

    def get_latency_ms(self):

        with self._lock:

            return self._last_inference_ms

    # ========================================================
    # INFERENCE FPS
    # ========================================================

    def get_fps(self):

        with self._lock:

            return self._inference_fps

    # ========================================================
    # DROPPED FRAME COUNT
    # ========================================================

    def get_dropped_frames(self):

        with self._lock:

            return self._dropped_frames

    # ========================================================
    # RESET STATISTICS
    # ========================================================

    def reset_stats(self):

        with self._lock:

            self._inference_count = 0

            self._submitted_frames = 0

            self._dropped_frames = 0

            self._last_inference_ms = 0.0

            self._inference_fps = 0.0

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self._stop_event.set()

        # Wake the worker if it is waiting.
        #
        # We don't need to enqueue a fake frame because
        # the timeout in _run() is only 0.1 sec.

        if self._thread.is_alive():

            self._thread.join(timeout=1.0)

        # ----------------------------------------------------
        # CLEAR QUEUE
        # ----------------------------------------------------

        try:

            while True:

                self._queue.get_nowait()

                try:

                    self._queue.task_done()

                except Exception:
                    pass

        except queue.Empty:

            pass
