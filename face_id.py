import os
import pickle
import queue
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceIdentifier:
    """
    Detects and recognizes faces inside person crops coming from the YOLO +
    CentroidTracker pipeline, using InsightFace (ArcFace embeddings) instead
    of dlib - no C++ build tools required on Windows.

    Recognition is throttled per track_id (recognize_every) so a busy scene
    with many tracked people doesn't re-run the model on every single frame -
    the result is cached and reused between checks.

    Faces that don't match the known watchlist are still fingerprinted: each
    gets a persistent "UNKNOWN-###" id based on its face embedding, stored in
    unknown_faces.pkl. If that same stranger reappears - a new track, or even
    a different run of the program - they get matched back to the same id
    instead of being treated as a brand new unknown person.
    """

    def __init__(
        self,
        known_faces_dir="known_faces",
        encodings_path="known_faces.pkl",
        similarity_threshold=0.45,
        unknown_faces_dir="unknown_faces",
        unknown_store_path="unknown_faces.pkl",
        unknown_similarity_threshold=0.45,
        recognize_every=10,
        det_size=(320, 320),
    ):
        self.known_faces_dir = known_faces_dir
        self.encodings_path = encodings_path
        self.similarity_threshold = similarity_threshold

        self.unknown_faces_dir = unknown_faces_dir
        self.unknown_store_path = unknown_store_path
        self.unknown_similarity_threshold = unknown_similarity_threshold

        self.recognize_every = recognize_every

        # ctx_id=0 uses GPU if onnxruntime-gpu is installed and a CUDA device
        # is available, otherwise falls back to CPU automatically.
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)

        self.known_encodings = []  # list of unit-normalized 512-d embeddings
        self.known_names = []
        self._track_cache = {}  # track_id -> (name, frames_since_check)

        self.unknown_encodings = (
            []
        )  # unit-normalized embeddings of strangers seen so far
        self.unknown_ids = []  # int id matching each entry above
        self.next_unknown_id = 1

        self._load_or_build_encodings()
        self._load_unknown_store()

    def _load_or_build_encodings(self):
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
            self.known_encodings = data.get("encodings", [])
            self.known_names = data.get("names", [])
            return

        self.build_encodings()

    def build_encodings(self):
        """
        Scans known_faces_dir for either:
          known_faces/<person_name>/*.jpg   (multiple photos per person), or
          known_faces/<person_name>.jpg     (one photo per person)
        and rebuilds known_faces.pkl. Call this again whenever the watchlist
        changes (enroll_face.py does this automatically).
        """
        encodings, names = [], []

        os.makedirs(self.known_faces_dir, exist_ok=True)

        for entry in os.listdir(self.known_faces_dir):
            entry_path = os.path.join(self.known_faces_dir, entry)

            if os.path.isdir(entry_path):
                person_name = entry
                image_paths = [
                    os.path.join(entry_path, f)
                    for f in os.listdir(entry_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            elif entry.lower().endswith((".jpg", ".jpeg", ".png")):
                person_name = os.path.splitext(entry)[0]
                image_paths = [entry_path]
            else:
                continue

            for image_path in image_paths:
                image = cv2.imread(image_path)

                if image is None:
                    continue

                faces = self.app.get(image)

                if faces:
                    # Largest detected face in the enrollment photo
                    best_face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                    encodings.append(best_face.normed_embedding)
                    names.append(person_name)

        self.known_encodings = encodings
        self.known_names = names

        with open(self.encodings_path, "wb") as f:
            pickle.dump({"encodings": encodings, "names": names}, f)

        return len(names)

    def _load_unknown_store(self):
        os.makedirs(self.unknown_faces_dir, exist_ok=True)

        if not os.path.exists(self.unknown_store_path):
            return

        with open(self.unknown_store_path, "rb") as f:
            data = pickle.load(f)

        self.unknown_encodings = data.get("encodings", [])
        self.unknown_ids = data.get("ids", [])
        self.next_unknown_id = data.get("next_id", 1)

    def _save_unknown_store(self):
        with open(self.unknown_store_path, "wb") as f:
            pickle.dump(
                {
                    "encodings": self.unknown_encodings,
                    "ids": self.unknown_ids,
                    "next_id": self.next_unknown_id,
                },
                f,
            )

    def _save_unknown_snapshot(self, unknown_id, face_crop, max_snapshots=5):
        """Keeps a handful of reference photos per unknown id so you can look
        back later and decide whether to enroll them as a named, known face."""
        if face_crop is None or face_crop.size == 0:
            return

        person_dir = os.path.join(self.unknown_faces_dir, f"UNKNOWN-{unknown_id:03d}")
        os.makedirs(person_dir, exist_ok=True)

        existing = [
            f
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(existing) >= max_snapshots:
            return

        filename = f"{int(time.time() * 1000)}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), face_crop)

    def _identify_or_register_unknown(self, embedding, face_crop):
        """Matches a non-watchlist face against previously seen strangers.
        Returns an "UNKNOWN-###" label, reusing an existing id when the face
        matches one already on file, or minting a new one otherwise."""
        if self.unknown_encodings:
            similarities = np.dot(self.unknown_encodings, embedding)
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            if best_similarity >= self.unknown_similarity_threshold:
                unknown_id = self.unknown_ids[best_idx]
                self._save_unknown_snapshot(unknown_id, face_crop)
                print(
                    f"UNKNOWN-{unknown_id:03d} detected again (similarity {best_similarity:.2f})"
                )
                return f"UNKNOWN-{unknown_id:03d}"

        unknown_id = self.next_unknown_id
        self.next_unknown_id += 1

        self.unknown_encodings.append(embedding)
        self.unknown_ids.append(unknown_id)
        self._save_unknown_snapshot(unknown_id, face_crop)
        self._save_unknown_store()

        print(f"New unknown face registered: UNKNOWN-{unknown_id:03d}")
        return f"UNKNOWN-{unknown_id:03d}"

    @staticmethod
    def _crop_to_bbox(crop_bgr, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, crop_bgr.shape[1]), min(y2, crop_bgr.shape[0])
        return crop_bgr[y1:y2, x1:x2]

    def identify_in_crop(self, crop_bgr):
        """
        Runs face detection + recognition on a single BGR crop (e.g. a YOLO
        person box). Returns a list of (name, confidence, bbox). name is a
        watchlist name when matched, otherwise a persistent "UNKNOWN-###" id.
        confidence is only meaningful for watchlist matches (0.0 for unknowns).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return []

        faces = self.app.get(crop_bgr)

        if not faces:
            return []

        results = []

        for face in faces:
            name = "UNKNOWN"
            confidence = 0.0
            matched_known = False

            if self.known_encodings:
                similarities = np.dot(self.known_encodings, face.normed_embedding)
                best_idx = int(np.argmax(similarities))
                best_similarity = float(similarities[best_idx])

                if best_similarity >= self.similarity_threshold:
                    name = self.known_names[best_idx]
                    confidence = best_similarity
                    matched_known = True

            if not matched_known:
                face_crop = self._crop_to_bbox(crop_bgr, face.bbox)
                name = self._identify_or_register_unknown(
                    face.normed_embedding, face_crop
                )

            results.append((name, confidence, face.bbox))

        return results

    def identify_for_track(self, track_id, crop_bgr):
        """
        Throttled per-track lookup: only runs recognition every
        `recognize_every` calls for a given track_id, returning the cached
        name in between. Returns None until the first check has run.
        """
        cached_name, frames_since_check = self._track_cache.get(
            track_id, (None, self.recognize_every)
        )

        if frames_since_check < self.recognize_every:
            self._track_cache[track_id] = (cached_name, frames_since_check + 1)
            return cached_name

        results = self.identify_in_crop(crop_bgr)
        name = results[0][0] if results else None

        self._track_cache[track_id] = (name, 0)
        return name

    def forget_track(self, track_id):
        self._track_cache.pop(track_id, None)

    def cleanup(self, active_track_ids):
        """Drop cached entries for tracks the tracker has already dropped."""
        stale = [tid for tid in self._track_cache if tid not in active_track_ids]
        for tid in stale:
            self.forget_track(tid)


class AsyncFaceWorker:
    """
    Runs FaceIdentifier recognition on a background thread so model inference
    never blocks the video display loop - the UI thread just submits crops
    and reads back whatever the latest known result is, without waiting.

    At most one recognition job is ever in flight per track_id: submit()
    silently drops the frame if a job for that track is already pending, so
    this also acts as the throttle (a busy track naturally gets checked as
    often as inference speed allows, no fixed frame-count needed).
    """

    def __init__(self, face_identifier, max_queue=4):
        self.face_identifier = face_identifier

        self._queue = queue.Queue(maxsize=max_queue)
        self._results = {}  # track_id -> name
        self._pending = set()  # track_ids with a job currently queued/running
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                track_id, crop = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                results = self.face_identifier.identify_in_crop(crop)
                name = results[0][0] if results else None

                if name is not None:
                    with self._lock:
                        self._results[track_id] = name
            finally:
                with self._lock:
                    self._pending.discard(track_id)
                self._queue.task_done()

    def submit(self, track_id, crop_bgr):
        """Non-blocking. Queues a crop for background recognition unless a
        job for this track is already pending or the queue is full."""
        if crop_bgr is None or crop_bgr.size == 0:
            return

        with self._lock:
            if track_id in self._pending:
                return
            self._pending.add(track_id)

        try:
            self._queue.put_nowait((track_id, crop_bgr.copy()))
        except queue.Full:
            with self._lock:
                self._pending.discard(track_id)

    def get_name(self, track_id):
        """Returns the latest known name for this track, or None if
        recognition hasn't produced a result for it yet."""
        with self._lock:
            return self._results.get(track_id)

    def forget_track(self, track_id):
        with self._lock:
            self._results.pop(track_id, None)
            self._pending.discard(track_id)

    def cleanup(self, active_track_ids):
        with self._lock:
            stale = [tid for tid in self._results if tid not in active_track_ids]
            for tid in stale:
                self._results.pop(tid, None)

    def stop(self):
        self._stop_event.set()
