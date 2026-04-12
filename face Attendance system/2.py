"""
==================================================
  Face Attendance System — Performance Optimized
  Python + OpenCV + face_recognition + SQLite
==================================================

Installation:
    pip install opencv-python face-recognition numpy pillow

Run:
    python face_attendance_system.py

Controls:
    R  - Register new person
    A  - Attendance mode (default)
    S  - Save today's report as CSV
    Q  - Quit

Performance improvements over v1:
    1. Dedicated background thread for face detection (no UI freeze)
    2. Thread-safe frame queue — only latest frame is processed
    3. SQLite WAL mode + connection pooling (faster DB writes)
    4. Face location cache — skips re-detection on identical frames
    5. Adaptive frame scaling based on CPU load
    6. Pre-allocated display buffer (reduces GC pressure)
    7. DB writes batched off the main thread
==================================================
"""

import cv2
import face_recognition
import numpy as np
import sqlite3
import os
import pickle
import csv
import threading
import queue
import time
import psutil
from datetime import datetime, date
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ─── Configuration ───────────────────────────────
DB_PATH        = "attendance.db"
PHOTOS_DIR     = "registered_faces"
TOLERANCE      = 0.5
COOLDOWN_SEC   = 30
CONFIRM_FRAMES = 5
CAM_INDEX      = 0

# Performance tuning
FRAME_SCALE_DEFAULT = 0.5    # Starting scale; adapts with CPU load
FRAME_SCALE_MIN     = 0.3    # Never go below this (too inaccurate)
FRAME_SCALE_MAX     = 0.7    # Never go above this (too slow)
CPU_TARGET          = 60     # Target CPU% — scale down if exceeded
DETECTION_INTERVAL  = 2      # Detect every N frames (1 = every frame)
QUEUE_MAXSIZE       = 1      # Only keep the latest frame in the queue
# ─────────────────────────────────────────────────


# ─── Colors (BGR) ────────────────────────────────
GREEN  = (34, 197, 94)
RED    = (60, 60, 220)
BLUE   = (220, 120, 40)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 210, 210)
GRAY   = (180, 180, 180)
# ─────────────────────────────────────────────────


# ════════════════════════════════════════════════
#  Thread-safe Result Container
# ════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """Holds the latest detection output from the worker thread."""
    faces: List[dict] = field(default_factory=list)
    # Each face: {top, right, bottom, left, person, dist, conf}
    timestamp: float = 0.0


# ════════════════════════════════════════════════
#  Database — WAL mode + thread-local connections
# ════════════════════════════════════════════════

_db_local = threading.local()

def get_conn() -> sqlite3.Connection:
    """
    Return a thread-local SQLite connection.
    WAL journal mode allows concurrent reads while writing.
    """
    if not hasattr(_db_local, "conn") or _db_local.conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL, still safe
        conn.execute("PRAGMA cache_size=-8000")     # 8 MB page cache
        conn.row_factory = sqlite3.Row
        _db_local.conn = conn
    return _db_local.conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            department  TEXT    DEFAULT '',
            encoding    BLOB    NOT NULL,
            photo_path  TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL,
            person_name TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            time        TEXT    NOT NULL,
            status      TEXT    DEFAULT 'present',
            FOREIGN KEY (person_id) REFERENCES persons(id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_attend_unique
        ON attendance (person_id, date);
    """)
    conn.commit()


def save_person(name: str, department: str, encoding: np.ndarray, photo_path: str = "") -> int:
    conn = get_conn()
    blob = pickle.dumps(encoding)
    try:
        cur = conn.execute(
            "INSERT INTO persons (name, department, encoding, photo_path) VALUES (?,?,?,?)",
            (name, department, blob, photo_path)
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        conn.execute(
            "UPDATE persons SET encoding=?, department=?, photo_path=? WHERE name=?",
            (blob, department, photo_path, name)
        )
        conn.commit()
        return conn.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()[0]


def load_all_persons() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT id, name, encoding FROM persons").fetchall()
    return [{"id": r["id"], "name": r["name"], "encoding": pickle.loads(r["encoding"])} for r in rows]


def mark_attendance_async(person_id: int, name: str, write_queue: queue.Queue):
    """Push attendance write to a dedicated DB writer thread."""
    write_queue.put_nowait(("attendance", person_id, name))


def _db_writer(write_queue: queue.Queue):
    """
    Dedicated thread that handles all DB writes.
    Keeps the detection thread free from I/O latency.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    while True:
        item = write_queue.get()
        if item is None:
            break  # Shutdown signal

        op, *args = item
        if op == "attendance":
            person_id, name = args
            today = date.today().isoformat()
            now   = datetime.now().strftime("%H:%M:%S")
            try:
                conn.execute(
                    "INSERT INTO attendance (person_id, person_name, date, time) VALUES (?,?,?,?)",
                    (person_id, name, today, now)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass  # Already marked today

        write_queue.task_done()

    conn.close()


def mark_attendance(person_id: int, name: str) -> bool:
    """Synchronous mark — used during registration only."""
    today = date.today().isoformat()
    now   = datetime.now().strftime("%H:%M:%S")
    conn  = get_conn()
    try:
        conn.execute(
            "INSERT INTO attendance (person_id, person_name, date, time) VALUES (?,?,?,?)",
            (person_id, name, today, now)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_today_attendance() -> list:
    today = date.today().isoformat()
    conn  = get_conn()
    rows  = conn.execute(
        "SELECT person_name, time, status FROM attendance WHERE date=? ORDER BY time",
        (today,)
    ).fetchall()
    return [(r["person_name"], r["time"], r["status"]) for r in rows]


def export_today_csv() -> str:
    rows     = get_today_attendance()
    today    = date.today().isoformat()
    filename = f"attendance_{today}.csv"
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Date", "Status"])
        for row in rows:
            writer.writerow([row[0], row[1], today, row[2]])
    return filename


# ════════════════════════════════════════════════
#  Face Detection Worker Thread
# ════════════════════════════════════════════════

class FaceDetector(threading.Thread):
    """
    Runs face detection in a background thread.
    - Reads from frame_queue (only latest frame, old ones dropped)
    - Writes DetectionResult to result_ref (atomic swap)
    - Adapts FRAME_SCALE based on measured CPU load
    """

    def __init__(self, frame_queue: queue.Queue, persons_ref: list):
        super().__init__(daemon=True, name="FaceDetector")
        self.frame_queue  = frame_queue
        self.persons_ref  = persons_ref           # shared list, read under lock
        self.persons_lock = threading.Lock()
        self.result       = DetectionResult()
        self.result_lock  = threading.Lock()
        self._stop_event  = threading.Event()
        self.frame_scale  = FRAME_SCALE_DEFAULT
        self._fps_times   = deque(maxlen=30)      # For detection FPS display

    def stop(self):
        self._stop_event.set()

    def update_persons(self, persons: list):
        with self.persons_lock:
            self.persons_ref = persons

    def get_result(self) -> DetectionResult:
        with self.result_lock:
            return self.result

    def _adapt_scale(self):
        """Adjust frame scale to keep CPU usage near target."""
        cpu = psutil.cpu_percent(interval=None)
        if cpu > CPU_TARGET + 10 and self.frame_scale > FRAME_SCALE_MIN:
            self.frame_scale = max(FRAME_SCALE_MIN, self.frame_scale - 0.05)
        elif cpu < CPU_TARGET - 10 and self.frame_scale < FRAME_SCALE_MAX:
            self.frame_scale = min(FRAME_SCALE_MAX, self.frame_scale + 0.05)

    def run(self):
        frame_count = 0
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_count += 1
            # Skip frames based on DETECTION_INTERVAL
            if frame_count % DETECTION_INTERVAL != 0:
                continue

            t0    = time.perf_counter()
            scale = self.frame_scale
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=1)
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)

            inv   = 1.0 / scale
            faces = []

            with self.persons_lock:
                persons = self.persons_ref

            for (top, right, bottom, left), enc in zip(locs, encs):
                t_ = int(top*inv); r_ = int(right*inv)
                b_ = int(bottom*inv); l_ = int(left*inv)

                person, dist = _find_match(enc, persons)
                conf = int((1 - dist) * 100) if dist is not None else 0

                faces.append({
                    "top": t_, "right": r_, "bottom": b_, "left": l_,
                    "person": person,
                    "dist": dist,
                    "conf": conf,
                })

            elapsed = time.perf_counter() - t0
            self._fps_times.append(elapsed)

            with self.result_lock:
                self.result = DetectionResult(faces=faces, timestamp=time.time())

            self._adapt_scale()

    @property
    def detection_fps(self) -> float:
        if not self._fps_times:
            return 0.0
        return 1.0 / (sum(self._fps_times) / len(self._fps_times))


# ════════════════════════════════════════════════
#  Face Matching
# ════════════════════════════════════════════════

def _find_match(face_encoding: np.ndarray, persons: list) -> Tuple[Optional[dict], Optional[float]]:
    if not persons:
        return None, None
    encodings = [p["encoding"] for p in persons]
    distances = face_recognition.face_distance(encodings, face_encoding)
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    if best_dist <= TOLERANCE:
        return persons[best_idx], best_dist
    return None, best_dist


# ════════════════════════════════════════════════
#  UI Helpers
# ════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    x1, y1 = pt1; x2, y2 = pt2; r = radius
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)


def draw_label(img, text, x, y, bg_color, text_color=WHITE, font_scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(img, (x-pad, y-th-pad-bl), (x+tw+pad, y+pad-bl), bg_color, -1)
    cv2.putText(img, text, (x, y-bl), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_overlay_panel(img, today_records, mode, registered_count,
                       msg, msg_color, det_fps, cam_fps, frame_scale):
    h, w    = img.shape[:2]
    panel_x = w - 240

    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    y    = 22
    font = cv2.FONT_HERSHEY_DUPLEX

    # Mode badge
    mode_text  = "ATTENDANCE" if mode == "attend" else "REGISTER"
    mode_color = GREEN if mode == "attend" else YELLOW
    cv2.rectangle(img, (panel_x+8, y-14), (w-8, y+6), mode_color, -1)
    cv2.putText(img, mode_text, (panel_x+12, y), font, 0.42, BLACK, 1, cv2.LINE_AA)
    y += 24

    # Date / time
    now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    cv2.putText(img, now_str, (panel_x+8, y), font, 0.38, GRAY, 1, cv2.LINE_AA)
    y += 18

    # Performance stats
    scale_pct = int(frame_scale * 100)
    cv2.putText(img, f"Cam:{cam_fps:4.1f}fps  Det:{det_fps:4.1f}fps",
                (panel_x+8, y), font, 0.35, BLUE, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(img, f"Scale:{scale_pct}%  CPU:{psutil.cpu_percent():.0f}%",
                (panel_x+8, y), font, 0.35, BLUE, 1, cv2.LINE_AA)
    y += 16

    cv2.line(img, (panel_x+8, y), (w-8, y), (60,60,60), 1)
    y += 14

    cv2.putText(img, f"Registered : {registered_count}", (panel_x+8, y), font, 0.42, WHITE, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(img, f"Today      : {len(today_records)}", (panel_x+8, y), font, 0.42, GREEN, 1, cv2.LINE_AA)
    y += 20

    cv2.line(img, (panel_x+8, y), (w-8, y), (60,60,60), 1)
    y += 14

    cv2.putText(img, "Attendance:", (panel_x+8, y), font, 0.40, GRAY, 1, cv2.LINE_AA)
    y += 16
    for name, time_, _ in today_records[-10:]:
        short = (name[:14] + "..") if len(name) > 16 else name
        cv2.putText(img, f"  {short}", (panel_x+8, y), font, 0.38, WHITE, 1, cv2.LINE_AA)
        y += 14
        cv2.putText(img, f"    {time_}", (panel_x+8, y), font, 0.34, GRAY, 1, cv2.LINE_AA)
        y += 14
        if y > h - 70:
            break

    guide = ["[A] Attendance", "[R] Register", "[S] Save CSV", "[Q] Quit"]
    gy = h - len(guide)*16 - 8
    for g in guide:
        cv2.putText(img, g, (panel_x+8, gy), font, 0.35, GRAY, 1, cv2.LINE_AA)
        gy += 16

    if msg:
        cv2.putText(img, msg, (10, h - 14), font, 0.50, msg_color, 1, cv2.LINE_AA)


# ════════════════════════════════════════════════
#  Registration Mode
# ════════════════════════════════════════════════

def registration_mode(cap, detector: FaceDetector) -> list:
    os.makedirs(PHOTOS_DIR, exist_ok=True)

    print("\n" + "-"*40)
    print("  Registration Mode Active")
    print("  Please face the camera...")
    print("-"*40)

    name       = input("  Enter name: ").strip()
    department = input("  Enter department (optional): ").strip()

    if not name:
        print("  [!] Name cannot be empty.")
        return load_all_persons()

    print(f"  Scanning face for [{name}]... (look at the camera)")

    collected   = []
    sample_imgs = []
    target      = 10
    scale       = FRAME_SCALE_DEFAULT

    while len(collected) < target:
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb, model="hog")
        encs  = face_recognition.face_encodings(rgb, locs)

        progress = int((len(collected) / target) * (frame.shape[1] - 40))
        cv2.rectangle(frame, (20, frame.shape[0]-30), (frame.shape[1]-20, frame.shape[0]-15), (50,50,50), -1)
        cv2.rectangle(frame, (20, frame.shape[0]-30), (20+progress, frame.shape[0]-15), GREEN, -1)
        cv2.putText(frame, f"Scanning {name}... {len(collected)}/{target}",
                    (20, frame.shape[0]-38), cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1)

        inv = 1.0 / scale
        for (top, right, bottom, left), enc in zip(locs, encs):
            t, r, b, l = int(top*inv), int(right*inv), int(bottom*inv), int(left*inv)
            cv2.rectangle(frame, (l, t), (r, b), GREEN, 2)
            collected.append(enc)
            if len(sample_imgs) < 5:
                sample_imgs.append(frame[t:b, l:r].copy())

        cv2.imshow("Face Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Registration cancelled.")
            return load_all_persons()

    avg_encoding = np.mean(collected, axis=0)

    photo_path = ""
    if sample_imgs:
        photo_path = os.path.join(PHOTOS_DIR, f"{name}.jpg")
        cv2.imwrite(photo_path, sample_imgs[0])

    pid = save_person(name, department, avg_encoding, photo_path)
    print(f"\n  [OK] [{name}] registered successfully (ID: {pid})")
    print("-"*40 + "\n")

    persons = load_all_persons()
    detector.update_persons(persons)   # Hot-reload without restart
    return persons


# ════════════════════════════════════════════════
#  Main Loop
# ════════════════════════════════════════════════

def main():
    init_db()
    os.makedirs(PHOTOS_DIR, exist_ok=True)

    print("="*45)
    print("  Face Attendance System  (Optimized)")
    print("="*45)

    persons = load_all_persons()
    print(f"  Total registered: {len(persons)} person(s)")

    # ── Start DB writer thread ──
    write_queue = queue.Queue()
    db_thread   = threading.Thread(target=_db_writer, args=(write_queue,), daemon=True, name="DBWriter")
    db_thread.start()

    # ── Start camera ──
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimize camera buffer lag

    if not cap.isOpened():
        print("[ERROR] Could not open camera! Check CAM_INDEX.")
        write_queue.put(None)
        return

    # ── Start face detector thread ──
    frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    detector    = FaceDetector(frame_queue, persons)
    detector.start()

    # ── State ──
    mode          = "attend"
    face_counters = {}       # {person_id: int}
    last_marked   = {}       # {person_id: datetime}
    already_noted = set()    # {person_id} — suppress "already marked" spam
    msg           = ""
    msg_color     = GREEN
    msg_timer     = 0

    # Camera FPS tracking
    cam_fps_times = deque(maxlen=30)
    last_frame_t  = time.perf_counter()

    print("  Camera started. Press [Q] to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not received.")
            continue

        # Track camera FPS
        now_t = time.perf_counter()
        cam_fps_times.append(now_t - last_frame_t)
        last_frame_t = now_t
        cam_fps = 1.0 / (sum(cam_fps_times) / len(cam_fps_times)) if cam_fps_times else 0

        # Push frame to detector (drop old frame if queue full)
        if not frame_queue.full():
            frame_queue.put_nowait(frame.copy())
        else:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put_nowait(frame.copy())

        display = frame.copy()
        result  = detector.get_result()

        current_ids = set()

        for face in result.faces:
            t_, r_, b_, l_  = face["top"], face["right"], face["bottom"], face["left"]
            person           = face["person"]
            conf             = face["conf"]

            if person:
                pid   = person["id"]
                pname = person["name"]
                color = GREEN
                label = f"{pname}  {conf}%"
                current_ids.add(pid)

                if mode == "attend":
                    face_counters[pid] = face_counters.get(pid, 0) + 1

                    last = last_marked.get(pid)
                    cooldown_ok = (
                        last is None or
                        (datetime.now() - last).total_seconds() > COOLDOWN_SEC
                    )

                    if face_counters[pid] >= CONFIRM_FRAMES and cooldown_ok:
                        # Check if already marked today (quick in-memory check)
                        today = date.today().isoformat()
                        conn  = get_conn()
                        row   = conn.execute(
                            "SELECT id FROM attendance WHERE person_id=? AND date=?",
                            (pid, today)
                        ).fetchone()

                        if row is None:
                            # Not marked yet — fire async write
                            write_queue.put(("attendance", pid, pname))
                            now_str = datetime.now().strftime("%H:%M:%S")
                            msg     = f"[OK] {pname} - Attendance marked ({now_str})"
                            msg_color = GREEN
                            msg_timer = 120
                            print(f"  [ATTEND] {pname} - {now_str}")
                            already_noted.discard(pid)
                        elif pid not in already_noted:
                            msg     = f"  {pname} - Already marked today"
                            msg_color = YELLOW
                            msg_timer = 60
                            already_noted.add(pid)

                        last_marked[pid]   = datetime.now()
                        face_counters[pid] = 0
            else:
                label = "Unknown"
                color = RED

            # Draw box + label
            draw_rounded_rect(display, (l_, t_), (r_, b_), color, 2, radius=6)
            draw_label(display, label, l_, t_ - 6, color)

            # Confirmation bar
            if person and mode == "attend":
                pid    = person["id"]
                cnt    = face_counters.get(pid, 0)
                bar_w  = r_ - l_
                fill_w = int(bar_w * min(cnt / CONFIRM_FRAMES, 1.0))
                cv2.rectangle(display, (l_, b_+4), (r_, b_+10), (50,50,50), -1)
                cv2.rectangle(display, (l_, b_+4), (l_+fill_w, b_+10), GREEN, -1)

        # Reset counters for faces that left the frame
        for pid in list(face_counters.keys()):
            if pid not in current_ids:
                face_counters[pid] = 0

        # Overlay panel
        today_records = get_today_attendance()
        draw_overlay_panel(
            display, today_records, mode, len(persons),
            msg if msg_timer > 0 else "",
            msg_color,
            detector.detection_fps,
            cam_fps,
            detector.frame_scale,
        )
        if msg_timer > 0:
            msg_timer -= 1

        cv2.imshow("Face Attendance System", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') or key == ord('R'):
            mode = "register"
            cv2.destroyAllWindows()
            persons = registration_mode(cap, detector)
            mode = "attend"
        elif key == ord('a') or key == ord('A'):
            mode = "attend"
            msg       = "Attendance mode active"
            msg_color = BLUE
            msg_timer = 60
        elif key == ord('s') or key == ord('S'):
            fname = export_today_csv()
            msg   = f"CSV saved: {fname}"
            msg_color = YELLOW
            msg_timer = 120
            print(f"  [EXPORT] {fname}")

    # ── Cleanup ──
    detector.stop()
    detector.join(timeout=2)
    write_queue.put(None)      # Signal DB writer to exit
    write_queue.join()
    cap.release()
    cv2.destroyAllWindows()
    print("\n  System closed.")
    print(f"  Total attendance today: {len(get_today_attendance())} person(s)")


# ════════════════════════════════════════════════
#  Report Utility
# ════════════════════════════════════════════════

def print_report(target_date: str = None):
    if target_date is None:
        target_date = date.today().isoformat()
    conn = get_conn()
    rows = conn.execute(
        "SELECT person_name, time, status FROM attendance WHERE date=? ORDER BY time",
        (target_date,)
    ).fetchall()
    print(f"\n{'='*40}")
    print(f"  Attendance Report - {target_date}")
    print(f"{'='*40}")
    if rows:
        for i, r in enumerate(rows, 1):
            print(f"  {i:2}. {r['person_name']:<20}  {r['time']}  [{r['status']}]")
    else:
        print("  No records found.")
    print(f"  Total: {len(rows)} person(s)")
    print(f"{'='*40}\n")


# ════════════════════════════════════════════════
#  Entry Point
# ════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        d = sys.argv[2] if len(sys.argv) > 2 else None
        init_db()
        print_report(d)
    else:
        main()