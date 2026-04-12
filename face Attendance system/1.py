"""
==================================================
  Face Attendance System
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
==================================================
"""

import cv2
import face_recognition
import numpy as np
import sqlite3
import os
import pickle
import csv
from datetime import datetime, date
from pathlib import Path


# ─── Configuration ───────────────────────────────
DB_PATH        = "attendance.db"
PHOTOS_DIR     = "registered_faces"
TOLERANCE      = 0.5       # Face matching strictness (lower = stricter)
COOLDOWN_SEC   = 30        # Seconds before re-marking the same person
CONFIRM_FRAMES = 5         # Frames required to confirm attendance
CAM_INDEX      = 0         # Camera index (0 = default)
FRAME_SCALE    = 0.5       # Scale factor for faster processing
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
#  Database Setup
# ════════════════════════════════════════════════

def init_db():
    """Create database and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            department  TEXT    DEFAULT '',
            encoding    BLOB    NOT NULL,
            photo_path  TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL,
            person_name TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            time        TEXT    NOT NULL,
            status      TEXT    DEFAULT 'present',
            FOREIGN KEY (person_id) REFERENCES persons(id)
        )
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_attend_unique
        ON attendance (person_id, date)
    """)

    conn.commit()
    conn.close()


def save_person(name: str, department: str, encoding: np.ndarray, photo_path: str = ""):
    """Save a new person to the database. Updates encoding if name already exists."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    blob = pickle.dumps(encoding)
    try:
        cur.execute(
            "INSERT INTO persons (name, department, encoding, photo_path) VALUES (?,?,?,?)",
            (name, department, blob, photo_path)
        )
        conn.commit()
        pid = cur.lastrowid
        return pid
    except sqlite3.IntegrityError:
        # Name already exists — update encoding
        cur.execute(
            "UPDATE persons SET encoding=?, department=?, photo_path=? WHERE name=?",
            (blob, department, photo_path, name)
        )
        conn.commit()
        cur.execute("SELECT id FROM persons WHERE name=?", (name,))
        return cur.fetchone()[0]
    finally:
        conn.close()


def load_all_persons():
    """Load all registered persons with their face encodings."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, encoding FROM persons")
    rows = cur.fetchall()
    conn.close()
    persons = []
    for pid, name, blob in rows:
        enc = pickle.loads(blob)
        persons.append({"id": pid, "name": name, "encoding": enc})
    return persons


def mark_attendance(person_id: int, name: str) -> bool:
    """
    Mark attendance for a person.
    Will not mark twice on the same day.
    Returns True if marked successfully, False if already marked.
    """
    today = date.today().isoformat()
    now   = datetime.now().strftime("%H:%M:%S")
    conn  = sqlite3.connect(DB_PATH)
    cur   = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO attendance (person_id, person_name, date, time) VALUES (?,?,?,?)",
            (person_id, name, today, now)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_today_attendance():
    """Fetch all attendance records for today."""
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    cur   = conn.cursor()
    cur.execute(
        "SELECT person_name, time, status FROM attendance WHERE date=? ORDER BY time",
        (today,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def export_today_csv():
    """Export today's attendance report to a CSV file."""
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
#  Face Matching Helper
# ════════════════════════════════════════════════

def find_match(face_encoding: np.ndarray, persons: list):
    """
    Compare face encoding against all registered persons.
    Returns the closest match and its distance.
    """
    if not persons:
        return None, None

    encodings = [p["encoding"] for p in persons]
    distances = face_recognition.face_distance(encodings, face_encoding)
    best_idx  = int(np.argmin(distances))
    best_dist = distances[best_idx]

    if best_dist <= TOLERANCE:
        return persons[best_idx], best_dist
    return None, best_dist


# ════════════════════════════════════════════════
#  UI Helpers
# ════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0, 90, color, thickness)


def draw_label(img, text, x, y, bg_color, text_color=WHITE, font_scale=0.55, thickness=1):
    """Draw a text label with a filled background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(img, (x-pad, y-th-pad-bl), (x+tw+pad, y+pad-bl), bg_color, -1)
    cv2.putText(img, text, (x, y-bl), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_overlay_panel(img, today_records, mode, registered_count, msg, msg_color):
    """Draw the info panel on the right side of the frame."""
    h, w = img.shape[:2]
    panel_x = w - 230

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    y = 22
    font = cv2.FONT_HERSHEY_DUPLEX

    # Mode badge
    mode_text  = "ATTENDANCE" if mode == "attend" else "REGISTER"
    mode_color = GREEN if mode == "attend" else YELLOW
    cv2.rectangle(img, (panel_x+8, y-14), (w-8, y+6), mode_color, -1)
    cv2.putText(img, mode_text, (panel_x+12, y), font, 0.42, BLACK, 1, cv2.LINE_AA)
    y += 24

    # Date and time
    now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    cv2.putText(img, now_str, (panel_x+8, y), font, 0.38, GRAY, 1, cv2.LINE_AA)
    y += 20

    cv2.line(img, (panel_x+8, y), (w-8, y), (60,60,60), 1)
    y += 14

    # Stats
    cv2.putText(img, f"Registered : {registered_count}", (panel_x+8, y), font, 0.42, WHITE, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(img, f"Today      : {len(today_records)}", (panel_x+8, y), font, 0.42, GREEN, 1, cv2.LINE_AA)
    y += 20

    cv2.line(img, (panel_x+8, y), (w-8, y), (60,60,60), 1)
    y += 14

    # Today's attendance list
    cv2.putText(img, "Attendance:", (panel_x+8, y), font, 0.40, GRAY, 1, cv2.LINE_AA)
    y += 16
    for name, time_, _ in today_records[-10:]:  # Show max 10 entries
        short = (name[:14] + "..") if len(name) > 16 else name
        cv2.putText(img, f"  {short}", (panel_x+8, y), font, 0.38, WHITE, 1, cv2.LINE_AA)
        y += 14
        cv2.putText(img, f"    {time_}", (panel_x+8, y), font, 0.34, GRAY, 1, cv2.LINE_AA)
        y += 14
        if y > h - 60:
            break

    # Hotkey guide at the bottom
    guide = ["[A] Attendance", "[R] Register", "[S] Save CSV", "[Q] Quit"]
    gy = h - len(guide)*16 - 8
    for g in guide:
        cv2.putText(img, g, (panel_x+8, gy), font, 0.35, GRAY, 1, cv2.LINE_AA)
        gy += 16

    # Notification message
    if msg:
        cv2.putText(img, msg, (10, h - 14), font, 0.50, msg_color, 1, cv2.LINE_AA)


# ════════════════════════════════════════════════
#  Registration Mode
# ════════════════════════════════════════════════

def registration_mode(cap, persons):
    """
    Registration mode:
    - Collects 10 frame encodings for reliability
    - Takes name and department input from terminal
    - Saves a photo to disk
    """
    os.makedirs(PHOTOS_DIR, exist_ok=True)

    print("\n" + "-"*40)
    print("  Registration Mode Active")
    print("  Please face the camera...")
    print("-"*40)

    name       = input("  Enter name: ").strip()
    department = input("  Enter department (optional): ").strip()

    if not name:
        print("  [!] Name cannot be empty.")
        return persons

    print(f"  Scanning face for [{name}]... (look at the camera)")

    collected   = []
    sample_imgs = []
    target      = 10   # Number of encodings to collect

    while len(collected) < target:
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb, model="hog")
        encs  = face_recognition.face_encodings(rgb, locs)

        progress = int((len(collected) / target) * (frame.shape[1] - 40))
        cv2.rectangle(frame, (20, frame.shape[0]-30), (frame.shape[1]-20, frame.shape[0]-15), (50,50,50), -1)
        cv2.rectangle(frame, (20, frame.shape[0]-30), (20+progress, frame.shape[0]-15), GREEN, -1)
        cv2.putText(frame, f"Scanning {name}... {len(collected)}/{target}",
                    (20, frame.shape[0]-38), cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1)

        scale = 1.0 / FRAME_SCALE
        for (top, right, bottom, left), enc in zip(locs, encs):
            t,r,b,l = int(top*scale), int(right*scale), int(bottom*scale), int(left*scale)
            cv2.rectangle(frame, (l,t), (r,b), GREEN, 2)
            collected.append(enc)
            if len(sample_imgs) < 5:
                sample_imgs.append(frame[t:b, l:r].copy())

        cv2.imshow("Face Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Registration cancelled.")
            return persons

    # Compute average encoding for better accuracy
    avg_encoding = np.mean(collected, axis=0)

    # Save the first captured photo
    photo_path = ""
    if sample_imgs:
        photo_path = os.path.join(PHOTOS_DIR, f"{name}.jpg")
        cv2.imwrite(photo_path, sample_imgs[0])

    pid = save_person(name, department, avg_encoding, photo_path)
    print(f"\n  [OK] [{name}] registered successfully (ID: {pid})")
    print("-"*40 + "\n")

    # Reload persons into memory
    return load_all_persons()


# ════════════════════════════════════════════════
#  Main Loop
# ════════════════════════════════════════════════

def main():
    init_db()
    os.makedirs(PHOTOS_DIR, exist_ok=True)

    print("="*45)
    print("  Face Attendance System Starting...")
    print("="*45)

    persons = load_all_persons()
    print(f"  Total registered: {len(persons)} person(s)")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        print("[ERROR] Could not open camera! Check CAM_INDEX.")
        return

    mode          = "attend"   # "attend" or "register"
    face_counters = {}         # {person_id: frame_count}
    last_marked   = {}         # {person_id: datetime}
    msg           = ""
    msg_color     = GREEN
    msg_timer     = 0
    process_this  = True

    print("  Camera started. Press [Q] to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not received.")
            continue

        display = frame.copy()
        h, w   = frame.shape[:2]

        # Process every other frame for better speed
        if process_this:
            small = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb, model="hog")
            encs  = face_recognition.face_encodings(rgb, locs)

            scale       = 1.0 / FRAME_SCALE
            current_ids = set()

            for (top, right, bottom, left), enc in zip(locs, encs):
                t = int(top*scale); r = int(right*scale)
                b = int(bottom*scale); l = int(left*scale)

                person, dist = find_match(enc, persons)

                if person:
                    pid   = person["id"]
                    pname = person["name"]
                    conf  = int((1 - dist) * 100)
                    color = GREEN
                    label = f"{pname}  {conf}%"
                    current_ids.add(pid)

                    if mode == "attend":
                        face_counters[pid] = face_counters.get(pid, 0) + 1

                        # Check cooldown
                        last = last_marked.get(pid)
                        cooldown_ok = (
                            last is None or
                            (datetime.now() - last).total_seconds() > COOLDOWN_SEC
                        )

                        if face_counters[pid] >= CONFIRM_FRAMES and cooldown_ok:
                            marked = mark_attendance(pid, pname)
                            if marked:
                                now_str = datetime.now().strftime("%H:%M:%S")
                                msg     = f"[OK] {pname} - Attendance marked ({now_str})"
                                msg_color = GREEN
                                msg_timer = 120
                                last_marked[pid] = datetime.now()
                                face_counters[pid] = 0
                                print(f"  [ATTEND] {pname} - {now_str}")
                            else:
                                msg     = f"  {pname} - Already marked today"
                                msg_color = YELLOW
                                msg_timer = 60
                                last_marked[pid] = datetime.now()
                                face_counters[pid] = 0
                else:
                    label = "Unknown"
                    color = RED

                # Draw bounding box and label
                draw_rounded_rect(display, (l, t), (r, b), color, 2, radius=6)
                draw_label(display, label, l, t - 6, color)

                # Confirmation progress bar
                if person and mode == "attend":
                    pid    = person["id"]
                    cnt    = face_counters.get(pid, 0)
                    bar_w  = r - l
                    fill_w = int(bar_w * min(cnt / CONFIRM_FRAMES, 1.0))
                    cv2.rectangle(display, (l, b+4), (r, b+10), (50,50,50), -1)
                    cv2.rectangle(display, (l, b+4), (l+fill_w, b+10), GREEN, -1)

            # Reset counters for faces no longer in frame
            for pid in list(face_counters.keys()):
                if pid not in current_ids:
                    face_counters[pid] = 0

        process_this = not process_this

        # Draw overlay panel
        today_records = get_today_attendance()
        draw_overlay_panel(
            display, today_records, mode,
            len(persons),
            msg if msg_timer > 0 else "",
            msg_color
        )
        if msg_timer > 0:
            msg_timer -= 1

        cv2.imshow("Face Attendance System", display)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') or key == ord('R'):
            mode = "register"
            cv2.destroyAllWindows()
            persons = registration_mode(cap, persons)
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

    cap.release()
    cv2.destroyAllWindows()
    print("\n  System closed.")
    print(f"  Total attendance today: {len(get_today_attendance())} person(s)")


# ════════════════════════════════════════════════
#  Report Utility (can be run standalone)
# ════════════════════════════════════════════════

def print_report(target_date: str = None):
    """
    Print attendance report in the terminal.
    target_date: format "2024-07-15", defaults to today if None.
    """
    if target_date is None:
        target_date = date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(
        "SELECT person_name, time, status FROM attendance WHERE date=? ORDER BY time",
        (target_date,)
    )
    rows = cur.fetchall()
    conn.close()

    print(f"\n{'='*40}")
    print(f"  Attendance Report - {target_date}")
    print(f"{'='*40}")
    if rows:
        for i, (name, time_, status) in enumerate(rows, 1):
            print(f"  {i:2}. {name:<20}  {time_}  [{status}]")
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
        # Show report only: python face_attendance_system.py report [date]
        d = sys.argv[2] if len(sys.argv) > 2 else None
        init_db()
        print_report(d)
    else:
        main()