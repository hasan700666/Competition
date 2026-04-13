"""
Child Face Boundary Monitor
============================
Detects a human face and alerts when it moves outside the camera frame boundary.

Requirements:
    pip install opencv-python numpy pygame

Usage:
    python child_monitor.py
"""

import cv2
import numpy as np
import time
import pygame

# ===================== SETTINGS =====================

# Boundary margin — how close to the edge triggers alert (0.0 to 0.3)
# 0.05 = alert when face goes within 5% of frame edge
BOUNDARY_MARGIN = 0.05

# How often the alert sound repeats (seconds)
ALERT_COOLDOWN = 2.0

# How many seconds without a face before "no face" state
FACE_LOST_THRESHOLD = 2.0

# Face detector scale factor (1.1 = more detections, slower)
SCALE_FACTOR = 1.1

# Minimum face size in pixels
MIN_FACE_SIZE = 60

# ====================================================

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)


def generate_beep():
    """Generate a two-tone alert beep"""
    sample_rate = 44100
    duration = 0.45
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)
    # High-pitched two-tone beep
    wave = (np.sin(2 * np.pi * 1000 * t) * 0.5 +
            np.sin(2 * np.pi * 1400 * t) * 0.5)
    # Fade out at end
    fade = np.linspace(1.0, 0.0, num_samples)
    wave = wave * fade * 0.8 * 32767
    wave = wave.astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)


def load_face_detector():
    """Load Haar cascade face detector"""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print("[ERROR] Could not load face detector!")
        return None
    print("[OK] Face detector loaded")
    return detector


def detect_faces(frame, detector):
    """Detect faces in frame, return list of (x, y, w, h)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve detection in low light
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return []
    return faces.tolist()


def get_face_center(face):
    """Get center point of a face bounding box"""
    x, y, w, h = face
    return x + w // 2, y + h // 2


def is_outside_boundary(face, frame_w, frame_h, margin):
    """Check if face bounding box is outside the safe boundary"""
    x, y, w, h = face
    mx = int(frame_w * margin)
    my = int(frame_h * margin)
    # Alert if any edge of face box crosses the boundary
    return (x < mx or
            y < my or
            x + w > frame_w - mx or
            y + h > frame_h - my)


def draw_ui(frame, faces, boundary_margin, face_outside, face_visible, alert_active):
    """Draw all UI elements on the frame"""
    h, w = frame.shape[:2]
    mx = int(w * boundary_margin)
    my = int(h * boundary_margin)

    # --- Safe zone boundary box ---
    boundary_color = (0, 60, 220) if face_outside else (0, 210, 0)
    cv2.rectangle(frame, (mx, my), (w - mx, h - my), boundary_color, 2)
    cv2.putText(frame, "Safe Zone", (mx + 6, my - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, boundary_color, 1)

    # --- Draw detected faces ---
    for face in faces:
        x, y, fw, fh = face
        cx, cy = get_face_center(face)
        outside = is_outside_boundary(face, w, h, boundary_margin)

        # Face bounding box
        face_color = (0, 60, 220) if outside else (0, 210, 0)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), face_color, 2)

        # Face label
        label = "Outside!" if outside else "Child"
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, face_color, 2)

        # Center dot
        cv2.circle(frame, (cx, cy), 5, (255, 220, 0), -1)

    # --- Top status bar ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if not face_visible:
        status = "Scanning for face..."
        color = (180, 180, 180)
    elif face_outside:
        status = "!! ALERT: Face outside boundary !!"
        color = (50, 80, 255)
    else:
        status = "Face detected — in safe zone"
        color = (50, 220, 50)

    cv2.putText(frame, status, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # --- Flashing red border when alert ---
    if alert_active and int(time.time() * 4) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)

    # --- Bottom hint ---
    cv2.putText(frame, "Press 'q' to quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (130, 130, 130), 1)

    return frame


def main():
    print("=" * 52)
    print("   Child Face Boundary Monitor")
    print("=" * 52)

    # Load face detector
    detector = load_face_detector()
    if detector is None:
        return

    # Generate alert sound
    alert_sound = generate_beep()
    print("[OK] Alert sound ready")

    # Open camera
    import platform
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    print("Searching for camera...", end=' ', flush=True)
    cap = None
    for index in range(3):
        test = cv2.VideoCapture(index, backend)
        if test.isOpened():
            cap = test
            print(f"found at index {index}")
            break
        test.release()
    if cap is None:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n[READY] Show a face to the camera.")
    print("When the face moves outside the green boundary — alert will sound.\n")

    last_alert_time = 0
    last_face_time = time.time()
    face_visible = False
    face_outside = False
    alert_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        h, w = frame.shape[:2]

        # --- Face detection (every frame) ---
        faces = detect_faces(frame, detector)

        current_time = time.time()

        if len(faces) > 0:
            last_face_time = current_time
            face_visible = True

            # Check if any face is outside boundary
            face_outside = any(
                is_outside_boundary(f, w, h, BOUNDARY_MARGIN)
                for f in faces
            )
        else:
            # No face detected
            if current_time - last_face_time > FACE_LOST_THRESHOLD:
                face_visible = False
                face_outside = False

        # --- Alert logic ---
        should_alert = face_visible and face_outside
        if should_alert:
            alert_active = True
            if current_time - last_alert_time >= ALERT_COOLDOWN:
                print(f"[ALERT] Face outside boundary! ({time.strftime('%H:%M:%S')})")
                if not pygame.mixer.get_busy():
                    alert_sound.play()
                last_alert_time = current_time
        else:
            alert_active = False

        # --- Draw UI ---
        frame = draw_ui(frame, faces, BOUNDARY_MARGIN,
                        face_outside, face_visible, alert_active)

        cv2.imshow("Child Face Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("\nProgram closed.")


if __name__ == "__main__":
    main()