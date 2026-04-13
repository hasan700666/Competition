import cv2
import time
import pygame
import numpy as np
from ultralytics import YOLO

# ================= SETTINGS =================
BOUNDARY_MARGIN = 0.05
ALERT_COOLDOWN = 2.0
LOST_THRESHOLD = 2.0
# ===========================================

# Load YOLO (auto डाउनलोड হবে যদি না থাকে)
model = YOLO("yolov8n.pt")

pygame.mixer.init()

def generate_beep():
    sr = 44100
    duration = 0.4
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = np.sin(2 * np.pi * 1200 * t) * 0.8 * 32767
    wave = wave.astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

def detect_persons(frame):
    results = model(frame, verbose=False)[0]
    persons = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append((x1, y1, x2 - x1, y2 - y1))

    return persons

def is_outside(box, w, h):
    x, y, bw, bh = box
    mx = int(w * BOUNDARY_MARGIN)
    my = int(h * BOUNDARY_MARGIN)

    return (
        x < mx or
        y < my or
        x + bw > w - mx or
        y + bh > h - my
    )

def main():
    cap = cv2.VideoCapture(0)
    beep = generate_beep()

    last_seen = time.time()
    last_alert = 0

    visible = False
    outside = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        persons = detect_persons(frame)
        now = time.time()

        if persons:
            visible = True
            last_seen = now

            outside = any(is_outside(p, w, h) for p in persons)
        else:
            if now - last_seen > LOST_THRESHOLD:
                visible = False
                outside = False

        # Alert
        if visible and outside:
            if now - last_alert > ALERT_COOLDOWN:
                if not pygame.mixer.get_busy():
                    beep.play()
                print("ALERT: Person outside!")
                last_alert = now

        # Draw boundary
        mx = int(w * BOUNDARY_MARGIN)
        my = int(h * BOUNDARY_MARGIN)
        color = (0, 0, 255) if outside else (0, 255, 0)
        cv2.rectangle(frame, (mx, my), (w - mx, h - my), color, 2)

        # Draw persons
        for (x, y, bw, bh) in persons:
            c = (0, 0, 255) if is_outside((x, y, bw, bh), w, h) else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), c, 2)
            cv2.putText(frame, "Person", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        # Status text
        if not visible:
            text = "No person"
        elif outside:
            text = "ALERT!"
        else:
            text = "Safe"

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLO Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()