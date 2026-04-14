import cv2
import time
import pygame
import numpy as np
import threading
from ultralytics import YOLO
from collections import deque

# ╔══════════════════════════════════════════════════════════╗
# ║                       SETTINGS                          ║
# ╠══════════════════════════════════════════════════════════╣
BOUNDARY_MARGIN  = 0.05
ALERT_COOLDOWN   = 2.0
LOST_THRESHOLD   = 2.0
INFER_EVERY_N    = 2
INFER_SCALE      = 0.5

CONFIDENCE       = 0.40
IOU_THRESHOLD    = 0.40

MODEL_NAME       = "yolov8s-pose.pt"

HEAD_KPS         = [0, 1, 2, 3, 4]
SHOULDER_KPS     = [5, 6]
HIP_KPS          = [11, 12]
KP_CONF_THRESH   = 0.30

EDGE_MARGIN_RATIO   = 0.03
CURTAIN_WIDTH_MAX   = 0.18
CURTAIN_ASPECT_MIN  = 3.0

ASPECT_MIN       = 0.25
ASPECT_MAX       = 5.0
MIN_HEIGHT_RATIO = 0.10
MIN_WIDTH_RATIO  = 0.04

CONFIRM_WINDOW   = 4
CONFIRM_FRAMES   = 2
# ╚══════════════════════════════════════════════════════════╝


_model_ready = threading.Event()
model = None

def _load_model():
    global model
    print(f"[INFO] Loading {MODEL_NAME} ...")
    m = YOLO(MODEL_NAME)
    m.fuse()
    model = m
    _model_ready.set()
    print("[INFO] YOLO-Pose ready")

threading.Thread(target=_load_model, daemon=True).start()
pygame.mixer.init()


def generate_beep() -> pygame.mixer.Sound:
    sr = 44100
    t  = np.linspace(0, 0.4, int(sr * 0.4), False)
    w  = (np.sin(2 * np.pi * 1200 * t) * 0.8 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([w, w]))


def has_upper_body(kps: np.ndarray) -> bool:
    def any_visible(indices):
        return any(kps[i][2] > KP_CONF_THRESH for i in indices)
    return any_visible(HEAD_KPS) and any_visible(SHOULDER_KPS) and any_visible(HIP_KPS)


def is_valid_shape(box: tuple, frame_w: int, frame_h: int) -> bool:
    x, y, bw, bh = box
    if bw <= 0 or bh <= 0:
        return False
    if bh < frame_h * MIN_HEIGHT_RATIO:
        return False
    if bw < frame_w * MIN_WIDTH_RATIO:
        return False
    aspect      = bh / bw
    box_w_ratio = bw / frame_w
    touching    = (x < frame_w * EDGE_MARGIN_RATIO or
                   (x + bw) > frame_w * (1 - EDGE_MARGIN_RATIO))
    if touching and box_w_ratio < CURTAIN_WIDTH_MAX and aspect > CURTAIN_ASPECT_MIN:
        return False
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return False
    return True


def detect_persons(frame_bgr: np.ndarray) -> list:
    h, w  = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)
    res   = model(small, verbose=False, conf=CONFIDENCE, iou=IOU_THRESHOLD)[0]
    scale = 1.0 / INFER_SCALE
    persons = []
    kps_data = res.keypoints.data if res.keypoints is not None else None

    for idx, box in enumerate(res.boxes):
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        b = (
            int(x1 * scale), int(y1 * scale),
            int((x2 - x1) * scale), int((y2 - y1) * scale),
        )
        conf_val = float(box.conf[0])
        if not is_valid_shape(b, w, h):
            continue
        if kps_data is not None and idx < len(kps_data):
            kps = kps_data[idx].cpu().numpy()
            if not has_upper_body(kps):
                continue
        else:
            continue
        persons.append((b, conf_val))

    persons.sort(key=lambda p: p[1], reverse=True)
    return [b for b, _ in persons]


def is_outside(box: tuple, w: int, h: int) -> bool:
    x, y, bw, bh = box
    mx, my = int(w * BOUNDARY_MARGIN), int(h * BOUNDARY_MARGIN)
    return x < mx or y < my or x + bw > w - mx or y + bh > h - my


class StabilityFilter:
    def __init__(self):
        self._history    = deque(maxlen=CONFIRM_WINDOW)
        self._last_boxes : list = []

    def update(self, boxes: list) -> list:
        self._history.append(bool(boxes))
        if sum(self._history) >= CONFIRM_FRAMES:
            if boxes:
                self._last_boxes = boxes
            return self._last_boxes
        self._last_boxes = []
        return []


class CameraStream:
    def __init__(self, src: int = 0):
        self._lock    = threading.Lock()
        self._frame   = None
        self._running = True
        self._ready   = threading.Event()
        threading.Thread(target=self._run, args=(src,), daemon=True).start()

    def _run(self, src):
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                self._ready.set()
        cap.release()

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False


class FPSCounter:
    def __init__(self, window=30):
        self._ts = deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


def draw_loading(win: str, cam_ok: bool, mdl_ok: bool, dots: int):
    img = np.zeros((220, 500, 3), dtype=np.uint8)
    items = [("Camera", cam_ok), ("YOLO-Pose model", mdl_ok)]
    for i, (label, ok) in enumerate(items):
        text  = f"[OK] {label}" if ok else f"[..] {label}" + "." * (dots % 4)
        color = (0, 220, 80) if ok else (0, 200, 255)
        cv2.putText(img, text, (30, 75 + i * 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(img, "Press Q to quit",
                (165, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    cv2.imshow(win, img)


def main():
    WIN = "YOLO Monitor"
    cv2.namedWindow(WIN)

    cam  = CameraStream(0)
    beep = generate_beep()
    fps  = FPSCounter()
    stab = StabilityFilter()

    dots = 0
    while not (cam._ready.is_set() and _model_ready.is_set()):
        draw_loading(WIN, cam._ready.is_set(), _model_ready.is_set(), dots)
        dots += 1
        if cv2.waitKey(80) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            pygame.quit()
            return

    print("[INFO] Ready — starting monitor.")

    last_seen      = time.time()
    last_alert     = 0.0
    persons        = []
    frame_idx      = 0

    # Track if person was ever detected (so we don't alert on startup)
    ever_seen      = False

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()
        frame_idx += 1

        if frame_idx % INFER_EVERY_N == 0:
            raw     = detect_persons(frame)
            persons = stab.update(raw)

        # ── State logic ──────────────────────────────────────────────────
        if persons:
            last_seen = now
            ever_seen = True
            visible   = True
            # Alert condition 1: person is inside frame but outside boundary box
            outside_box = any(is_outside(p, w, h) for p in persons)
            left_frame  = False
        else:
            time_since_seen = now - last_seen
            # Alert condition 2: person was seen before but now left the frame
            left_frame  = ever_seen and (time_since_seen <= LOST_THRESHOLD + ALERT_COOLDOWN)
            visible     = ever_seen and (time_since_seen <= LOST_THRESHOLD)
            outside_box = False

        # ── Trigger alert if outside boundary OR left the frame ──────────
        should_alert = ever_seen and (outside_box or left_frame)

        if should_alert and (now - last_alert) > ALERT_COOLDOWN:
            if not pygame.mixer.get_busy():
                beep.play()
            reason = "left frame" if left_frame else "outside boundary"
            print(f"[ALERT] {time.strftime('%H:%M:%S')} — person {reason}")
            last_alert = now

        # ── Draw boundary box ─────────────────────────────────────────────
        mx, my  = int(w * BOUNDARY_MARGIN), int(h * BOUNDARY_MARGIN)
        if left_frame:
            b_color = (0, 100, 255)   # Orange — left frame
        elif outside_box:
            b_color = (0, 0, 255)     # Red — outside boundary
        else:
            b_color = (0, 255, 0)     # Green — safe

        cv2.rectangle(frame, (mx, my), (w - mx, h - my), b_color, 2)

        # ── Draw person boxes ─────────────────────────────────────────────
        for (x, y, bw, bh) in persons:
            c = (0, 0, 255) if is_outside((x, y, bw, bh), w, h) else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), c, 2)
            cv2.putText(frame, "Person", (x, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        # ── HUD ───────────────────────────────────────────────────────────
        if left_frame:
            status  = "ALERT! Left frame"
            s_color = (0, 100, 255)
        elif outside_box:
            status  = "ALERT! Outside boundary"
            s_color = (0, 60, 255)
        elif visible:
            status  = "Safe"
            s_color = (255, 255, 255)
        else:
            status  = "Waiting for person..."
            s_color = (160, 160, 160)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, s_color, 2)
        cv2.putText(frame, f"FPS : {fps.tick():.1f}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, f"Det : {len(persons)}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Flashing border when alert
        if should_alert and int(now * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)

        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()