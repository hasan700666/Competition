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

# ── YOLOv8-pose model (detects person + 17 keypoints) ───────
MODEL_NAME       = "yolov8s-pose.pt"   # auto-downloads if not present

# ── Upper-body keypoint rule ─────────────────────────────────
# COCO keypoint indices:
#   0=nose  1=left_eye  2=right_eye  3=left_ear  4=right_ear
#   5=left_shoulder     6=right_shoulder
#   7=left_elbow        8=right_elbow
#   9=left_wrist        10=right_wrist
#  11=left_hip          12=right_hip
#  13=left_knee         14=right_knee
#  15=left_ankle        16=right_ankle
#
# Rule: person is valid ONLY when ALL of these are visible:
#   • At least 1 of: nose / eye / ear   (HEAD)
#   • At least 1 of: left/right shoulder (SHOULDER)
#   • At least 1 of: left/right hip      (HIP / WAIST)
#
# This means: head + shoulder + hip must all be present.
# Just a head or neck alone → rejected.

HEAD_KPS         = [0, 1, 2, 3, 4]    # nose, eyes, ears
SHOULDER_KPS     = [5, 6]              # left/right shoulder
HIP_KPS          = [11, 12]            # left/right hip
KP_CONF_THRESH   = 0.30               # minimum keypoint confidence to count as "visible"

# ── Curtain / background rejection ──────────────────────────
EDGE_MARGIN_RATIO   = 0.03
CURTAIN_WIDTH_MAX   = 0.18
CURTAIN_ASPECT_MIN  = 3.0

# ── Shape filter ─────────────────────────────────────────────
ASPECT_MIN       = 0.25
ASPECT_MAX       = 5.0
MIN_HEIGHT_RATIO = 0.10
MIN_WIDTH_RATIO  = 0.04

# ── Stability ────────────────────────────────────────────────
CONFIRM_WINDOW   = 4
CONFIRM_FRAMES   = 2
# ╚══════════════════════════════════════════════════════════╝


# ── Model loads in background ─────────────────────────────────────────────────
_model_ready = threading.Event()
model = None

def _load_model():
    global model
    print(f"[INFO] Loading {MODEL_NAME} ...")
    m = YOLO(MODEL_NAME)
    m.fuse()
    model = m
    _model_ready.set()
    print("[INFO] YOLO-Pose ready ✓")

threading.Thread(target=_load_model, daemon=True).start()
pygame.mixer.init()


# ── Audio ─────────────────────────────────────────────────────────────────────
def generate_beep() -> pygame.mixer.Sound:
    sr = 44100
    t  = np.linspace(0, 0.4, int(sr * 0.4), False)
    w  = (np.sin(2 * np.pi * 1200 * t) * 0.8 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([w, w]))


# ── Keypoint upper-body check ─────────────────────────────────────────────────
def has_upper_body(kps: np.ndarray) -> bool:
    """
    kps: shape (17, 3) — each row is [x, y, confidence]

    Returns True ONLY if:
      • at least 1 head keypoint   is visible (conf > KP_CONF_THRESH)
      • at least 1 shoulder keypoint is visible
      • at least 1 hip keypoint    is visible

    This enforces: head → shoulder → hip (waist) must all be present.
    A floating head or neck-only detection is rejected.
    """
    def any_visible(indices):
        return any(kps[i][2] > KP_CONF_THRESH for i in indices)

    head_ok     = any_visible(HEAD_KPS)
    shoulder_ok = any_visible(SHOULDER_KPS)
    hip_ok      = any_visible(HIP_KPS)

    return head_ok and shoulder_ok and hip_ok


# ── Shape / curtain filter ────────────────────────────────────────────────────
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

    # Narrow + tall + touching edge → curtain / clothes
    if touching and box_w_ratio < CURTAIN_WIDTH_MAX and aspect > CURTAIN_ASPECT_MIN:
        return False

    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return False

    return True


# ── Main detection pipeline ───────────────────────────────────────────────────
def detect_persons(frame_bgr: np.ndarray) -> list:
    """
    1. Run YOLOv8-pose on downscaled frame
    2. For each 'person' detection:
       a. Shape / curtain filter
       b. Keypoint upper-body check (head + shoulder + hip)
    3. Return validated boxes in original-frame coordinates
    """
    h, w  = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)
    res   = model(small, verbose=False, conf=CONFIDENCE, iou=IOU_THRESHOLD)[0]
    scale = 1.0 / INFER_SCALE

    persons = []

    # res.keypoints is None if model has no pose — safe fallback
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

        # ── Shape filter ──────────────────────────────────────────────────
        if not is_valid_shape(b, w, h):
            continue

        # ── Keypoint upper-body check ─────────────────────────────────────
        if kps_data is not None and idx < len(kps_data):
            kps = kps_data[idx].cpu().numpy()   # shape: (17, 3)
            if not has_upper_body(kps):
                continue
        # If keypoints unavailable for this box → skip (don't accept blindly)
        else:
            continue

        persons.append((b, conf_val))

    persons.sort(key=lambda p: p[1], reverse=True)
    return [b for b, _ in persons]


def is_outside(box: tuple, w: int, h: int) -> bool:
    x, y, bw, bh = box
    mx, my = int(w * BOUNDARY_MARGIN), int(h * BOUNDARY_MARGIN)
    return x < mx or y < my or x + bw > w - mx or y + bh > h - my


# ── Temporal stability filter ─────────────────────────────────────────────────
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


# ── Threaded camera ───────────────────────────────────────────────────────────
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


# ── FPS counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window=30):
        self._ts = deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


# ── Loading screen ────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
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

    last_seen  = time.time()
    last_alert = 0.0
    persons    = []
    frame_idx  = 0

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

        # State
        if persons:
            last_seen = now
            visible   = True
            outside   = any(is_outside(p, w, h) for p in persons)
        else:
            visible = (now - last_seen) <= LOST_THRESHOLD
            outside = False

        # Alert
        if visible and outside and (now - last_alert) > ALERT_COOLDOWN:
            if not pygame.mixer.get_busy():
                beep.play()
            print(f"[ALERT] {time.strftime('%H:%M:%S')} -- person outside boundary")
            last_alert = now

        # ── Draw ──────────────────────────────────────────────────────────
        mx, my  = int(w * BOUNDARY_MARGIN), int(h * BOUNDARY_MARGIN)
        b_color = (0, 0, 255) if outside else (0, 255, 0)
        cv2.rectangle(frame, (mx, my), (w - mx, h - my), b_color, 2)

        for (x, y, bw, bh) in persons:
            c = (0, 0, 255) if is_outside((x, y, bw, bh), w, h) else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), c, 2)
            cv2.putText(frame, "Person", (x, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        # ── HUD ───────────────────────────────────────────────────────────
        status  = ("ALERT!" if (visible and outside)
                   else "Safe" if visible
                   else "No person")
        s_color = (0, 60, 255) if status == "ALERT!" else (255, 255, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (210, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, s_color, 2)
        cv2.putText(frame, f"FPS : {fps.tick():.1f}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, f"Det : {len(persons)}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()