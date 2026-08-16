import cv2
import numpy as np
from .config import SKELETON, DRAW_CONF as KP_CONF

LEVEL = {"ok": (120, 220, 90), "warn": (0, 165, 255), "alert": (60, 60, 240)}
BONE = (230, 200, 120)


def draw_person(img, person, ann):
    color = LEVEL.get(ann.level, LEVEL["ok"])
    kp = person.kp
    valid = [(a, b) for a, b in SKELETON if kp[a, 2] >= KP_CONF and kp[b, 2] >= KP_CONF]
    lens = [float(np.linalg.norm(kp[a, :2] - kp[b, :2])) for a, b in valid]
    if kp[5, 2] >= KP_CONF and kp[6, 2] >= KP_CONF:
        cap = 3.0 * float(np.linalg.norm(kp[5, :2] - kp[6, :2]))
    else:
        cap = 2.5 * np.median(lens) if lens else 0
    for (a, b), L in zip(valid, lens):
        if L <= cap:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)),
                     BONE, 2, cv2.LINE_AA)
    for x, y, c in kp:
        if c >= KP_CONF:
            cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    x1, y1, x2, y2 = person.box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    _label(img, f"#{person.track_id} {ann.label}", x1, y1, color)


def _label(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, h + 6)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(img, mode, scene, fps=None):
    lines = [mode.title.upper()]
    lines += [f"{k}: {v}" for k, v in scene.kpis.items()]
    if fps:
        lines.append(f"fps: {fps:.1f}")
    _panel(img, lines, 12, 12)
    if scene.alerts:
        _banner(img, "  |  ".join(scene.alerts))


def _panel(img, lines, x, y):
    pad, lh = 10, 22
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in lines)
    box = img[y:y + lh * len(lines) + pad, x:x + w + 2 * pad].copy()
    if box.size:
        img[y:y + lh * len(lines) + pad, x:x + w + 2 * pad] = (box * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        col = (120, 220, 90) if i == 0 else (240, 240, 240)
        cv2.putText(img, t, (x + pad, y + pad + lh * i + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)


def _banner(img, text):
    h, w = img.shape[:2]
    y = h - 44
    strip = img[y:h].copy()
    img[y:h] = (strip * 0.25 + np.array([0, 0, 90])).clip(0, 255).astype(np.uint8)
    cv2.putText(img, "! " + text, (16, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (60, 60, 250), 2, cv2.LINE_AA)
