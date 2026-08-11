import numpy as np
from ..config import KP, KP_CONF


def _pt(kp, name):
    p = kp[KP[name]]
    return p[:2] if p[2] >= KP_CONF else None


def _mid(kp, a, b):
    pa, pb = _pt(kp, a), _pt(kp, b)
    if pa is None and pb is None:
        return None
    if pa is None:
        return pb
    if pb is None:
        return pa
    return (pa + pb) / 2.0


def angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba, bc = a - b, c - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-6:
        return None
    cos = np.clip(np.dot(ba, bc) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _height(box):
    return max(float(box[3] - box[1]), 1.0)


def norm_one(kp, box):
    """Single-frame keypoints centered on the hip and scaled by torso size."""
    hip = _mid(kp, "left_hip", "right_hip")
    sho = _mid(kp, "left_shoulder", "right_shoulder")
    if hip is None:
        hip = kp[:, :2].mean(0)
    scale = np.linalg.norm(sho - hip) if sho is not None else None
    if not scale or scale < 1e-3:
        scale = _height(box) * 0.5
    return (kp[:, :2] - hip) / scale


def normalized(track):
    """Keypoints centered on the hip and scaled by torso size -> (T,17,2)."""
    return np.asarray([norm_one(kp, track.box[i]) for i, kp in enumerate(track.kp)],
                      np.float32)


def compute(track):
    kp = track.kp[-1]
    box = track.box[-1]
    h = _height(box)
    w = float(box[2] - box[0])

    torso = _mid(kp, "left_shoulder", "right_shoulder")
    hip = _mid(kp, "left_hip", "right_hip")
    center = _mid(kp, "left_hip", "right_hip")
    if center is None:
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    torso_angle = 90.0
    if torso is not None and hip is not None:
        v = torso - hip
        torso_angle = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))

    dt = (track.t[-1] - track.t[0]) or 1e-6
    prev_c = _mid(track.kp[0], "left_hip", "right_hip")
    if prev_c is None:
        prev_c = center
    v_down = float((center[1] - prev_c[1]) / dt / h)      # +down, per height/s

    norm = normalized(track)
    if len(norm) >= 2:
        disp = np.linalg.norm(np.diff(norm, axis=0), axis=2)
        motion = float(np.nanmean(disp))
        limb_idx = [KP[n] for n in ("left_wrist", "right_wrist",
                                    "left_ankle", "right_ankle")]
        limb = float(np.nanmean(np.linalg.norm(np.diff(norm[:, limb_idx], axis=0), axis=2)))
    else:
        motion = limb = 0.0

    cadence = _cadence(track)
    knee = _best(angle(_pt(kp, "left_hip"), _pt(kp, "left_knee"), _pt(kp, "left_ankle")),
                 angle(_pt(kp, "right_hip"), _pt(kp, "right_knee"), _pt(kp, "right_ankle")))
    elbow = _best(angle(_pt(kp, "left_shoulder"), _pt(kp, "left_elbow"), _pt(kp, "left_wrist")),
                  angle(_pt(kp, "right_shoulder"), _pt(kp, "right_elbow"), _pt(kp, "right_wrist")))

    return {
        "center": center, "box": box, "height": h,
        "aspect": w / h, "torso_angle": torso_angle,
        "v_down": v_down, "motion": motion, "limb_speed": limb,
        "cadence": cadence, "knee_angle": knee, "elbow_angle": elbow,
        "norm": norm,
    }


def _best(a, b):
    vals = [x for x in (a, b) if x is not None]
    return float(np.mean(vals)) if vals else None


def _cadence(track):
    if len(track.kp) < 6:
        return 0.0
    ys = []
    for kp in track.kp:
        a = _mid(kp, "left_ankle", "right_ankle")
        ys.append(np.nan if a is None else a[1])
    ys = np.asarray(ys)
    if np.isnan(ys).all():
        return 0.0
    ys = ys - np.nanmean(ys)
    signs = np.sign(ys[~np.isnan(ys)])
    return float(np.count_nonzero(np.diff(signs) != 0)) / max(len(signs), 1)
