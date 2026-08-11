KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

WINDOW = 32          # frames kept per track for temporal analysis
KP_CONF = 0.30       # min keypoint confidence to trust a joint

from .modes import MODES, get_mode  # noqa: E402

__all__ = ["KEYPOINTS", "KP", "SKELETON", "WINDOW", "KP_CONF", "MODES", "get_mode"]
