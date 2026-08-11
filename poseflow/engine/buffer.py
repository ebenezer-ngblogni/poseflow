from collections import deque
import numpy as np
from ..config import WINDOW


class Track:
    def __init__(self, track_id, maxlen=WINDOW):
        self.id = track_id
        self.kp = deque(maxlen=maxlen)        # (17,3) per frame
        self.box = deque(maxlen=maxlen)       # (4,)
        self.t = deque(maxlen=maxlen)         # seconds
        self.last_seen = 0

    def push(self, kp, box, t, frame_idx):
        self.kp.append(kp.astype(np.float32))
        self.box.append(np.asarray(box, np.float32))
        self.t.append(float(t))
        self.last_seen = frame_idx

    def ready(self, n=5):
        return len(self.kp) >= n


class TrackStore:
    def __init__(self, fps=30.0, ttl=30):
        self.fps = fps
        self.ttl = ttl
        self.tracks = {}
        self.frame_idx = 0

    def update(self, people):
        t = self.frame_idx / max(self.fps, 1e-6)
        seen = set()
        for p in people:
            if p.track_id < 0:
                continue
            tr = self.tracks.get(p.track_id)
            if tr is None:
                tr = self.tracks[p.track_id] = Track(p.track_id)
            tr.push(p.kp, p.box, t, self.frame_idx)
            seen.add(p.track_id)
        for tid in [k for k, v in self.tracks.items()
                    if self.frame_idx - v.last_seen > self.ttl]:
            del self.tracks[tid]
        self.frame_idx += 1
        return seen
