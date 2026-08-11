import time
from .engine import PoseEstimator, TrackStore, features
from .modes import get_mode, Annotation
from .render import draw_person, draw_hud


class PoseFlow:
    def __init__(self, mode="safety", model="yolo11n-pose.pt", device=None,
                 fps=30.0, imgsz=640, min_frames=6, **mode_kw):
        self.estimator = PoseEstimator(model, device, imgsz)
        self.mode = get_mode(mode, **mode_kw)
        self.store = TrackStore(fps=fps)
        self.min_frames = min_frames
        self._t = None
        self._fps = 0.0

    def step(self, frame, people):
        self.store.update(people)
        feats, anns = {}, {}
        for p in people:
            tr = self.store.tracks.get(p.track_id)
            if tr is None or not tr.ready(self.min_frames):
                continue
            f = features.compute(tr)
            feats[p.track_id] = f
            anns[p.track_id] = self.mode.classify(p.track_id, f)
        scene = self.mode.scene(anns, feats)
        for p in people:
            draw_person(frame, p, anns.get(p.track_id, Annotation(str(p.track_id))))
        draw_hud(frame, self.mode, scene, fps=self._tick())
        return frame, scene, anns

    def run(self, source, vid_stride=1):
        for frame, people in self.estimator.stream(source, vid_stride):
            yield self.step(frame, people)

    def _tick(self):
        now = time.perf_counter()
        if self._t is not None:
            inst = 1.0 / max(now - self._t, 1e-6)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._t = now
        return self._fps
