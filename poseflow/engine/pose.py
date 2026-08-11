from dataclasses import dataclass
import numpy as np


@dataclass
class Person:
    track_id: int
    box: np.ndarray        # (4,) xyxy
    kp: np.ndarray         # (17, 3) x, y, conf


class PoseEstimator:
    def __init__(self, model="yolo11n-pose.pt", device=None, imgsz=640,
                 conf=0.25, tracker="bytetrack.yaml"):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = tracker

    def stream(self, source, vid_stride=1):
        results = self.model.track(
            source=source, stream=True, persist=True, verbose=False,
            tracker=self.tracker, imgsz=self.imgsz, conf=self.conf,
            device=self.device, vid_stride=vid_stride,
        )
        for r in results:
            yield r.orig_img, _people(r)


def _people(r):
    out = []
    if r.keypoints is None or r.boxes is None:
        return out
    kps = r.keypoints.data.cpu().numpy()          # (N, 17, 3)
    boxes = r.boxes.xyxy.cpu().numpy()            # (N, 4)
    ids = r.boxes.id
    ids = ids.cpu().numpy().astype(int) if ids is not None else -np.ones(len(boxes), int)
    for i in range(len(boxes)):
        out.append(Person(int(ids[i]), boxes[i], kps[i]))
    return out
