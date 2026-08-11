"""Build a skeleton-action dataset from labeled clips.

Layout:  data/<action_label>/<clip>.mp4
Output:  dataset.npz  with X (N,T,34), y (N,), labels (list)
"""
import argparse
import glob
import os
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseflow.engine import PoseEstimator, features  # noqa: E402
from poseflow.config import WINDOW  # noqa: E402


def clip_windows(est, path, T, step):
    seqs = defaultdict(list)          # track_id -> list of (17,2)
    for _, people in est.stream(path):
        for p in people:
            if p.track_id >= 0:
                seqs[p.track_id].append(features.norm_one(p.kp, p.box))
    out = []
    for seq in seqs.values():
        seq = np.asarray(seq, np.float32)
        for s in range(0, max(1, len(seq) - T + 1), step):
            w = seq[s:s + T]
            if len(w) == T:
                out.append(w.reshape(T, -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    args = ap.parse_args()

    labels = sorted(d for d in os.listdir(args.data)
                    if os.path.isdir(os.path.join(args.data, d)))
    est = PoseEstimator(args.model)
    X, y = [], []
    for ci, lab in enumerate(labels):
        clips = glob.glob(os.path.join(args.data, lab, "*.*"))
        for clip in clips:
            w = clip_windows(est, clip, args.window, args.step)
            X += w; y += [ci] * len(w)
            print(f"{lab:>16} | {os.path.basename(clip):30} -> {len(w)} windows")
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    np.savez_compressed(args.out, X=X, y=y, labels=labels)
    print(f"[saved] {args.out}  X={X.shape}  classes={labels}")


if __name__ == "__main__":
    main()
