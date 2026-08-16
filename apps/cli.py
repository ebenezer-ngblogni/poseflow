import argparse
import json
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseflow import PoseFlow, MODES  # noqa: E402


def probe_fps(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if 1 < fps < 240 else 25.0


def main():
    ap = argparse.ArgumentParser(description="PoseFlow, pose + action sur video")
    ap.add_argument("--source", required=True, help="video path, RTSP url, or webcam index")
    ap.add_argument("--mode", default="safety", choices=list(MODES))
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--exercise", default="squat", choices=["squat", "pushup"])
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    fps = probe_fps(args.source)
    kw = {"exercise": args.exercise} if args.mode == "fitness" else {}
    flow = PoseFlow(mode=args.mode, model=args.model, device=args.device,
                    fps=fps, imgsz=args.imgsz, **kw)
    src = int(args.source) if str(args.source).isdigit() else args.source

    writer, events, last = None, [], None
    for i, (frame, scene, _) in enumerate(flow.run(src, vid_stride=args.stride)):
        if not args.no_save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps / args.stride, (w, h))
            writer.write(frame)
        key = tuple(scene.alerts)
        if key and key != last:
            events.append({"frame": i, "t": round(i / fps, 2), "alerts": scene.alerts,
                           "kpis": scene.kpis})
        last = key
        if i % 30 == 0:
            print(f"\rframe {i:5d} | {scene.kpis} | {' '.join(scene.alerts)}", end="")
        if args.max_frames and i + 1 >= args.max_frames:
            break

    if writer:
        writer.release()
    ev_path = os.path.splitext(args.out)[0] + ".events.json"
    with open(ev_path, "w") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"\n[done] mode={args.mode} frames={i + 1} events={len(events)}")
    if not args.no_save:
        print(f"[out]  {args.out}\n[events] {ev_path}")


if __name__ == "__main__":
    main()
