import os
import sys
import time
import cv2

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseflow import PoseFlow, MODES  # noqa: E402

DEFAULT_SRC = os.environ.get("POSEFLOW_SOURCE", "samples/people-detection.mp4")
MODEL = os.environ.get("POSEFLOW_MODEL", "yolo11n-pose.pt")
app = FastAPI(title="PoseFlow API")

_latest = {"kpis": {}, "alerts": []}


def _frames(mode, source, loop=True):
    while True:
        flow = PoseFlow(mode=mode, model=MODEL, fps=25)
        src = int(source) if str(source).isdigit() else source
        for frame, scene, _ in flow.run(src):
            _latest["kpis"], _latest["alerts"] = scene.kpis, scene.alerts
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        if not loop or str(source).isdigit():
            break
        time.sleep(0.2)


@app.get("/health")
def health():
    return {"status": "ok", "modes": MODES, "model": MODEL}


@app.get("/state")
def state():
    return JSONResponse(_latest)


@app.get("/stream")
def stream(mode: str = Query("safety"), source: str = Query(DEFAULT_SRC)):
    if mode not in MODES:
        return JSONResponse({"error": "bad mode", "modes": list(MODES)}, 400)
    return StreamingResponse(_frames(mode, source),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index(mode: str = "safety"):
    opts = "".join(
        f'<option value="{k}" {"selected" if k == mode else ""}>{v}</option>'
        for k, v in MODES.items())
    return f"""<!doctype html><meta charset=utf-8>
<title>PoseFlow</title>
<style>body{{background:#0d0f14;color:#e8e8e8;font-family:system-ui;margin:0;padding:24px}}
h1{{font-weight:700;letter-spacing:-.5px}} .sub{{color:#8a94a6}}
select{{background:#1a1f2b;color:#fff;border:1px solid #2a3242;padding:8px;border-radius:8px}}
img{{width:100%;max-width:900px;border-radius:12px;border:1px solid #222}}</style>
<h1>PoseFlow <span class=sub>&middot; un moteur, 4 applications</span></h1>
<p>Mode&nbsp;
<select onchange="location='/?mode='+this.value">{opts}</select></p>
<img src="/stream?mode={mode}">"""
