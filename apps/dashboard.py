import os
import sys
import glob
import time
import tempfile
import cv2
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseflow import PoseFlow, MODES  # noqa: E402

st.set_page_config(page_title="PoseFlow", layout="wide")
st.markdown("## PoseFlow")
st.caption("Un moteur temps réel, quatre applications · pose multi-personnes + action · YOLO11-pose + ByteTrack")

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Mode", list(MODES), format_func=lambda k: MODES[k])
    exercise = st.selectbox("Exercice", ["squat", "pushup"]) if mode == "fitness" else "squat"
    samples = sorted(glob.glob("samples/*.mp4"))
    up = st.file_uploader("Vidéo (.mp4)", type=["mp4", "mov", "avi"])
    src = st.selectbox("Ou un échantillon", samples) if samples else None
    imgsz = st.select_slider("Résolution", [320, 416, 512, 640], value=512)
    run = st.button("▶ Lancer", type="primary", use_container_width=True)

video = st.empty()
c1, c2, c3, c4 = st.columns(4)
kpi_slots = [c1.empty(), c2.empty(), c3.empty(), c4.empty()]
chart = st.empty()
alert_box = st.empty()

if run:
    if up is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(up.read()); source = tmp.name
    else:
        source = src
    if not source:
        st.warning("Choisis une vidéo."); st.stop()

    kw = {"exercise": exercise} if mode == "fitness" else {}
    flow = PoseFlow(mode=mode, fps=25, imgsz=imgsz, **kw)
    hist, alerts_log = [], []
    for i, (frame, scene, _) in enumerate(flow.run(source)):
        if i % 2:
            continue
        video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        for slot, (k, v) in zip(kpi_slots, list(scene.kpis.items())[:4]):
            slot.metric(k, v)
        row = {k: v for k, v in scene.kpis.items() if isinstance(v, (int, float))}
        row["t"] = round(i / 25, 1); hist.append(row)
        if scene.alerts:
            alerts_log.append(f"t={row['t']}s · " + " · ".join(scene.alerts))
        if len(hist) > 3 and i % 6 == 0:
            df = pd.DataFrame(hist).set_index("t")
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines"))
            fig.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h"), template="plotly_dark")
            chart.plotly_chart(fig, use_container_width=True)
        if alerts_log:
            alert_box.error("🚨 " + "  |  ".join(alerts_log[-3:]))
    st.success(f"Terminé, {len(hist)} frames analysées.")
