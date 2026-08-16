import numpy as np
from .base import Mode, Annotation, Scene
from ..actions import heuristics as H


class CrowdSafety(Mode):
    key = "safety"
    title = "Crowd Safety"
    description = "Chute, panique et bagarre en temps reel dans une foule."

    def classify(self, tid, f):
        if H.posture(f) == "fallen":
            return Annotation("FALL" if H.fall_event(f) else "down", "alert")
        loco = H.locomotion(f)
        if loco == "running":
            return Annotation("running", "warn")
        return Annotation(loco)

    def scene(self, anns, feats):
        heat = {tid: H.intensity(f) for tid, f in feats.items()}
        fallen = [t for t, a in anns.items() if a.level == "alert"]
        running = [t for t, a in anns.items() if a.label == "running"]
        alerts = []
        if fallen:
            alerts.append(f"CHUTE detectee ({len(fallen)})")
        if len(running) >= max(3, int(0.5 * len(anns) + 1)):
            alerts.append("MOUVEMENT DE PANIQUE")
        for a, b in _fights(feats, heat):
            anns[a].level = anns[b].level = "alert"
            anns[a].label = anns[b].label = "FIGHT"
            alerts.append("BAGARRE probable")
        return Scene(
            kpis={"people": len(anns), "fallen": len(fallen), "running": len(running)},
            alerts=sorted(set(alerts)), heat=heat)


def _fights(feats, heat, thr=0.72):
    items = list(feats.items())
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ta, fa = items[i]
            tb, fb = items[j]
            if heat[ta] < thr or heat[tb] < thr:
                continue
            if not (fa["lower_vis"] and fb["lower_vis"]):
                continue
            if fa["n_vis"] < 9 or fb["n_vis"] < 9:
                continue
            d = np.linalg.norm(fa["center"] - fb["center"])
            reach = 0.55 * (fa["height"] + fb["height"]) / 2
            if d < reach:
                pairs.append((ta, tb))
    return pairs
