import numpy as np
from .base import Mode, Annotation, Scene
from ..actions import heuristics as H


class Flow(Mode):
    key = "flow"
    title = "People Flow"
    description = "Comptage, temps d'attente et detection d'attroupements."

    def classify(self, tid, f):
        st = self.state.setdefault(tid, {"idle": 0})
        if H.locomotion(f) == "idle":
            st["idle"] += 1
        else:
            st["idle"] = max(0, st["idle"] - 2)
        if st["idle"] > 25:
            return Annotation("waiting", "warn")
        return Annotation("moving")

    def scene(self, anns, feats):
        waiting = [t for t, a in anns.items() if a.label == "waiting"]
        groups = _clusters(feats)
        alerts = []
        big = max((len(g) for g in groups), default=0)
        if big >= 5:
            alerts.append(f"ATTROUPEMENT ({big} personnes)")
        return Scene(
            kpis={"people": len(anns), "waiting": len(waiting),
                  "groups": len(groups)},
            alerts=alerts, heat={t: min(1.0, self.state[t]["idle"] / 40)
                                 for t in anns if t in self.state})


def _clusters(feats):
    items = list(feats.items())
    parent = {t: t for t, _ in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ta, fa = items[i]
            tb, fb = items[j]
            d = np.linalg.norm(fa["center"] - fb["center"])
            if d < 1.1 * (fa["height"] + fb["height"]) / 2:
                parent[find(ta)] = find(tb)
    groups = {}
    for t, _ in items:
        groups.setdefault(find(t), []).append(t)
    return [g for g in groups.values() if len(g) >= 3]
