from .base import Mode, Annotation, Scene

ALERT_LABELS = {"fall", "fight", "faint"}


class LearnedAction(Mode):
    key = "action"
    title = "Learned Actions (TCN)"
    description = "Reconnaissance d'action par reseau temporel entraine (Tier-2)."

    def __init__(self, weights="weights/action.pt", conf=0.4):
        super().__init__()
        from ..actions.classifier import ActionClassifier
        self.clf = ActionClassifier(weights)
        self.conf = conf

    def classify(self, tid, f):
        lab, p = self.clf.predict(f["norm"])
        if p < self.conf:
            return Annotation("...")
        lvl = "alert" if lab in ALERT_LABELS else ("warn" if p > 0.8 else "ok")
        return Annotation(f"{lab} {p:.0%}", lvl)

    def scene(self, anns, feats):
        counts = {}
        for a in anns.values():
            key = a.label.split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
        alerts = [f"{k.upper()}" for k in counts if k in ALERT_LABELS]
        return Scene(kpis={"people": len(anns), **counts}, alerts=alerts)
