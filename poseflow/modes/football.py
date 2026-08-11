from .base import Mode, Annotation, Scene
from ..actions import heuristics as H


class Football(Mode):
    key = "football"
    title = "Football Analytics"
    description = "Sprint, saut et tacle/chute des joueurs sur un clip de match."

    def classify(self, tid, f):
        if H.jumping(f):
            return Annotation("jump", "warn")
        if H.posture(f) == "fallen":
            return Annotation("tackle/down", "alert" if H.fall_event(f) else "warn")
        loco = H.locomotion(f)
        if loco == "running":
            return Annotation("sprint" if f["limb_speed"] > 0.16 else "run", "warn")
        return Annotation("jog" if loco == "walking" else "stand")

    def scene(self, anns, feats):
        heat = {tid: H.intensity(f) for tid, f in feats.items()}
        c = {k: 0 for k in ("sprint", "run", "jump", "tackle/down", "jog", "stand")}
        for a in anns.values():
            c[a.label] = c.get(a.label, 0) + 1
        return Scene(
            kpis={"players": len(anns), "sprints": c["sprint"],
                  "jumps": c["jump"], "downs": c["tackle/down"]},
            alerts=[], heat=heat)
