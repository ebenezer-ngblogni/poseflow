from .base import Mode, Annotation, Scene


class Fitness(Mode):
    key = "fitness"
    title = "Fitness Coach"
    description = "Comptage de repetitions et controle de posture (squat/pushup)."

    JOINT = {"squat": "knee_angle", "pushup": "elbow_angle"}

    def __init__(self, exercise="squat"):
        super().__init__()
        self.exercise = exercise
        self.joint = self.JOINT[exercise]

    def classify(self, tid, f):
        ang = f[self.joint]
        st = self.state.setdefault(tid, {"reps": 0, "phase": "up", "bad": False})
        if ang is None:
            return Annotation(f"reps {st['reps']}")
        if ang < 95:
            st["phase"] = "down"
            st["bad"] = ang < 55
        elif ang > 155 and st["phase"] == "down":
            st["phase"] = "up"
            st["reps"] += 1
        lvl = "warn" if st["bad"] and st["phase"] == "down" else "ok"
        tag = "TOO DEEP" if lvl == "warn" else st["phase"]
        return Annotation(f"reps {st['reps']} [{tag}]", lvl)

    def scene(self, anns, feats):
        total = sum(v["reps"] for v in self.state.values())
        return Scene(kpis={"athletes": len(anns), "total_reps": total,
                           "exercise": self.exercise}, alerts=[])
