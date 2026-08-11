from dataclasses import dataclass, field


@dataclass
class Annotation:
    label: str
    level: str = "ok"          # ok | warn | alert


@dataclass
class Scene:
    kpis: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    heat: dict = field(default_factory=dict)   # track_id -> 0..1 intensity


class Mode:
    key = "base"
    title = "Base"
    description = ""

    def __init__(self):
        self.state = {}

    def reset(self):
        self.state = {}

    def classify(self, tid, f) -> Annotation:
        return Annotation("person")

    def scene(self, anns, feats) -> Scene:
        return Scene(kpis={"people": len(anns)})
