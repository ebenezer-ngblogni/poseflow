"""Tier-1 explainable action signals derived from skeleton geometry."""


def posture(f):
    if f["n_vis"] < 7 or not f["lower_vis"]:
        return "upright"                       # trop occlus pour juger d'une chute
    if f["torso_angle"] > 55 and f["kp_aspect"] > 1.3:
        return "fallen"
    return "upright"


def fall_event(f):
    return posture(f) == "fallen" and f["v_down"] > 0.35


def locomotion(f):
    m, limb, cad = f["motion"], f["limb_speed"], f["cadence"]
    if m < 0.012 and limb < 0.02:
        return "idle"
    if f["lower_vis"] and (limb > 0.13 or (cad > 0.45 and limb > 0.07)):
        return "running"
    return "walking"


def jumping(f):
    return f["v_down"] < -0.4 and f["motion"] > 0.04


def intensity(f):
    """0..1 overall movement intensity, for heatmap / fight scoring."""
    return min(1.0, f["limb_speed"] * 4 + f["motion"] * 6)
