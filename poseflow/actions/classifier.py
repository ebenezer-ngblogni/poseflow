import json
import numpy as np
import torch
from .model import TCN
from ..config import WINDOW


class ActionClassifier:
    """Wraps a trained TCN for skeleton-window action recognition."""

    def __init__(self, weights, labels=None, window=WINDOW, device="cpu"):
        ck = torch.load(weights, map_location=device)
        self.labels = labels or ck.get("labels", [])
        self.window = ck.get("window", window)
        self.device = device
        self.net = TCN(n_classes=len(self.labels)).to(device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()

    @torch.no_grad()
    def predict(self, norm_window):
        """norm_window: (T,17,2) normalized keypoints -> (label, prob)."""
        x = _prep(norm_window, self.window)
        logit = self.net(torch.from_numpy(x).float().to(self.device))
        p = torch.softmax(logit, 1)[0]
        i = int(p.argmax())
        return self.labels[i], float(p[i])


def _prep(win, T):
    win = np.nan_to_num(np.asarray(win, np.float32))
    if len(win) < T:
        win = np.concatenate([np.repeat(win[:1], T - len(win), 0), win])
    win = win[-T:].reshape(T, -1)          # (T, 34)
    return win[None]                        # (1, T, 34)
