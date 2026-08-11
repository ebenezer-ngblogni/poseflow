import torch
import torch.nn as nn


class TCN(nn.Module):
    """Compact temporal CNN over skeleton windows: input (B, T, 34)."""

    def __init__(self, in_dim=34, n_classes=4, ch=64, layers=3, k=5, p=0.2):
        super().__init__()
        blocks, c = [], in_dim
        for i in range(layers):
            d = 2 ** i
            blocks += [
                nn.Conv1d(c, ch, k, padding=(k // 2) * d, dilation=d),
                nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(p),
            ]
            c = ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):                 # x: (B, T, 34)
        x = x.transpose(1, 2)             # -> (B, 34, T)
        x = self.tcn(x).mean(dim=2)       # global temporal pool
        return self.head(x)
