"""Train the TCN action classifier. CPU/GPU, Colab-ready.

  python train/train.py --data dataset.npz --out weights/action.pt --epochs 40
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseflow.actions.model import TCN  # noqa: E402


def synth(n=1200, T=32):
    """Fallback synthetic set so the pipeline is runnable without clips."""
    labels = ["idle", "wave", "walk", "fall"]
    X, y = [], []
    t = np.linspace(0, 2 * np.pi, T)
    for _ in range(n):
        c = np.random.randint(4)
        base = np.random.randn(17, 2) * 0.1
        seq = np.repeat(base[None], T, 0)
        if c == 1:
            seq[:, 9] += np.stack([np.sin(t) * 0.6, np.cos(t) * 0.6], 1)
        elif c == 2:
            seq[:, [15, 16]] += np.sin(t)[:, None, None] * 0.4
        elif c == 3:
            seq += np.linspace(0, 1.2, T)[:, None, None] * np.array([0.3, 1.0])
        X.append((seq + np.random.randn(T, 17, 2) * 0.02).reshape(T, -1))
        y.append(c)
    return np.asarray(X, np.float32), np.asarray(y, np.int64), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset.npz (omit for synthetic demo)")
    ap.add_argument("--out", default="weights/action.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if args.data and os.path.exists(args.data):
        d = np.load(args.data, allow_pickle=True)
        X, y, labels = d["X"], d["y"], list(d["labels"])
        print(f"[data] {args.data}  X={X.shape}  classes={labels}")
    else:
        X, y, labels = synth()
        print(f"[data] synthetic  X={X.shape}  classes={labels}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    nv = max(1, int(0.2 * len(ds)))
    tr, va = random_split(ds, [len(ds) - nv, nv])
    dltr = DataLoader(tr, batch_size=args.bs, shuffle=True)
    dlva = DataLoader(va, batch_size=256)

    net = TCN(in_dim=X.shape[-1], n_classes=len(labels)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, args.epochs + 1):
        net.train()
        for xb, yb in dltr:
            opt.zero_grad()
            loss = lossf(net(xb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
        net.eval(); ok = tot = 0
        with torch.no_grad():
            for xb, yb in dlva:
                pred = net(xb.to(dev)).argmax(1).cpu()
                ok += (pred == yb).sum().item(); tot += len(yb)
        acc = ok / tot
        if ep % 5 == 0 or ep == args.epochs:
            print(f"epoch {ep:3d} | val_acc {acc:.3f}")
        if acc >= best:
            best = acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "labels": labels,
                        "window": X.shape[1]}, args.out)
    print(f"[saved] {args.out}  best_val_acc={best:.3f}")


if __name__ == "__main__":
    main()
