"""
Component-level external validation on a real chemosensor drift benchmark.

The publicly distributed gas-sensor drift collection contains 13 910
measurements from a sixteen-element metal-oxide array recorded over thirty-six
months and grouped into ten batches, and is the standard benchmark for
long-term chemosensor drift.  Only aggregated descriptors are distributed -
steady-state response together with exponential-moving-average transients at
three time constants for the rising and the decaying phase - so the raw response
curve is not available and the recurrent branches of the proposed model cannot
be exercised on it.  What can be tested externally is the component that does
not require a sequence: the soft low-rank nuisance projection with its
co-operative and adversarial heads.

The experiment trains on the earliest batches and tests on each later batch, so
the shift between training and test is genuine instrument drift rather than a
simulated perturbation.
"""
from __future__ import annotations
import os, json, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C
from model import grad_reverse

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_ext")
N_FEAT = 128


def load_batches(path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(path, "batch*.dat")),
                   key=lambda p: int("".join(c for c in os.path.basename(p)
                                             if c.isdigit())))
    X, Y, Bidx = [], [], []
    for bi, f in enumerate(files):
        xs, ys = [], []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            head, *rest = line.split()
            lab = int(head.split(";")[0]) - 1
            v = np.zeros(N_FEAT, dtype=np.float32)
            for tok in rest:
                k, val = tok.split(":")
                k = int(k) - 1
                if 0 <= k < N_FEAT:
                    v[k] = float(val)
            xs.append(v); ys.append(lab)
        X.append(np.stack(xs)); Y.append(np.array(ys)); Bidx.append(np.full(len(ys), bi))
    return X, Y, Bidx


class SoftProjection(nn.Module):
    """The disentangling projection of the proposed model, applied to a feature vector."""

    def __init__(self, n_feat, rank, init=None):
        super().__init__()
        B = torch.randn(n_feat, rank) / np.sqrt(n_feat) if init is None \
            else torch.as_tensor(init, dtype=torch.float32)
        self.U = nn.Parameter(B.clone())
        self.gate_logit = nn.Parameter(torch.full((rank,), 2.0))

    def forward(self, x):
        Q, _ = torch.linalg.qr(self.U)
        a = x @ Q
        g = torch.sigmoid(self.gate_logit)
        return x - (a * g) @ Q.transpose(0, 1), a


class Net(nn.Module):
    def __init__(self, n_feat, n_cls, rank=0, adv=False, hidden=128):
        super().__init__()
        self.proj = SoftProjection(n_feat, rank) if rank > 0 else None
        self.enc = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(hidden, 64), nn.GELU())
        self.cls = nn.Linear(64, n_cls)
        self.adv = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)) if adv else None
        self.coop = nn.Sequential(nn.Linear(rank, 16), nn.GELU(),
                                  nn.Linear(16, 1)) if rank > 0 else None

    def forward(self, x, lam=0.0):
        a = None
        if self.proj is not None:
            x, a = self.proj(x)
        h = self.enc(x)
        out = dict(logits=self.cls(h))
        if self.adv is not None and lam > 0:
            out["adv"] = self.adv(grad_reverse(h, lam))
        if self.coop is not None:
            out["coop"] = self.coop(a)
        return out


def drift_init(Xtr, btr, rank):
    """Data-driven analogue of the physical nuisance basis: batch-mean directions."""
    ms = [Xtr[btr == b].mean(0) for b in np.unique(btr)]
    Mm = np.stack(ms) - np.mean(ms, 0)
    U = np.linalg.svd(Mm.T @ Mm)[0][:, :rank]
    return U.astype(np.float32)


def run(train_batches=(0, 1), rank=4, epochs=90, seeds=(0, 1, 2, 3, 4), log=print):
    X, Y, Bi = load_batches()
    n_cls = int(max(y.max() for y in Y)) + 1
    Xtr = np.concatenate([X[b] for b in train_batches])
    Ytr = np.concatenate([Y[b] for b in train_batches])
    Btr = np.concatenate([Bi[b] for b in train_batches])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    U0 = drift_init(Ztr, Btr, rank)
    bz = (Btr - Btr.mean()) / (Btr.std() + 1e-6)
    test_ids = [b for b in range(len(X)) if b not in train_batches]
    variants = {"MLP": dict(rank=0, adv=False),
                "MLP + adversarial batch head": dict(rank=0, adv=True),
                "MLP + hard projection": dict(rank=rank, adv=False, hard=True),
                "MLP + proposed projection": dict(rank=rank, adv=True)}
    out = {}
    for tag, kw in variants.items():
        hard = kw.pop("hard", False)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            net = Net(N_FEAT, n_cls, **kw)
            if kw["rank"] > 0:
                with torch.no_grad():
                    net.proj.U.copy_(torch.from_numpy(U0))
                    if hard:
                        net.proj.gate_logit.fill_(8.0)
                        net.proj.gate_logit.requires_grad_(False)
                        net.proj.U.requires_grad_(False)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            xt = torch.from_numpy(Ztr).float()
            yt = torch.from_numpy(Ytr).long()
            nt = torch.from_numpy(bz).float().unsqueeze(1)
            n = len(yt)
            for ep in range(epochs):
                lam = float(2.0 / (1.0 + np.exp(-6.0 * ep / epochs)) - 1.0)
                perm = torch.randperm(n)
                for i in range(0, n, 128):
                    idx = perm[i:i + 128]
                    o = net(xt[idx], lam)
                    loss = F.cross_entropy(o["logits"], yt[idx])
                    if "adv" in o:
                        loss = loss + 0.2 * F.mse_loss(o["adv"], nt[idx])
                    if "coop" in o:
                        loss = loss + 0.2 * F.mse_loss(o["coop"], nt[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            accs = []
            with torch.no_grad():
                for b in test_ids:
                    zb = torch.from_numpy(((X[b] - mu) / sd)).float()
                    pred = net(zb)["logits"].argmax(-1).numpy()
                    accs.append(float((pred == Y[b]).mean()))
            out.setdefault(tag, []).append(accs)
            log(f"[E10] {tag:30s} seed {seed} mean acc {np.mean(accs):.3f}")
    res = dict(test_batches=[b + 1 for b in test_ids],
               train_batches=[b + 1 for b in train_batches],
               n_per_batch=[int(len(y)) for y in Y],
               results={k: np.array(v).tolist() for k, v in out.items()})
    with open(os.path.join(C.RESULTS_DIR, "E10_external.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    r = run()
    for k, v in r["results"].items():
        a = np.array(v)
        print(f"{k:32s} {a.mean():.3f} +- {a.mean(1).std(ddof=1):.3f}")
