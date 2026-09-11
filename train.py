"""Training engine shared by the proposed model and every neural baseline."""
from __future__ import annotations
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
import config as C
import losses as L


class Augmenter:
    """
    Physically motivated augmentation applied identically to every neural model.

    Three corruptions are injected during training: additive read-out noise, a
    random excursion along the computed nuisance directions, and a small
    acquisition-timing jitter.  All three are corruptions the instrument itself
    produces, so the augmentation encodes measurement invariance rather than a
    generic regulariser, and applying it to every model keeps the comparison
    fair.
    """

    def __init__(self, basis, noise=0.05, drift=0.35, shift=2, p=0.85):
        self.B = torch.as_tensor(basis).float()
        self.noise, self.drift, self.shift, self.p = noise, drift, shift, p

    def __call__(self, x):
        B, T, K = x.shape
        m = (torch.rand(B, 1, 1) < self.p).float()
        out = x + m * self.noise * torch.randn_like(x)
        amp = self.drift * torch.randn(B, self.B.shape[1])
        ramp = torch.linspace(0.0, 1.0, T).view(1, T, 1)
        walk = torch.cumsum(torch.randn(B, T, 1), dim=1) / (T ** 0.5)
        prof = (0.6 * ramp + 0.4 * walk) * m
        out = out + prof * (amp @ self.B.T).unsqueeze(1)
        if self.shift > 0:
            k = int(torch.randint(-self.shift, self.shift + 1, (1,)).item())
            if k:
                out = torch.roll(out, k, dims=1)
        return out


def class_weights(y, n_classes):
    """Inverse-frequency weights, normalised to mean one."""
    cnt = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    w = cnt.sum() / (n_classes * np.maximum(cnt, 1.0))
    return w / w.mean()
from model import KINDRA, KindraAblation

torch.set_num_threads(2)


def _probabilities(out, n_classes):
    if "alpha" in out:
        a = out["alpha"]
        return a / a.sum(-1, keepdim=True)
    return F.softmax(out["logits"], -1)


def evaluate_model(net, loader, n_classes, adv_lambda=0.0):
    net.eval()
    P, Y, U = [], [], []
    with torch.no_grad():
        for x, y, nu in loader:
            out = net(x)
            p = _probabilities(out, n_classes)
            P.append(p.numpy()); Y.append(y.numpy())
            if "alpha" in out:
                a = out["alpha"]
                U.append((n_classes / a.sum(-1)).numpy())
            else:
                pl = p.clamp_min(1e-12)
                U.append((-(pl * pl.log()).sum(-1) / np.log(n_classes)).numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(U)


def train_model(net, loaders, n_classes, cfg=None, is_kindra=False,
                use_evi=True, verbose=False, seed=0, class_weight=None,
                augment=None):
    cfg = cfg or C.TRAIN
    torch.manual_seed(seed)
    tr, va, te = loaders
    w = None if class_weight is None else torch.as_tensor(class_weight).float()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(cfg.epochs):
        net.train()
        lam = cfg.lambda_adv and float(2.0 / (1.0 + np.exp(-6.0 * ep / cfg.epochs)) - 1.0)
        for x, y, nu in tr:
            if augment is not None:
                x = augment(x)
            out = net(x, adv_lambda=lam)
            if is_kindra:
                loss, _ = L.total_loss(out, y, nu, cfg, n_classes, ep,
                                       use_evi=use_evi, weight=w)
            else:
                loss = F.cross_entropy(out["logits"], y, weight=w)
                if "nuis_adv" in out:
                    loss = loss + cfg.lambda_adv * F.mse_loss(out["nuis_adv"], nu)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        p, y, _ = evaluate_model(net, va, n_classes)
        from sklearn.metrics import balanced_accuracy_score
        score = balanced_accuracy_score(y, p.argmax(1))
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if verbose:
            print(f"  ep {ep:02d} val {score:.3f} best {best:.3f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    return dict(val_score=best, epochs_run=ep + 1, seconds=time.time() - t0)


def build_kindra(n_ch, n_classes, init_basis, cfg=None, exposure=None, **kw):
    cfg = cfg or C.TRAIN
    if kw:
        return KindraAblation(n_ch, n_classes, cfg, init_basis, 2, exposure, **kw)
    return KINDRA(n_ch, n_classes, cfg, init_basis, 2, exposure)


def count_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def inference_latency(net, x, repeats=12):
    """Median wall-clock latency of a single-record forward pass, milliseconds."""
    net.eval()
    xs = torch.as_tensor(x[:1]).float()
    ts = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.time()
            net(xs)
            ts.append((time.time() - t0) * 1000.0)
    return float(np.median(ts))
