"""
Reference models.

The set spans four families: classical feature-based classifiers, plain gated
recurrent networks, the stacked LSTM-GRU hybrid that the sensor literature
commonly adopts, and modern time-series classification architectures.  Two
drift-aware competitors are included so that the proposed disentanglement is
compared against an established alternative rather than only against models
that ignore drift.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import grad_reverse


# ----------------------------------------------------------------------
# feature extraction for the classical baselines
# ----------------------------------------------------------------------
def handcrafted_features(X, exposure):
    """Steady-state and transient descriptors of every channel."""
    on = exposure > 0.5
    idx = np.where(on)[0]
    first_on = idx[0]
    seg = slice(first_on, first_on + int(on.sum() // 3))
    off = slice(seg.stop, seg.stop + (seg.stop - seg.start))
    f = [X.max(1), X.min(1), X.mean(1), X.std(1),
         X[:, on].mean(1), X[:, ~on].mean(1),
         X[:, seg].mean(1), X[:, off].mean(1),
         X[:, seg].mean(1) - X[:, off].mean(1),
         X[:, -6:].mean(1) - X[:, :4].mean(1),
         np.diff(X, axis=1).max(1), np.diff(X, axis=1).min(1)]
    return np.concatenate(f, axis=1)


# ----------------------------------------------------------------------
# recurrent baselines
# ----------------------------------------------------------------------
class _SeqHead(nn.Module):
    def __init__(self, n_feat, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 96), nn.GELU(),
                                 nn.Dropout(0.15), nn.Linear(96, n_classes))

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, n_ch, n_classes, hidden=64, layers=1):
        super().__init__()
        self.rnn = nn.LSTM(n_ch, hidden, layers, batch_first=True)
        self.head = _SeqHead(2 * hidden, n_classes)

    def forward(self, x, adv_lambda=0.0):
        o, _ = self.rnn(x)
        f = torch.cat([o[:, -1], o.mean(1)], -1)
        return dict(logits=self.head(f), feat=f)


class GRUNet(nn.Module):
    def __init__(self, n_ch, n_classes, hidden=64, layers=1):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, layers, batch_first=True)
        self.head = _SeqHead(2 * hidden, n_classes)

    def forward(self, x, adv_lambda=0.0):
        o, _ = self.rnn(x)
        f = torch.cat([o[:, -1], o.mean(1)], -1)
        return dict(logits=self.head(f), feat=f)


class StackedLSTMGRU(nn.Module):
    """The conventional hybrid: an LSTM layer followed by a GRU layer."""

    def __init__(self, n_ch, n_classes, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, batch_first=True)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = _SeqHead(2 * hidden, n_classes)

    def forward(self, x, adv_lambda=0.0):
        o, _ = self.lstm(x)
        o, _ = self.gru(o)
        f = torch.cat([o[:, -1], o.mean(1)], -1)
        return dict(logits=self.head(f), feat=f)


class ParallelLSTMGRU(nn.Module):
    """Concatenation hybrid: independent LSTM and GRU branches on the same input."""

    def __init__(self, n_ch, n_classes, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, batch_first=True)
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.head = _SeqHead(4 * hidden, n_classes)

    def forward(self, x, adv_lambda=0.0):
        a, _ = self.lstm(x)
        b, _ = self.gru(x)
        f = torch.cat([a[:, -1], a.mean(1), b[:, -1], b.mean(1)], -1)
        return dict(logits=self.head(f), feat=f)


class MultiRateGRU(nn.Module):
    """
    Fixed multi-rate recurrence in the spirit of a clockwork network: three GRU
    modules updated every 1, 2 and 4 samples respectively.
    """

    def __init__(self, n_ch, n_classes, hidden=32, periods=(1, 2, 4)):
        super().__init__()
        self.periods = periods
        self.cells = nn.ModuleList([nn.GRUCell(n_ch, hidden) for _ in periods])
        self.head = _SeqHead(len(periods) * hidden, n_classes)
        self.hidden = hidden

    def forward(self, x, adv_lambda=0.0):
        B, T, _ = x.shape
        hs = [x.new_zeros(B, self.hidden) for _ in self.periods]
        for t in range(T):
            for i, p in enumerate(self.periods):
                if t % p == 0:
                    hs[i] = self.cells[i](x[:, t], hs[i])
        f = torch.cat(hs, -1)
        return dict(logits=self.head(f), feat=f)


class GRUPhase(nn.Module):
    """
    Gated recurrent encoder whose pooling includes the same per-spot phase
    statistics the proposed model uses, so that the contribution of the pooling
    choice can be separated from the contribution of the architecture.
    """

    def __init__(self, n_ch, n_classes, hidden=64, exposure=None):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, batch_first=True)
        if exposure is None:
            exposure = torch.ones(1)
        self.register_buffer("exposure", torch.as_tensor(exposure).float())
        self.norm = nn.BatchNorm1d(2 * hidden + 12 * n_ch)
        self.head = _SeqHead(2 * hidden + 12 * n_ch, n_classes)

    def forward(self, x, adv_lambda=0.0):
        from model import phase_pool
        o, _ = self.rnn(x)
        f = self.norm(torch.cat([o[:, -1], o.mean(1),
                                 phase_pool(x, self.exposure)], -1))
        return dict(logits=self.head(f), feat=f)


class GRUDANN(nn.Module):
    """Gated recurrent encoder with a gradient-reversal nuisance head."""

    def __init__(self, n_ch, n_classes, hidden=64, n_nuisance=2):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, batch_first=True)
        self.head = _SeqHead(2 * hidden, n_classes)
        self.adv = nn.Sequential(nn.Linear(2 * hidden, 48), nn.GELU(),
                                 nn.Linear(48, n_nuisance))

    def forward(self, x, adv_lambda=0.0):
        o, _ = self.rnn(x)
        f = torch.cat([o[:, -1], o.mean(1)], -1)
        out = dict(logits=self.head(f), feat=f)
        if adv_lambda > 0:
            out["nuis_adv"] = self.adv(grad_reverse(f, adv_lambda))
        return out


# ----------------------------------------------------------------------
# convolutional and attention baselines
# ----------------------------------------------------------------------
class LSTMFCN(nn.Module):
    def __init__(self, n_ch, n_classes, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, 96, 8, padding=4), nn.BatchNorm1d(96), nn.ReLU(),
            nn.Conv1d(96, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 96, 3, padding=1), nn.BatchNorm1d(96), nn.ReLU())
        self.head = _SeqHead(hidden + 96, n_classes)

    def forward(self, x, adv_lambda=0.0):
        o, _ = self.lstm(x)
        c = self.conv(x.transpose(1, 2)).mean(-1)
        f = torch.cat([o[:, -1], c], -1)
        return dict(logits=self.head(f), feat=f)


class _Chomp(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, x):
        return x[:, :, :-self.s] if self.s > 0 else x


class TCN(nn.Module):
    def __init__(self, n_ch, n_classes, hidden=64, levels=4, k=5):
        super().__init__()
        layers, cin = [], n_ch
        for i in range(levels):
            d = 2 ** i
            pad = (k - 1) * d
            layers += [nn.Conv1d(cin, hidden, k, padding=pad, dilation=d),
                       _Chomp(pad), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1)]
            cin = hidden
        self.net = nn.Sequential(*layers)
        self.head = _SeqHead(2 * hidden, n_classes)

    def forward(self, x, adv_lambda=0.0):
        h = self.net(x.transpose(1, 2))
        f = torch.cat([h[:, :, -1], h.mean(-1)], -1)
        return dict(logits=self.head(f), feat=f)


class _Inception(nn.Module):
    def __init__(self, cin, nf=16, kernels=(5, 11, 23)):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 else nn.Identity()
        cb = nf if cin > 1 else cin
        self.convs = nn.ModuleList(
            [nn.Conv1d(cb, nf, k, padding=k // 2, bias=False) for k in kernels])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(kernels) + 1))

    def forward(self, x):
        b = self.bottleneck(x)
        z = torch.cat([c(b) for c in self.convs] + [self.pool(x)], 1)
        return F.relu(self.bn(z))


class InceptionTime(nn.Module):
    """
    Single Inception network.  The published method ensembles five such models;
    a five-fold inference cost is not affordable for a point-of-care read-out,
    so one network is used and the difference is stated when the result is
    reported.
    """

    def __init__(self, n_ch, n_classes, nf=16, depth=3):
        super().__init__()
        blocks, cin = [], n_ch
        for _ in range(depth):
            blocks.append(_Inception(cin, nf)); cin = nf * 4
        self.blocks = nn.ModuleList(blocks)
        self.head = _SeqHead(cin, n_classes)

    def forward(self, x, adv_lambda=0.0):
        h = x.transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        f = h.mean(-1)
        return dict(logits=self.head(f), feat=f)


class TSTransformer(nn.Module):
    def __init__(self, n_ch, n_classes, d_model=64, heads=4, layers=2, T=90):
        super().__init__()
        self.inp = nn.Linear(n_ch, d_model)
        pe = torch.zeros(T, d_model)
        pos = torch.arange(T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model, heads, 128, 0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = _SeqHead(2 * d_model, n_classes)

    def forward(self, x, adv_lambda=0.0):
        h = self.enc(self.inp(x) + self.pe[:, :x.shape[1]])
        f = torch.cat([h[:, -1], h.mean(1)], -1)
        return dict(logits=self.head(f), feat=f)


# ----------------------------------------------------------------------
# random convolutional kernel transform
# ----------------------------------------------------------------------
def rocket_transform(X, n_kernels=1200, seed=0):
    """
    Random dilated convolutional kernels with max and proportion-of-positive
    pooling, following the ROCKET construction.
    """
    rng = np.random.default_rng(seed)
    n, T, C_ = X.shape
    feats = np.empty((n, 2 * n_kernels), dtype=np.float32)
    Xt = X.transpose(0, 2, 1)
    for k in range(n_kernels):
        klen = int(rng.choice([7, 9, 11]))
        ch = int(rng.integers(1, C_ + 1))
        chan = rng.choice(C_, size=ch, replace=False)
        w = rng.standard_normal((ch, klen)).astype(np.float32)
        w -= w.mean()
        b = rng.uniform(-1, 1)
        max_d = max(1, int(2 ** rng.uniform(0, np.log2((T - 1) / (klen - 1)))))
        pad = ((klen - 1) * max_d) // 2 if rng.random() < 0.5 else 0
        sig = Xt[:, chan, :]
        if pad:
            sig = np.pad(sig, ((0, 0), (0, 0), (pad, pad)))
        L = sig.shape[-1] - (klen - 1) * max_d
        if L <= 0:
            feats[:, 2 * k:2 * k + 2] = 0.0
            continue
        acc = np.zeros((n, L), dtype=np.float32)
        for j in range(klen):
            acc += (sig[:, :, j * max_d: j * max_d + L] * w[None, :, j:j + 1]).sum(1)
        acc += b
        feats[:, 2 * k] = acc.max(1)
        feats[:, 2 * k + 1] = (acc > 0).mean(1)
    return feats


TORCH_BASELINES = {
    "LSTM": LSTMNet,
    "GRU": GRUNet,
    "LSTM-GRU (stacked)": StackedLSTMGRU,
    "LSTM-GRU (parallel)": ParallelLSTMGRU,
    "Multi-rate GRU": MultiRateGRU,
    "GRU + phase pooling": GRUPhase,
    "GRU-DANN": GRUDANN,
    "LSTM-FCN": LSTMFCN,
    "TCN": TCN,
    "InceptionTime": InceptionTime,
    "Time-series Transformer": TSTransformer,
}
