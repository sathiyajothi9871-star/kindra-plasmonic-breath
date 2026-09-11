"""Cohort loading, subject-level partitioning and normalisation."""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import config as C

CACHE = os.path.join(C.CACHE_DIR, "cohort.npz")


def load_raw():
    d = np.load(CACHE)
    return {k: d[k] for k in d.files}


def subject_split(subject, y, seed=0, frac=(0.70, 0.10, 0.20)):
    """
    Partition at the level of the subject, stratified by class.

    Every session of a subject falls in exactly one partition, which prevents
    the same person contributing to both training and test and removes the
    optimistic bias a session-level random split would introduce.
    """
    rng = np.random.default_rng(seed)
    subj_label = {}
    for s, lab in zip(subject, y):
        subj_label[s] = lab
    tr, va, te = [], [], []
    for c in np.unique(y):
        ss = np.array([s for s, l in subj_label.items() if l == c])
        rng.shuffle(ss)
        n = len(ss)
        n_tr = int(round(frac[0] * n))
        n_va = int(round(frac[1] * n))
        tr += list(ss[:n_tr]); va += list(ss[n_tr:n_tr + n_va]); te += list(ss[n_tr + n_va:])
    idx = lambda pool: np.isin(subject, np.array(pool))
    return idx(tr), idx(va), idx(te)


def device_split(device, held_out=(6, 7)):
    """Leave-devices-out partition used for the cross-chip generalisation test."""
    m_out = np.isin(device, np.array(held_out))
    return ~m_out, m_out


def standardise(X_tr, *others):
    """
    Isotropic scaling of the resonance-shift traces.

    A single global factor is used rather than a per-channel one so that the
    geometry of the array response is preserved: the physically derived nuisance
    directions keep their orientation relative to the surface-binding directions,
    which a per-channel rescaling would destroy.  Channel-specific gain is left
    to the first learned layer of each model, so no model is disadvantaged.
    """
    sd = float(X_tr.std()) + 1e-12
    out = [X_tr / sd] + [o / sd for o in others]
    return out, (0.0, sd)


def rescale_basis_iso(basis):
    """Orthonormal nuisance basis in the isotropically scaled coordinates."""
    Q, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64))
    return Q.astype(np.float32)


def rescale_basis(basis, sd):
    """
    Express a physically derived response direction in the standardised
    coordinates the network actually sees.
    """
    B = np.asarray(basis, dtype=np.float64) / np.asarray(sd, dtype=np.float64)[:, None]
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    Q, _ = np.linalg.qr(B)
    return Q.astype(np.float32)


def nuisance_targets(raw, mask=None):
    """Logged operating conditions used by the drift heads."""
    t = raw["temp"].astype(np.float32)
    m = raw["months"].astype(np.float32)
    z = np.stack([(t - t.mean()) / (t.std() + 1e-8),
                  (m - m.mean()) / (m.std() + 1e-8)], axis=1)
    return z if mask is None else z[mask]


def make_loaders(X, y, nu, masks, batch_size=None, shuffle_train=True):
    bs = batch_size or C.TRAIN.batch_size
    out = []
    for i, m in enumerate(masks):
        ds = TensorDataset(torch.from_numpy(X[m]).float(),
                           torch.from_numpy(y[m]).long(),
                           torch.from_numpy(nu[m]).float())
        out.append(DataLoader(ds, batch_size=bs,
                              shuffle=(i == 0 and shuffle_train), drop_last=False))
    return out
