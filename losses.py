"""Objective terms of the KINDRA training criterion."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dirichlet_risk(alpha, target, n_classes):
    """
    Bayes risk of the squared-error loss under a Dirichlet belief.

        L = sum_k (y_k - p_k)^2 + p_k (1 - p_k) / (S + 1),   p = alpha / S

    The first term is the usual fit; the second is the variance of the belief,
    so a prediction is penalised both for being wrong and for being confidently
    wrong.
    """
    y = F.one_hot(target, n_classes).float()
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    err = (y - p).pow(2).sum(-1)
    var = (p * (1.0 - p) / (S + 1.0)).sum(-1)
    return (err + var).mean()


def dirichlet_ml(alpha, target, n_classes, weight=None):
    """
    Type-II maximum-likelihood form of the evidential objective,

        L = sum_k y_k [ psi(S) - psi(alpha_k) ],

    which supplies a stronger gradient than the Bayes-risk form when the total
    evidence is still small and therefore trains more reliably from scratch.
    """
    y = F.one_hot(target, n_classes).float()
    S = alpha.sum(-1, keepdim=True)
    per = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
    if weight is not None:
        w = weight[target]
        return (per * w).sum() / w.sum()
    return per.mean()


def dirichlet_ce(alpha, target, n_classes, weight=None):
    """
    Negative log-likelihood of the Dirichlet mean.

    Training the belief directly rather than through the digamma expectation
    keeps the decision boundary as sharp as a softmax head while leaving the
    evidence, and therefore the uncertainty mass, available at inference.
    """
    S = alpha.sum(-1, keepdim=True)
    p = (alpha / S).clamp_min(1e-9)
    per = -torch.log(p.gather(-1, target.unsqueeze(-1)).squeeze(-1))
    if weight is not None:
        w = weight[target]
        return (per * w).sum() / w.sum()
    return per.mean()


def dirichlet_kl(alpha, target, n_classes):
    """KL of the belief with the evidence of the true class removed to a flat prior."""
    y = F.one_hot(target, n_classes).float()
    a = y + (1.0 - y) * alpha
    S = a.sum(-1, keepdim=True)
    K = float(n_classes)
    t1 = torch.lgamma(S).squeeze(-1) - torch.lgamma(a).sum(-1) - torch.lgamma(
        torch.tensor(K, device=alpha.device))
    t2 = ((a - 1.0) * (torch.digamma(a) - torch.digamma(S))).sum(-1)
    return (t1 + t2).mean()


def kinetic_residual(recon, z):
    """
    Consistency between the drift-free trace and the trace that a physically
    admissible coverage trajectory can reproduce.
    """
    return F.mse_loss(recon, z)


def total_loss(out, y, nu, cfg, n_classes, epoch, use_evi=True, weight=None):
    parts = {}
    if use_evi and "alpha" in out:
        alpha = out["alpha"]
        anneal = min(1.0, epoch / max(cfg.warmup_epochs, 1))
        parts["cls"] = dirichlet_ce(alpha, y, n_classes, weight)
        parts["evi"] = cfg.lambda_evi * anneal * dirichlet_kl(alpha, y, n_classes)
    else:
        parts["cls"] = F.cross_entropy(out["logits"], y, weight=weight)
        parts["evi"] = torch.zeros((), device=y.device)
    parts["kin"] = cfg.lambda_kin * kinetic_residual(out["recon"], out["z"].detach())
    parts["coop"] = cfg.lambda_adv * F.mse_loss(out["nuis_coop"], nu)
    if "nuis_adv" in out:
        parts["adv"] = cfg.lambda_adv * F.mse_loss(out["nuis_adv"], nu)
    else:
        parts["adv"] = torch.zeros((), device=y.device)
    return sum(parts.values()), parts
