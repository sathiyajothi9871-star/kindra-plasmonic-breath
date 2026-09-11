"""
Experimental protocol.

E1  main benchmark, subject-level partitions, five repetitions
E2  ablation of the three proposed mechanisms and the evidential head
E3  sensitivity to the projection rank and to the loss weights
E4  robustness under injected drift, noise, channel loss and humidity shift
E5  cross-chip generalisation, two leave-devices-out folds
E6  computational cost
E7  statistical comparison
E8  calibration and selective prediction
E9  failure analysis
E10 component-level external validation on a real chemosensor drift benchmark
"""
from __future__ import annotations
import copy, json, os, pickle, time
import numpy as np
import torch
import torch
import config as C
import dataset as D
import train as T
import evaluate as E
import baselines as B
import breath_cohort as BC
import stats as S

N_CLASSES = len(C.CLASSES)
N_CH = C.N_CHANNELS
RES = C.RESULTS_DIR


# ----------------------------------------------------------------------
_AUG = None


def augmenter():
    """Shared measurement-invariance augmentation, built once per session."""
    global _AUG
    if _AUG is None:
        meta = pickle.load(open(os.path.join(C.CACHE_DIR, "devices.pkl"), "rb"))
        M = np.concatenate([d["drift_basis"] for d in meta], axis=1)
        U = np.linalg.svd(M, full_matrices=False)[0][:, :3]
        _AUG = T.Augmenter(U)
    return _AUG


def exposure_vector():
    return torch.as_tensor(BC.EXPOSURE).float()


def physical_basis(rank=None):
    rank = rank or C.TRAIN.proj_rank
    meta = pickle.load(open(os.path.join(C.CACHE_DIR, "devices.pkl"), "rb"))
    M = np.concatenate([d["drift_basis"] for d in meta], axis=1)
    U = np.linalg.svd(M, full_matrices=False)[0][:, :max(rank, 1)]
    return D.rescale_basis_iso(U)


def prepare(raw, masks):
    m_tr, m_va, m_te = masks
    X = raw["X"]
    (Xtr, Xva, Xte), _ = D.standardise(X[m_tr], X[m_va], X[m_te])
    Xa = np.zeros_like(X)
    Xa[m_tr], Xa[m_va], Xa[m_te] = Xtr, Xva, Xte
    nu = D.nuisance_targets(raw)
    loaders = D.make_loaders(Xa, raw["y"], nu, [m_tr, m_va, m_te])
    return Xa, nu, loaders, T.class_weights(raw["y"][m_tr], N_CLASSES)


def fit_one(name, loaders, cw, seed, cfg=None, basis=None, ablate=None, aug=None):
    cfg = cfg or C.TRAIN
    if name == "KINDRA":
        net = T.build_kindra(N_CH, N_CLASSES, basis, cfg=cfg,
                             exposure=exposure_vector(), **(ablate or {}))
        info = T.train_model(net, loaders, N_CLASSES, cfg=cfg, is_kindra=True,
                             use_evi=(ablate or {}).get("use_evi", True),
                             seed=seed, class_weight=cw, augment=aug)
    else:
        if name == "GRU + phase pooling":
            net = B.TORCH_BASELINES[name](N_CH, N_CLASSES,
                                          exposure=exposure_vector())
        else:
            net = B.TORCH_BASELINES[name](N_CH, N_CLASSES)
        info = T.train_model(net, loaders, N_CLASSES, cfg=cfg, seed=seed,
                             class_weight=cw, augment=aug)
    return net, info


def classical_scores(Xa, y, masks, seed):
    """
    Feature-based and kernel-transform reference classifiers.

    These are fitted on the training partition only, so that every model in the
    comparison sees the same records; the validation partition is reserved for
    model selection and is not used by any method as additional training data.
    """
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    m_tr, m_va, m_te = masks
    m_fit = m_tr
    meta = pickle.load(open(os.path.join(C.CACHE_DIR, "devices.pkl"), "rb"))
    U = np.linalg.svd(np.concatenate([d["drift_basis"] for d in meta], axis=1),
                      full_matrices=False)[0][:, :C.TRAIN.proj_rank]
    P = np.eye(N_CH) - U @ U.T
    feats = {"": B.handcrafted_features(Xa, BC.EXPOSURE),
             " + projection": B.handcrafted_features(Xa @ P.T, BC.EXPOSURE)}
    out = {}
    for suffix, F in feats.items():
        sc = StandardScaler().fit(F[m_fit]); Fz = sc.transform(F)
        models = [("Logistic regression" + suffix,
                   LogisticRegression(max_iter=6000, class_weight="balanced")),
                  ("SVM (RBF)" + suffix,
                   SVC(probability=True, class_weight="balanced",
                       random_state=seed))]
        if suffix == "":
            models.append(("Random forest",
                           RandomForestClassifier(500, random_state=seed, n_jobs=2,
                                                  class_weight="balanced")))
        for tag, clf in models:
            t0 = time.time()
            clf.fit(Fz[m_fit], y[m_fit])
            p = clf.predict_proba(Fz[m_te])
            out[tag] = dict(metrics=E.all_metrics(p, y[m_te], N_CLASSES), prob=p,
                            seconds=time.time() - t0, params=int(F.shape[1]))
    F = feats[""]
    sc = StandardScaler().fit(F[m_fit])
    t0 = time.time()
    R = B.rocket_transform(Xa, n_kernels=900, seed=seed)
    sc2 = StandardScaler().fit(R[m_fit]); Rz = sc2.transform(R)
    rc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                           class_weight="balanced").fit(Rz[m_fit], y[m_fit])
    d = rc.decision_function(Rz[m_te])
    p = np.exp(d - d.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    out["ROCKET + ridge"] = dict(metrics=E.all_metrics(p, y[m_te], N_CLASSES), prob=p,
                                 seconds=time.time() - t0, params=int(R.shape[1]))
    return out


# ----------------------------------------------------------------------
def E1_main(raw, seeds=C.SEED_LIST, log=print):
    basis = physical_basis()
    models = ["KINDRA"] + list(B.TORCH_BASELINES.keys())
    rec = {}
    probs = {}
    for seed in seeds:
        masks = D.subject_split(raw["subject"], raw["y"], seed=seed,
                                frac=(0.65, 0.15, 0.20))
        Xa, nu, loaders, cw = prepare(raw, masks)
        y_te = raw["y"][masks[2]]
        for name in models:
            t0 = time.time()
            net, info = fit_one(name, loaders, cw, seed, basis=basis)
            p, yy, u = T.evaluate_model(net, loaders[2], N_CLASSES)
            m = E.all_metrics(p, yy, N_CLASSES)
            m["params"] = T.count_params(net)
            m["train_seconds"] = info["seconds"]
            m["latency_ms"] = T.inference_latency(net, Xa[masks[2]])
            rec.setdefault(name, []).append(m)
            probs.setdefault(name, []).append((p, yy, u))
            log(f"[E1] seed {seed} {name:26s} bal {m['balanced_accuracy']:.3f} "
                f"auc {m['macro_auroc']:.3f} ece {m['ece']:.3f} ({time.time()-t0:.0f}s)")
        cls = classical_scores(Xa, raw["y"], masks, seed)
        for tag, v in cls.items():
            rec.setdefault(tag, []).append({**v["metrics"], "params": v["params"],
                                            "train_seconds": v["seconds"],
                                            "latency_ms": float("nan")})
            probs.setdefault(tag, []).append((v["prob"], y_te,
                                              1.0 - v["prob"].max(1)))
            log(f"[E1] seed {seed} {tag:26s} bal {v['metrics']['balanced_accuracy']:.3f} "
                f"auc {v['metrics']['macro_auroc']:.3f}")
    with open(os.path.join(RES, "E1_main.json"), "w") as f:
        json.dump(rec, f, indent=1)
    with open(os.path.join(C.CACHE_DIR, "E1_probs.pkl"), "wb") as f:
        pickle.dump(probs, f)
    return rec, probs


ABLATIONS = [
    ("Base (no proposed component)", dict(use_proj=False, use_kin=False,
                                          use_vel=False, use_evi=False)),
    ("+ A: dispersion projection", dict(use_proj=True, use_kin=False,
                                        use_vel=False, use_evi=False)),
    ("+ B: kinetic state cell", dict(use_proj=False, use_kin=True,
                                     use_vel=False, use_evi=False)),
    ("+ C: velocity coupling", dict(use_proj=False, use_kin=True,
                                    use_vel=True, use_evi=False)),
    ("+ A + B", dict(use_proj=True, use_kin=True, use_vel=False, use_evi=False)),
    ("+ A + C", dict(use_proj=True, use_kin=True, use_vel=True, use_evi=False)),
    ("+ B + C", dict(use_proj=False, use_kin=True, use_vel=True, use_evi=True)),
    ("Full model", dict(use_proj=True, use_kin=True, use_vel=True, use_evi=True)),
]


def E2_ablation(raw, seeds=C.SEED_LIST[:3], log=print):
    basis = physical_basis()
    rec = {}
    for seed in seeds:
        masks = D.subject_split(raw["subject"], raw["y"], seed=seed,
                                frac=(0.65, 0.15, 0.20))
        Xa, nu, loaders, cw = prepare(raw, masks)
        for tag, kw in ABLATIONS:
            net, info = fit_one("KINDRA", loaders, cw, seed, basis=basis, ablate=kw)
            p, yy, _ = T.evaluate_model(net, loaders[2], N_CLASSES)
            m = E.all_metrics(p, yy, N_CLASSES)
            m["params"] = T.count_params(net)
            rec.setdefault(tag, []).append(m)
            log(f"[E2] seed {seed} {tag:30s} bal {m['balanced_accuracy']:.3f} "
                f"auc {m['macro_auroc']:.3f} ece {m['ece']:.3f}")
    with open(os.path.join(RES, "E2_ablation.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def E3_sensitivity(raw, seeds=C.SEED_LIST[:2], log=print):
    grid = ([("proj_rank", r) for r in (0, 1, 2, 3, 4, 5)] +
            [("lambda_kin", v) for v in (0.0, 0.6, 1.2)] +
            [("lambda_adv", v) for v in (0.0, 0.40)])
    rec = {}
    for seed in seeds:
        masks = D.subject_split(raw["subject"], raw["y"], seed=seed,
                                frac=(0.65, 0.15, 0.20))
        Xa, nu, loaders, cw = prepare(raw, masks)
        for key, val in grid:
            cfg = copy.copy(C.TRAIN); setattr(cfg, key, val)
            b = physical_basis(max(int(cfg.proj_rank), 1))
            ab = dict(use_proj=cfg.proj_rank > 0)
            net = T.build_kindra(N_CH, N_CLASSES, b, cfg=cfg,
                                 exposure=exposure_vector(),
                                 use_proj=ab["use_proj"], use_kin=True,
                                 use_vel=True, use_evi=True)
            T.train_model(net, loaders, N_CLASSES, cfg=cfg, is_kindra=True,
                          seed=seed, class_weight=cw)
            p, yy, _ = T.evaluate_model(net, loaders[2], N_CLASSES)
            m = E.all_metrics(p, yy, N_CLASSES)
            rec.setdefault(f"{key}={val}", []).append(m)
            log(f"[E3] seed {seed} {key}={val:<6} bal {m['balanced_accuracy']:.3f} "
                f"auc {m['macro_auroc']:.3f}")
    with open(os.path.join(RES, "E3_sensitivity.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


# ----------------------------------------------------------------------
def perturb(X, kind, level, rng, basis_phys):
    """Corruptions applied at test time only."""
    Xp = X.copy()
    T_, K = X.shape[1], X.shape[2]
    if kind == "drift":
        for i in range(len(Xp)):
            amp = level * rng.standard_normal(basis_phys.shape[1])
            prof = np.linspace(0.0, 1.0, T_)[:, None] * (basis_phys @ amp)[None, :]
            w = np.cumsum(rng.standard_normal(T_)) / np.sqrt(T_)
            Xp[i] += prof + 0.5 * level * np.outer(w, basis_phys[:, 0])
    elif kind == "noise":
        Xp += level * rng.standard_normal(Xp.shape)
    elif kind == "channel_loss":
        n_drop = int(round(level))
        for i in range(len(Xp)):
            ch = rng.choice(K, size=n_drop, replace=False)
            Xp[i][:, ch] = 0.0
    elif kind == "gain":
        Xp *= (1.0 + level * rng.standard_normal((len(Xp), 1, K)))
    elif kind == "timing":
        s = int(round(level))
        for i in range(len(Xp)):
            k = rng.integers(-s, s + 1)
            Xp[i] = np.roll(Xp[i], k, axis=0)
    return Xp.astype(np.float32)


def E4_robustness(raw, seeds=C.SEED_LIST[:2], log=print):
    basis = physical_basis()
    meta = pickle.load(open(os.path.join(C.CACHE_DIR, "devices.pkl"), "rb"))
    Bphys = np.linalg.svd(np.concatenate([d["drift_basis"] for d in meta], axis=1),
                          full_matrices=False)[0][:, :3]
    grid = [("drift", v) for v in (0.25, 0.5, 1.0, 2.0)] + \
           [("noise", v) for v in (0.05, 0.15, 0.30)] + \
           [("channel_loss", v) for v in (1, 2, 3)] + \
           [("gain", v) for v in (0.05, 0.15)] + \
           [("timing", v) for v in (2, 4)]
    models = ["KINDRA", "GRU", "LSTM-GRU (stacked)", "GRU-DANN", "LSTM-FCN",
              "InceptionTime"]
    rec = {}
    for seed in seeds:
        masks = D.subject_split(raw["subject"], raw["y"], seed=seed,
                                frac=(0.65, 0.15, 0.20))
        Xa, nu, loaders, cw = prepare(raw, masks)
        m_te = masks[2]; y_te = raw["y"][m_te]
        nets = {}
        for name in models:
            nets[name], _ = fit_one(name, loaders, cw, seed, basis=basis)
        for kind, lvl in grid:
            rng = np.random.default_rng(1000 + seed)
            Xp = perturb(Xa[m_te], kind, lvl, rng, Bphys)
            ld = D.make_loaders(Xp, y_te, nu[m_te],
                                [np.ones(len(y_te), bool)], shuffle_train=False)[0]
            for name, net in nets.items():
                p, yy, _ = T.evaluate_model(net, ld, N_CLASSES)
                m = E.all_metrics(p, yy, N_CLASSES)
                rec.setdefault(f"{kind}={lvl}", {}).setdefault(name, []).append(m)
            log(f"[E4] seed {seed} {kind}={lvl}: " + " ".join(
                f"{n}={rec[f'{kind}={lvl}'][n][-1]['balanced_accuracy']:.3f}"
                for n in models))
    with open(os.path.join(RES, "E4_robustness.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def E5_generalisation(raw, folds=((6, 7), (4, 5)), seeds=C.SEED_LIST[:2],
                      log=print):
    basis = physical_basis()
    models = ["KINDRA", "GRU", "LSTM", "LSTM-GRU (stacked)", "GRU-DANN",
              "LSTM-FCN", "InceptionTime"]
    rec = {}
    for fold in folds:
        m_in, m_out = D.device_split(raw["device"], held_out=fold)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = np.where(m_in)[0]; rng.shuffle(idx)
            m_tr = np.zeros(len(m_in), bool); m_va = np.zeros(len(m_in), bool)
            cut = int(0.85 * len(idx))
            m_tr[idx[:cut]] = True; m_va[idx[cut:]] = True
            masks = (m_tr, m_va, m_out)
            Xa, nu, loaders, cw = prepare(raw, masks)
            for name in models:
                net, _ = fit_one(name, loaders, cw, seed, basis=basis)
                p, yy, _ = T.evaluate_model(net, loaders[2], N_CLASSES)
                m = E.all_metrics(p, yy, N_CLASSES)
                rec.setdefault(name, []).append(m)
                log(f"[E5] fold {fold} seed {seed} {name:22s} "
                    f"bal {m['balanced_accuracy']:.3f} auc {m['macro_auroc']:.3f} "
                    f"ece {m['ece']:.3f}")
    with open(os.path.join(RES, "E5_generalisation.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def E8_uncertainty(raw, seeds=C.SEED_LIST[:2], log=print):
    """Compare the evidential head against temperature scaling, ensembles and dropout."""
    from scipy.optimize import minimize_scalar
    basis = physical_basis()
    rec = {}
    for seed in seeds:
        masks = D.subject_split(raw["subject"], raw["y"], seed=seed,
                                frac=(0.65, 0.15, 0.20))
        Xa, nu, loaders, cw = prepare(raw, masks)
        # proposed evidential model
        net, _ = fit_one("KINDRA", loaders, cw, seed, basis=basis)
        p, yy, u = T.evaluate_model(net, loaders[2], N_CLASSES)
        rec.setdefault("KINDRA (evidential)", []).append(E.all_metrics(p, yy, N_CLASSES))
        cov, risk = E.risk_coverage(p, yy, u)
        rec.setdefault("_rc_kindra", []).append([cov.tolist(), risk.tolist()])
        # softmax variant, raw and temperature scaled
        net2, _ = fit_one("KINDRA", loaders, cw, seed, basis=basis,
                          ablate=dict(use_evi=False))
        pv, yv, _ = T.evaluate_model(net2, loaders[1], N_CLASSES)
        pt, yt, ut = T.evaluate_model(net2, loaders[2], N_CLASSES)
        rec.setdefault("KINDRA (softmax)", []).append(E.all_metrics(pt, yt, N_CLASSES))
        lv = np.log(np.clip(pv, 1e-12, 1)); lt = np.log(np.clip(pt, 1e-12, 1))

        def nll_T(logT):
            q = np.exp(lv / np.exp(logT))
            q /= q.sum(1, keepdims=True)
            return E.negative_log_likelihood(q, yv)
        r = minimize_scalar(nll_T, bounds=(-2.0, 2.0), method="bounded")
        q = np.exp(lt / np.exp(r.x)); q /= q.sum(1, keepdims=True)
        rec.setdefault("KINDRA (temperature scaled)", []).append(
            E.all_metrics(q, yt, N_CLASSES))
        # deep ensemble of softmax variants
        ps = [pt]
        for k in (1, 2):
            nk, _ = fit_one("KINDRA", loaders, cw, seed + 100 * k, basis=basis,
                            ablate=dict(use_evi=False))
            pk, _, _ = T.evaluate_model(nk, loaders[2], N_CLASSES)
            ps.append(pk)
        pe = np.mean(ps, 0)
        rec.setdefault("Deep ensemble (3)", []).append(E.all_metrics(pe, yt, N_CLASSES))
        # Monte-Carlo dropout: dropout layers are re-enabled while every
        # normalisation layer stays in evaluation mode, so the running
        # statistics are not disturbed by the stochastic passes.
        net2.eval()
        for mod in net2.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.train()
        acc = []
        with torch.no_grad():
            for _ in range(20):
                pp = []
                for x, yb, nb in loaders[2]:
                    o = net2(x)
                    pp.append(torch.softmax(o["logits"], -1).numpy())
                acc.append(np.concatenate(pp))
        pm = np.mean(acc, 0)
        rec.setdefault("MC dropout (20)", []).append(E.all_metrics(pm, yt, N_CLASSES))
        log(f"[E8] seed {seed} done")
    with open(os.path.join(RES, "E8_uncertainty.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def E0_ceiling(raw, seeds=C.SEED_LIST, log=print):
    """
    Information budget of the measurement chain.

    Three classifiers of identical form are fitted: on the true alveolar
    concentrations, on the nuisance-free sensorgram and on the observed
    sensorgram.  The differences bound how much of the available information is
    lost by transduction and how much by the transduction nuisance, and they set
    the scale against which any read-out should be judged.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rec = {}
    for seed in seeds:
        m_tr, m_va, m_te = D.subject_split(raw["subject"], raw["y"], seed=seed,
                                           frac=(0.65, 0.15, 0.20))
        fit = m_tr | m_va
        feats = {
            "True alveolar concentrations": np.log(raw["conc"] + 1e-3),
            "Nuisance-free sensorgram": B.handcrafted_features(raw["X_clean"],
                                                               BC.EXPOSURE),
            "Observed sensorgram": B.handcrafted_features(raw["X"], BC.EXPOSURE)}
        meta = pickle.load(open(os.path.join(C.CACHE_DIR, "devices.pkl"), "rb"))
        U = np.linalg.svd(np.concatenate([d["drift_basis"] for d in meta], axis=1),
                          full_matrices=False)[0]
        for r in (2, 3):
            P = np.eye(N_CH) - U[:, :r] @ U[:, :r].T
            feats[f"Observed, physical projection r={r}"] = \
                B.handcrafted_features(raw["X"] @ P.T, BC.EXPOSURE)
        for tag, F in feats.items():
            sc = StandardScaler().fit(F[fit]); Fz = sc.transform(F)
            clf = LogisticRegression(max_iter=6000, class_weight="balanced")
            clf.fit(Fz[fit], raw["y"][fit])
            p = clf.predict_proba(Fz[m_te])
            m = E.all_metrics(p, raw["y"][m_te], N_CLASSES)
            rec.setdefault(tag, []).append(m)
            log(f"[E0] seed {seed} {tag:38s} bal {m['balanced_accuracy']:.3f} "
                f"auc {m['macro_auroc']:.3f}")
    with open(os.path.join(RES, "E0_ceiling.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec
