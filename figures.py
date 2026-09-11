"""
Publication figures.

Every panel is drawn at final column width, in a grayscale-safe palette with
redundant marker and line-style encoding, so that no comparison depends on
colour reproduction.
"""
from __future__ import annotations
import json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linestyle": ":", "legend.frameon": False})

COL = 3.45      # single column width, inches
DCOL = 7.16     # double column width
GREY = ["#111111", "#4d4d4d", "#7a7a7a", "#a6a6a6", "#c9c9c9"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "d"]
LSTY = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def _save(fig, name):
    p = os.path.join(C.FIGURE_DIR, name)
    fig.savefig(p + ".png"); plt.close(fig)
    return p + ".png"


def load(name):
    with open(os.path.join(C.RESULTS_DIR, name)) as f:
        return json.load(f)


def agg(rec, metric):
    out = {}
    for k, v in rec.items():
        if k.startswith("_"):
            continue
        a = np.array([m[metric] for m in v], float)
        out[k] = (a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0, a)
    return out


# ----------------------------------------------------------------------
def fig_architecture(path="fig1_architecture"):
    fig, ax = plt.subplots(figsize=(DCOL, 2.65))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, t, fc="white", fs=6.4, lw=0.7, style="round,pad=0.25"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                    fc=fc, ec="black", lw=lw))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, style="-|>", ls="-", lw=0.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=7, lw=lw, ls=ls,
                                     color="black", shrinkA=1, shrinkB=1))

    box(1, 16, 12, 11, "Breath sample\n3 exhalation cycles\nambient purge", "#f0f0f0")
    box(15.5, 16, 13, 11, "Plasmonic array\n12 functionalised\nspots, Kretschmann", "#f0f0f0")
    box(31, 16, 11, 11, "Resonance\ntracking\n$\\lambda_{res}(t)$", "#f0f0f0")
    arrow(13, 21.5, 15.5, 21.5); arrow(28.5, 21.5, 31, 21.5); arrow(42, 21.5, 45, 21.5)

    box(45, 15, 13, 13, "Dispersion\ndisentangling\nprojection\n(mechanism A)", "#dcdcdc")
    arrow(51.5, 15, 51.5, 11.5, ls=":")
    box(44, 4, 15, 6.5, "co-operative and\nadversarial drift heads", "#f7f7f7", 5.9)

    box(61, 26, 15, 10, "Langmuir kinetic\nstate cell\n(mechanism B)", "#dcdcdc")
    box(61, 12, 15, 10, "velocity-modulated\ngated branch\n(mechanism C)", "#dcdcdc")
    arrow(58, 24, 61, 31); arrow(58, 20, 61, 17)
    arrow(68.5, 26, 68.5, 22, style="-|>", ls="--")
    ax.text(69.4, 24.0, "$\\Delta\\theta_t$", fontsize=6.0, ha="left", va="center")
    ax.text(60.5, 8.6, "residual $\\mathbf{z}_t-\\hat{\\mathbf{z}}_t$", fontsize=5.9)
    arrow(64, 12, 64, 10.2, style="<|-", ls="-.")

    box(79, 19, 9.5, 9, "fusion", "#f0f0f0")
    arrow(76, 31, 79, 25); arrow(76, 17, 79, 22)
    box(90.5, 19, 8.5, 9, "Dirichlet\nevidence", "#dcdcdc")
    arrow(88.5, 23.5, 90.5, 23.5)
    ax.text(94.7, 15.5, "class belief\n+ abstention", ha="center", va="top", fontsize=5.9)
    ax.text(50, 39.4, "acquisition", ha="center", fontsize=7.0, style="italic")
    ax.text(84, 39.4, "decision", ha="center", fontsize=7.0, style="italic")
    ax.plot([44, 44], [1, 37], color="black", lw=0.5, ls=":")
    ax.plot([77.5, 77.5], [1, 37], color="black", lw=0.5, ls=":")
    return _save(fig, path)


def fig_physics(dev_meta, example, path="fig2_physics"):
    fig = plt.figure(figsize=(DCOL, 3.55))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    for i, (k, (lam, R)) in enumerate(zip([1, 6, 12], example["spectra"])):
        lam = np.asarray(lam); R = np.asarray(R)
        ax.plot(lam, R, color=GREY[i], ls=LSTY[i], label=f"spot {k}")
        j = int(np.argmin(R))
        ax.plot([lam[j]], [R[j]], marker=MARK[i], ms=3.0, color=GREY[i])
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("reflectance")
    ax.set_ylim(-0.03, 1.0); ax.legend(loc="lower right", handlelength=1.8)
    ax.set_title("(a) computed resonance", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    k = np.arange(1, 13)
    ax.bar(k - 0.2, dev_meta["s_bulk"] / 1000.0, 0.4, color=GREY[1],
           edgecolor="black", lw=0.4, label="$S^{\\mathrm{blk}}$")
    ax.bar(k + 0.2, dev_meta["s_surf"] / 1000.0, 0.4, color=GREY[3],
           edgecolor="black", lw=0.4, hatch="///", label="$S^{\\mathrm{srf}}$")
    ax.set_xlabel("sensing spot")
    ax.set_ylabel("sensitivity (10$^{3}$ nm RIU$^{-1}$)")
    ax.set_xticks(k[::2]); ax.legend(loc="upper left")
    ax.set_title("(b) response coefficients", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    r = np.asarray(example["rank_curve"])
    ax.plot(r[:, 0], r[:, 1], color=GREY[0], ls=LSTY[0], marker=MARK[0], ms=2.8,
            label="nuisance")
    ax.plot(r[:, 0], r[:, 2], color=GREY[2], ls=LSTY[1], marker=MARK[1], ms=2.8,
            label="binding")
    ax.plot(r[:, 0], r[:, 3], color=GREY[1], ls=LSTY[2], marker=MARK[2], ms=2.8,
            label="between-class")
    ax.axvline(3, color="0.55", lw=0.7, ls=":")
    ax.set_xlabel("projection rank $r$"); ax.set_ylabel("retained energy")
    ax.set_ylim(-0.03, 1.05); ax.legend(loc="upper right")
    ax.set_title("(c) what the projection removes", loc="left")

    ax = fig.add_subplot(gs[1, :2])
    t = example["t"]
    ax.plot(t, example["obs"], color=GREY[0], lw=0.9,
            label="observed $\\Delta\\lambda_k(t)$")
    ax.plot(t, example["surf"], color=GREY[2], ls="--", lw=0.9,
            label="binding component")
    ax.plot(t, example["nuis"], color=GREY[1], ls=":", lw=0.9,
            label="transduction nuisance")
    for a, b in example["windows"]:
        ax.axvspan(a, b, color="0.90", zorder=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("$\\Delta\\lambda$ (nm)")
    ax.legend(ncol=3, loc="upper left", handlelength=2.0)
    ax.set_title("(d) decomposition of one sensorgram "
                 "(spot 1, shaded: exhalation)", loc="left")

    ax = fig.add_subplot(gs[1, 2])
    tf, ts = example["tau"]
    bins = np.linspace(np.log10(min(tf.min(), ts.min())),
                       np.log10(max(tf.max(), ts.max())), 22)
    ax.hist(np.log10(tf), bins=bins, color=GREY[3], edgecolor="black", lw=0.4,
            density=True, label="physisorption")
    ax.hist(np.log10(ts), bins=bins, histtype="step", color=GREY[0], lw=1.1,
            density=True, label="carbonyl capture")
    ax.set_xlabel("$\\log_{10}\\tau_d$ (s)"); ax.set_ylabel("density")
    ax.legend(loc="upper center")
    ax.set_title("(e) two desorption regimes", loc="left")
    return _save(fig, path)


def fig_main(rec, path="fig4_benchmark"):
    ba = agg(rec, "balanced_accuracy"); au = agg(rec, "macro_auroc")
    order = sorted(ba, key=lambda k: ba[k][0])
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.9), sharey=True)
    ypos = np.arange(len(order))
    for ax, d, lab in zip(axes, (ba, au),
                          ("balanced accuracy", "macro AUROC")):
        vals = [d[k][0] for k in order]; err = [d[k][1] for k in order]
        cols = [GREY[0] if k == "KINDRA" else GREY[3] for k in order]
        ax.barh(ypos, vals, xerr=err, color=cols, edgecolor="black", lw=0.5,
                error_kw=dict(lw=0.7, capsize=1.8))
        ax.set_xlabel(lab)
        lo = min(v - e for v, e in zip(vals, err))
        ax.set_xlim(max(0.0, lo - 0.03), max(vals) + max(err) + 0.02)
        for i, k in enumerate(order):
            ax.text(vals[i] + err[i] + 0.004, i, f"{vals[i]:.3f}", va="center",
                    fontsize=5.6)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(order)
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    return _save(fig, path)


def fig_ablation(rec, path="fig5_ablation"):
    ba = agg(rec, "balanced_accuracy"); au = agg(rec, "macro_auroc")
    ece = agg(rec, "ece")
    order = list(rec.keys())
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.6))
    ax = axes[0]
    ax.bar(x - 0.2, [ba[k][0] for k in order], 0.4, yerr=[ba[k][1] for k in order],
           color=GREY[1], edgecolor="black", lw=0.5, label="balanced accuracy",
           error_kw=dict(lw=0.7, capsize=1.6))
    ax.bar(x + 0.2, [au[k][0] for k in order], 0.4, yerr=[au[k][1] for k in order],
           color=GREY[3], edgecolor="black", lw=0.5, hatch="///", label="macro AUROC",
           error_kw=dict(lw=0.7, capsize=1.6))
    ax.set_xticks(x); ax.set_xticklabels([o.replace(" (no proposed component)", "")
                                          for o in order], rotation=38, ha="right")
    ax.set_ylim(0.55, 0.96); ax.legend(loc="upper left"); ax.set_ylabel("score")
    ax.set_title("(a) component ablation", loc="left")
    ax = axes[1]
    ax.bar(x, [ece[k][0] for k in order], 0.55, yerr=[ece[k][1] for k in order],
           color=GREY[2], edgecolor="black", lw=0.5, error_kw=dict(lw=0.7, capsize=1.6))
    ax.set_xticks(x); ax.set_xticklabels([o.replace(" (no proposed component)", "")
                                          for o in order], rotation=38, ha="right")
    ax.set_ylabel("expected calibration error")
    ax.set_title("(b) calibration", loc="left")
    return _save(fig, path)


def fig_sensitivity(rec, path="fig6_sensitivity"):
    groups = {}
    for k, v in rec.items():
        key, val = k.split("=")
        groups.setdefault(key, []).append((float(val), v))
    fig, axes = plt.subplots(1, len(groups), figsize=(DCOL, 2.0))
    titles = {"proj_rank": "projection rank $r$", "lambda_kin": "$\\lambda_{\\mathrm{kin}}$",
              "lambda_adv": "$\\lambda_{\\mathrm{drift}}$", "hidden_slow": "sites $P$"}
    for ax, (key, items) in zip(np.atleast_1d(axes), groups.items()):
        items.sort()
        xs = [i[0] for i in items]
        m = [np.mean([d["balanced_accuracy"] for d in i[1]]) for i in items]
        s = [np.std([d["balanced_accuracy"] for d in i[1]], ddof=1) for i in items]
        ax.errorbar(xs, m, yerr=s, color=GREY[0], marker="o", ms=3.0, lw=0.9,
                    capsize=2.0, elinewidth=0.7)
        ax.set_xlabel(titles.get(key, key)); ax.set_ylabel("balanced accuracy")
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel("")
    return _save(fig, path)


def fig_robustness(rec, path="fig7_robustness"):
    kinds = {}
    for k, per_model in rec.items():
        kind, lvl = k.split("=")
        kinds.setdefault(kind, []).append((float(lvl), per_model))
    keep = ["drift", "noise", "channel_loss", "gain"]
    names = {"drift": "injected drift amplitude", "noise": "additive noise $\\sigma$",
             "channel_loss": "failed spots", "gain": "gain mismatch"}
    fig, axes = plt.subplots(1, len(keep), figsize=(DCOL, 2.0))
    models = None
    for ax, kind in zip(axes, keep):
        items = sorted(kinds[kind])
        models = list(items[0][1].keys())
        for i, mname in enumerate(models):
            xs = [it[0] for it in items]
            ys = [np.mean([d["balanced_accuracy"] for d in it[1][mname]]) for it in items]
            ax.plot(xs, ys, color=GREY[i % len(GREY)], ls=LSTY[i % len(LSTY)],
                    marker=MARK[i % len(MARK)], ms=2.6, lw=0.9, label=mname)
        ax.set_xlabel(names[kind])
    axes[0].set_ylabel("balanced accuracy")
    axes[0].legend(loc="lower left", ncol=1, handlelength=2.2, fontsize=5.2)
    return _save(fig, path)


def fig_generalisation(rec5, ext, path="fig8_generalisation"):
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.4))
    ba = agg(rec5, "balanced_accuracy"); au = agg(rec5, "macro_auroc")
    order = sorted(ba, key=lambda k: ba[k][0])
    y = np.arange(len(order))
    ax = axes[0]
    ax.barh(y - 0.2, [ba[k][0] for k in order], 0.4, xerr=[ba[k][1] for k in order],
            color=GREY[1], edgecolor="black", lw=0.5, label="balanced accuracy",
            error_kw=dict(lw=0.6, capsize=1.5))
    ax.barh(y + 0.2, [au[k][0] for k in order], 0.4, xerr=[au[k][1] for k in order],
            color=GREY[3], edgecolor="black", lw=0.5, hatch="///", label="macro AUROC",
            error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(0.4, 0.95)
    ax.legend(loc="lower right"); ax.set_xlabel("score")
    ax.set_title("(a) leave-chips-out", loc="left")
    ax = axes[1]
    tb = ext["test_batches"]
    for i, (k, v) in enumerate(ext["results"].items()):
        a = np.array(v)
        ax.plot(tb, a.mean(0), color=GREY[i % len(GREY)], ls=LSTY[i % len(LSTY)],
                marker=MARK[i], ms=2.8, lw=0.9, label=k)
    ax.set_xlabel("test batch (increasing elapsed time)")
    ax.set_ylabel("accuracy"); ax.legend(loc="lower left", fontsize=5.2, handlelength=2.2)
    ax.set_title("(b) real chemosensor drift benchmark", loc="left")
    return _save(fig, path)


def fig_uncertainty(rec8, probs, path="fig9_uncertainty"):
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.15))
    p, y, u = probs["KINDRA"][0]
    conf = p.max(1); correct = (p.argmax(1) == y)
    edges = np.linspace(0, 1, 11)
    accs, confs, ws = [], [], []
    for i in range(10):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() < 5:
            continue
        accs.append(correct[m].mean()); confs.append(conf[m].mean()); ws.append(m.mean())
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY[3], ls="--", lw=0.8)
    ax.plot(confs, accs, color=GREY[0], marker="o", ms=3.0, lw=1.0, label="evidential")
    p2, y2, _ = probs["GRU"][0]
    c2 = p2.max(1); k2 = (p2.argmax(1) == y2)
    a2, f2 = [], []
    for i in range(10):
        m = (c2 > edges[i]) & (c2 <= edges[i + 1])
        if m.sum() < 5:
            continue
        a2.append(k2[m].mean()); f2.append(c2[m].mean())
    ax.plot(f2, a2, color=GREY[2], marker="s", ms=3.0, lw=1.0, ls="-.", label="GRU softmax")
    ax.set_xlabel("confidence"); ax.set_ylabel("empirical accuracy")
    ax.legend(loc="upper left"); ax.set_title("(a) reliability", loc="left")

    ax = axes[1]
    for i, (name, pr) in enumerate([("KINDRA", probs["KINDRA"][0]),
                                    ("GRU", probs["GRU"][0]),
                                    ("LSTM-GRU (stacked)", probs["LSTM-GRU (stacked)"][0])]):
        pp, yy, uu = pr
        import evaluate as EV
        cov, risk = EV.risk_coverage(pp, yy, uu)
        ax.plot(cov, risk, color=GREY[i], ls=LSTY[i], lw=1.0, label=name)
    ax.set_xlabel("coverage"); ax.set_ylabel("selective error")
    ax.legend(loc="upper left", fontsize=5.4); ax.set_title("(b) risk-coverage", loc="left")

    ax = axes[2]
    labels = list(rec8.keys())
    labels = [l for l in labels if not l.startswith("_")]
    ec = [np.mean([d["ece"] for d in rec8[l]]) for l in labels]
    es = [np.std([d["ece"] for d in rec8[l]], ddof=1) for l in labels]
    ax.barh(np.arange(len(labels)), ec, xerr=es, color=GREY[2], edgecolor="black",
            lw=0.5, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([l.replace("KINDRA ", "") for l in labels], fontsize=5.6)
    ax.set_xlabel("expected calibration error")
    ax.set_title("(c) uncertainty methods", loc="left")
    return _save(fig, path)


def fig_failure(probs, path="fig10_failure"):
    import evaluate as EV
    from sklearn.metrics import f1_score
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.15))
    P = np.concatenate([p for p, y, u in probs["KINDRA"]])
    Y = np.concatenate([y for p, y, u in probs["KINDRA"]])
    U = np.concatenate([u for p, y, u in probs["KINDRA"]])
    cm = EV.confusion(P, Y).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    ax = axes[0]
    ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(C.CLASSES, rotation=40, ha="right"); ax.set_yticklabels(C.CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=5.4,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.grid(False); ax.set_ylabel("true"); ax.set_xlabel("predicted")
    ax.set_title("(a) confusion", loc="left")

    ax = axes[1]
    f_k = f1_score(Y, P.argmax(1), average=None)
    Pg = np.concatenate([p for p, y, u in probs["GRU"]])
    Yg = np.concatenate([y for p, y, u in probs["GRU"]])
    f_g = f1_score(Yg, Pg.argmax(1), average=None)
    x = np.arange(4)
    ax.bar(x - 0.2, f_k, 0.4, color=GREY[0], edgecolor="black", lw=0.5, label="KINDRA")
    ax.bar(x + 0.2, f_g, 0.4, color=GREY[3], edgecolor="black", lw=0.5, hatch="///",
           label="GRU")
    ax.set_xticks(x); ax.set_xticklabels(C.CLASSES, rotation=40, ha="right")
    ax.set_ylabel("$F_1$"); ax.legend(loc="lower left")
    ax.set_title("(b) per-class", loc="left")

    ax = axes[2]
    ok = P.argmax(1) == Y
    bins = np.linspace(U.min(), U.max(), 22)
    ax.hist(U[ok], bins=bins, color=GREY[3], edgecolor="black", lw=0.4,
            density=True, label="correct")
    ax.hist(U[~ok], bins=bins, color=GREY[1], edgecolor="black", lw=0.4,
            density=True, alpha=0.8, hatch="\\\\", label="incorrect")
    ax.set_xlabel("evidential uncertainty"); ax.set_ylabel("density")
    ax.legend(loc="upper left"); ax.set_title("(c) uncertainty separation", loc="left")
    return _save(fig, path)
