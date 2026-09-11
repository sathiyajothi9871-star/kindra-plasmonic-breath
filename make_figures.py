"""Generate every manuscript figure."""
from __future__ import annotations
import json, os, pickle
import numpy as np
import config as C
import figures as F


def physics_example():
    import optics, kinetics as KIN, breath_cohort as BC
    dev = BC.Device(0, np.random.default_rng(7)); BC.calibrate_offset(dev)
    spectra = []
    for k in (0, 5, 11):
        ch = dev.array["channels"][k]
        spectra.append((ch.lam.tolist(), ch.reflectance(1.00027).tolist()))
    import dataset as D, pickle, os
    raw = D.load_raw()
    meta = pickle.load(open(os.path.join(C.CACHE_DIR, "devices.pkl"), "rb"))
    G = np.linalg.svd(np.concatenate([m["drift_basis"] for m in meta], axis=1),
                      full_matrices=False)[0]
    N = raw["X"] - raw["X_clean"]
    Sg = raw["X_clean"]
    Mc = np.stack([Sg[raw["y"] == c].mean(axis=(0, 1)) for c in range(4)])
    Mc = Mc - Mc.mean(0)
    curve = []
    for r in range(0, 7):
        P = np.eye(C.N_CHANNELS) - (G[:, :r] @ G[:, :r].T if r else 0.0)
        curve.append([r, ((N @ P) ** 2).sum() / (N ** 2).sum(),
                      ((Sg @ P) ** 2).sum() / (Sg ** 2).sum(),
                      ((Mc @ P) ** 2).sum() / (Mc ** 2).sum()])
    amb = np.array([BC.HEALTHY_GM[n] for n in KIN.VOC_NAMES[:-1]]) * BC.AMBIENT_FRACTION
    c = BC.sample_subject("LungCancer", np.random.default_rng(5))
    x, clean, lat = BC.simulate_session(dev, c, amb, 0.42, 0.95, 2.0, 9.0,
                                        np.random.default_rng(3), return_latent=True)
    t = np.arange(C.T_STEPS) * 2.0
    win, s = [], C.BASELINE_S
    for _ in range(C.N_CYCLES):
        win.append((s, s + C.EXPOSURE_S)); s += C.EXPOSURE_S + C.PURGE_S
    amine = KIN.GROUP_GAIN[:C.N_CHANNELS, 0] > 1.0
    ald = np.array([v[5] > 0.5 for v in KIN.VOC_PANEL])
    chem = np.zeros_like(dev.tau, dtype=bool)
    chem[np.ix_(amine, ald)] = True
    tau_slow = dev.tau[chem].ravel()
    tau_fast = dev.tau[~chem].ravel()
    return dict(spectra=spectra, rank_curve=curve, t=t.tolist(),
                obs=x[:, 0].tolist(), surf=clean[:, 0].tolist(),
                nuis=(x[:, 0] - clean[:, 0]).tolist(), windows=win,
                tau=[tau_fast, tau_slow]), dict(
                    s_bulk=dev.s_bulk, s_surf=dev.s_surf)


def fig_mechanism(path="fig3_mechanism"):
    """Schematic of the kinetic cell and an illustration of the two rates."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import kinetics as KIN, breath_cohort as BC
    fig = plt.figure(figsize=(F.DCOL, 2.35))
    gs = fig.add_gridspec(1, 3, wspace=0.34, width_ratios=[1.25, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.grid(False)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def box(x, y, w, h, t, fc="white", fs=5.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                    fc=fc, ec="black", lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    def arr(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.75, ls=ls, color="black"))
    box(2, 74, 30, 16, "$\\mathbf{z}_t$", "#f2f2f2")
    box(2, 46, 30, 18, "shared encoder\n$\\mathbf{u}_t\\in\\mathbb{R}_{+}^{J'}$", "#e2e2e2")
    box(38, 46, 30, 18, "flux\n$\\mathbf{k}^{a}\\!\\odot\\!\\mathbf{u}_t$", "#e2e2e2")
    box(38, 74, 30, 16, "$\\mathbf{k}^{a}=\\mathbf{K}_s\\!\\odot\\!\\mathbf{k}^{d}$",
        "#f2f2f2")
    box(20, 18, 56, 20, "per-spot coverage $\\boldsymbol{\\vartheta}_t$\n"
        "$\\vartheta^{*}+(\\vartheta_{t-1}-\\vartheta^{*})e^{-(k^{a}u+k^{d})\\Delta t}$",
        "#cfcfcf")
    box(78, 46, 20, 18, "$s_{t-1}$", "#f2f2f2")
    arr(17, 74, 17, 64); arr(32, 55, 38, 55); arr(53, 74, 53, 64)
    arr(53, 46, 48, 38); arr(88, 46, 70, 34)
    arr(48, 18, 48, 8); ax.text(50, 4, "$\\Delta\\boldsymbol{\\vartheta}_t$ to fast branch",
                                fontsize=5.6, ha="center")
    ax.add_patch(FancyArrowPatch((20, 28), (8, 28), arrowstyle="-|>",
                                 connectionstyle="arc3,rad=0.55", mutation_scale=6,
                                 lw=0.75, color="black", ls=":"))
    ax.text(4, 36, "$\\boldsymbol{\\vartheta}_{t-1}$", fontsize=5.8)
    ax.set_title("(a) Langmuir kinetic state cell", loc="left")

    dev = BC.Device(0, np.random.default_rng(7)); BC.calibrate_offset(dev)
    amb = np.array([BC.HEALTHY_GM[n] for n in KIN.VOC_NAMES[:-1]]) * BC.AMBIENT_FRACTION
    ax = fig.add_subplot(gs[0, 1])
    t = np.arange(C.T_STEPS) * 2.0
    for i, lab in enumerate(("Healthy", "LungCancer")):
        c = BC.sample_subject(lab, np.random.default_rng(5))
        _, cl, lat = BC.simulate_session(dev, c, amb, 0.42, 0.95, 0.0, 4.0,
                                         np.random.default_rng(3), return_latent=True)
        ax.plot(t, lat["occ"][:, 4], color=F.GREY[i], ls=F.LSTY[i], lw=1.0,
                label=f"{lab}, amine spot")
        ax.plot(t, lat["occ"][:, 0], color=F.GREY[i], ls=F.LSTY[i], lw=1.0,
                alpha=0.55, marker=F.MARK[i], ms=1.8, markevery=9,
                label=f"{lab}, dispersive spot")
    s = C.BASELINE_S
    for _ in range(C.N_CYCLES):
        ax.axvspan(s, s + C.EXPOSURE_S, color="0.90", zorder=0)
        s += C.EXPOSURE_S + C.PURGE_S
    ax.set_xlabel("time (s)"); ax.set_ylabel("weighted occupancy $\\Theta_k$")
    ax.set_ylim(-0.02, 1.42)
    ax.legend(loc="upper center", ncol=2, fontsize=4.4, handlelength=1.8,
              columnspacing=0.8, handletextpad=0.4)
    ax.set_title("(b) two adsorption regimes", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    P = C.TRAIN.latent_analytes
    kd = np.exp(np.linspace(np.log(1 / 240.0), np.log(1 / 4.0), P))
    for i, u in enumerate((0.15, 0.6, 2.0)):
        rate = kd * (1.0 + u)
        ax.semilogy(np.arange(1, P + 1), 1.0 / rate, color=F.GREY[i],
                    ls=F.LSTY[i], lw=1.0, marker=F.MARK[i], ms=3.0,
                    label=f"$K_s u={u}$")
    ax.set_xticks(np.arange(1, P + 1))
    ax.set_xlabel("latent component index $j$")
    ax.set_ylabel("relaxation time (s)")
    ax.legend(loc="upper left"); ax.set_title("(c) timescale ordering preserved",
                                              loc="left")
    return F._save(fig, path)


def main(which="123"):
    out = {}
    if "1" in which:
        out["fig1"] = F.fig_architecture()
    if "2" in which:
        ex, dm = physics_example()
        out["fig2"] = F.fig_physics(dm, ex)
    if "3" in which:
        out["fig3"] = fig_mechanism()
    if "4" in which:
        out["fig4"] = F.fig_main(F.load("E1_main.json"))
    if "5" in which:
        out["fig5"] = F.fig_ablation(F.load("E2_ablation.json"))
    if "6" in which:
        out["fig6"] = F.fig_sensitivity(F.load("E3_sensitivity.json"))
    if "7" in which:
        out["fig7"] = F.fig_robustness(F.load("E4_robustness.json"))
    if "8" in which:
        out["fig8"] = F.fig_generalisation(F.load("E5_generalisation.json"),
                                           F.load("E10_external.json"))
    if "9" in which:
        probs = pickle.load(open(os.path.join(C.CACHE_DIR, "E1_probs.pkl"), "rb"))
        out["fig9"] = F.fig_uncertainty(F.load("E8_uncertainty.json"), probs)
    if "0" in which:
        probs = pickle.load(open(os.path.join(C.CACHE_DIR, "E1_probs.pkl"), "rb"))
        out["fig10"] = F.fig_failure(probs)
    return out


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1] if len(sys.argv) > 1 else "123"))
