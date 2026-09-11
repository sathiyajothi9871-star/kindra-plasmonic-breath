# -*- coding: utf-8 -*-
"""Table content assembled from the configuration and from the result files."""
from __future__ import annotations
import json, os, pickle
import numpy as np
import config as C
import breath_cohort as BC
import kinetics as KIN

ANCHOR = {
 "Acetone": "[[5]], [[6]]", "Isoprene": "[[6]]", "Ethanol": "[[6]]",
 "1-Propanol": "[[5]], [[6]]", "Toluene": "[[15]]", "Benzene": "[[5]]",
 "Ethylbenzene": "[[5]], [[6]]", "Hexanal": "[[6]], [[7]]",
 "Pentanal": "[[6]], [[7]]", "2-Pentanone": "[[5]], [[6]]",
 "2-Butanone": "[[5]]", "Butane": "[[6]]",
}


def voc_table():
    rows = [["Volatile", "Healthy (ppb)", "Lung cancer", "COPD", "Asthma", "GSD",
             "Anchor"]]
    for n in KIN.VOC_NAMES[:-1]:
        h = BC.HEALTHY_GM[n]
        rows.append([n, f"{h:.4g}",
                     f"{h*BC.CLASS_RATIO['LungCancer'][n]:.4g}",
                     f"{h*BC.CLASS_RATIO['COPD'][n]:.4g}",
                     f"{h*BC.CLASS_RATIO['Asthma'][n]:.4g}",
                     f"{BC.GSD[n]:.2f}", ANCHOR[n]])
    rows.append(["Water vapour", "1.3$\\times$10$^{7}$", "as healthy", "as healthy",
                 "as healthy", "--", "[[9]]"])
    return rows


def array_table(meta):
    d = meta[0]
    rows = [["Spot", "Coating", "Chemistry", "$\\lambda^{\\mathrm{res}}$ (nm)",
             "$S^{\\mathrm{blk}}$", "$S^{\\mathrm{srf}}$", "$\\ell$ (nm)"]]
    chem = (["dispersive"] * 4 + ["amine (carbonyl)"] * 4 +
            ["aromatic"] * 2 + ["hydroxyl"] * 2)
    for k in range(C.N_CHANNELS):
        rows.append([str(k + 1), f"{C.COATING_THICKNESS_NM[k]:.0f} nm, "
                     f"$n$={C.COATING_INDEX[k]:.2f}", chem[k],
                     f"{d['lambda0'][k]:.1f}", f"{d['s_bulk'][k]:.1f}",
                     f"{d['s_surf'][k]:.1f}", f"{d['l_d'][k]:.0f}"])
    return rows


def cohort_table(raw):
    y = raw["y"]
    rows = [["Property", "Healthy", "Lung cancer", "COPD", "Asthma", "Total"]]
    subj = [len(np.unique(raw["subject"][y == i])) for i in range(4)]
    sess = [int((y == i).sum()) for i in range(4)]
    rows.append(["Subjects"] + [str(s) for s in subj] + [str(sum(subj))])
    rows.append(["Recording sessions"] + [str(s) for s in sess] + [str(sum(sess))])
    rows.append(["Sessions per subject", "2", "2", "2", "2", "2"])
    rows.append(["Sensing spots per record", "12", "12", "12", "12", "12"])
    rows.append(["Samples per record", "90", "90", "90", "90", "90"])
    rows.append(["Record duration (s)", "180", "180", "180", "180", "180"])
    rows.append(["Chips used", "8", "8", "8", "8", "8"])
    rows.append(["Elapsed age range (months)", "0-18", "0-18", "0-18", "0-18", "0-18"])
    rows.append(["Sites", "3", "3", "3", "3", "3"])
    tr = [int(round(0.65 * s)) for s in sess]
    va = [int(round(0.15 * s)) for s in sess]
    te = [s - a - b for s, a, b in zip(sess, tr, va)]
    rows.append(["Train / val / test subjects",
                 *[f"{int(round(0.65*s))}/{int(round(0.15*s))}/"
                   f"{s-int(round(0.65*s))-int(round(0.15*s))}" for s in subj],
                 f"{sum(int(round(0.65*s)) for s in subj)}/"
                 f"{sum(int(round(0.15*s)) for s in subj)}/"
                 f"{sum(s-int(round(0.65*s))-int(round(0.15*s)) for s in subj)}"])
    return rows


def implementation_table():
    t = C.TRAIN
    return [
     ["Setting", "Value", "Setting", "Value"],
     ["Framework", "PyTorch 2.14, CPU only", "Optimiser", "AdamW"],
     ["Initial learning rate", f"{t.lr:.1e}", "Weight decay", f"{t.weight_decay:.0e}"],
     ["Schedule", "cosine over 60 epochs", "Minibatch", str(t.batch_size)],
     ["Maximum epochs", str(t.epochs), "Early stopping", f"{t.patience} epochs"],
     ["Gradient clipping", f"{t.grad_clip:.0f}", "Model selection",
      "best validation balanced accuracy"],
     ["Latent components $J\'$", str(t.latent_analytes), "Fast branch width $H$",
      str(t.hidden_fast)],
     ["Projection rank $r$", str(t.proj_rank), "Gate initialisation",
      "$\\sigma(2)=0.88$"],
     ["$\\lambda_{\\mathrm{evi}}$", f"{t.lambda_evi:.2f}",
      "$\\lambda_{\\mathrm{kin}}$", f"{t.lambda_kin:.2f}"],
     ["$\\lambda_{\\mathrm{drift}}$", f"{t.lambda_adv:.2f}", "KL warm-up",
      f"{t.warmup_epochs} epochs"],
     ["Head width", str(t.head_width), "Head dropout", f"{t.head_dropout:.2f}"],
     ["Class weighting", "inverse frequency", "Descriptor normalisation",
      "batch"],
     ["Partitions", "5 subject-level draws", "Augmentation",
      "none in reported runs"],
     ["Seeds", ", ".join(str(s) for s in C.SEED_LIST), "Bootstrap resamples",
      "2000"],
    ]
