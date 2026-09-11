# -*- coding: utf-8 -*-
"""Load every result file and expose the quantities the manuscript refers to."""
from __future__ import annotations
import json, os, pickle
import numpy as np
import config as C
import stats as S

RES = C.RESULTS_DIR
PROPOSED = "KINDRA"


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def agg(rec, metric):
    out = {}
    for k, v in rec.items():
        if k.startswith("_"):
            continue
        a = np.array([m[metric] for m in v], float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            continue
        out[k] = dict(mean=float(a.mean()),
                      sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                      raw=a)
    return out


def fmt(x, n=3):
    return f"{x:.{n}f}"


def pm(d, n=3):
    return f"{d['mean']:.{n}f} $\\pm$ {d['sd']:.{n}f}"


class Summary:
    def __init__(self):
        self.e1 = _load("E1_main.json")
        self.e2 = _load("E2_ablation.json")
        self.e3 = _load("E3_sensitivity.json")
        self.e4 = _load("E4_robustness.json")
        self.e5 = _load("E5_generalisation.json")
        self.e8 = _load("E8_uncertainty.json")
        self.e10 = _load("E10_external.json")
        self.ceiling = _load("E0_ceiling.json")
        pp = os.path.join(C.CACHE_DIR, "E1_probs.pkl")
        self.probs = pickle.load(open(pp, "rb")) if os.path.exists(pp) else None

    # -- main benchmark ---------------------------------------------------
    def main_table(self, metrics=("balanced_accuracy", "macro_f1", "mcc",
                                  "macro_auroc", "macro_auprc", "ece", "brier")):
        A = {m: agg(self.e1, m) for m in metrics}
        order = sorted(A["balanced_accuracy"],
                       key=lambda k: -A["balanced_accuracy"][k]["mean"])
        return A, order

    def best_baseline(self, metric="balanced_accuracy"):
        A = agg(self.e1, metric)
        cand = {k: v for k, v in A.items() if k != PROPOSED}
        k = max(cand, key=lambda x: cand[x]["mean"])
        return k, cand[k]

    def gain_over(self, name, metric="balanced_accuracy"):
        A = agg(self.e1, metric)
        return A[PROPOSED]["mean"] - A[name]["mean"]

    def pairwise(self, metric="balanced_accuracy"):
        A = agg(self.e1, metric)
        rows, ps = [], []
        for k, v in A.items():
            if k == PROPOSED:
                continue
            a, b = A[PROPOSED]["raw"], v["raw"]
            n = min(len(a), len(b))
            p, _ = S.wilcoxon_pair(a[:n], b[:n])
            rows.append([k, v["mean"], A[PROPOSED]["mean"] - v["mean"],
                         S.paired_cohen_dz(a[:n], b[:n]),
                         S.cliffs_delta(a[:n], b[:n]), p])
            ps.append(p)
        rej, adj = S.benjamini_hochberg(ps)
        for r, q, sig in zip(rows, adj, rej):
            r.append(q); r.append(sig)
        rows.sort(key=lambda r: -r[1])
        return rows

    def friedman(self, metric="balanced_accuracy"):
        A = agg(self.e1, metric)
        names = [k for k in A if len(A[k]["raw"]) == len(C.SEED_LIST)]
        M = np.stack([A[k]["raw"] for k in names], axis=1)
        stat, p, ranks = S.friedman(M)
        cd = S.nemenyi_critical_difference(len(names), M.shape[0])
        return names, stat, p, ranks, cd

    # -- ablation ---------------------------------------------------------
    def ablation_table(self):
        A = {m: agg(self.e2, m) for m in ("balanced_accuracy", "macro_auroc",
                                          "ece", "params")}
        return A, list(self.e2.keys())

    def component_gain(self, tag_with, tag_without, metric="balanced_accuracy"):
        A = agg(self.e2, metric)
        return A[tag_with]["mean"] - A[tag_without]["mean"]

    # -- robustness -------------------------------------------------------
    def robustness_table(self, models=None):
        rows = []
        keys = list(self.e4.keys())
        models = models or list(self.e4[keys[0]].keys())
        for k in keys:
            kind, lvl = k.split("=")
            row = [kind.replace("_", " "), lvl]
            for m in models:
                a = np.array([d["balanced_accuracy"] for d in self.e4[k][m]])
                row.append(f"{a.mean():.3f}")
            rows.append(row)
        return models, rows

    def robustness_area(self, model, kind):
        vals = []
        for k, per in self.e4.items():
            if k.split("=")[0] == kind:
                vals.append(np.mean([d["balanced_accuracy"] for d in per[model]]))
        return float(np.mean(vals))

    # -- generalisation ---------------------------------------------------
    def gen_table(self):
        A = {m: agg(self.e5, m) for m in ("balanced_accuracy", "macro_auroc", "ece")}
        order = sorted(A["balanced_accuracy"],
                       key=lambda k: -A["balanced_accuracy"][k]["mean"])
        return A, order

    # -- efficiency -------------------------------------------------------
    def efficiency_table(self):
        A = {m: agg(self.e1, m) for m in ("params", "train_seconds", "latency_ms",
                                          "balanced_accuracy")}
        order = sorted(A["balanced_accuracy"],
                       key=lambda k: -A["balanced_accuracy"][k]["mean"])
        return A, order

    # -- external ---------------------------------------------------------
    def external_table(self):
        r = self.e10
        rows = [["Variant"] + [f"B{b}" for b in r["test_batches"]] + ["mean"]]
        for k, v in r["results"].items():
            a = np.array(v)
            rows.append([k] + [f"{x:.3f}" for x in a.mean(0)] +
                        [f"{a.mean():.3f} $\\pm$ {a.mean(1).std(ddof=1):.3f}"])
        return rows

    # -- attributes used by the builder -----------------------------------
    @property
    def devmeta(self):
        if not hasattr(self, "_dm"):
            self._dm = pickle.load(open(os.path.join(C.CACHE_DIR, "devices.pkl"),
                                        "rb"))
        return self._dm

    @property
    def raw(self):
        if not hasattr(self, "_raw"):
            import dataset as D
            self._raw = D.load_raw()
        return self._raw

    # -- table rows --------------------------------------------------------
    def budget_table(self):
        ba = agg(self.ceiling, "balanced_accuracy")
        au = agg(self.ceiling, "macro_auroc")
        rows = [["Representation", "Balanced acc.", "Macro AUROC"]]
        for k in ["True alveolar concentrations", "Nuisance-free sensorgram",
                  "Observed sensorgram", "Observed, physical projection r=2",
                  "Observed, physical projection r=3"]:
            rows.append([k.replace("r=", "$r$="), pm(ba[k]), pm(au[k])])
        return rows

    def main_rows(self):
        A, order = self.main_table()
        rows = [["Model", "Balanced acc.", "Macro $F_1$", "MCC", "Macro AUROC",
                 "Macro AUPRC", "ECE"]]
        for k in order:
            rows.append([k, pm(A["balanced_accuracy"][k]), pm(A["macro_f1"][k]),
                         pm(A["mcc"][k]), pm(A["macro_auroc"][k]),
                         pm(A["macro_auprc"][k]), pm(A["ece"][k])])
        return rows

    def stats_rows(self):
        rows = [["Comparator", "Balanced acc.", "$\\Delta$", "$d_z$",
                 "Cliff's $\\delta$", "$p$", "$q$ (BH)"]]
        for r in self.pairwise():
            rows.append([r[0], fmt(r[1]), fmt(r[2]), fmt(r[3], 2), fmt(r[4], 2),
                         f"{r[5]:.3f}", f"{r[6]:.3f}" + ("*" if r[7] else "")])
        return rows

    def ablation_rows(self):
        A, order = self.ablation_table()
        rows = [["Variant", "Balanced acc.", "Macro AUROC", "ECE", "Params"]]
        for k in order:
            rows.append([k, pm(A["balanced_accuracy"][k]), pm(A["macro_auroc"][k]),
                         pm(A["ece"][k]),
                         f"{int(A['params'][k]['mean']):,}"])
        return rows

    def sensitivity_rows(self):
        rows = [["Setting", "Balanced acc.", "Macro AUROC"]]
        pretty = {"proj_rank": "projection rank $r$",
                  "lambda_kin": "$\\lambda_{\\mathrm{kin}}$",
                  "lambda_adv": "$\\lambda_{\\mathrm{drift}}$"}
        for k, v in self.e3.items():
            key, val = k.split("=")
            ba = np.array([d["balanced_accuracy"] for d in v])
            au = np.array([d["macro_auroc"] for d in v])
            rows.append([f"{pretty.get(key,key)} = {val}",
                         f"{ba.mean():.3f} $\\pm$ {ba.std(ddof=1):.3f}",
                         f"{au.mean():.3f} $\\pm$ {au.std(ddof=1):.3f}"])
        return rows

    def robustness_rows(self):
        models, rows = self.robustness_table()
        head = ["Corruption", "Level"] + models
        return [head] + rows

    def gen_rows(self):
        A, order = self.gen_table()
        rows = [["Model", "Balanced acc.", "Macro AUROC", "ECE"]]
        for k in order:
            rows.append([k, pm(A["balanced_accuracy"][k]), pm(A["macro_auroc"][k]),
                         pm(A["ece"][k])])
        return rows

    def external_rows(self):
        return self.external_table()

    def efficiency_rows(self):
        A, order = self.efficiency_table()
        rows = [["Model", "Parameters", "Train (s)", "Latency (ms)"]]
        for k in order:
            lat = A["latency_ms"][k]["mean"] if k in A["latency_ms"] else float("nan")
            rows.append([k, f"{int(A['params'][k]['mean']):,}",
                         f"{A['train_seconds'][k]['mean']:.0f}",
                         "--" if np.isnan(lat) else f"{lat:.1f}"])
        return rows

    def uncertainty_rows(self):
        A = {m: agg(self.e8, m) for m in ("balanced_accuracy", "ece", "brier",
                                          "nll")}
        rows = [["Method", "Balanced acc.", "ECE", "Brier"]]
        for k in A["ece"]:
            rows.append([k, pm(A["balanced_accuracy"][k]), pm(A["ece"][k]),
                         pm(A["brier"][k])])
        return rows
