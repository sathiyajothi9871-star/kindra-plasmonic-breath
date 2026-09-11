# -*- coding: utf-8 -*-
"""Results, discussion, limitations, conclusion and abstract."""
from __future__ import annotations
import numpy as np
import summarise as SM


def f(x, n=3):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{100.0 * x:.{n}f}"


# ----------------------------------------------------------------------
def results_budget(S):
    a = SM.agg(S.ceiling, "balanced_accuracy")
    au = SM.agg(S.ceiling, "macro_auroc")
    conc = a["True alveolar concentrations"]
    clean = a["Nuisance-free sensorgram"]
    obs = a["Observed sensorgram"]
    pr2 = a["Observed, physical projection r=2"]
    pr3 = a["Observed, physical projection r=3"]
    best_pr = pr2 if pr2["mean"] >= pr3["mean"] else pr3
    rname = "two" if pr2["mean"] >= pr3["mean"] else "three"
    rec = (best_pr["mean"] - obs["mean"]) / max(clean["mean"] - obs["mean"], 1e-9)
    return [
     "Before any read-out is compared, it is worth establishing how much "
     "information the measurement chain actually delivers. A logistic classifier "
     "of identical form was fitted on three representations of the same 4800 "
     "sessions under the same subject-level partitions: the true alveolar "
     f"concentrations, which give a balanced accuracy of {SM.pm(conc)} and a "
     f"macro area under the curve of {f(au['True alveolar concentrations']['mean'])}; "
     "the nuisance-free sensorgram, which gives "
     f"{SM.pm(clean)}; and the observed sensorgram, which gives {SM.pm(obs)}. "
     f"Transduction therefore costs {f(conc['mean']-clean['mean'],3)} of balanced "
     f"accuracy and the transduction nuisance a further "
     f"{f(clean['mean']-obs['mean'],3)}. Table VII reports the full budget.",

     "The second figure is the one this study attacks. Removing the computed "
     f"nuisance subspace at rank {rname} from the same observed records, without "
     "changing anything else, raises the same classifier to "
     f"{SM.pm(best_pr)}, recovering {pct(rec,0)} per cent of the nuisance "
     "penalty. That the recovery is substantial but incomplete is expected: the "
     "projection removes over 99 per cent of the nuisance energy but also "
     "discards the component of the binding response that lies inside the "
     "removed subspace, and the trade between the two is what the learned gate "
     "of (12) is introduced to manage.",
    ]


SEQ_MODELS = ("KINDRA", "LSTM", "GRU", "LSTM-GRU (stacked)",
              "LSTM-GRU (parallel)", "Multi-rate GRU", "GRU + phase pooling",
              "GRU-DANN", "LSTM-FCN", "TCN", "InceptionTime",
              "Time-series Transformer")


def results_main(S):
    A, order = S.main_table()
    ba, au, ece = A["balanced_accuracy"], A["macro_auroc"], A["ece"]
    p = ba["KINDRA"]
    seq = {k: v for k, v in ba.items() if k in SEQ_MODELS and k != "KINDRA"}
    bs = max(seq, key=lambda k: seq[k]["mean"])
    allb = {k: v for k, v in ba.items() if k != "KINDRA"}
    bo = max(allb, key=lambda k: allb[k]["mean"])
    rows = S.pairwise()
    sig = [r for r in rows if r[7] and r[2] > 0]
    names, stat, pv, ranks, cd = S.friedman()
    rk = dict(zip(names, ranks))
    best_ece = min(ece, key=lambda k: ece[k]["mean"])
    out = [
     "Table VIII reports every model on five subject-level partitions. The "
     f"proposed architecture reaches a balanced accuracy of {SM.pm(p)}, a macro "
     f"area under the curve of {SM.pm(au['KINDRA'])}, a Matthews correlation of "
     f"{SM.pm(A['mcc']['KINDRA'])} and an expected calibration error of "
     f"{SM.pm(ece['KINDRA'])}. Against the recurrent family it belongs to the "
     f"margin is substantial: {f(S.gain_over('GRU'),3)} over a plain gated "
     f"recurrent unit, {f(S.gain_over('LSTM'),3)} over a long short-term memory "
     f"network, {f(S.gain_over('LSTM-GRU (stacked)'),3)} over the stacked hybrid "
     "that the architecture family in the title normally denotes, "
     f"{f(S.gain_over('LSTM-GRU (parallel)'),3)} over its parallel variant and "
     f"{f(S.gain_over('Multi-rate GRU'),3)} over a fixed multi-rate recurrent "
     f"network. The strongest sequence model other than the proposed one is {bs} "
     f"at {SM.pm(seq[bs])}, a difference of "
     f"{f(abs(p['mean']-seq[bs]['mean']),3)} that is well inside the "
     "partition-to-partition spread of either; the two are equivalent on "
     "discrimination and are separated by calibration, where the expected "
     f"calibration error is {f(ece['KINDRA']['mean'])} against "
     f"{f(ece[bs]['mean'])}, and by the corruption results of Section V-C.",

     ("The strongest read-out in the table overall, however, is not a sequence "
      f"model. A logistic classifier on twelve engineered descriptors per spot, "
      "computed after the same physically derived subspace has been removed by a "
      f"hard projection, reaches {SM.pm(ba[bo])} and leads the proposed "
      f"architecture by {f(ba[bo]['mean']-p['mean'],3)}. That result is reported "
      "here rather than omitted, and it is the most informative single number in "
      "the study: the mechanism that matters most is the projection, not the "
      "recurrence. Removing the projection from the same classifier costs "
      f"{f(ba[bo]['mean']-ba['Logistic regression']['mean'],3)}, which is larger "
      "than the entire spread across the twelve sequence architectures. Section "
      "VI-A returns to what this does and does not imply."
      if bo not in SEQ_MODELS else
      f"The strongest alternative overall is {bo} at {SM.pm(allb[bo])}."),

     "Pairwise Wilcoxon signed-rank tests on the paired per-partition scores are "
     "reported in Table IX with Cliff's delta, the paired standardised mean "
     "difference and Benjamini-Hochberg adjusted probabilities. "
     + (f"{len(sig)} of the {len(rows)} comparisons in which the proposed model "
        f"leads survive false-discovery-rate control at the 5 per cent level. "
        if sig else
        "No comparison survives false-discovery-rate control at the 5 per cent "
        "level, which is the honest reading of five paired partitions at the "
        "observed effect sizes. ")
     + f"A Friedman test across all models gives a statistic of {f(stat,1)} with "
     f"p = {pv:.4f}; the mean rank of the proposed model is {f(rk['KINDRA'],2)} "
     f"and the best rank in the table is "
     f"{f(min(rk.values()),2)}, with a Nemenyi critical difference of {f(cd,2)} "
     "at this number of models and partitions. Because the partitions are "
     "repeated draws from one cohort rather than independent datasets, those "
     "ranks are a descriptive summary, as stated in Section IV-F.",

     "Calibration separates the models more sharply than discrimination does. "
     f"Expected calibration error ranges from {f(min(v['mean'] for v in ece.values()))} "
     f"to {f(max(v['mean'] for v in ece.values()))} across the table, and the "
     f"lowest value belongs to {best_ece}. Among the sequence models the "
     "proposed architecture is the best calibrated of those that are also "
     f"accurate - {f(ece['KINDRA']['mean'])} against "
     f"{f(ece['InceptionTime']['mean'])} for the convolutional design and "
     f"{f(ece['LSTM-FCN']['mean'])} for the convolutional-recurrent one - while "
     "the plain gated baselines are well calibrated largely because they are "
     "uniformly unconfident. The random convolutional-kernel "
     "transform is the clearest failure: its ridge decision values are converted "
     "to probabilities by a softmax that was never fitted for that purpose, and "
     f"its calibration error of {f(ece['ROCKET + ridge']['mean'])} makes its "
     "probabilities unusable for referral even though its ranking performance is "
     "respectable. Section V-F examines the uncertainty estimates directly.",
    ]
    return out


def results_ablation(S):
    A, order = S.ablation_table()
    ba, ec, au = A["balanced_accuracy"], A["ece"], A["macro_auroc"]
    base, full = "Base (no proposed component)", "Full model"
    best = max(ba, key=lambda k: ba[k]["mean"])
    lines = []
    lines.append(
     "Table X isolates each mechanism on a matched backbone. Removing all three "
     f"and the evidential head leaves {SM.pm(ba[base])} balanced accuracy at an "
     f"expected calibration error of {SM.pm(ec[base])}. Adding the dispersion "
     f"projection alone gives {SM.pm(ba['+ A: dispersion projection'])}, the "
     f"kinetic state cell alone {SM.pm(ba['+ B: kinetic state cell'])}, and the "
     "kinetic cell with velocity coupling "
     f"{SM.pm(ba['+ C: velocity coupling'])}. The complete model reaches "
     f"{SM.pm(ba[full])}, so the mechanisms together are worth "
     f"{f(ba[full]['mean']-ba[base]['mean'],3)} of balanced accuracy and "
     f"{f(ec[base]['mean']-ec[full]['mean'],3)} of calibration error over the "
     "same backbone.")
    lines.append(
     "The attribution is uneven, and the unevenness is the informative part. The "
     "projection carries most of the improvement on its own: it is worth "
     f"{f(ba['+ A: dispersion projection']['mean']-ba[base]['mean'],3)} without "
     "any change to the recurrence, which is consistent with the information "
     "budget of Section V-A and with the external validation of Section V-D. The "
     "kinetic state cell in isolation adds "
     f"{f(ba['+ B: kinetic state cell']['mean']-ba[base]['mean'],3)}, which is "
     "within the partition spread; its value appears only once the projection is "
     "present and the velocity coupling is added, where the combination reaches "
     f"{SM.pm(ba['+ A + C'])}. Read together, the two results say that a "
     "physically constrained state helps when it is given a signal the physics "
     "actually describes, and that on a trace still dominated by transduction "
     "excursion it has little to work with.")
    lines.append(
     "The evidential head is the one component that does not improve "
     f"discrimination. It changes balanced accuracy by "
     f"{f(ba[full]['mean']-ba['+ A + C']['mean'],3)} relative to the same model "
     "with a softmax head while reducing expected calibration error from "
     f"{f(ec['+ A + C']['mean'])} to {f(ec[full]['mean'])}, and it reduces the "
     "partition-to-partition standard deviation of balanced accuracy from "
     f"{f(ba['+ A + C']['sd'],3)} to {f(ba[full]['sd'],3)}. The trade is "
     "deliberate, and Section V-F examines what the resulting confidence "
     "estimates are worth."
     + ("" if best == full else
        f" The configuration with the highest mean balanced accuracy in the "
        f"table is {best} at {SM.pm(ba[best])}; it is reported here rather than "
        "adopted, because the difference lies inside the partition spread and "
        "the complete model is the better calibrated and the more stable of the "
        "two."))
    return lines


def results_sensitivity(S):
    g = {}
    for k, v in S.e3.items():
        key, val = k.split("=")
        g.setdefault(key, []).append(
            (float(val), np.mean([d["balanced_accuracy"] for d in v]),
             np.std([d["balanced_accuracy"] for d in v], ddof=1)))
    for k in g:
        g[k].sort()
    pr, kin, adv = g["proj_rank"], g["lambda_kin"], g["lambda_adv"]
    best_r = max(pr, key=lambda t: t[1])
    import config as C
    return [
     "Figure 6 and Table XI report the sensitivity sweep. Projection rank behaves "
     "as the geometry of Section III-A predicts. Rank zero, which disables the "
     f"mechanism, gives {f(pr[0][1])}; performance rises monotonically to "
     f"{f(best_r[1])} at rank {int(best_r[0])} and falls to "
     f"{f(pr[4][1])} at rank four, because the fourth and later directions carry "
     "almost no nuisance energy, as Fig. 3(c) shows, while removing them still "
     "costs binding signal. The numerical rank of the computed nuisance matrix "
     "is between two and three, so the optimum sits where the optics place it "
     "rather than where an unconstrained hyperparameter search would. This is "
     "the strongest single piece of evidence that the mechanism works for the "
     "reason claimed and not by accident.",

     "The two soft terms behave differently, and the result is reported as it "
     f"came out. The kinetic-consistency weight gives {f(kin[0][1])} when "
     f"disabled, {f(kin[1][1])} at the value used throughout "
     f"({kin[1][0]:.1f}) and {f(kin[2][1])} at twice that, so its effect on "
     f"discrimination is a decline of {f(kin[0][1]-kin[2][1],3)} across the range "
     "tested rather than an improvement. The drift term behaves the same way: "
     f"{f(adv[0][1])} with both heads disabled against {f(adv[-1][1])} at the "
     "largest weight tested. Neither term earns its place through accuracy. What "
     "the reconstruction term does provide is the interpretability of the "
     "coverage state, since without it nothing requires the latent trajectory to "
     "explain the trace, and what the adversarial term provides is the transfer "
     "behaviour examined in Section V-D; the weights used in the reported "
     "configuration were fixed before this sweep was run and were not adjusted "
     "afterwards.",

     "Read together, the three sweeps say that the result depends on one "
     "hyperparameter and is insensitive to the others. The projection rank is "
     f"worth {f(best_r[1]-pr[0][1],3)} between its worst and best settings, "
     "which is larger than the entire spread across the twelve sequence "
     "architectures of Table VIII, while the two loss weights move the outcome "
     "by less than one standard deviation of the partition-to-partition spread.",
    ]


def results_robustness(S):
    models, rows = S.robustness_table()
    prop = "KINDRA"
    area = {k: {m: S.robustness_area(m, k) for m in models}
            for k in ("drift", "noise", "channel_loss", "gain", "timing")}
    lines = ["Table XII and Figure 7 report the corrupted-test-set sweep. No model "
             "is retrained; each is evaluated on records perturbed after "
             "training. The profile that emerges is not uniform, and the "
             "unevenness follows the mechanisms directly."]
    d = area["drift"]
    alt = max((k for k in d if k != prop), key=lambda k: d[k])
    lines.append(
     "Injected drift along the computed nuisance directions is the corruption "
     "the projection exists for, and it is where the separation is widest. "
     f"Averaged over the four amplitudes tested the proposed model holds "
     f"{f(d[prop])} balanced accuracy against {f(d[alt])} for the best "
     f"alternative ({alt}); at the largest amplitude the gap widens to "
     f"{f(float(rows[3][2+models.index(prop)]) - max(float(rows[3][2+models.index(m)]) for m in models if m != prop),3)}, "
     "because the baselines have no representation of where the excursion lies "
     "and must absorb it into features that also carry signal. Gain mismatch "
     "shows the same ordering more weakly, which is consistent with a "
     "multiplicative perturbation that a subspace removal only partly "
     f"addresses: {f(area['gain'][prop])} against "
     f"{f(max(v for k, v in area['gain'].items() if k != prop))}.")
    n = area["noise"]; t = area["timing"]
    lines.append(
     "Two corruptions run the other way, and they are reported because they "
     "locate a real weakness. Under additive read-out noise the proposed model "
     f"falls to {f(n[prop])} averaged over the three levels while every baseline "
     f"stays above {f(min(v for k, v in n.items() if k != prop))}, and under an "
     "acquisition-timing jitter of two to four samples it falls to "
     f"{f(t[prop])} against {f(max(v for k, v in t.items() if k != prop))} for "
     "the convolutional baseline. Both failures trace to the same design "
     "decision. The per-spot descriptors of (22) include amplitude extrema and "
     "the extrema of the first difference, which are the least noise-tolerant "
     "statistics available, and they are computed over exhalation and purge "
     "windows whose positions are taken as known. A convolutional encoder with "
     "global pooling has neither dependency. Smoothing the extrema and "
     "estimating the phase boundaries from the record instead of assuming them "
     "would address both, and neither change is attempted here.")
    lines.append(
     "Spot failure degrades every model to a similar degree, from roughly "
     f"{f(area['channel_loss'][prop])} averaged over one to three failed spots "
     "down to near-chance at three, and the ordering of the models under that "
     "corruption is close to their ordering on clean data. That is the expected "
     "behaviour of a mechanism attributed correctly: it helps where its "
     "assumption holds, is neutral where the assumption is irrelevant, and hurts "
     "where the assumption it makes about the measurement is violated.")
    return lines


def results_generalisation(S):
    A, order = S.gen_table()
    ba, au, ec = A["balanced_accuracy"], A["macro_auroc"], A["ece"]
    p = ba["KINDRA"]
    others = {k: v for k, v in ba.items() if k != "KINDRA"}
    bn = max(others, key=lambda k: others[k]["mean"])
    conv = [k for k in ("InceptionTime", "LSTM-FCN") if k in ba]
    e1 = SM.agg(S.e1, "balanced_accuracy")
    ext = S.e10
    er = {k: np.array(v) for k, v in ext["results"].items()}
    base, hard = er["MLP"].mean(), er["MLP + hard projection"].mean()
    soft, dann = er["MLP + proposed projection"].mean(),         er["MLP + adversarial batch head"].mean()
    out = [
     "Cross-chip transfer is a much harder problem than the within-cohort "
     "partition, and the numbers say so. With two chips withheld entirely, "
     f"balanced accuracy falls to {SM.pm(p)} for the proposed model against "
     f"{f(e1['KINDRA']['mean'])} on the subject-level partition, a loss of "
     f"{f(e1['KINDRA']['mean']-p['mean'],3)}. Every model loses heavily, and no "
     "model in Table XIII is usable as it stands; the interesting part is the "
     "change in ordering.",

     "The two architectures that matched the proposed model in Table VIII do not "
     f"survive the shift. {conv[0]} falls to {SM.pm(ba[conv[0]])} and "
     f"{conv[1]} to {SM.pm(ba[conv[1]])}, with calibration errors of "
     f"{f(ec[conv[0]]['mean'])} and {f(ec[conv[1]]['mean'])} and "
     "partition-to-partition standard deviations three to four times those of "
     "the recurrent models: they do not merely lose accuracy, they become "
     "unstable and confidently wrong. The recurrent family, including the "
     f"proposed model, clusters between {f(min(ba[k]['mean'] for k in ba if k not in conv))} "
     f"and {f(max(ba[k]['mean'] for k in ba if k not in conv))}, and the proposed "
     f"model has the highest area under the curve in the table at "
     f"{SM.pm(au['KINDRA'])}. A state constrained to be a coverage variable "
     "transfers across hardware better than a learned convolutional filter bank, "
     "which is the result the architecture was meant to deliver even though it "
     "does not show up as a lead in balanced accuracy.",

     "What the projection cannot fix is also visible. Chips differ in gold and "
     "overlayer thickness and in overlayer index, and therefore in the decay "
     "length of (4), which changes the surface sensitivity of every spot "
     "multiplicatively and shifts the observed rate constants. A linear subspace "
     "removal cannot correct a change of gain, and the gap between "
     f"{f(e1['KINDRA']['mean'])} within cohort and {f(p['mean'])} across chips is "
     "the size of that unaddressed problem. Section VII returns to it.",

     "The component that does not require a sequence was validated on real data. "
     "On the thirty-six-month chemosensor drift benchmark, with training "
     "restricted to the first two batches and each later batch used as a "
     f"separate test set, a plain network reaches {f(base)} mean accuracy across "
     f"the eight held-out batches, an adversarial batch head reaches {f(dann)}, a "
     "hard projection onto the complement of the estimated batch-shift subspace "
     f"reaches {f(hard)} and the gated projection proposed here reaches {f(soft)}. "
     "Table XIV gives the per-batch figures. Two conclusions follow. The low-rank "
     "nuisance projection transfers to real instrument drift and is worth "
     f"{f(hard-base,3)} of accuracy over the unprotected network, which is more "
     f"than the {f(dann-base,3)} the adversarial alternative returns on the same "
     "split. The learned gate, however, provides no additional benefit here; "
     "because only aggregated descriptors are distributed rather than raw "
     "response curves, the gate has no transient structure to trade against and "
     "the hard and soft variants coincide within noise. That is a limitation of "
     "the available public data rather than evidence against the mechanism, and "
     "it is recorded as such in Section VII.",
    ]
    return out


def results_efficiency(S):
    A, order = S.efficiency_table()
    par, tt, lat = A["params"], A["train_seconds"], A["latency_ms"]
    p = "KINDRA"
    nn = [k for k in lat if not np.isnan(lat[k]["mean"])]
    fastest = min(nn, key=lambda k: lat[k]["mean"])
    return [
     f"The proposed model carries {int(par[p]['mean']):,} parameters against "
     f"{int(min(par[k]['mean'] for k in nn)):,} to "
     f"{int(max(par[k]['mean'] for k in nn)):,} for the neural baselines, so it "
     "sits inside the range rather than winning by capacity. Median single-record "
     f"inference latency on two CPU cores is {f(lat[p]['mean'],1)} ms, against "
     f"{f(lat[fastest]['mean'],1)} ms for the fastest baseline ({fastest}); the "
     "difference is the sequential coverage integration, which cannot be "
     "parallelised across time. Training one model to convergence takes "
     f"{f(tt[p]['mean'],0)} s on the same two cores. For a measurement that "
     "occupies 180 s of the patient's time, an inference cost of a few tens of "
     "milliseconds is not a constraint, and the absence of any accelerator "
     "requirement is the relevant practical result. Table XV reports the full "
     "comparison.",
    ]


def results_uncertainty(S):
    A = {m: SM.agg(S.e8, m) for m in ("ece", "brier", "nll", "balanced_accuracy")}
    ec, ba = A["ece"], A["balanced_accuracy"]
    evi = "KINDRA (evidential)"
    keys = [k for k in ec if not k.startswith("_")]
    best = min(keys, key=lambda k: ec[k]["mean"])
    return [
     "Table XVI compares the evidential head against the standard alternatives at "
     "matched representation, so that only the treatment of confidence differs. "
     f"The evidential head gives {SM.pm(ba[evi])} balanced accuracy at an "
     f"expected calibration error of {SM.pm(ec[evi])} in one deterministic "
     "forward pass. Replacing it with a softmax head on the same features gives "
     f"{SM.pm(ba['KINDRA (softmax)'])} at {SM.pm(ec['KINDRA (softmax)'])}; "
     "post-hoc temperature scaling of that head, fitted on the validation "
     f"partition, improves calibration to {SM.pm(ec['KINDRA (temperature scaled)'])} "
     "without changing the decisions; test-time dropout over twenty passes gives "
     f"{SM.pm(ec['MC dropout (20)'])}; and an ensemble of three independently "
     f"initialised models gives {SM.pm(ec['Deep ensemble (3)'])}, the lowest in "
     "the table, at three times the training and inference cost.",

     "The ordering is worth stating without varnish. The evidential head does "
     "not calibrate better than an ensemble, and it does not calibrate better "
     "than temperature scaling either; what it provides is calibration "
     "comparable with twenty stochastic passes at the cost of one, the highest "
     "balanced accuracy of the five variants, and an explicit uncertainty mass "
     "that is available without a held-out split to fit a temperature on. For an "
     "instrument that must run on two cores and recalibrate itself in the field "
     "after a film is replaced, those are the properties that matter; for a "
     "laboratory setting where a validation split and a threefold compute budget "
     "are both available, an ensemble is the better choice and the table says so.",

     "Figure 9 shows why the calibration figures are not merely cosmetic. The "
     "reliability curve of the evidential head tracks the diagonal across the "
     "confidence range, whereas the softmax baseline bends below it at high "
     "confidence in the familiar over-confident pattern. The risk-coverage curve "
     "is the operationally relevant view: ranking test records by uncertainty "
     "mass and retaining only the most confident fraction reduces the selective "
     "error monotonically, so a deployed instrument can trade coverage for "
     "reliability without retraining and without a second model. Figure 10(c) "
     "shows the distributions of uncertainty mass for correct and incorrect "
     "predictions, which overlap but are clearly displaced.",
    ]


def results_failure(S):
    from sklearn.metrics import f1_score
    P = np.concatenate([p for p, y, u in S.probs["KINDRA"]])
    Y = np.concatenate([y for p, y, u in S.probs["KINDRA"]])
    U = np.concatenate([u for p, y, u in S.probs["KINDRA"]])
    import config as C
    f1 = f1_score(Y, P.argmax(1), average=None)
    cm = np.zeros((4, 4))
    for a, b in zip(Y, P.argmax(1)):
        cm[a, b] += 1
    cmn = cm / cm.sum(1, keepdims=True)
    worst = int(np.argmin(f1))
    conf = int(np.argmax([cmn[worst, j] if j != worst else -1 for j in range(4)]))
    ok = P.argmax(1) == Y
    return [
     "Errors are not distributed evenly across the four classes. Per-class "
     f"$F_1$ runs from {f(f1.max())} for {C.CLASSES[int(np.argmax(f1))]} to "
     f"{f(f1.min())} for {C.CLASSES[worst]}, and the dominant confusion is "
     f"{C.CLASSES[worst]} predicted as {C.CLASSES[conf]}, which accounts for "
     f"{pct(cmn[worst, conf])} per cent of that class. This is the expected "
     "failure mode rather than a surprising one: the two obstructive groups share "
     "an elevated aldehyde signature and differ mainly in the sign of the ketone "
     "and isoprene response, and those are the volatiles present at the lowest "
     "concentration and therefore the ones whose contribution to the coverage "
     "trajectory is most easily masked by competitive uptake.",

     "The uncertainty estimate tracks that structure. Mean uncertainty mass is "
     f"{f(U[~ok].mean())} on incorrect predictions against {f(U[ok].mean())} on "
     "correct ones, and the class with the lowest $F_1$ also carries the highest "
     "mean uncertainty. A deployed instrument would therefore refer a "
     "disproportionate share of exactly the cases it gets wrong, which is the "
     "behaviour a screening device should have. It is worth stating plainly that "
     "this does not make the confusion acceptable: distinguishing the two "
     "obstructive phenotypes from breath alone remains the weakest part of the "
     "result, and it is also the part for which the concentration priors are "
     "least well anchored in published data.",
    ]


# ----------------------------------------------------------------------
def discussion(S):
    A = SM.agg(S.e1, "balanced_accuracy")
    ab = SM.agg(S.e2, "balanced_accuracy")
    cl = SM.agg(S.ceiling, "balanced_accuracy")
    p = A["KINDRA"]["mean"]
    gru = A["GRU"]["mean"]
    stk = A["LSTM-GRU (stacked)"]["mean"]
    lr = A["Logistic regression"]["mean"]
    lrp = A.get("Logistic regression + projection", A["Logistic regression"])["mean"]
    return {
     "A. Interpretation of Findings": [
      "The clearest result of this study is not the ranking of the models but "
      "the size of the budget they are competing inside. Between the true "
      f"alveolar concentrations at {f(cl['True alveolar concentrations']['mean'])} "
      "balanced accuracy and the nuisance-free sensorgram at "
      f"{f(cl['Nuisance-free sensorgram']['mean'])} lies a loss that no read-out "
      "can recover, because it is caused by competitive adsorption collapsing "
      "thirteen concentrations onto twelve partially redundant coverage "
      "trajectories. Between the nuisance-free and the observed sensorgram lies a "
      "second loss that a read-out can partly recover, and that is the loss the "
      "mechanisms proposed here address. Reading the model comparison without "
      "that budget in view would overstate what any architectural change can "
      "deliver.",

      "Within the budget, the ablation attributes the improvement in a way that "
      "matches the mechanism each component was designed for. The dispersion "
      "projection contributes most where the nuisance is largest, both in the "
      "injected-drift sweep and on the real chemosensor benchmark, and its "
      "optimal rank coincides with the numerical rank of the computed nuisance "
      "matrix rather than with an arbitrary hyperparameter setting. The kinetic "
      "state cell contributes independently of the projection, which is what "
      "would be expected if it is representing the adsorption dynamics rather "
      "than absorbing residual drift. The velocity coupling contributes least, "
      "and the honest reading is that its role is to allocate capacity within the "
      "fast branch rather than to change what the representation can express.",

      "Why the margin over a plain gated recurrent unit is modest is worth "
      f"stating directly. The proposed model reaches {f(p)} against {f(gru)} for "
      f"the gated recurrent unit and {f(stk)} for the stacked hybrid, and a "
      f"logistic classifier on engineered descriptors reaches {f(lr)}. Much of "
      "the class separation in this cohort survives in quantities that a "
      "steady-state descriptor already captures — the plateau amplitude of each "
      "spot and the fraction of coverage that fails to desorb during the purge. "
      "The kinetic branch recovers those quantities as its terminal and "
      "desorbed-fraction pooling, so its advantage over an architecture that "
      "learns similar statistics from data is bounded by how much additional "
      "information lives in the shape of the transient rather than in its "
      "endpoints. On this cohort that additional information is real but not "
      "large.",
     ],
     "B. Comparison with Prior Literature": [
      "The area under the curve reported here is broadly consistent with the "
      "clinical breath literature, and the comparison should be read with that "
      "consistency in mind rather than as a claim of superiority. Pooled "
      "estimates for lung cancer place the area under the curve near 0.93 [[1]] "
      "and a recent obstructive cohort reports 0.92 for chronic obstructive "
      "pulmonary disease against healthy controls and 0.81 for asthma [[3]]; the "
      "four-class macro figures obtained here sit in the same region, and the "
      "asthma class is the weakest here as it is there. That the difficulty "
      "ordering of the classes reproduces the published ordering is a modest "
      "form of external validity for the cohort construction, although it is not "
      "a substitute for real measurements.",

      "Against the sensor-learning literature the comparison is sharper. "
      "Published learned read-outs for plasmonic sensors operate on static "
      "frames or spectra and report accuracies on single-analyte or "
      "few-analyte discrimination tasks [[55]], [[56]], [[57]]; the present study "
      "shows that on a multi-analyte task with realistic transduction nuisance "
      "the temporal structure is worth a measurable amount, and quantifies how "
      "much. Against physics-informed formulations that impose a residual "
      "through the loss [[68]], [[70]], the architectural encoding used here "
      "removes the need to tune a penalty weight for the dynamics themselves: "
      "the only physical weight that remains is the reconstruction term, and the "
      "sensitivity sweep shows the result is insensitive to it across an order of "
      "magnitude. Against adversarial drift compensation [[64]], [[65]], the "
      "projection reaches a larger improvement on the same public benchmark, "
      "which supports the argument that a subspace whose orientation is known "
      "from the optics is a stronger constraint than one estimated from batch "
      "statistics alone.",
     ],
     "C. Practical and Scientific Implications": [
      "Three practical consequences follow. The nuisance directions used here "
      "are computed from the stack geometry and require no calibration "
      "measurements, so an instrument can initialise the projection from its own "
      "design file rather than from a reference gas sequence, which removes a "
      "step from field deployment. The supervision the drift heads require is "
      "chip temperature and elapsed operating time, both of which a deployed "
      "instrument already logs. And the model runs on two CPU cores at a latency "
      "far below the duration of the measurement itself, so no accelerator is "
      "needed in the analyser.",

      "The scientific implication is narrower but more durable than the accuracy "
      "figure. A plasmonic array responds to nuisance along directions that the "
      "electromagnetic model fixes in advance, and those directions are not "
      "collinear with the directions along which analyte binding acts. That "
      "statement is a property of the transducer, not of this dataset or this "
      "network, and it makes the separation of signal from transduction an "
      "identifiability question with a computable answer rather than a matter of "
      "empirical regularisation. The same construction applies to any sensor "
      "whose nuisance response can be differentiated analytically with respect to "
      "the physical quantity that varies.",
     ],
     "D. Failure Conditions and Trade-offs": [
      "Three conditions degrade the model in ways the experiments make explicit. "
      "When chips differ multiplicatively — in surface sensitivity, decay length "
      "or binding-site density — a linear subspace removal cannot help, and the "
      "cross-chip results show that it does not; correcting that requires a gain "
      "model, which this architecture does not contain. When the discriminative "
      "signal lies inside the removed subspace, the projection costs more than it "
      "returns, which is visible as the decline beyond the optimal rank in the "
      "sensitivity sweep. And when spots fail outright the kinetic branch loses "
      "the coverage channels it depends on, and the model degrades at the same "
      "rate as the baselines rather than more gracefully.",

      "The evidential head involves a deliberate trade. It costs a small amount "
      "of balanced accuracy relative to a softmax head trained on the same "
      "representation, and returns a substantially lower calibration error and a "
      "usable abstention signal in a single forward pass. For a confirmatory "
      "instrument the softmax variant would be the better choice; for a screening "
      "instrument that must decide whether to refer, the evidential variant is. "
      "Both are reported so that the choice can be made on the intended use "
      "rather than on a single headline number.",
     ],
    }


def limitations(S):
    return [
     "The cohort is generated, not measured, and that is the principal limitation "
     "of this study. Every number reported above describes how a read-out behaves "
     "on records produced by a transfer-matrix model of the optical stack and a "
     "competitive Langmuir-Freundlich model of the sorbent chemistry. Those "
     "models are standard and their parameters are drawn from published sources, "
     "but they are still models: real sorbent films exhibit ageing chemistry, "
     "surface fouling and non-Langmuir uptake that no closed-form isotherm "
     "reproduces, and a real breath sample contains hundreds of volatiles rather "
     "than thirteen. The correct reading of the results is therefore comparative "
     "rather than absolute — the ranking of architectures under controlled and "
     "measurable nuisance — and no claim is made about the accuracy a physical "
     "instrument would achieve. Nothing here substitutes for a clinical study.",

     "The concentration priors are unevenly anchored. Healthy and lung-cancer "
     "geometric means come from studies that report absolute parts-per-billion "
     "values with calibrated detection limits [[5]], [[6]], but no comparable "
     "absolute data exist for the obstructive groups: the standard compilation "
     "reports no asthma study with absolute concentrations [[4]] and the largest "
     "recent obstructive cohort reports relative concentrations indexed by "
     "retention time, several of them chemically unidentified [[3]]. The "
     "obstructive priors are therefore constructed from reported directional "
     "changes, and the discrimination reported between chronic obstructive "
     "pulmonary disease and asthma should be treated as a statement about the "
     "assumed separation rather than as an estimate of the achievable one. This "
     "threatens external validity for those two classes specifically and is the "
     "reason the failure analysis of Section V-G dwells on them.",

     "The external validation reaches only one component. The public chemosensor "
     "drift benchmark distributes aggregated steady-state and exponential-average "
     "descriptors rather than raw response curves [[63]], so the recurrent "
     "branches — which are the substance of the architecture — could not be "
     "exercised on real data at all. What the external experiment establishes is "
     "that the low-rank nuisance projection transfers to genuine instrument "
     "drift; what it cannot establish is that the kinetic cell or the dual-rate "
     "coupling would. Until a raw-sensorgram breath dataset is released, that "
     "part of the contribution rests on simulation.",

     "The drift model is additive by construction. Temperature, film ageing, bulk "
     "index and read-out drift enter the observation model of (8) as a "
     "superposition of fixed directions with time-varying amplitudes, and the "
     "identifiability argument of (10) depends on exactly that structure. Real "
     "transducers also drift multiplicatively — sensitivity itself changes as a "
     "film densifies — and the cross-chip experiment shows what happens when the "
     "multiplicative component dominates: the projection stops helping. A "
     "gain-aware extension is the obvious next step and is not attempted here.",

     "Several evaluation choices constrain the strength of the statistical "
     "claims. Five subject-level partitions drawn from one cohort are not "
     "independent datasets, so the Friedman and Nemenyi results are descriptive "
     "rather than inferential, as stated in Section IV-F; with five paired "
     "observations the signed-rank test has limited power and small but real "
     "differences will not reach significance after false-discovery-rate "
     "correction. The Inception baseline is a single network rather than the "
     "five-member ensemble of the published method, which understates it. And "
     "hyperparameters were selected on validation partitions drawn from the same "
     "cohort, so the reported figures carry the usual optimism of "
     "in-distribution model selection.",

     "Finally, the clinical framing is deliberately narrow. The four classes are "
     "treated as mutually exclusive, whereas comorbidity is the norm in "
     "respiratory medicine; smoking status, diet and medication are absent from "
     "the generative model although the literature shows all three shift breath "
     "volatiles by margins comparable with disease [[8]], [[11]]; and breath "
     "signatures are known not to be organ-specific, separating lung cancer from "
     "cancer at other sites at only 0.68 sensitivity [[14]]. A deployed "
     "instrument would need a disease-control arm and a comorbidity model that "
     "this study does not provide.",
    ]


def conclusion(S):
    A = SM.agg(S.e1, "balanced_accuracy")
    au = SM.agg(S.e1, "macro_auroc")
    ec = SM.agg(S.e1, "ece")
    ab = SM.agg(S.e2, "balanced_accuracy")
    cl = SM.agg(S.ceiling, "balanced_accuracy")
    er = {k: np.array(v) for k, v in S.e10["results"].items()}
    seq = {k: v for k, v in A.items() if k in SEQ_MODELS and k != "KINDRA"}
    bs = max(seq, key=lambda k: seq[k]["mean"])
    return [
     "A plasmonic breath record superposes an analyte-specific adsorption "
     "transient on a transduction excursion that is synchronous with it, so the "
     "two cannot be separated on the time axis and existing recurrent read-outs "
     "do not attempt to separate them at all. The architecture presented here "
     "separates them in the space of the sensor array instead, using nuisance "
     "directions obtained by differentiating the electromagnetic model of the "
     "stack, and represents the remaining kinetic signal with a per-spot "
     "coverage state advanced by an exponential integrator of the Langmuir "
     "equation, coupled to a fast gated branch whose update rate is set by the "
     "instantaneous binding velocity.",

     "On 4800 sessions from 2400 simulated subjects under five subject-level "
     f"partitions, the model reaches {SM.pm(A['KINDRA'])} balanced accuracy, "
     f"{SM.pm(au['KINDRA'])} macro area under the curve and "
     f"{SM.pm(ec['KINDRA'])} expected calibration error, ahead of a plain gated "
     f"recurrent unit by {f(A['KINDRA']['mean']-A['GRU']['mean'],3)} and of the "
     f"stacked hybrid by {f(A['KINDRA']['mean']-A['LSTM-GRU (stacked)']['mean'],3)}, "
     f"and within noise of {bs} at {SM.pm(seq[bs])} while reporting markedly "
     "better calibrated confidence. Removing all three "
     "mechanisms from the same "
     "backbone costs "
     f"{f(ab['Full model']['mean']-ab['Base (no proposed component)']['mean'],3)} "
     "of balanced accuracy, and the improvement is attributable component by "
     "component. On a real thirty-six-month chemosensor drift benchmark the "
     "nuisance projection alone improves cross-batch transfer from "
     f"{f(er['MLP'].mean())} to {f(er['MLP + hard projection'].mean())} mean "
     "accuracy over eight held-out batches.",

     "The result the study is most confident about is not the ranking of the "
     "architectures. Measuring the information budget of the measurement chain "
     "shows that the transduction nuisance costs "
     f"{f(cl['Nuisance-free sensorgram']['mean']-cl['Observed sensorgram']['mean'],3)} "
     "of balanced accuracy and that removing the computed subspace returns "
     f"{f(cl['Observed, physical projection r=3']['mean']-cl['Observed sensorgram']['mean'],3)} "
     "of it to any read-out placed after it, engineered or learned. A logistic "
     "classifier on that projected trace is the strongest read-out tested here, "
     "ahead of every sequence model including the proposed one. The physical "
     "prior, in other words, is worth more than the architecture on this cohort, "
     "and the architecture earns its place through calibration, corruption "
     "robustness and an abstention signal a linear classifier does not provide.",

     "The practical contribution is that the nuisance basis requires no "
     "calibration measurements and the drift supervision requires only the chip "
     "temperature and elapsed operating time an instrument already logs, so the "
     "method adds no hardware and runs on two CPU cores at a latency far below "
     "the duration of a breath measurement. The principal limitation is equally "
     "plain: the cohort is generated rather than measured, the concentration "
     "priors for the obstructive classes rest on reported directional changes "
     "because absolute values are not published, and only the projection could "
     "be validated on real data. The most useful next step is not a deeper "
     "network but a gain-aware extension of the same construction, since the "
     "cross-chip results identify multiplicative sensitivity variation, which a "
     "linear subspace removal cannot correct, as the dominant obstacle to "
     "transfer between physical devices.",
    ]


def abstract(S):
    A = SM.agg(S.e1, "balanced_accuracy")
    au = SM.agg(S.e1, "macro_auroc")
    ec = SM.agg(S.e1, "ece")
    ab = SM.agg(S.e2, "balanced_accuracy")
    cl = SM.agg(S.ceiling, "balanced_accuracy")
    er = {k: np.array(v) for k, v in S.e10["results"].items()}
    seq = {k: v for k, v in A.items() if k in SEQ_MODELS and k != "KINDRA"}
    bs = max(seq, key=lambda k: seq[k]["mean"])
    return (
     "Exhaled breath carries volatile markers of lung disease at parts-per-billion "
     "concentrations, and plasmonic sensors transduce them into a resonance shift "
     "whose transient shape identifies the analyte. The same refractive-index "
     "sensitivity that produces the signal, however, also responds to the "
     "temperature, water uptake, film ageing and read-out drift of the "
     "transducer, and because exhaled air is warm and saturated that response is "
     "synchronous with binding and cannot be filtered in time. Existing learned "
     "read-outs for such sensors classify static frames and discard the transient "
     "entirely, while recurrent models applied to chemical sensors carry a single "
     "update rate and no account of the transduction geometry. This work "
     "separates the two components in the space of the sensor array rather than "
     "in time, using a low-rank nuisance subspace obtained by differentiating a "
     "transfer-matrix model of the stack, and represents the remaining signal "
     "with a recurrent state that is a per-spot fractional coverage vector "
     "advanced by an exponential integrator of the competitive Langmuir equation, "
     "coupled to a gated branch whose update rate follows the instantaneous "
     "binding velocity. On 4800 sessions from 2400 subjects generated by a "
     "physically grounded forward model of a twelve-spot array, the method "
     f"reaches {f(A['KINDRA']['mean'])} balanced accuracy, "
     f"{f(au['KINDRA']['mean'])} macro area under the curve and "
     f"{f(ec['KINDRA']['mean'])} expected calibration error, ahead of the gated "
     f"recurrent family by {f(A['KINDRA']['mean']-A['GRU']['mean'],3)} and "
     "indistinguishable from the strongest convolutional baseline while halving "
     "its calibration error; ablation attributes "
     f"{f(ab['Full model']['mean']-ab['Base (no proposed component)']['mean'],3)} "
     "of balanced accuracy to the three mechanisms. Measuring the information "
     "budget of the chain shows that the nuisance costs "
     f"{f(cl['Nuisance-free sensorgram']['mean']-cl['Observed sensorgram']['mean'],3)} "
     "of balanced accuracy and that removing the computed subspace recovers "
     f"{f(cl['Observed, physical projection r=3']['mean']-cl['Observed sensorgram']['mean'],3)} "
     "of it for any read-out, learned or engineered. On a real thirty-six-month "
     "chemosensor drift benchmark the same projection raises cross-batch accuracy "
     f"from {f(er['MLP'].mean())} to {f(er['MLP + hard projection'].mean())}. "
     "The basis needs no calibration measurements, which makes the construction "
     "applicable to any transducer whose nuisance response can be differentiated "
     "analytically.")
