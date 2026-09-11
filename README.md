# KINDRA — Kinetics-Informed Dual-Rate Architecture for Plasmonic Breath Analysis

Reference implementation for *Lung Disease Diagnosis Using Plasmonic Sensors
Integrated with a Hybrid LSTM–GRU Architecture*.

The repository contains everything needed to reproduce the manuscript end to
end: the electromagnetic forward model of the sensor array, the competitive
adsorption model of the sorbent chemistry, the cohort generator, the proposed
architecture, thirteen baselines, the full experimental protocol, the
statistical analysis, every figure and every table.

---

## What the model does

A plasmonic breath record superposes an analyte-specific adsorption transient on
a transduction excursion caused by temperature, water uptake, film ageing and
read-out drift. Because exhaled air is warm and saturated, the excursion is
synchronous with the signal and cannot be separated on the time axis. It can be
separated in the space of the sensor array, because the direction along which
the twelve resonances move under a nuisance is fixed by the optics and is not
collinear with the direction produced by surface binding.

Three mechanisms follow from that observation.

1. **Dispersion-disentangling projection.** A low-rank basis initialised from
   the computed thermal, ageing, bulk-index, read-out and fabrication response
   directions is softly removed from the input, held responsible for the
   nuisance by a co-operative head and excluded from the retained representation
   by a gradient-reversal head.
2. **Langmuir kinetic state cell.** The slow branch is a recurrent cell whose
   state is a fractional-coverage vector advanced by an exponential integrator
   of the competitive Langmuir equation. The forget term is the physical
   relaxation factor, not a free gate. Parameterising by affinity and desorption
   rate (`k_a = K_s * k_d`) keeps the timescale ordering stable during training.
3. **Velocity-modulated dual-rate coupling.** The fast gated branch consumes the
   residual the kinetic branch cannot explain, and its update gate is modulated
   by the instantaneous binding velocity of the slow branch.

A Dirichlet head returns a class belief together with an uncertainty mass, so an
ambiguous record can be referred rather than assigned.


## Headline results

Cohort: 2400 simulated subjects, 4800 sessions, 12 plasmonic spots, 90 samples per
record. Five subject-level partitions; mean +- SD across partitions.

| read-out | balanced acc. | macro AUROC | ECE |
|---|---|---|---|
| Logistic regression + physical projection | 0.794 +- 0.016 | 0.947 | 0.031 |
| Logistic regression | 0.755 +- 0.017 | 0.935 | 0.036 |
| InceptionTime | 0.746 +- 0.015 | 0.929 | 0.060 |
| **KINDRA (this work)** | **0.742 +- 0.017** | **0.925** | **0.033** |
| LSTM-GRU (stacked) | 0.697 +- 0.023 | 0.907 | 0.039 |
| GRU | 0.686 +- 0.026 | 0.906 | 0.040 |
| LSTM | 0.681 +- 0.022 | 0.901 | 0.036 |

Information budget of the measurement chain (same classifier, three representations):

| representation | balanced acc. |
|---|---|
| true alveolar concentrations | 0.891 |
| nuisance-free sensorgram | 0.823 |
| observed sensorgram | 0.758 |
| observed + physical projection (r=3) | 0.801 |

Ablation on a matched backbone (3 partitions): base 0.716,
+ projection 0.743, + kinetic cell 0.719,
full model 0.749.

Cross-chip transfer (two chips withheld): KINDRA 0.589, InceptionTime
0.503, LSTM-FCN 0.491 - the convolutional models
that match KINDRA in distribution collapse under hardware shift.

External validation on the real 36-month chemosensor drift benchmark, mean accuracy
over eight held-out batches: plain network 0.668, adversarial batch head
0.684, hard projection 0.708,
gated projection 0.700.

**The honest summary:** the physically derived nuisance projection is the decisive
mechanism and it helps every read-out, learned or engineered; a linear classifier on
engineered descriptors of the projected trace is the strongest read-out tested. The
proposed architecture leads the recurrent family by about five points, matches the best
convolutional baseline in distribution while halving its calibration error, and is far
more stable than it under cross-chip shift.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU only is sufficient; the whole study was produced on two cores.

---

## Reproducing the study

```bash
python make_data.py                # generate and cache the cohort (~30 s)
python run_all.py 012345 8         # run experiments E0-E5 and E8
python external_uci.py             # external validation on real drift data
python make_figures.py 1234567890  # all ten figures
python build_paper.py              # assemble the manuscript
```

`run_all.py` takes a string of experiment identifiers:

| id | experiment |
|----|------------|
| 0 | information budget of the measurement chain |
| 1 | main benchmark, five subject-level partitions |
| 2 | ablation of the three mechanisms and the evidential head |
| 3 | sensitivity to projection rank and loss weights |
| 4 | robustness under test-time corruption |
| 5 | cross-chip generalisation |
| 8 | calibration and uncertainty comparison |

Results are written as JSON to `results/`, figures as PNG to `figures/`.

### External data

`external_uci.py` expects the ten batch files of the public gas-sensor drift
collection (Vergara et al., *Sens. Actuators B*, 2012) in `data_ext/` as
`batch1.dat` … `batch10.dat` in the distributed LIBSVM-style format.

---

## Files

| file | contents |
|------|----------|
| `config.py` | every optical, chemical, cohort and training parameter |
| `optics.py` | Kretschmann stack, Abelès transfer matrix, Rakić gold, SF10 Sellmeier, sensitivities and nuisance response directions |
| `kinetics.py` | volatile panel, coating selectivity, competitive Langmuir–Freundlich integration |
| `breath_cohort.py` | concentration priors, chips, sessions, forward simulation |
| `make_data.py` | generates and caches the cohort |
| `dataset.py` | subject-level and leave-chips-out partitioning, scaling |
| `model.py` | the proposed architecture and its ablation variants |
| `baselines.py` | thirteen reference models and the handcrafted descriptors |
| `losses.py` | evidential, kinetic-consistency and drift objectives |
| `train.py` | shared training engine, augmentation, class weighting |
| `evaluate.py` | metrics, calibration, bootstrap, risk–coverage |
| `stats.py` | Wilcoxon, Friedman, Nemenyi, effect sizes, FDR control |
| `experiments.py` | experiments E0–E5 and E8 |
| `external_uci.py` | external validation of the projection on real drift data |
| `run_all.py` | driver |
| `figures.py`, `make_figures.py` | all publication figures |
| `tables.py`, `summarise.py` | table assembly and result aggregation |
| `omml.py`, `docx_build.py`, `build_paper.py` | manuscript assembly with native Word equations |
| `equations.py`, `refs.py`, `cite.py`, `manuscript_*.py` | manuscript content |

---

## Notes on scope

The cohort is **generated**, not measured. The optical and adsorption models are
standard and their parameters are anchored to published breath measurements, but
no clinical data were used and no claim is made about the accuracy a physical
instrument would achieve. The external validation exercises the nuisance
projection on real instrument drift; the recurrent branches could not be
validated externally because no public dataset distributes raw plasmonic breath
sensorgrams.

## Licence

MIT. See `LICENSE`.
