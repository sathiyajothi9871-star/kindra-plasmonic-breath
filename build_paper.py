# -*- coding: utf-8 -*-
"""Assemble the manuscript."""
from __future__ import annotations
import os, re
import numpy as np
import config as C
import docx_build as DB
import cite as CITE
import refs as REFS
import equations as EQ
import summarise as SM
import tables as TB
import manuscript_a as MA
import manuscript_b as MB
import manuscript_c as MC
import manuscript_d as MD
import manuscript_e as ME

TPL = ("/root/.claude/uploads/49c2f1a3-253a-560d-847e-42fa5115ad85/"
       "3831b319-Q1_SCI_IEEE_Transactions_Master_Template.docx")
OUT = "/mnt/user-data/outputs/Lung_Disease_Plasmonic_LSTM_GRU.docx"
FIG = C.FIGURE_DIR

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
         "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII"]
EQD = {n: l for n, l in EQ.EQUATIONS}


class Builder:
    def __init__(self):
        self.d = DB.Doc(TPL)
        self.cit = CITE.Renumber()
        self.tab = 0
        self.fig = 0

    # -- helpers ---------------------------------------------------------
    def t(self, s):
        return self.cit.scan_and_replace(s)

    def para(self, s, **kw):
        self.d.para(self.t(s), **kw)

    def eq(self, n):
        self.d.equation(EQD[n], n)

    def table(self, rows, caption, cols=2, widths=None, size=16, align=None):
        self.tab += 1
        self.d.caption(self.t(f"TABLE {ROMAN[self.tab]}\n{caption}"),
                       "TableCaption", cols=cols, keep=True)
        self.d.table([[self.t(c) for c in r] for r in rows], widths=widths,
                     cols=cols, size=size, align=align)

    def figure(self, name, caption, cols=1, width=None):
        self.fig += 1
        w = width or (6.55 if cols == 1 else 3.30)
        self.d.image(os.path.join(FIG, name), w, cols=cols)
        self.d.caption(self.t(f"Fig. {self.fig}. {caption}"), "FigureCaption",
                       cols=cols)

    # -- document --------------------------------------------------------
    def build(self, S):
        d, D = self.d, self.d
        # --- front matter (single column) ---
        d._add('<w:p><w:pPr><w:pStyle w:val="PaperTitle"/></w:pPr>'
               + DB._runs(MA.TITLE) + "</w:p>", 1)
        d.para(MA.AUTHORS, style="Authors", cols=1)
        d.para(MA.AFFIL, style="Affiliations", cols=1)
        d.blank(1)
        self.d.para("Abstract—" + self.t(ME.abstract(S)), indent=False, cols=1)
        d.blank(1)
        d.para("Index Terms—" + MA.INDEX_TERMS, indent=False, cols=1)
        d.blank(1)

        # --- I. Introduction ---
        d.section("I. INTRODUCTION")
        for p in MA.introduction(S):
            self.para(p)
        d.subsection("A. Research Objectives")
        for o in MA.OBJECTIVES:
            self.para(o)
        d.subsection("B. Contributions")
        for i, cbody in enumerate(MA.CONTRIBUTIONS, 1):
            self.para(f"C{i}. {cbody}")
        self.para(
         "The remainder of the paper is organised as follows. Section II reviews "
         "the literature by methodological theme and states the research gap. "
         "Section III develops the transducer model, the three mechanisms and the "
         "training objective. Section IV describes the cohort, the partitions, "
         "the baselines and the statistical protocol. Section V reports the "
         "results, Section VI interprets them, Section VII states the "
         "limitations and Section VIII concludes.")

        # --- II. Related work ---
        d.section("II. RELATED WORK")
        for title, paras in MB.RELATED:
            d.subsection(title)
            for p in paras:
                self.para(p)
        d.subsection("F. Research Gap")
        self.para(MB.GAP)
        self.table(MB.COMPARISON_TABLE,
                   "POSITION OF THE PROPOSED WORK RELATIVE TO THE CLOSEST METHODS",
                   cols=1, widths=[900, 1900, 2600, 1900, 3100], size=15,
                   align=["c", "l", "l", "l", "l"])

        # --- III. Methodology ---
        d.section("III. PROPOSED METHODOLOGY")
        d.subsection("A. Problem Formulation and Transducer Model")
        for i, p in enumerate(MC.PROBLEM):
            self.para(p)
            if i == 1:
                for n in (1, 2, 3):
                    self.eq(n)
            elif i == 2:
                for n in (4, 5):
                    self.eq(n)
            elif i == 3:
                for n in (6, 7):
                    self.eq(n)
            elif i == 4:
                for n in (8, 9):
                    self.eq(n)
            elif i == 5:
                self.eq(10)
        self.table(MC.NOTATION, "NOTATION USED THROUGHOUT THE MANUSCRIPT",
                   cols=2, widths=[900, 1750, 800], size=15,
                   align=["c", "l", "c"])

        d.subsection("B. System Overview")
        self.para(
         "Figure 1 shows the complete chain. A breath sample is drawn across the "
         "functionalised array through a sampling line, the twelve resonances are "
         "tracked and referenced to their pre-exhalation baseline, and the "
         "resulting record enters the read-out. Inside the read-out the "
         "dispersion projection acts first, because every downstream component "
         "should see a signal from which the transduction excursion has already "
         "been reduced; the kinetic branch then produces a coverage trajectory "
         "and a reconstruction of the projected trace, and the fast branch "
         "consumes the reconstruction residual under the control of the binding "
         "velocity. Pooled statistics of both branches feed a Dirichlet head that "
         "returns a belief over the four classes together with an uncertainty "
         "mass, and an abstention rule is applied to that mass rather than to the "
         "maximum probability.")
        self.figure("fig1_architecture.png",
                    "End-to-end organisation of the proposed read-out. Solid "
                    "arrows carry signals, the dashed arrow carries the binding "
                    "velocity that modulates the fast branch, the dash-dotted "
                    "arrow carries the kinetic reconstruction residual, and the "
                    "dotted arrow carries the nuisance coefficients to the "
                    "co-operative and adversarial drift heads. Mechanisms A, B "
                    "and C are the three components ablated in Section V-B.")

        d.subsection("C. Core Proposed Mechanisms")
        d.subsubsection("1) Dispersion-disentangling projection")
        for i, p in enumerate(MC.MECH_A):
            self.para(p)
            if i == 0:
                self.eq(11); self.eq(12)
        d.subsubsection("2) Langmuir kinetic state cell")
        for i, p in enumerate(MC.MECH_B):
            self.para(p)
            if i == 0:
                self.eq(13); self.eq(14)
            elif i == 1:
                self.eq(15); self.eq(16); self.eq(17)
            elif i == 2:
                self.eq(18)
        d.subsubsection("3) Velocity-modulated dual-rate coupling")
        for i, p in enumerate(MC.MECH_C):
            self.para(p)
            if i == 0:
                self.eq(19); self.eq(20); self.eq(21)
        self.figure("fig3_mechanism.png",
                    "The kinetic state cell and the two adsorption regimes it is "
                    "built to represent. (a) One update of the coverage state; "
                    "the forget term is the physical relaxation factor and the "
                    "input term an association flux. (b) Weighted occupancy of an "
                    "amine-functionalised spot and a dispersive spot for a healthy "
                    "and a lung-cancer sample on the same chip; shading marks "
                    "exhalation windows. (c) Relaxation time of each latent site "
                    "under three drive amplitudes, showing that the affinity "
                    "parameterisation of (14) preserves the timescale ordering "
                    "imposed at initialisation.")
        d.subsubsection("4) Pooling and evidential head")
        for i, p in enumerate(MC.HEAD):
            self.para(p)
            if i == 0:
                self.eq(22)
            elif i == 1:
                self.eq(23); self.eq(24)

        d.subsection("D. Objective Function and Optimisation")
        for i, p in enumerate(MC.OBJECTIVE):
            self.para(p)
            if i == 0:
                self.eq(25); self.eq(26)
            else:
                self.eq(27); self.eq(28); self.eq(29)

        d.subsection("E. Training and Inference Algorithms")
        self.para(
         "Algorithms 1 and 2 give the two procedures in full. Every operation "
         "corresponds to a numbered equation, and the inference procedure is the "
         "training procedure with the loss, the reversal and the parameter update "
         "removed and the abstention rule added.")
        d.algorithm("Algorithm 1  Training", [self.t(x) for x in MC.ALG_TRAIN])
        d.blank()
        d.algorithm("Algorithm 2  Inference with abstention",
                    [self.t(x) for x in MC.ALG_INFER])
        d.blank()

        d.subsection("F. Complexity Analysis")
        for p in MC.COMPLEXITY:
            self.para(p)
        self.eq(30)

        # --- IV. Data and protocol ---
        d.section("IV. DATA AND EXPERIMENTAL PROTOCOL")
        d.subsection("A. Cohort Construction and Array Design")
        for p in MD.COHORT:
            self.para(p)
        self.table(TB.voc_table(),
                   "VOLATILE PANEL, CLASS-CONDITIONAL GEOMETRIC MEANS AND THE "
                   "PUBLISHED MEASUREMENTS THEY ARE ANCHORED TO",
                   cols=1, widths=[1500, 1400, 1400, 1200, 1200, 900, 1800],
                   size=15, align=["l", "c", "c", "c", "c", "c", "c"])
        self.table(TB.array_table(S.devmeta),
                   "COMPUTED RESPONSE COEFFICIENTS OF ONE CHIP",
                   cols=1, widths=[700, 1900, 1900, 1600, 1200, 1200, 1000],
                   size=15, align=["c", "l", "l", "c", "c", "c", "c"])
        self.figure("fig2_physics.png",
                    "The transducer model and the geometry the read-out "
                    "exploits. (a) Computed reflectance of three spots. (b) Bulk "
                    "and surface index sensitivity of every spot. (c) Absolute "
                    "cosine between the four nuisance response directions and the "
                    "mean binding direction. (d) One recorded sensorgram "
                    "decomposed into its binding and transduction components. "
                    "(e) Distribution of desorption time constants, showing the "
                    "separation between physisorption and carbonyl capture.")
        d.subsection("B. Preprocessing and Data Partitioning")
        for p in MD.PARTITION:
            self.para(p)
        self.table(TB.cohort_table(S.raw), "COHORT AND ACQUISITION SUMMARY",
                   cols=2, widths=[1250, 500, 550, 500, 500, 550], size=15,
                   align=["l", "c", "c", "c", "c", "c"])
        d.subsection("C. Baselines")
        for p in MD.BASELINES:
            self.para(p)
        d.subsection("D. Implementation Details")
        for p in MD.IMPLEMENTATION:
            self.para(p)
        self.table(TB.implementation_table(), "IMPLEMENTATION AND TRAINING SETTINGS",
                   cols=1, widths=[2400, 2400, 2400, 2400], size=15,
                   align=["l", "l", "l", "l"])
        d.subsection("E. Evaluation Metrics")
        for p in MD.METRICS_TEXT:
            self.para(p)
        self.eq(31); self.eq(32); self.eq(33)
        d.subsection("F. Statistical Analysis")
        for p in MD.STATS_TEXT:
            self.para(p)
        self.eq(34)

        # --- V. Results ---
        d.section("V. RESULTS")
        d.subsection("A. Information Budget and Main Benchmark Comparison")
        for p in ME.results_budget(S):
            self.para(p)
        self.table(S.budget_table(), "INFORMATION BUDGET OF THE MEASUREMENT CHAIN",
                   cols=2, widths=[1800, 800, 750], size=15,
                   align=["l", "c", "c"])
        for p in ME.results_main(S):
            self.para(p)
        self.table(S.main_rows(), "MAIN BENCHMARK COMPARISON OVER FIVE "
                   "SUBJECT-LEVEL PARTITIONS (MEAN $\\pm$ SD)", cols=1,
                   widths=[2100, 1500, 1350, 1200, 1500, 1400, 1250], size=15,
                   align=["l", "c", "c", "c", "c", "c", "c"])
        self.figure("fig4_benchmark.png",
                    "Main benchmark. Bars are means over five subject-level "
                    "partitions and whiskers one standard deviation; the proposed "
                    "model is shown in the darker shade.")
        self.table(S.stats_rows(), "PAIRWISE COMPARISON AGAINST THE PROPOSED "
                   "MODEL (WILCOXON SIGNED-RANK, BENJAMINI-HOCHBERG ADJUSTED)",
                   cols=1, widths=[2400, 1400, 1400, 1300, 1300, 1300, 1300],
                   size=15, align=["l", "c", "c", "c", "c", "c", "c"])

        d.subsection("B. Ablation Study")
        for p in ME.results_ablation(S):
            self.para(p)
        self.table(S.ablation_rows(), "ABLATION OF THE THREE MECHANISMS AND THE "
                   "EVIDENTIAL HEAD", cols=1,
                   widths=[3000, 1700, 1700, 1500, 1500], size=15,
                   align=["l", "c", "c", "c", "c"])
        self.figure("fig5_ablation.png",
                    "Component ablation. (a) Discrimination as mechanisms are "
                    "added to a matched backbone. (b) Expected calibration error "
                    "for the same variants.")

        d.subsection("C. Sensitivity and Robustness Analysis")
        for p in ME.results_sensitivity(S):
            self.para(p)
        self.table(S.sensitivity_rows(), "SENSITIVITY TO THE PROJECTION RANK AND "
                   "THE LOSS WEIGHTS", cols=2, widths=[1500, 950, 900], size=15,
                   align=["l", "c", "c"])
        self.figure("fig6_sensitivity.png",
                    "Sensitivity sweeps. Points are means over the repeated "
                    "partitions and bars one standard deviation.")
        for p in ME.results_robustness(S):
            self.para(p)
        self.table(S.robustness_rows(), "BALANCED ACCURACY UNDER TEST-TIME "
                   "CORRUPTION (NO RETRAINING)", cols=1,
                   widths=[1600, 900, 1300, 1200, 1600, 1400, 1300, 1500],
                   size=15, align=["l", "c", "c", "c", "c", "c", "c", "c"])
        self.figure("fig7_robustness.png",
                    "Robustness to four test-time corruptions. Line style and "
                    "marker identify the model; no model is retrained.")

        d.subsection("D. Generalisation and External Validation")
        for p in ME.results_generalisation(S):
            self.para(p)
        self.table(S.gen_rows(), "CROSS-CHIP GENERALISATION WITH TWO CHIPS "
                   "WITHHELD", cols=2, widths=[1550, 850, 800, 650], size=15,
                   align=["l", "c", "c", "c"])
        self.table(S.external_rows(), "EXTERNAL VALIDATION OF THE NUISANCE "
                   "PROJECTION ON A REAL 36-MONTH CHEMOSENSOR DRIFT BENCHMARK",
                   cols=1, widths=[2600, 800, 800, 800, 800, 800, 800, 800, 800,
                                   1400], size=14,
                   align=["l"] + ["c"] * 9)
        self.figure("fig8_generalisation.png",
                    "Transfer under hardware shift. (a) Leave-chips-out "
                    "evaluation on the simulated cohort. (b) Per-batch accuracy "
                    "on the real chemosensor drift benchmark, with training "
                    "restricted to the first two batches.")

        d.subsection("E. Efficiency Analysis")
        for p in ME.results_efficiency(S):
            self.para(p)
        self.table(S.efficiency_rows(), "COMPUTATIONAL COST ON TWO CPU CORES",
                   cols=2, widths=[1550, 900, 850, 800], size=15,
                   align=["l", "c", "c", "c"])

        d.subsection("F. Calibration and Selective Prediction")
        for p in ME.results_uncertainty(S):
            self.para(p)
        self.table(S.uncertainty_rows(), "UNCERTAINTY ESTIMATION COMPARED AT "
                   "MATCHED REPRESENTATION", cols=2,
                   widths=[1600, 850, 800, 800], size=15,
                   align=["l", "c", "c", "c"])
        self.figure("fig9_uncertainty.png",
                    "Confidence behaviour. (a) Reliability of the evidential head "
                    "against a softmax baseline. (b) Selective error against "
                    "coverage when records are ranked by uncertainty. "
                    "(c) Expected calibration error of the alternatives compared "
                    "at matched representation.")

        d.subsection("G. Error Structure")
        for p in ME.results_failure(S):
            self.para(p)
        self.figure("fig10_failure.png",
                    "Error structure of the proposed model, pooled over the five "
                    "partitions. (a) Row-normalised confusion. (b) Per-class "
                    "$F_1$ against a gated recurrent baseline. (c) Distribution "
                    "of evidential uncertainty for correct and incorrect "
                    "predictions.")

        # --- VI. Discussion ---
        d.section("VI. DISCUSSION")
        for title, paras in ME.discussion(S).items():
            d.subsection(title)
            for p in paras:
                self.para(p)

        # --- VII. Limitations ---
        d.section("VII. LIMITATIONS")
        for p in ME.limitations(S):
            self.para(p)

        # --- VIII. Conclusion ---
        d.section("VIII. CONCLUSION")
        for p in ME.conclusion(S):
            self.para(p)

        # --- back matter ---
        d.section("DATA AVAILABILITY")
        self.para(
         "The generator that produces the cohort, together with the "
         "configuration file that fixes every parameter reported in Tables III "
         "to VI, is released with the source code, so the 2400 sessions used here "
         "can be reproduced exactly from the seeds listed in Table VI. No human "
         "subjects were involved and no clinical data were used. The external "
         "benchmark is the publicly distributed gas-sensor drift collection of "
         "[[63]] and is obtained from its published source.")
        d.section("CODE AVAILABILITY")
        self.para(
         "The complete implementation is released as a single directory "
         "containing the optical forward model, the adsorption model, the cohort "
         "generator, the proposed architecture, every baseline, the experimental "
         "protocol, the statistical analysis and the figure scripts, together "
         "with a requirements file and a description of how to reproduce each "
         "table and figure. Repository URL and archived release identifier to be "
         "supplied on acceptance.")
        d.section("ETHICS STATEMENT")
        self.para(
         "This study used no human participants, no animal subjects and no "
         "identifiable data. The cohort is generated by the forward model "
         "described in Section IV-A. No ethics approval was therefore required, "
         "and none is claimed.")
        d.section("AUTHOR CONTRIBUTIONS")
        self.para(
         "Conceptualisation, methodology, software, formal analysis, "
         "investigation, data curation, writing of the original draft, review and "
         "editing, and visualisation were carried out by the author.")
        d.section("CONFLICT OF INTEREST")
        self.para("The author declares no competing financial or non-financial "
                  "interests.")
        d.section("ACKNOWLEDGMENT")
        self.para(
         "The author thanks the maintainers of the publicly distributed "
         "chemosensor drift collection used for external validation, and the "
         "authors of the breath-analysis studies whose reported concentrations "
         "made an anchored generative model possible.")

        # --- references ---
        d.section("REFERENCES")
        for i, r in enumerate(self.cit.final_list(REFS.REFERENCES), 1):
            d._add('<w:p><w:pPr><w:jc w:val="both"/><w:spacing w:after="20"/>'
                   '<w:ind w:left="260" w:hanging="260"/></w:pPr>'
                   f'{DB._runs(f"[{i}] {r}")}</w:p>', 2)
        return d.save(OUT)
