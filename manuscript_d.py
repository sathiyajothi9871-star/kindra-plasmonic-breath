# -*- coding: utf-8 -*-
"""Data and experimental protocol."""

COHORT = [
 "No public collection of raw plasmonic breath sensorgrams exists, and the "
 "published breath literature reports either aggregated descriptors or "
 "chromatographic concentrations rather than time-resolved optical traces. The "
 "evaluation cohort is therefore generated from the forward model of Section "
 "III-A: the transfer-matrix computation of (1)-(5) supplies the response "
 "coefficients of every sensing spot, the competitive law of (6) supplies the "
 "coverage trajectories, and (8) supplies the observed record. This is a "
 "simulation, and the consequences of that choice are stated without hedging in "
 "Section VII. What the construction does provide is a testbed in which the "
 "quantity of interest — how much diagnostic information survives transduction — "
 "is measurable, because the nuisance-free trace and the true concentrations are "
 "both available for reference.",

 "Concentration priors are anchored to published measurements rather than "
 "invented. Geometric means for the healthy group are taken from breath studies "
 "that report absolute parts-per-billion values, and the lung-cancer group is "
 "obtained by applying the ratios those same studies report between cancer "
 "patients and matched controls [[5]], [[6]], [[15]]. Between-subject variability "
 "is log-normal with geometric standard deviations between 1.7 and 2.4, which is "
 "the range the primary studies exhibit. Because absolute-concentration ground "
 "truth for obstructive disease is not available anywhere in the published "
 "record — the most recent and best-powered obstructive cohort reports only "
 "relative concentrations indexed by retention time, several of them chemically "
 "unidentified [[3]], and the standard compilation lists no asthma study with "
 "absolute values at all [[4]] — the obstructive priors are constructed from the "
 "reported directional changes, with oxidative-stress aldehydes and alkanes "
 "elevated and the ketone and isoprene responses differing in sign between the "
 "two obstructive groups. That asymmetry is a property of the field and is "
 "carried forward as a limitation rather than concealed. Table III lists the panel "
 "and the anchors.",

 "The array is twelve functionalised spots on a common gold film in the "
 "attenuated-total-reflection geometry. Four spots carry generic sorbents spanning "
 "a dispersive-to-polar gradient, four are amine-functionalised and capture "
 "carbonyls through Schiff-base condensation, two are aromatic-selective and two "
 "favour short-chain alcohols; this mirrors the cross-reactive layouts used in "
 "plasmonic optoelectronic noses and the carbonyl-capture chemistry used in "
 "Raman breath substrates [[25]], [[32]]. Affinities are generated from "
 "physicochemical descriptors of the volatiles — partition coefficient, dipole "
 "moment, molar volume, hydrogen-bond capacity — combined with the functional "
 "selectivity of each coating, which reproduces the low-rank affinity structure "
 "that gives real cross-reactive arrays their multicollinearity. Desorption time "
 "constants span 2.5 to 28 s for physisorbed volatiles and are lengthened by an "
 "order of magnitude on the amine spots for aldehydes, consistent with the "
 "reported irreversibility of Schiff-base capture [[32]]. Table IV lists the "
 "computed response coefficients of one chip.",

 "A recording lasts 180 s: a 30 s ambient purge, three cycles of 20 s exhalation "
 "and 20 s purge, and a 30 s recovery, sampled at 0.5 Hz as one full spectral "
 "scan every 2 s. Exhaled air enters at 34 degrees Celsius and 95 per cent "
 "relative humidity against ambient conditions near 25 degrees and 42 per cent, "
 "and the sampling line contributes a 2.2 s transport lag. Eight physical chips "
 "differ in gold thickness by 1.6 nm standard deviation, in coating thickness by "
 "5.5 per cent and in spot-to-spot affinity by 0.045 decades, and each session "
 "carries an elapsed-age between zero and eighteen months that reduces "
 "binding-site density by 1.3 per cent per month. Three notional sites differ in "
 "ambient temperature offset, humidity and background volatile burden, following "
 "the reported dependence of room-air composition on location [[10]]. Read-out "
 "noise is 2.5 pm root-mean-square with 0.5 pm quantisation, and the spectrograph "
 "wavelength scale drifts with an offset and a dispersive component. Table V "
 "summarises the cohort.",
]

PARTITION = [
 "Every session of a subject falls in exactly one partition. Splits are drawn at "
 "the level of the subject and stratified by class, in the proportion 65, 15 and "
 "20 per cent, and are repeated over five random draws with the seeds fixed in "
 "the released configuration. A session-level random split would place two "
 "recordings of the same person on both sides of the partition and is not used "
 "anywhere in this study; the difference is not cosmetic, because within-subject "
 "repeatability is far tighter than between-subject variability.",

 "The cross-chip protocol is stricter still. Two of the eight chips are withheld "
 "entirely, the remaining six supply training and validation data in the ratio "
 "85 to 15, and the withheld chips supply the test set; three disjoint folds are "
 "used so that six of the eight chips appear in a held-out role. Because chip "
 "identity is confounded with elapsed age and site in the generated cohort, this "
 "protocol measures transfer under simultaneous geometric, ageing and ambient "
 "shift rather than under geometric shift alone.",

 "Records are scaled by a single global factor estimated on the training "
 "partition. A per-channel standardisation is deliberately avoided: it would "
 "rescale the array axes independently and destroy the geometric relationship "
 "between the nuisance directions and the binding directions on which (10) "
 "depends. Channel-specific gain is left to the first learned layer of every "
 "model, so no architecture is disadvantaged by this choice.",
]

BASELINES = [
 "Fifteen reference models are used, in four families. Classical read-out is a logistic regression, a "
 "random forest and a radial-basis support vector machine on twelve descriptors "
 "per spot covering steady-state amplitude, exposure and purge means, the "
 "difference between the first exposure and the following purge, the net drift "
 "across the record and the extremes of the first difference; these are the "
 "descriptors a conventional plasmonic read-out would compute. Plain gated "
 "recurrence is represented by a long short-term memory network and a gated "
 "recurrent unit [[41]], [[42]]. The hybrid architecture named in the title of "
 "this study is represented twice, once stacked and once in parallel, matching "
 "the two ways it is assembled in the applied literature [[53]], [[54]].",

 "Modern time-series classification supplies a two-branch convolutional and "
 "recurrent network, a dilated causal convolutional network, a single Inception "
 "network and a transformer encoder with fixed positional encoding [[45]], "
 "[[48]], [[49]], together with a random convolutional-kernel transform followed "
 "by a ridge classifier [[46]], [[47]]. The Inception model is used as a single "
 "network rather than the five-member ensemble of the original method, because a "
 "fivefold inference cost is not compatible with the point-of-care setting this "
 "study targets; the difference is noted wherever that model is compared. Two "
 "further baselines address the same problems as the proposed mechanisms from a "
 "different direction: a fixed multi-rate recurrent network with clock periods of "
 "one, two and four samples [[50]], and a gated recurrent encoder with a "
 "gradient-reversal nuisance head, which is the recurrent counterpart of "
 "adversarial drift compensation [[64]], [[66]].",

 "All neural models share the training procedure, the class weighting, the "
 "partitions, the seeds and the model-selection criterion. Reported capacities "
 "lie between 22 900 and 144 300 parameters, so the comparison is not confounded "
 "by an order-of-magnitude difference in size.",
]

IMPLEMENTATION = [
 "Models are implemented in PyTorch 2.14 on a two-core x86-64 container with 7 GB "
 "of memory and no accelerator, which is a deliberate choice: an architecture "
 "intended for a portable breath analyser should be shown to train and run "
 "without a graphics processor. Optimisation uses AdamW with an initial learning "
 "rate of 1.8 multiplied by ten to the minus three, weight decay of ten to the "
 "minus four, a cosine schedule over 60 epochs, minibatches of 64 records and "
 "gradient-norm clipping at 2. The state achieving the best validation balanced "
 "accuracy is retained and training stops after fifteen epochs without "
 "improvement. Loss weights are 0.02 for the evidential regulariser, 0.60 for "
 "kinetic consistency and 0.15 for the drift term, with the reversal coefficient "
 "ramped over training; the projection rank is three and the kinetic cell carries "
 "forty latent sites against a forty-eight-unit fast branch. Section V-C reports "
 "the sensitivity of the result to each of these. Table VI lists the settings in "
 "full.",
]

METRICS_TEXT = [
 "Because the cohort is unbalanced, accuracy alone is not reported as a primary "
 "outcome. Balanced accuracy and the Matthews correlation coefficient of (33) are "
 "the headline discrimination measures, supported by macro-averaged $F_1$, "
 "one-versus-rest macro area under the receiver operating characteristic curve "
 "and macro average precision; the last is included because it is the more "
 "informative of the two ranking measures when a class is small. Calibration is "
 "reported as the fifteen-bin expected calibration error of (31), the multi-class "
 "Brier score and the negative log-likelihood of (32) [[76]], [[77]]. Efficiency "
 "is reported as parameter count, wall-clock training time and median "
 "single-record inference latency on the same two-core machine.",
]

STATS_TEXT = [
 "Every neural result is the outcome of five independent subject-level "
 "partitions. Point estimates are reported as mean and standard deviation across "
 "those partitions, and percentile bootstrap intervals over 2000 resamples of the "
 "test sessions are given for the primary comparison. Pairwise comparison against "
 "the proposed model uses the two-sided Wilcoxon signed-rank test on the paired "
 "per-partition scores, with Cliff's delta and the paired standardised mean "
 "difference of (34) as effect sizes, and Benjamini-Hochberg control of the false "
 "discovery rate across the family of pairwise tests [[78]], [[79]]. A Friedman "
 "test across all models over the partitions is followed by a Nemenyi critical "
 "difference, computed from the same expression in (34).",

 "One deviation from the recommended protocol is stated explicitly. The Friedman "
 "and Nemenyi procedures assume independent datasets, whereas the partitions used "
 "here are repeated draws from one cohort and are therefore correlated. The "
 "resulting ranks are reported as a descriptive summary rather than as a formal "
 "multiple-comparison inference, and the pairwise signed-rank results with "
 "false-discovery-rate correction carry the statistical weight. The cross-chip "
 "folds, which are genuinely disjoint in hardware, are used for the comparison "
 "that most nearly satisfies the independence assumption.",
]

VOC_TABLE_HEADER = ["Volatile", "Healthy (ppb)", "Lung cancer", "COPD", "Asthma",
                    "Anchor"]
