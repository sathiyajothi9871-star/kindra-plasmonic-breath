# -*- coding: utf-8 -*-
"""Related work."""

RELATED = [
 ("A. Breath Volatiles as a Diagnostic Substrate", [
  "Evidence that exhaled volatiles carry information about lung disease is now "
  "quantitative rather than anecdotal. Pooled analyses of screening studies place "
  "lung-cancer sensitivity and specificity near 0.85 and pulmonary-infection "
  "sensitivity near 0.94, with areas under the curve of 0.93 and 0.96 "
  "respectively [[1]], [[2]]; a cross-sectional obstructive-disease cohort reaches "
  "0.92 for chronic obstructive pulmonary disease against healthy controls and "
  "0.81 for asthma [[3]]. What the pooled figures conceal is the heterogeneity "
  "beneath them. A compilation of absolute concentrations across roughly "
  "twenty-five primary studies reports the same compound in parts per billion, "
  "nanomoles per litre and nanograms per litre in different papers, and lists "
  "cyclohexane, ethyl acetate and 2-pentanone as lower in disease while aldehydes "
  "and aromatics are higher [[4]]. Calibrated gas-chromatographic measurements on "
  "a cancer cohort and matched controls give detection limits below one part per "
  "billion and linear ranges up to a few hundred [[5]], and a separate cohort "
  "stratified by smoking shows ethanol, isoprene and 1-propanol responding to "
  "tobacco use by margins that overlap the disease effect [[6]]. Saturated "
  "aldehydes rise several-fold in cancer patients relative to never-smoking and "
  "former-smoking controls [[7]], while a normative study of 504 healthy subjects "
  "measured on one instrument finds significant dependence on sex, age and "
  "smoking history for several of the same compounds [[8]].",

  "Standardisation has narrowed but not closed the gap between laboratories. The "
  "European Respiratory Society technical standard prescribes flow control, "
  "alveolar fractioning and ambient correction for exhaled biomarkers [[9]], and "
  "a survey of 113 room-air volatiles across hospital locations finds that "
  "although room air separates strongly by location the paired breath samples do "
  "not, which bounds the ambient confounder more tightly than had been assumed "
  "[[10]]. The washout requirement is less forgiving: breathprints recorded before "
  "and after a single cigarette separate with 89.6 per cent accuracy after one "
  "hour, so a fixed abstinence window is not sufficient in general [[11]]. The "
  "cumulative effect of this heterogeneity is visible in a meta-analysis that "
  "abandoned quantitative pooling altogether and recoded every study as presence "
  "or absence of each compound, recovering 93 per cent classification accuracy "
  "only after grouping volatiles by chemical function [[12]].",

  "Instrumented alternatives to mass spectrometry inherit the same variability. "
  "A systematic survey of electronic-nose studies across lung cancer, obstructive "
  "disease, interstitial disease and cystic fibrosis reports sensitivities from "
  "0.71 to 0.99 and specificities from 0.13 to 1.00, and declines to pool them "
  "because device and design heterogeneity make the estimates incommensurable "
  "[[13]]. Organ specificity is a further limitation: breath signatures separate "
  "lung cancer from cancer at other sites with only 0.68 sensitivity and 0.69 "
  "specificity [[14]]. Semiconducting-oxide arrays reach the concentration window "
  "the application requires and map cleanly onto the aromatic, aldehyde and "
  "ketone targets, but operate at elevated temperature and respond strongly to "
  "humidity [[15]], and a consolidation of nanobiosensor breath platforms places "
  "achievable detection limits between one and one hundred parts per billion "
  "while naming humidity interference, missing standardisation and thin clinical "
  "validation as the barriers to deployment [[16]].",
 ]),

 ("B. Plasmonic Transduction of Gas-Phase Analytes", [
  "The attenuated-total-reflection geometry that underlies most laboratory "
  "plasmonic sensors dates to a two-page note on the radiative decay of "
  "non-radiative surface plasmons [[17]], and the sensing formalism built on it "
  "defines the resonance condition, the evanescent penetration depth and the "
  "distinction between bulk and surface refractive-index sensitivity [[18]]. "
  "Localised resonances on nanostructures trade propagation length for a shorter "
  "decay length and therefore for greater surface selectivity [[19]]. Converting "
  "a measured shift into adsorbed material requires the two-layer effective-index "
  "approximation, which relates the shift to layer thickness through the "
  "evanescent decay length and is the identity that licenses treating a resonance "
  "trace as a proxy for surface coverage [[20]]. Quantitative modelling of such a "
  "stack additionally requires dispersion data for the metal and the coupling "
  "prism, for which a Lorentz-Drude parameterisation of gold and a curated "
  "database of optical constants are the standard sources [[21]], [[22]].",

  "Applied to lung-cancer biomarkers, plasmonic platforms have been surveyed "
  "across resonance, localised-resonance and Raman modalities, with the review "
  "noting that reported detection limits are not standardised and that the "
  "translation from a gas-phase response to a clinical decision remains "
  "unresolved [[23]]. The dominant engineering strategy is a sorbent overlayer "
  "that preconcentrates analyte inside the near field: a framework film on silver "
  "nanoparticles raises the shift by roughly an order of magnitude over the bare "
  "sensor [[24]], and a cross-reactive microarray of peptide and organic "
  "receptors read by resonance imaging discriminates volatiles differing by a "
  "single carbon and resolves simple mixtures [[25]]. Nanohole arrays offer "
  "collinear transmission read-out compatible with lens-free and complementary "
  "metal-oxide-semiconductor integration, at the cost of a fabrication tolerance "
  "that sets the resonance position directly and therefore makes chip-to-chip "
  "variability a systematic calibration burden [[26]].",

  "Sensitivity remains the binding constraint. Coupling surface plasmon "
  "polaritons to nanometre-thick framework films reaches parts-per-million "
  "detection and quantifies the trade of sensitivity against response time "
  "through overlayer thickness [[27]]; a like-for-like comparison of "
  "grating-coupled resonance against interferometry on a shared framework film "
  "reports shifts of 32 and 63 nanometres with a detection limit near four "
  "thousand parts per million and concedes poor selectivity across volatile "
  "classes [[28]]. Printed subpixel arrays read by hyperspectral imaging "
  "discriminate four volatiles and map their spatial distribution, but the "
  "authors state that at comparable concentrations two gases perturb the same "
  "resonance jointly and cannot be separated [[29]]. Framework-coated gold "
  "nano-urchins reach detection limits of 12.7, 14.5 and 36.3 parts per million "
  "for three volatiles and show a response rising linearly with relative humidity "
  "above 20 per cent together with residual signal after purging [[30]]. A "
  "benchmarking parameter proposed to make such figures comparable is itself "
  "derived from inert-gas bulk-index exchange, which characterises the very "
  "channel that confounds a breath measurement [[31]].",

  "Raman-active plasmonic substrates take a different route to specificity. "
  "Nanoimprinted semiopen bimetallic nanotube arrays combine confinement with "
  "multimodal enhancement and capture aldehydes covalently through Schiff-base "
  "chemistry, reaching a detection limit of ten parts per billion for "
  "benzaldehyde with 4.9 per cent relative standard deviation over fifty days "
  "[[32]]; the same chemistry is not readily reversible, which limits reuse across "
  "successive breath samples. A dedicated review of Raman breath analysis across "
  "lung cancer, gastric cancer and diabetes identifies background signal and "
  "false positives in clinical testing as the outstanding problems and describes "
  "portable real-time operation as aspirational [[33]]. Thermal cross-sensitivity "
  "is treated separately: dual-channel, grating and photonic-crystal-fibre "
  "schemes decouple temperature from analyte response, but require reference "
  "hardware that compact analysers omit [[34]]. That ppb-level detection is "
  "achievable in breath-relevant humidity has been shown for hydrogen sulfide on "
  "gold-nanoparticle-functionalised nanotube arrays, although through "
  "chemiresistive rather than optical transduction [[35]].",
 ]),

 ("C. Adsorption Kinetics as an Information Channel", [
  "Fitting a first-order association and dissociation model to a real-time "
  "resonance trace is the founding technique of quantitative plasmonic sensing "
  "[[36]], and the statistically correct estimator for the rate constants is "
  "global nonlinear least squares over both phases rather than a linearised "
  "transform [[37]]. Both treatments assume a single homogeneous and fully "
  "reversible site, which gas-phase physisorption on a functionalised film "
  "violates; the multi-component isotherm literature supplies the competitive and "
  "heterogeneity-corrected forms, in which a shared denominator couples the "
  "uptake of every analyte and an interferent with large affinity suppresses a "
  "target without itself being detected [[38]].",

  "The diagnostic value of the transient, as opposed to the endpoint, has been "
  "demonstrated directly. Fitting a surface-reaction kinetic model to transient "
  "response curves recovers per-analyte kinetic parameters that discriminate four "
  "volatiles from a single chemiresistive element, which is the clearest evidence "
  "that kinetics carry analyte identity and not merely concentration [[39]]. "
  "Reading only the first ten seconds of an adsorptive response and forecasting "
  "the steady state enables breath-by-breath operation, and the same study finds "
  "a stretched-exponential form empirically superior to the monoexponential "
  "Langmuir prediction, which is a direct statement that real sorbent transients "
  "are heterogeneous [[40]]. Neither line of work has been carried into a learned "
  "sequence model for a plasmonic array.",
 ]),

 ("D. Learned Read-Out and Sequence Models for Sensor Signals", [
  "The gated recurrent architectures on which sensor sequence models are built "
  "are long-established. A constant-error carousel with multiplicative gates "
  "makes gradient flow tractable over hundreds of steps [[41]]; a reset-and-update "
  "formulation removes the separate cell state at roughly a quarter fewer "
  "parameters [[42]]; and the standard head-to-head comparison finds the two "
  "comparable on stationary, densely sampled benchmarks [[43]]. Self-attention "
  "replaces recurrence with a constant path length between any two positions but "
  "assumes a regular grid through fixed positional encodings and requires more "
  "data than a clinical cohort typically provides [[44]]. For classification of "
  "multivariate series specifically, a two-branch convolutional and recurrent "
  "design, an ensemble of multi-scale inception modules and a random "
  "convolutional-kernel transform followed by a linear model define the "
  "competitive baselines [[45,46,47]], with dilated causal convolutions and masked "
  "pretraining of a transformer completing the set [[48]], [[49]].",

  "None of these carries a mechanism for two coexisting timescales. The "
  "architectures that do carry one fix it in advance. Partitioning a hidden layer "
  "into modules with exponentially spaced clock periods is the cleanest such "
  "precedent, but the periods are hyperparameters chosen before the data are seen "
  "[[50]]; a learned oscillatory time gate handles long quiet intervals and "
  "irregular sampling at low update cost, yet is periodic by construction and "
  "therefore mismatched to a monotonic thermal ramp [[51]]; and a continuous-time "
  "latent formulation removes the grid entirely but defines autonomous stationary "
  "dynamics, so a drift term must be absorbed into the trajectory rather than "
  "factored out [[52]]. Where hybrids of the two gated cells have been reported "
  "for applied classification they are stacked or concatenated, with the second "
  "cell consuming the output of the first at the same update rate [[53]], [[54]].",

  "Applied to sensors, learned read-out has so far improved the observation step "
  "rather than the temporal model. Feeding a whole plasmonic spectrum rather than "
  "a peak-shift descriptor to a network recovers information lost at low "
  "concentration and holds a hydrogen sensor stable for 142 hours in humid air "
  "[[56]]; a convolutional network classifies gas-phase acetone from camera images "
  "of a framework-coated nanopillar array at 95 to 98 per cent accuracy, but "
  "operates on static frames and never models the adsorption transient [[55]]; "
  "and a network trained on angular scans improves refractive-index resolution "
  "over centroid peak finding while remaining a per-frame regressor with no "
  "temporal component at all [[57]]. Deliberate modulation of the excitation turns "
  "a single sensor's transient into a discriminative series and reaches "
  "97 per cent classification at 0.53 milliwatts, although the modulation is fast "
  "and periodic and the analysis window too short to encounter inter-session "
  "drift [[58]].",

  "Two studies come closest to the position taken here. Coupling a physical "
  "surface-state model of the sensing interface to a gated recurrent regressor "
  "constrains the mapping from response curve to concentration by physics rather "
  "than by data alone, but the surface-state model is specific to metal-oxide "
  "chemisorption, the recurrence is single-scale, and drift is treated as noise "
  "to be regressed away [[59]]. Reading the early dynamic portion of a biosensor "
  "response with a theory-guided network recovers the endpoint faster and more "
  "accurately than waiting for equilibrium, which is the core argument for "
  "transient-based read-out, yet the validation is single-analyte and under "
  "controlled conditions [[60]]. At the application level, a twenty-element hybrid "
  "array with a convolutional classifier reaches 97.8 per cent accuracy on 181 "
  "breath samples, and a hierarchical two-stage convolutional network separates "
  "healthy, lung-cancer and gastric-cancer subjects at areas under the curve of "
  "0.89 to 0.92 on 206 subjects; both treat the array response as a static "
  "feature map and neither is externally validated [[61]], [[62]].",
 ]),

 ("E. Drift Compensation, Physical Priors and Calibrated Confidence", [
  "The reference benchmark for long-term chemosensor drift distributes 13 910 "
  "measurements from a sixteen-element array recorded over thirty-six months, "
  "together with a weighted ensemble baseline, and establishes both the magnitude "
  "of the problem and the convention of reporting per-batch transfer [[63]]. "
  "Recent treatments are adversarial: aligning drifted target batches to a source "
  "domain while rejecting unseen gas classes, and smoothing a conditional domain "
  "adversarial objective, both improve cross-batch accuracy [[64]], [[65]]. Each "
  "inherits the gradient-reversal construction that learns representations "
  "predictive of the label yet indiscriminate between domains [[66]], and each "
  "operates on the same aggregated steady-state descriptors as the benchmark "
  "itself, so the alignment happens in a static feature space and gives no "
  "guidance on how a recurrent model should adapt when the binding dynamics "
  "change. That unsupervised factorisation cannot be relied upon without an "
  "explicit inductive bias has been established at scale, which is the argument "
  "for supplying the nuisance directions from the optics rather than hoping a "
  "network discovers them [[67]].",

  "Physical priors enter learned models either through the loss or through the "
  "architecture. The canonical soft formulation adds a differential-equation "
  "residual as a penalty and identifies parameters by the same mechanism [[68]]; "
  "the distinction between observational, architectural and loss-encoded bias is "
  "drawn explicitly in the subsequent review, which argues that architectural "
  "encoding is the stronger constraint [[69]]. The nearest published application "
  "to plasmonic sensing informs a convolutional network with a synthesised "
  "complex-frequency-wave transform and more than halves the mean relative error "
  "of protein-dynamics readout, but the prior is spectral and static rather than "
  "kinetic and temporal, and no uncertainty calibration is reported [[70]].",

  "Confidence is the last requirement a screening instrument must meet. Modern "
  "networks are systematically over-confident, and the reliability-diagram and "
  "expected-calibration-error protocol together with single-parameter temperature "
  "scaling is the standard remedy, although it is post-hoc and does not survive "
  "distribution shift [[71]]. Independently initialised ensembles remain the "
  "strongest general estimator and are notably robust under shift, at a cost "
  "linear in ensemble size [[72]]; test-time dropout gives epistemic uncertainty "
  "from a single trained network but underestimates it far from the training "
  "distribution [[73]]. Replacing the softmax with a Dirichlet belief yields class "
  "evidence and an explicit uncertainty mass in one deterministic pass, which "
  "supports abstention rather than a forced decision [[74]], and the regression "
  "analogue separates aleatoric from epistemic components in the same way [[75]]. "
  "The binned calibration-error estimator itself originates in Bayesian binning "
  "[[76]], and the proper scoring rule used alongside it is older still [[77]]. "
  "For comparing models the recommended protocol is non-parametric — signed-rank "
  "testing between pairs and a Friedman test with post-hoc analysis across many "
  "— [[78]], with false-discovery-rate control preferred to Bonferroni correction "
  "when many pairwise comparisons are made [[79]].",
 ]),
]

GAP = (
 "Three observations follow from the literature above. Plasmonic transduction "
 "supplies a resonance trace whose transient shape carries analyte identity, and "
 "that shape is currently discarded by every learned read-out reported for such "
 "sensors. The nuisance that contaminates the trace is not additive noise but a "
 "structured excursion driven by temperature, water uptake, film ageing and "
 "read-out drift, and because exhaled air is warm and saturated the excursion is "
 "synchronous with the signal, so no temporal filter can separate them. The "
 "array, however, responds to that nuisance along directions that are fixed by "
 "the electromagnetic model of the stack and are not collinear with the "
 "directions produced by surface binding, which makes the separation identifiable "
 "in the space of the sensing spots even where it is not identifiable in time. "
 "Existing sequence models neither exploit that geometry nor represent the two "
 "adsorption timescales that a functionalised array necessarily contains, because "
 "a single gated cell carries one update rate and the architectures that carry "
 "more fix them before any measurement is made. The research gap addressed here "
 "is therefore the absence of a read-out that separates the kinetic component of "
 "a plasmonic sensorgram from its transduction component using the known response "
 "geometry of the array, and that models the fast and slow components of the "
 "kinetic part at their own rates rather than at a shared one.")

COMPARISON_TABLE = [
 ["Reference", "Problem addressed", "Method", "Data", "Principal limitation for the present task"],
 ["[[25]]", "VOC discrimination", "SPR imaging of a cross-reactive receptor array",
  "Laboratory volatiles", "Pattern recognition over static responses; no transient model, no drift treatment"],
 ["[[55]]", "Gas-phase acetone", "Convolutional network on camera images of an LSPR array",
  "0.5-80 umol/mol acetone", "Per-frame classification; adsorption dynamics unused"],
 ["[[56]]", "Hydrogen at 100 ppm in humid air", "Deep network on the full plasmonic spectrum",
  "142 h humid-air series", "Single analyte, purely data-driven, no calibration analysis"],
 ["[[57]]", "Refractive-index resolution", "Deep network on SPR angular scans",
  "Simulated and measured scans", "Static per-frame regression; no sequence-level model"],
 ["[[59]]", "Concentration regression", "Surface-state physical model coupled to a GRU",
  "Metal-oxide sensor curves", "Prior specific to chemisorption; single timescale; drift treated as noise"],
 ["[[60]]", "Early-response biosensing", "Theory-guided network on partial response",
  "Single-analyte binding assays", "No multi-analyte competition; assumes calibrated kinetics"],
 ["[[61]], [[62]]", "Breath-based cancer screening", "Convolutional and hierarchical convolutional networks",
  "181 and 206 subjects", "Static feature maps; single-site; no external validation"],
 ["[[64]], [[65]]", "Chemosensor drift", "Adversarial domain alignment",
  "36-month drift benchmark", "Aggregated descriptors only; no temporal adaptation"],
 ["[[70]]", "Protein dynamics from a plasmonic sensor", "Physics-informed convolutional network",
  "Mid-infrared plasmonic spectra", "Spectral and static prior; no kinetic prior, no uncertainty"],
 ["This work", "Lung disease from a plasmonic breath array",
  "Kinetics-informed dual-rate recurrent network with dispersion disentangling",
  "2400 subjects in silico; real 36-month drift benchmark for the drift component",
  "Cohort is simulated; clinical validation outstanding"],
]
