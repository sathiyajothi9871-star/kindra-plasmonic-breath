# -*- coding: utf-8 -*-
"""Front matter, introduction and related work."""

TITLE = ("Lung Disease Diagnosis Using Plasmonic Sensors Integrated with a "
         "Hybrid LSTM–GRU Architecture")

AUTHORS = "S. Subaranjani"
AFFIL = ("Department of Computer Science and Engineering, [Institution], [City], "
         "[Country]  |  Corresponding author: stroresearchadvisor@gmail.com  |  "
         "ORCID: [0000-0000-0000-0000]")

INDEX_TERMS = ("Adsorption kinetics, breath analysis, evidential deep learning, "
               "gated recurrent networks, lung disease diagnosis, sensor drift, "
               "surface plasmon resonance, volatile organic compounds.")


def introduction(R):
    p = []
    p.append(
        "Lung disease is diagnosed late far more often than it is diagnosed early, "
        "and the cost of that delay is measured in survival. Exhaled breath offers "
        "a sampling route that is painless, repeatable and cheap enough to be used "
        "for surveillance rather than confirmation, and pooled evidence across "
        "twenty-five studies and more than four thousand participants places the "
        "discriminative ceiling of breath volatiles for lung cancer at a sensitivity "
        "of 0.85 and a specificity of 0.86 [[1]]. Comparable pooling for pulmonary "
        "infection reports a sensitivity of 0.94 and an area under the receiver "
        "operating characteristic curve of 0.96 [[2]], and a recent cross-sectional "
        "cohort of 367 participants separates chronic obstructive pulmonary disease "
        "from healthy controls with an area under the curve of 0.92 while reaching "
        "only 0.81 for asthma against the same controls [[3]]. Breath therefore "
        "carries genuine diagnostic information, and the obstacle to using it is "
        "not the biology but the instrument.")
    p.append(
        "What makes the measurement hard is the combination of concentration and "
        "chemistry. The volatiles that carry disease information sit between "
        "fractions of a part per billion and a few hundred parts per billion, they "
        "are outnumbered by water vapour at percent level, and several of them move "
        "in opposite directions between conditions rather than rising uniformly with "
        "disease [[4]]. Absolute concentrations reported for saturated aldehydes, "
        "short-chain alcohols and aromatics differ between cohorts by more than an "
        "order of magnitude [[5]], [[6]], smoking status shifts the same compounds by "
        "amounts comparable with the disease effect [[6]], [[11]], and healthy baselines "
        "themselves vary with sex and age across five hundred subjects measured on a "
        "single instrument [[8]]. Sampling protocol, ambient background and washout "
        "period each contribute further variance that standardisation efforts have "
        "reduced but not removed [[9]], [[10]], [[11]]. A meta-analysis of the breath "
        "literature was forced to discard all quantitative information and recode "
        "every study as presence or absence, because the reported concentrations "
        "could not be pooled at all [[12]].")
    p.append(
        "Sensor arrays were introduced to sidestep the cost and latency of mass "
        "spectrometry, and they have reached the clinic in prototype form. Reviews "
        "of electronic-nose studies across lung cancer, obstructive disease and "
        "interstitial disease report sensitivities from 0.71 to 0.99 against "
        "specificities from 0.13 to 1.00, a spread wide enough that no pooled "
        "estimate is offered [[13]]; breath signatures also fail to separate lung "
        "cancer from cancer at other sites better than 0.68 sensitivity [[14]]. "
        "Semiconducting oxide arrays reach the required detection limits for "
        "aromatics and aldehydes but operate hot and respond strongly to humidity "
        "[[15]], [[16]]. Optical transduction avoids the heater and the baseline "
        "resistance, and surface plasmon resonance in particular converts a "
        "refractive-index change at a functionalised metal film into a resonance "
        "shift with a well-understood electromagnetic model [[17]], [[18]], [[19]].")
    p.append(
        "Plasmonic arrays for gas-phase sensing have advanced quickly. Sorbent "
        "overlayers concentrate analyte inside the evanescent field and raise the "
        "response by an order of magnitude [[23]], [[24]], [[27]]; cross-reactive receptor "
        "microarrays read by plasmon-resonance imaging discriminate volatiles that "
        "differ by a single carbon [[25]]; nanohole, nanoparticle and nano-urchin "
        "geometries have pushed fabrication towards printable, lens-free formats "
        "[[26]], [[29]], [[30]]. Detection limits, however, remain the binding constraint: "
        "parts-per-million rather than parts-per-billion for most sorbent-coated "
        "plasmonic transducers [[28]], [[30]], and the same near-field confinement that "
        "produces the sensitivity also produces cross-response, so two volatiles at "
        "similar concentration perturb one resonance jointly and cannot be separated "
        "from the resonance position alone [[29]], [[31]].")
    p.append(
        "A second difficulty is more fundamental and receives less attention. The "
        "quantity that makes a plasmonic sensor sensitive is its refractive-index "
        "sensitivity, and that same quantity makes it sensitive to everything else "
        "that changes the index: the temperature of the stack, the water content of "
        "the sorbent, the slow densification of the film and the wavelength scale of "
        "the spectrograph. Dual-channel and grating schemes exist to compensate "
        "thermal drift, but they require reference hardware that compact analysers "
        "usually omit, and the compensation degrades as the film ages [[34]]. "
        "Published plasmonic gas sensors record humidity-proportional baselines and "
        "incomplete recovery after purging [[30]], [[35]], and a benchmarking study "
        "notes that the figure of merit conventionally reported is derived from bulk "
        "index exchange, which is precisely the channel that confounds a breath "
        "measurement [[31]]. Over a multi-month deployment the drift accumulated by a "
        "chemical sensor array is large enough that classification collapses without "
        "explicit compensation [[63]].")
    p.append(
        "Learned read-out has been applied to both problems, with real but partial "
        "success. Feeding a whole plasmonic spectrum to a network rather than a "
        "single peak-shift descriptor recovers information lost at low concentration "
        "and holds a hydrogen sensor stable for 142 hours in humid air [[56]]; a "
        "convolutional network classifies acetone from camera images of a "
        "framework-coated nanopillar array at 95 to 98 per cent accuracy [[55]]; a "
        "network trained on angular scans improves refractive-index resolution over "
        "conventional curve fitting [[57]]. All three operate frame by frame. They "
        "improve the observation step and leave the temporal structure of the "
        "measurement untouched, even though the transient shape of an adsorption "
        "curve is known to identify an analyte that its endpoint cannot [[39]], [[40]].")
    p.append(
        "Sequence models are the obvious remedy, and the sensor literature has "
        "adopted them. Gated recurrent architectures, convolutional-recurrent "
        "hybrids and stacked long short-term memory followed by a gated recurrent "
        "unit have all been reported for chemical and biomedical time series [[53]], "
        "[[54]], [[61]]. Their weakness in this setting is structural rather than "
        "empirical. A single gated cell carries one update rate, so the same "
        "parameters must represent an adsorption transient that completes inside one "
        "exhalation and a baseline excursion that develops over the whole record; "
        "architectures that do fix multiple rates fix them in advance, either as "
        "exponentially spaced clock periods or as a periodic time gate, neither of "
        "which matches a monotonic and aperiodic instrument drift [[41]], [[50]], [[51]]. "
        "Continuous-time formulations remove the need for a regular grid but absorb "
        "drift into the latent trajectory rather than factoring it out [[52]]. Domain "
        "adaptation attacks the drift directly, and adversarial batch alignment "
        "improves cross-batch transfer on real chemosensor data, yet it operates on "
        "aggregated descriptors in a static feature space and offers no account of "
        "how the binding dynamics themselves change as a surface ages [[64]], [[65]].")
    p.append(
        "The gap that remains is specific. A plasmonic breath record is the sum of "
        "two things that live in the same time window: an analyte-specific "
        "adsorption transient governed by competitive Langmuir kinetics, and an "
        "analyte-independent excursion driven by the thermal, hygroscopic, ageing "
        "and read-out behaviour of the transducer. Exhaled air is warm and saturated, "
        "so the nuisance excursion is synchronous with the signal and cannot be "
        "removed by any filter that works in time alone. It can, however, be "
        "separated in the space of the array, because the direction along which the "
        "twelve resonances move under a temperature change is fixed by the optics "
        "and is not the direction along which they move under surface binding. No "
        "published model exploits that separation, and no published recurrent model "
        "for chemical sensors runs its branches at the two rates the physics "
        "actually contains.")
    p.append(
        "The architecture developed here, referred to as KINDRA, addresses that gap "
        "with three changes to the recurrent read-out rather than with a deeper "
        "stack. A low-rank subspace initialised from the electromagnetic model of "
        "the array is softly removed from the input and held responsible for the "
        "nuisance by a co-operative head, while the retained representation is "
        "pushed away from it by a gradient-reversal head. The slow branch is a long "
        "short-term memory cell whose state is a fractional-coverage vector advanced "
        "by an exponential integrator of the Langmuir equation, so its forget term "
        "is the physical relaxation factor rather than a free gate. The fast branch "
        "is a gated recurrent unit that consumes only what the kinetic branch cannot "
        "explain and whose update gate is modulated by the instantaneous binding "
        "velocity of the slow branch, which is what couples the two rates instead of "
        "concatenating two encoders. A Dirichlet head completes the model so that "
        "an ambiguous sample can be declined rather than guessed.")
    p.append(
        "Evaluation uses a physically grounded in-silico cohort of 2400 subjects and "
        "4800 recording sessions, built from a transfer-matrix model of the "
        "Kretschmann stack, a competitive Langmuir–Freundlich adsorption model of "
        "twelve functionalised spots, and concentration priors anchored to published "
        "breath measurements. Because no public dataset of raw plasmonic breath "
        "sensorgrams exists, the component that does not require a sequence is "
        "additionally validated on a real thirty-six-month chemosensor drift "
        "benchmark [[63]]. The limits this imposes on the conclusions are stated "
        "plainly in Section VII rather than left implicit.")
    p.append(
        "One finding is stated here because it shapes how the rest should be "
        "read. Measuring the information budget of the measurement chain shows "
        "that the transduction nuisance is expensive and that removing the "
        "computed subspace returns most of that cost to any read-out placed "
        "after it, engineered or learned. A logistic classifier on engineered "
        "descriptors of the projected trace turns out to be the strongest "
        "read-out tested, ahead of every sequence architecture including the one "
        "proposed here. The physical prior therefore matters more than the "
        "recurrence on this cohort, and the architecture earns its place through "
        "calibration, robustness to corruption and an abstention signal that a "
        "linear classifier does not supply. Presenting that comparison rather "
        "than omitting it is deliberate.")
    return p


OBJECTIVES = [
 "O1. Quantify how much of the achievable diagnostic information a plasmonic "
 "breath array loses to transduction nuisance, by comparing classification from "
 "the observed sensorgram against classification from the nuisance-free "
 "sensorgram and from the true alveolar concentrations.",
 "O2. Determine whether a nuisance subspace derived from the electromagnetic "
 "model of the array, rather than estimated from data alone, recovers a "
 "measurable part of that loss.",
 "O3. Establish whether replacing the free gate of a recurrent cell with an "
 "exponential integrator of the adsorption equation improves discrimination "
 "relative to conventional and hybrid gated architectures at matched capacity.",
 "O4. Assess whether coupling a fast gated branch to the binding velocity of the "
 "slow branch improves robustness to injected drift, spot failure and gain "
 "mismatch, and whether the resulting confidence estimates support selective "
 "prediction.",
]

CONTRIBUTIONS = [
 "A dispersion-disentangling projection whose basis is computed, not estimated: "
 "the thermal, ageing, bulk-index, read-out and fabrication response directions "
 "of the array are obtained by differentiating the transfer-matrix model of the "
 "stack, and the resulting low-rank subspace is softly removed under a learned "
 "gate with paired co-operative and adversarial heads. Prior drift compensation "
 "for chemical sensors aligns feature distributions between batches; the "
 "subspace used here is fixed by the optics before any measurement is made. Its "
 "value is established three times over: as an information-budget experiment in "
 "Section V-A, as an ablation in Section V-B, and on a real thirty-six-month "
 "chemosensor drift benchmark in Section V-D.",

 "A Langmuir kinetic state cell in which the recurrent state is a "
 "fractional-coverage vector held at the granularity the adsorption equation is "
 "written for, one population of latent sites per sensing spot. The input term "
 "is an association flux and the forget term is the physical relaxation factor. "
 "Parameterising the cell by affinity and desorption rate rather than by two "
 "independent constants keeps the relaxation rate proportional to the desorption "
 "rate, so a site initialised as slow remains slow whatever the drive amplitude, "
 "and spreading the affinities over four decades keeps the population responsive "
 "across the three orders of magnitude of concentration a breath sample spans. "
 "This is an architectural constraint, not a penalty added to a loss, which is "
 "what distinguishes it from physics-informed formulations that enforce a "
 "residual softly [[68]], [[69]], [[70]]. Its contribution is measured in "
 "Section V-B.",

 "A dual-rate coupling in which the fast gated branch receives the residual left "
 "by the kinetic branch and has its update gate modulated by the binding "
 "velocity of that branch, so the two-timescale decomposition is driven by the "
 "state of the adsorption process rather than by clock periods fixed in advance "
 "[[50]], [[51]]. The effect is quantified in the ablation and in the robustness "
 "sweep of Section V-C.",

 "A complete and deliberately unflattering evaluation protocol. Fifteen "
 "reference models are compared on five subject-level partitions with "
 "confidence intervals, non-parametric tests and false-discovery-rate control; "
 "the information budget of the measurement chain is measured so that every "
 "improvement can be read against what is achievable; and the classical "
 "read-outs are given the same projection the proposed model uses, which is what "
 "makes the finding of Section V-A - that the physical prior contributes more "
 "than the architecture - visible rather than hidden. Calibration, selective "
 "prediction, corruption robustness, cross-chip transfer and computational cost "
 "are reported alongside discrimination.",
]
