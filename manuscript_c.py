# -*- coding: utf-8 -*-
"""Proposed methodology: problem formulation, mechanisms, objective, algorithms."""

NOTATION = [
 ["Symbol", "Meaning", "Domain"],
 ["$K,\;J,\;T$", "sensing spots, volatiles, samples per record", "$12,\;13,\;90$"],
 ["$P,\;H,\;r,\;C$", "latent sites, fast state width, projection rank, classes", "$40,\;48,\;3,\;4$"],
 ["$c_j(t)$", "concentration of volatile $j$ at the sensing surface", "ppb"],
 ["$\\theta_{k,j}(t)$", "fractional coverage of volatile $j$ on spot $k$", "$[0,1]$"],
 ["$k^{a}_{k,j},\;k^{d}_{k,j}$", "association and dissociation rate constants",
  "ppb$^{-n_j}$s$^{-1}$, s$^{-1}$"],
 ["$n_j$", "Langmuir-Freundlich heterogeneity exponent", "$(0,1]$"],
 ["$\\lambda^{\\mathrm{res}}_k$", "resonance wavelength of spot $k$", "nm"],
 ["$S^{\\mathrm{blk}}_k,\;S^{\\mathrm{srf}}_k$", "bulk and surface index sensitivity",
  "nm RIU$^{-1}$"],
 ["$\\ell_k$", "evanescent decay length in the sorbent overlayer", "nm"],
 ["$x_k(t)$", "baseline-corrected resonance shift of spot $k$", "nm"],
 ["$\\Psi,\;\\xi_m(t)$", "nuisance response directions and their amplitudes",
  "$\\mathbb{R}^{K\\times M}$"],
 ["$\\mathbf{U},\;\\mathbf{g}$", "learned nuisance basis and its gate",
  "$\\mathbb{R}^{K\\times r},\;(0,1)^{r}$"],
 ["$\\mathbf{z}_t,\;\\mathbf{a}_t$", "retained signal and nuisance coefficients",
  "$\\mathbb{R}^{K},\;\\mathbb{R}^{r}$"],
 ["$\\mathbf{u}_t$", "latent pseudo-concentrations driving the cell", "$\\mathbb{R}_{+}^{J'}$"],
 ["$\\boldsymbol{\\vartheta}_t$", "per-spot coverage state", "$[0,1]^{K\\times J'}$"],
 ["$\\mathbf{K}_{\\!s},\;\\mathbf{k}^{d}$", "site affinity and desorption rate",
  "$\\mathbb{R}_{+}^{P}$"],
 ["$s_{t,k}$", "occupied fraction of the binding sites of spot $k$", "$[0,1]$"],
 ["$\\hat{\\mathbf{z}}_t,\;\\mathbf{e}_t$", "kinetic reconstruction and its residual",
  "$\\mathbb{R}^{K}$"],
 ["$\\mathbf{h}_t$", "state of the velocity-modulated branch", "$\\mathbb{R}^{H}$"],
 ["$\\boldsymbol{\\phi}$", "pooled descriptor", "$\\mathbb{R}^{5KJ'+12K+2H}$"],
 ["$\\boldsymbol{\\alpha},\;S,\;\\hat{\\mathbf{p}},\;\\upsilon$",
  "Dirichlet parameters, strength, belief, uncertainty mass", "$\\mathbb{R}_{+}^{C}$"],
 ["$\\boldsymbol{\\nu}$", "logged operating conditions used by the drift heads",
  "$\\mathbb{R}^{2}$"],
]

EQ = {}   # filled by the builder; equation numbers are assigned in order


PROBLEM = [
 "A single measurement is a record $\\mathbf{X}\\in\\mathbb{R}^{T\\times K}$ whose "
 "entry $x_k(t)$ is the shift of the resonance of sensing spot $k$ relative to its "
 "pre-exhalation baseline, sampled every $\\Delta t=2$ s for $T=90$ samples across "
 "three exhalation cycles separated by ambient purges. The task is to assign the "
 "record to one of $C=4$ respiratory classes and, jointly, to report a belief that "
 "supports declining the decision. The difficulty is not that the mapping is "
 "nonlinear but that $x_k(t)$ is a sum of two processes that occupy the same time "
 "window, only one of which carries diagnostic content.",

 "Under the phase-matching condition of the attenuated-total-reflection geometry, "
 "the resonance of spot $k$ satisfies (1), in which $n_p$ is the prism index, "
 "$\\varepsilon_m$ the complex permittivity of the metal film and $\\varepsilon_d$ "
 "that of the adjacent dielectric [[17]], [[18]]. Reflectance of the full stack "
 "follows from the characteristic-matrix formulation for $p$-polarised light, with "
 "the per-layer matrix of (2) assembled into the stack matrix and converted to the "
 "amplitude coefficient of (3); the resonance wavelength is the minimiser of "
 "$R(\\lambda)$ at fixed incidence. Here $d_j$ and $\\varepsilon_j$ are the "
 "thickness and permittivity of layer $j$, and $q_j$ is its $p$-polarised "
 "admittance.",

 "Two derived quantities govern everything that follows. The evanescent field "
 "decays into the sensing overlayer with the length of (4), and a thin adsorbed "
 "film raises the effective index of that overlayer by the two-layer expression of "
 "(5), where $n_a$ is the index of the condensed adsorbate, $n_c$ that of the bare "
 "sorbent and $d_a$ the adlayer thickness [[20]]. Because $\\ell_k$ depends on the "
 "resonance wavelength while $S^{\\mathrm{blk}}_k$ and $S^{\\mathrm{srf}}_k$ depend "
 "on it differently, the vector of shifts produced across the array by a bulk index "
 "change is not parallel to the vector produced by surface binding. That "
 "non-collinearity is the property the read-out exploits.",

 "Uptake is competitive because the volatiles of a breath sample share the binding "
 "sites of one sorbent film. Coverage therefore obeys (6), a Langmuir-Freundlich "
 "law in which the term $1-\\sum_i\\theta_{k,i}$ couples every analyte on spot $k$ "
 "and $n_j$ absorbs the site-energy heterogeneity of a real film [[38]], [[40]]. At "
 "equilibrium (6) reduces to the competitive isotherm of (7), whose shared "
 "denominator is the mechanistic origin of cross-sensitivity: an interferent of "
 "large affinity suppresses a target without itself being reported. Water vapour, "
 "present at percent level against analytes at parts per billion, is carried "
 "explicitly as one of the $J$ competitors rather than as an external disturbance, "
 "because it occupies the same sites.",

 "Writing $\\Theta_k(t)=\\sum_j w_j\\theta_{k,j}(t)$ for the volume-weighted "
 "occupancy of spot $k$ and substituting (5) gives the observation model of (8). "
 "The first term is the diagnostic signal; the second is a superposition of "
 "$M$ nuisance processes, each entering along a fixed direction "
 "$\\boldsymbol{\\psi}_m\\in\\mathbb{R}^{K}$ with a time-varying amplitude "
 "$\\xi_m(t)$; the third is read-out noise. The directions are not free parameters "
 "but derivatives of the resonance position with respect to the physical quantity "
 "that varies, as in (9). Four such processes matter in practice: the thermo-optic "
 "and thermal-expansion response of the stack to the temperature of exhaled air, "
 "the densification of the sorbent film with elapsed months, the bulk index of the "
 "gas, and the offset and dispersive drift of the spectrograph wavelength scale.",

 "The consequence is stated in (10). Exhaled breath is roughly nine to eleven "
 "kelvin warmer than ambient and close to saturation, so $\\xi_{\\mathrm{th}}(t)$ "
 "rises and falls with the exhalation envelope and is therefore indistinguishable "
 "from the binding transient by any operation on the time axis alone. The signal "
 "and nuisance subspaces are, however, distinct in $\\mathbb{R}^{K}$: the "
 "component of the binding response that lies in the orthogonal complement of "
 "$\\mathrm{span}(\\Psi)$ is recoverable without knowing $\\xi_m(t)$, and the "
 "recoverable fraction is set by the principal angle between the two subspaces. "
 "For the array studied here that angle leaves the discriminative content largely "
 "intact while the nuisance is confined to a subspace of numerical rank three.",
]

MECH_A = [
 "The first mechanism removes $\\mathrm{span}(\\Psi)$ from the input without "
 "removing more than it must. A basis $\\mathbf{U}\\in\\mathbb{R}^{K\\times r}$ is "
 "initialised from the left singular vectors of the computed nuisance directions, "
 "orthonormalised at every forward pass, and used to extract the coefficients of "
 "(11). Removal is soft, as in (12): a learned gate "
 "$\\mathbf{g}=\\sigma(\\boldsymbol{\\gamma})$ decides how much of each component "
 "is subtracted, so a direction that carries some binding response is not "
 "discarded outright. A hard projection is the special case $\\mathbf{g}=\\mathbf{1}$ "
 "with $\\mathbf{U}$ frozen, and Section V-C shows that the soft form is the better "
 "of the two on the in-distribution partition.",

 "Two heads keep the factorisation honest. A co-operative head predicts the logged "
 "operating conditions $\\boldsymbol{\\nu}$ from the nuisance coefficients, and an "
 "adversarial head predicts the same quantities from the pooled descriptor through "
 "a gradient-reversal layer [[66]]. The first forces $\\mathbf{U}$ to explain the "
 "drift; the second forces the retained representation not to. Recording chip "
 "temperature and elapsed operating time is standard instrumentation practice, so "
 "the supervision this requires is available at training time without additional "
 "labelling. That an explicit inductive bias is necessary, rather than optional, "
 "follows from the impossibility of unsupervised factorisation without one [[67]]; "
 "here the bias is the electromagnetic model of the array itself.",
]

MECH_B = [
 "The second mechanism replaces the free gating of a long short-term memory cell "
 "with the integrator of the adsorption equation, and it does so while keeping "
 "the two roles that (6) assigns to different quantities separate. Concentration "
 "is common to every spot and is what couples them; coverage belongs to one spot "
 "and its chemistry. The cell follows that division. A shared encoder maps the "
 "projected array response to a nonnegative vector of $J'$ latent "
 "pseudo-concentrations by (13), which is the only place information from "
 "different spots is allowed to mix. Each spot then carries its own coverage of "
 "each latent component, so the state is never mixed across spots and remains a "
 "coverage variable rather than a general hidden vector.",

 "Rather than learning an association and a dissociation constant independently, "
 "the cell is parameterised by affinity and desorption rate as in (14). This is "
 "not cosmetic. The relaxation rate of the Langmuir equation is $k^{a}u+k^{d}$; "
 "if $k^{a}$ is free, a large drive drags a component that was initialised as "
 "slow into the fast regime and the long-memory branch of the decomposition "
 "disappears during the first epochs. Under (14) the relaxation rate is "
 "$k^{d}(K u+1)$, so the timescale ordering imposed at initialisation survives "
 "training, as Fig. 2(c) shows. Affinities are spread over four decades at "
 "initialisation for the same reason the Freundlich exponent of (6) is below "
 "unity: a real film presents a distribution of site energies, and a population "
 "with a single affinity saturates all at once and loses the amplitude "
 "resolution that separates the classes.",

 "Holding the drive constant across one sampling interval, the equation admits "
 "the closed-form update of (16) with the equilibrium point of (15), where the "
 "per-spot occupancy $s_{t,k}$ of (17) plays the role that "
 "$1-\\sum_i\\theta_{k,i}$ plays in (6). The recurrence is therefore gated, but "
 "its forget term is a physical relaxation factor and its input term an "
 "association flux. Coverage is confined to $[0,1]$ by construction and the state "
 "is directly interpretable, which is what distinguishes an architecturally "
 "encoded prior from a residual added to a loss [[68]], [[69]].",

 "A per-spot linear read-out reconstructs the drift-free trace from the coverage "
 "trajectory and defines the residual, both in (18); the form of that read-out is "
 "the thin-film limit of (5), in which the shift of a spot is affine in its "
 "occupancy. The reconstruction term is what the kinetic-consistency loss acts "
 "on: the branch is required to explain the retained signal with a coverage "
 "trajectory that a competitive Langmuir system could actually produce, and "
 "whatever it cannot explain is passed on rather than discarded.",
]


MECH_C = [
 "The third mechanism sets the rate of the fast branch from the state of the slow "
 "one. Its input is the concatenation of the kinetic residual and the first "
 "difference of the retained signal, so it sees only what the Langmuir model "
 "leaves over together with the instantaneous rate of change of the observation. "
 "Its update gate follows (19), in which the flattened binding velocity "
 "$\\Delta\\boldsymbol{\\vartheta}_t$ of every spot enters through a learned "
 "map $\\Gamma$. Reset gate, candidate and state update follow (20) and (21) in "
 "the usual form [[42]].",

 "The behaviour this produces is the intended one. While binding is in progress "
 "the velocity term is large, the gate opens and the fast branch tracks the "
 "transport and mixing transients that occur within an exhalation; once the "
 "surface approaches equilibrium the velocity falls, the gate closes and the "
 "branch holds its state instead of re-encoding a plateau. The coupling is "
 "therefore driven by the physical state of the adsorption process rather than by "
 "a clock period fixed in advance [[50]], [[51]], and the two branches are not two "
 "encoders whose outputs are concatenated but one decomposition in which the "
 "second stage is conditioned on the first.",
]

HEAD = [
 "Pooling follows (22). Three statistics of the coverage trajectory are retained "
 "in addition to its terminal and mean values: the maximum coverage, the "
 "difference between maximum and terminal coverage, which is the desorbed "
 "fraction and is the quantity that separates reversible physisorption from "
 "chemical capture, and the peak binding velocity. All five are kept per spot "
 "and per site rather than averaged, and the fast branch contributes its "
 "terminal and mean states. The concatenation is normalised across the batch "
 "rather than across features, because the descriptor is dominated by coverage "
 "amplitudes and a within-sample normalisation would remove precisely the "
 "amplitude scale that carries concentration information; Section V-B reports "
 "what that choice is worth.",

 "Classification uses a Dirichlet belief rather than a softmax. Evidence is "
 "produced by (23) and converted to a belief and a strength; the uncertainty mass "
 "of (24) is the share of the belief that no class claims, and it is the quantity "
 "on which selective prediction operates [[74]]. For a screening instrument this "
 "matters more than a marginal accuracy gain: a sample whose evidence is weak in "
 "every direction should be referred rather than assigned.",
]

OBJECTIVE = [
 "Four terms make up the criterion. Classification uses the type-II maximum "
 "likelihood form of the evidential objective, (25), weighted by inverse class "
 "frequency because the cohort is unbalanced; this form supplies a stronger "
 "gradient than the Bayes-risk alternative when total evidence is still small and "
 "was found to train more reliably from scratch. Over-confident evidence for "
 "incorrect classes is discouraged by the Kullback-Leibler term of (26), applied "
 "to the belief with the evidence of the true class removed and annealed over the "
 "first eight epochs [[74]].",

 "Kinetic consistency is imposed by (27), which is the only place a soft physical "
 "constraint appears: the architecture carries the adsorption dynamics, and the "
 "loss only asks that the coverage trajectory be sufficient to reproduce the "
 "retained trace. The drift term of (28) combines the co-operative and adversarial "
 "heads, with the reversal coefficient ramped from zero over training in the usual "
 "schedule. The complete objective is (29).",
]

COMPLEXITY = [
 "Both branches are linear in sequence length. The kinetic cell performs a "
 "$K\\times P$ projection and $\\mathcal{O}(P)$ elementwise operations per sample; "
 "the velocity-modulated branch performs $\\mathcal{O}(KH+H^{2}+PH)$ work per "
 "sample; the projection costs a $K\\times r$ product and a thin "
 "orthonormalisation. Total inference cost is therefore (30), which for the "
 "configuration used here is dominated by the $H^{2}$ term. Memory during "
 "training is $\\mathcal{O}(BT(P+H+K))$ for batch size $B$, and no attention "
 "matrix is formed, so the quadratic term in $T$ that a transformer incurs is "
 "absent. Measured parameter counts and single-record latencies are reported in "
 "Section V-E.",
]

ALG_TRAIN = [
 "Input: records $\\{\\mathbf{X}^{(i)}\\}$, labels $y^{(i)}$, logged conditions "
 "$\\boldsymbol{\\nu}^{(i)}$; computed nuisance directions $\\Psi$",
 "Output: parameters $\\Xi$ of projection, kinetic cell, fast branch and heads",
 "1: $\\mathbf{U}\\leftarrow$ first $r$ left singular vectors of $\\Psi$; "
 "$\\boldsymbol{\\gamma}\\leftarrow 2\\cdot\\mathbf{1}$",
 "2: for every spot, $\\log\\mathbf{k}^{d}\\leftarrow$ linspace over "
 "$[\\log(1/240),\\log(1/4)]$; $\\log\\mathbf{K}\\leftarrow$ linspace over "
 "$[-2,2]$",
 "3: for epoch $=1$ to $E$ do",
 "4:   $\;\;\\lambda_{\\mathrm{rev}}\\leftarrow 2/(1+e^{-6\\,\\mathrm{epoch}/E})-1$",
 "5:   for each minibatch $\\mathcal{B}$ do",
 "6:     $\;\;\\mathbf{z}_t,\\mathbf{a}_t\\leftarrow$ project $\\mathbf{x}_t$ by (11), (12)",
 "7:     $\;\;\\boldsymbol{\\vartheta}_t,\\Delta\\boldsymbol{\\vartheta}_t"
 "\\leftarrow$ integrate (13)-(17) for $t=1..T$",
 "8:     $\;\;\\hat{\\mathbf{z}}_t,\\mathbf{e}_t\\leftarrow$ (18)",
 "9:     $\;\;\\mathbf{h}_t\\leftarrow$ recur (19)-(21) for $t=1..T$",
 "10:    $\;\;\\boldsymbol{\\phi}\\leftarrow$ (22); "
 "$\\boldsymbol{\\alpha}\\leftarrow$ (23)",
 "11:    $\;\;\\mathcal{L}\\leftarrow$ (29) with reversal $\\lambda_{\\mathrm{rev}}$",
 "12:    $\;\;\\Xi\\leftarrow\\mathrm{AdamW}(\\Xi,\\nabla\\mathcal{L})$ after "
 "clipping $\\lVert\\nabla\\mathcal{L}\\rVert$ to 2",
 "13:  end for",
 "14:  evaluate balanced accuracy on the validation partition; retain the best "
 "state; stop after 15 epochs without improvement",
 "15: end for",
 "16: return $\\Xi$",
]

ALG_INFER = [
 "Input: record $\\mathbf{X}$, parameters $\\Xi$, abstention threshold $\\tau$",
 "Output: predicted class $\\hat{y}$ or abstention, uncertainty mass $\\upsilon$",
 "1: $\\mathbf{z}_t,\\mathbf{a}_t\\leftarrow$ (11), (12)",
 "2: $\\boldsymbol{\\vartheta}_0\\leftarrow\\mathbf{0}$; "
 "$\\mathbf{h}_0\\leftarrow\\mathbf{0}$",
 "3: for $t=1$ to $T$ do",
 "4:   $\;\;\\boldsymbol{\\vartheta}_t,\\Delta\\boldsymbol{\\vartheta}_t"
 "\\leftarrow$ (15)-(17)",
 "5:   $\;\;\\mathbf{e}_t\\leftarrow\\mathbf{z}_t-\\hat{\\mathbf{z}}_t$ by (18)",
 "6:   $\;\;\\mathbf{h}_t\\leftarrow$ (19)-(21)",
 "7: end for",
 "8: $\\boldsymbol{\\alpha}\\leftarrow$ (23); "
 "$\\hat{\\mathbf{p}}\\leftarrow\\boldsymbol{\\alpha}/S$; "
 "$\\upsilon\\leftarrow C/S$ by (24)",
 "9: if $\\upsilon>\\tau$ then return abstention",
 "10: return $\\hat{y}=\\arg\\max_c\\hat{p}_c$ and $\\upsilon$",
]
