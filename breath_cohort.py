"""
Construction of the plasmonic breath cohort.

Concentration priors are anchored to published exhaled-breath measurements.
Geometric means for the healthy and lung-cancer groups are taken from the
GC-MS/SPME breath literature that reports absolute ppb values; the obstructive
groups are built as multiplicative perturbations of the healthy prior, because
absolute-concentration ground truth for COPD and asthma is not available in the
published record and only directional or relative changes are reported.  That
asymmetry is a property of the field, not of this study, and is stated as such
in the manuscript.

The generator produces, for every session, a (T, K) matrix of resonance-shift
traces together with the latent quantities needed for the physics-informed
losses and for the drift analyses.
"""
from __future__ import annotations
import numpy as np
import config as C
import optics
import kinetics as KIN

# ----------------------------------------------------------------------
# Concentration priors, parts per billion, geometric means
# ----------------------------------------------------------------------
HEALTHY_GM = {
    "Acetone": 278.0, "Isoprene": 66.0, "Ethanol": 256.0, "1-Propanol": 11.8,
    "Toluene": 35.0, "Benzene": 2.50, "Ethylbenzene": 10.4, "Hexanal": 0.35,
    "Pentanal": 0.40, "2-Pentanone": 5.05, "2-Butanone": 2.10, "Butane": 25.8,
}
# multiplicative deviation from the healthy geometric mean
CLASS_RATIO = {
    "Healthy":    {k: 1.0 for k in HEALTHY_GM},
    "LungCancer": {"Acetone": 1.05, "Isoprene": 1.30, "Ethanol": 1.28,
                   "1-Propanol": 4.64, "Toluene": 2.57, "Benzene": 2.80,
                   "Ethylbenzene": 1.88, "Hexanal": 12.9, "Pentanal": 14.8,
                   "2-Pentanone": 1.49, "2-Butanone": 1.10, "Butane": 2.83},
    "COPD":       {"Acetone": 1.42, "Isoprene": 0.70, "Ethanol": 1.05,
                   "1-Propanol": 1.30, "Toluene": 1.10, "Benzene": 1.75,
                   "Ethylbenzene": 1.20, "Hexanal": 5.20, "Pentanal": 6.20,
                   "2-Pentanone": 0.78, "2-Butanone": 1.35, "Butane": 2.60},
    "Asthma":     {"Acetone": 0.82, "Isoprene": 1.42, "Ethanol": 1.00,
                   "1-Propanol": 1.08, "Toluene": 0.90, "Benzene": 1.05,
                   "Ethylbenzene": 1.02, "Hexanal": 4.10, "Pentanal": 4.60,
                   "2-Pentanone": 1.95, "2-Butanone": 1.10, "Butane": 2.05},
}
# geometric standard deviation of the between-subject distribution
GSD = {"Acetone": 1.95, "Isoprene": 1.72, "Ethanol": 2.40, "1-Propanol": 2.30,
       "Toluene": 2.20, "Benzene": 2.05, "Ethylbenzene": 2.10, "Hexanal": 1.98,
       "Pentanal": 2.02, "2-Pentanone": 1.80, "2-Butanone": 1.90, "Butane": 2.05}

AMBIENT_FRACTION = 0.12     # room-air background as a fraction of the healthy prior
WITHIN_SUBJECT_GSD = 1.18   # session-to-session repeatability of a given subject
BREATH_TO_BREATH_CV = 0.08

SAT_KPA_25C = 3.17
SAT_KPA_34C = 5.32
P_ATM_KPA = 101.325


DT = 1.0 / C.FS_HZ


def _timeline():
    """Exposure indicator sampled at the spectrometer read-out rate."""
    T = C.T_STEPS
    expo = np.zeros(T)
    t = int(round(C.BASELINE_S / DT))
    ne = int(round(C.EXPOSURE_S / DT))
    npg = int(round(C.PURGE_S / DT))
    for _ in range(C.N_CYCLES):
        expo[t:t + ne] = 1.0
        t += ne + npg
    return expo


EXPOSURE = _timeline()


def _lowpass(x, tau_s, dt=2.0):
    """First-order sampling-line transport lag."""
    a = np.exp(-dt / tau_s)
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = a * acc + (1.0 - a) * x[i]
        y[i] = acc
    return y


def concentration_trace(c_subject_ppb, c_ambient_ppb, rh_amb, rh_breath, rng):
    """
    (T, J) concentration trace in ppb, including water vapour.

    Water is carried as an explicit competing adsorbate rather than as an
    external nuisance, because uptake of water by the sorbent overlayer occupies
    the same binding sites as the volatiles of interest.
    """
    T = C.T_STEPS
    J = KIN.N_VOC
    conc = np.zeros((T, J), dtype=float)
    breath_gain = 1.0 + BREATH_TO_BREATH_CV * rng.standard_normal(T)
    for j in range(J - 1):
        raw = c_ambient_ppb[j] + (c_subject_ppb[j] - c_ambient_ppb[j]) * EXPOSURE
        raw = raw * np.clip(breath_gain, 0.6, 1.4)
        conc[:, j] = np.maximum(_lowpass(raw, 2.2, DT), 0.0)
    w_amb = rh_amb * SAT_KPA_25C / P_ATM_KPA * 1e9
    w_brt = rh_breath * SAT_KPA_34C / P_ATM_KPA * 1e9
    conc[:, KIN.WATER_INDEX] = _lowpass(w_amb + (w_brt - w_amb) * EXPOSURE, 2.2, DT)
    return conc


# One coating recipe is shared by the whole fabrication run, so every chip
# carries the same sorbent chemistry and differs only through geometric
# tolerance, spot-to-spot coating variability, thermal history and ageing.
RECIPE_SEED = 20250101


def coating_recipe():
    r = np.random.default_rng(RECIPE_SEED)
    sel = KIN.coating_selectivity(C.N_CHANNELS, r)
    logK = KIN.affinity_matrix(sel, r)
    n_exp = KIN.heterogeneity(r, KIN.N_VOC)
    tau_base = KIN.base_tau(C.N_CHANNELS, KIN.N_VOC, r)
    return sel, logK, n_exp, tau_base


_RECIPE = coating_recipe()


class Device:
    """One physical sensor chip: geometry, sensitivities and nuisance directions."""

    def __init__(self, device_id: int, rng: np.random.Generator):
        self.id = device_id
        d_au = C.AU_THICKNESS_NM + rng.normal(0.0, 1.6, size=C.N_CHANNELS)
        d_coat = np.array(C.COATING_THICKNESS_NM) * (
            1.0 + rng.normal(0.0, 0.055, size=C.N_CHANNELS))
        n_coat = np.array(C.COATING_INDEX) + rng.normal(0.0, 0.006, size=C.N_CHANNELS)
        self.array = optics.build_array(d_au, d_coat, n_coat, C.INCIDENCE_DEG,
                                        C.LAMBDA_SCAN_NM, C.LAMBDA_STEPS)
        self.s_bulk = self.array["s_bulk"]
        self.s_surf = self.array["s_surf"]
        self.l_d = self.array["l_d"]
        self.n_coat = n_coat
        self.d_coat = d_coat
        self.therm_dir = optics.thermal_direction(self.array)
        self.age_dir = optics.ageing_direction(self.array)
        self.spec_basis = optics.spectrograph_basis(self.array)
        self.fab_basis = optics.fabrication_directions(self.array)
        sel, logK, n_exp, tau_base = _RECIPE
        self.sel = sel
        # spot-to-spot variability of an otherwise identical coating recipe
        self.logK = logK + 0.045 * rng.standard_normal(logK.shape)
        self.logK[:, KIN.WATER_INDEX] = -6.62 + 0.05 * rng.standard_normal(C.N_CHANNELS)
        self.n_exp = np.clip(n_exp + 0.012 * rng.standard_normal(n_exp.shape), 0.55, 1.0)
        self.q_scale = np.clip(rng.normal(1.0, 0.06, C.N_CHANNELS), 0.75, 1.25)
        self.tau = tau_base * np.exp(rng.normal(0.0, 0.09, size=tau_base.shape))
        self.k_d = 1.0 / self.tau
        self.k_a = (10.0 ** self.logK) * self.k_d
        self.pore_fraction = np.clip(rng.normal(0.35, 0.03, C.N_CHANNELS), 0.2, 0.5)

    def drift_basis(self):
        """Physically derived nuisance directions, columns normalised."""
        M = np.concatenate([np.stack([self.therm_dir, self.age_dir, self.s_bulk],
                                     axis=1), self.spec_basis, self.fab_basis],
                           axis=1)
        M = M / np.linalg.norm(M, axis=0, keepdims=True)
        return M


def calibrate_offset(device: Device, target_median=0.012):
    """
    Set the affinity offset so that a healthy sample drives the array into the
    sub-saturation regime where the transient shape is informative.
    """
    c = np.array([HEALTHY_GM[n] for n in KIN.VOC_NAMES[:-1]])
    cn = np.power(c, device.n_exp[:-1])
    K = 10.0 ** device.logK[:, :-1]
    cur = np.median(K * cn[None, :])
    shift = np.log10(target_median / max(cur, 1e-12))
    device.logK[:, :-1] += shift
    device.k_a = (10.0 ** device.logK) * device.k_d
    return shift


def simulate_session(device: Device, c_subject, c_ambient, rh_amb, rh_breath,
                     temp_offset_K, months, rng, return_latent=False):
    """Generate one (T, K) resonance-shift record in nanometres."""
    conc = concentration_trace(c_subject, c_ambient, rh_amb, rh_breath, rng)
    q_eff = device.q_scale * max(1.0 - C.CAP_LOSS_PER_MONTH * months, 0.55)
    theta = KIN.integrate_coverage(conc, device.k_a, device.k_d, device.n_exp,
                                   q_eff, dt=DT, substeps=8)
    v = np.array([p[3] for p in KIN.VOC_PANEL])
    occ = (theta * v[None, None, :]).sum(axis=2)                    # (T, K)
    dn_coat = (C.N_ADLAYER - device.n_coat)[None, :] * device.pore_fraction[None, :] * occ
    ageing_gain = 1.0 + 0.004 * months
    surf = (device.s_surf * ageing_gain)[None, :] * dn_coat

    # transduction nuisances
    # Stack temperature: a static session offset, the exhalation transient, a slow
    # within-record ramp from the ambient environment and a short-term fluctuation.
    ramp = rng.normal(0.0, C.TEMP_RAMP_K) * np.linspace(0.0, 1.0, C.T_STEPS)
    a_T = np.exp(-DT / C.TEMP_OU_TAU_S)
    wT = np.zeros(C.T_STEPS)
    eT = rng.standard_normal(C.T_STEPS) * C.TEMP_OU_K * np.sqrt(1 - a_T * a_T)
    for i in range(1, C.T_STEPS):
        wT[i] = a_T * wT[i - 1] + eT[i]
    dT = (temp_offset_K + C.BREATH_TEMP_RISE_K * _lowpass(EXPOSURE, 6.0, DT)
          + ramp + wT)
    therm = device.therm_dir[None, :] * dT[:, None]
    age = device.age_dir[None, :] * months
    dn_amb = optics.DN_AMB_DT * dT
    t = np.arange(C.T_STEPS) * DT
    ou = np.zeros(C.T_STEPS)
    a = np.exp(-DT / C.DRIFT_OU_TAU_S)
    e = rng.standard_normal(C.T_STEPS) * C.DRIFT_OU_SIGMA_RIU * np.sqrt(1 - a * a)
    for i in range(1, C.T_STEPS):
        ou[i] = a * ou[i - 1] + e[i]
    dn_amb = dn_amb + C.DRIFT_LINEAR_RIU_PER_S * t * rng.normal(1.0, 0.35) + ou
    bulk = device.s_bulk[None, :] * dn_amb[:, None]

    # read-out drift of the spectrograph wavelength scale
    spec = np.zeros((C.T_STEPS, C.N_CHANNELS))
    for m in range(device.spec_basis.shape[1]):
        amp = C.SPEC_DRIFT_NM[m] * rng.standard_normal()
        w = np.zeros(C.T_STEPS)
        a_s = np.exp(-DT / C.SPEC_DRIFT_TAU_S)
        eps = rng.standard_normal(C.T_STEPS) * np.sqrt(1 - a_s * a_s)
        for i in range(1, C.T_STEPS):
            w[i] = a_s * w[i - 1] + eps[i]
        prof = amp * (0.55 * w + 0.45 * np.linspace(0.0, 1.0, C.T_STEPS)
                      * rng.normal(1.0, 0.4))
        spec += np.outer(prof, device.spec_basis[:, m])

    nuisance = therm + age + bulk + spec
    lam = surf + nuisance
    lam = lam + rng.normal(0.0, C.WAVELENGTH_NOISE_NM, size=lam.shape)
    lam = np.round(lam / C.QUANT_NM) * C.QUANT_NM

    n_base = max(int(round(C.BASELINE_S / DT)) - 2, 3)
    base = np.median(lam[:n_base], axis=0, keepdims=True)
    x = (lam - base).astype(np.float32)
    clean = surf + rng.normal(0.0, C.WAVELENGTH_NOISE_NM, size=surf.shape)
    clean = (clean - np.median(clean[:n_base], axis=0, keepdims=True)).astype(np.float32)
    if not return_latent:
        return x, clean
    return x, clean, dict(theta=theta.astype(np.float32), conc=conc.astype(np.float32),
                          occ=occ.astype(np.float32), dT=dT.astype(np.float32),
                          dn_amb=dn_amb.astype(np.float32),
                          nuisance=nuisance.astype(np.float32))


def sample_subject(label: str, rng: np.random.Generator):
    """Draw one subject's true alveolar concentration vector, ppb."""
    c = np.zeros(KIN.N_VOC - 1)
    for j, name in enumerate(KIN.VOC_NAMES[:-1]):
        gm = HEALTHY_GM[name] * CLASS_RATIO[label][name]
        sigma = np.log(GSD[name])
        c[j] = gm * np.exp(rng.normal(0.0, sigma))
    return c


def build_cohort(seed: int = 0, verbose: bool = False):
    """
    Assemble the full study: subjects, devices, sessions and signals.

    Returns a dictionary of arrays suitable for the torch dataset wrapper.
    """
    rng = np.random.default_rng(seed)
    devices = []
    for d in range(C.N_DEVICES):
        dev = Device(d, np.random.default_rng(seed * 131 + d))
        calibrate_offset(dev)
        devices.append(dev)

    X, Xc, Ctrue, y, subj, dev_id, month_arr, temp_arr, site_arr = [], [], [], [], [], [], [], [], []
    sid = 0
    ambient_base = np.array([HEALTHY_GM[n] for n in KIN.VOC_NAMES[:-1]]) * AMBIENT_FRACTION
    for ci, label in enumerate(C.CLASSES):
        for _ in range(C.N_SUBJECTS[label]):
            c_true = sample_subject(label, rng)
            site = int(rng.integers(0, 3))
            for s in range(C.SESSIONS_PER_SUBJECT):
                d = int(rng.integers(0, C.N_DEVICES))
                months = float(rng.uniform(0.0, C.STUDY_MONTHS))
                temp_off = float(rng.normal(0.0, 6.5) + (site - 1) * 4.0)
                rh_amb = float(np.clip(rng.normal(0.42 + 0.05 * (site - 1), 0.07), 0.2, 0.7))
                rh_brt = float(np.clip(rng.normal(0.95, 0.02), 0.85, 0.99))
                c_sess = c_true * np.exp(
                    rng.normal(0.0, np.log(WITHIN_SUBJECT_GSD), size=c_true.shape))
                c_amb = ambient_base * np.exp(rng.normal(0.0, 0.35, size=c_true.shape)) \
                        * (1.0 + 0.25 * (site - 1))
                x, xc = simulate_session(devices[d], c_sess, c_amb, rh_amb, rh_brt,
                                         temp_off, months, rng)
                X.append(x); Xc.append(xc); Ctrue.append(c_sess.astype(np.float32))
                y.append(ci); subj.append(sid); dev_id.append(d)
                month_arr.append(months); temp_arr.append(temp_off); site_arr.append(site)
            sid += 1
        if verbose:
            print(f"  {label}: {C.N_SUBJECTS[label]} subjects done")
    return dict(X=np.stack(X), X_clean=np.stack(Xc), conc=np.stack(Ctrue),
                y=np.array(y), subject=np.array(subj),
                device=np.array(dev_id), months=np.array(month_arr),
                temp=np.array(temp_arr), site=np.array(site_arr),
                devices=devices)
