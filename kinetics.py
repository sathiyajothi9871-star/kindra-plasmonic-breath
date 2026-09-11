"""
Adsorption model of the functionalised plasmonic array.

Each sensing spot carries a sorbent overlayer with a finite number of binding
sites shared by every volatile in the sample.  Uptake follows a competitive
Langmuir-Freundlich law, so the fractional coverage of one analyte depends on
the concentration of all the others through a common denominator.  That shared
denominator is the mechanistic origin of cross-sensitivity in a breath matrix
and the reason a per-channel steady-state amplitude cannot identify an analyte
on its own: the transient shape, governed by the analyte-specific association
and dissociation rate constants, carries the additional information.

Affinity constants are generated from physicochemical descriptors of the
volatiles (octanol-water partition coefficient, dipole moment, hydrogen-bond
capacity) and coating-specific selectivity weights, which produces the low-rank
affinity structure that real cross-reactive sensor arrays exhibit.
"""
from __future__ import annotations
import numpy as np

# name, logP, dipole moment (D), relative molar volume, H-bond capacity,
# aldehyde indicator, aromatic indicator
VOC_PANEL = [
    ("Acetone",       -0.24, 2.88, 0.74, 0.48, 0.0, 0.0),
    ("Isoprene",       2.42, 0.25, 1.02, 0.02, 0.0, 0.0),
    ("Ethanol",       -0.31, 1.69, 0.58, 0.92, 0.0, 0.0),
    ("1-Propanol",     0.25, 1.68, 0.75, 0.90, 0.0, 0.0),
    ("Toluene",        2.73, 0.36, 1.07, 0.06, 0.0, 1.0),
    ("Benzene",        2.13, 0.00, 0.90, 0.04, 0.0, 1.0),
    ("Ethylbenzene",   3.15, 0.37, 1.24, 0.05, 0.0, 1.0),
    ("Hexanal",        1.78, 2.60, 1.24, 0.44, 1.0, 0.0),
    ("Pentanal",       1.31, 2.60, 1.06, 0.45, 1.0, 0.0),
    ("2-Pentanone",    0.91, 2.70, 1.07, 0.46, 0.0, 0.0),
    ("2-Butanone",     0.29, 2.78, 0.90, 0.47, 0.0, 0.0),
    ("Butane",         2.89, 0.00, 0.97, 0.00, 0.0, 0.0),
    ("Water",         -1.38, 1.85, 0.19, 1.00, 0.0, 0.0),
]
VOC_NAMES = [v[0] for v in VOC_PANEL]
N_VOC = len(VOC_PANEL)
WATER_INDEX = VOC_NAMES.index("Water")

_DESC = np.array([[v[1], v[2], v[3], v[4]] for v in VOC_PANEL], dtype=float)
_DESC_Z = (_DESC - _DESC.mean(0)) / _DESC.std(0)
# binary functional-group indicators are kept unstandardised so that a
# chemically specific coating expresses a genuine affinity bonus rather than a
# shift of the whole selectivity axis
_GROUP = np.array([[v[5], v[6]] for v in VOC_PANEL], dtype=float)

# Array layout.  Four generic sorbents span a dispersive-to-polar gradient,
# four amine-functionalised spots capture carbonyls through Schiff-base
# formation, two aromatic-selective spots exploit pi-pi stacking and two
# hydrogen-bonding spots respond preferentially to short-chain alcohols.
GROUP_GAIN = np.array([
    [0.00, 0.00], [0.00, 0.00], [0.00, 0.00], [0.00, 0.00],
    [2.35, 0.00], [2.10, 0.00], [1.95, 0.00], [2.25, 0.00],
    [0.00, 1.85], [0.00, 1.70],
    [0.55, 0.00], [0.40, 0.00]])


def coating_selectivity(n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """
    Selectivity weights of each coating over the four descriptors.

    The twelve coatings are laid out on a deterministic grid so that the array
    spans a dispersive-to-polar selectivity range, with a small random
    perturbation representing spot-to-spot coating variability.
    """
    base = np.zeros((n_channels, 4))
    ang = np.linspace(0.0, np.pi, n_channels, endpoint=False)
    base[:, 0] = 1.15 * np.cos(ang)                 # dispersive / hydrophobic
    base[:, 1] = 1.05 * np.sin(ang)                 # dipolar
    base[:, 2] = 0.45 * np.cos(2.0 * ang)           # size exclusion
    base[:, 3] = 0.85 * np.sin(2.0 * ang + 0.6)     # hydrogen bonding
    base[10:, 3] += 1.10                            # alcohol-selective spots
    base = base + 0.07 * rng.standard_normal(base.shape)
    return base


def affinity_matrix(sel: np.ndarray, rng: np.random.Generator,
                    log_offset: float = 1.05) -> np.ndarray:
    """log10 K[k, j] for channel k and volatile j (K in ppb^-1 units)."""
    n_ch = sel.shape[0]
    gain = GROUP_GAIN[:n_ch] * (1.0 + 0.05 * rng.standard_normal((n_ch, 2)))
    logK = sel @ _DESC_Z.T + gain @ _GROUP.T
    logK = logK + log_offset + 0.06 * rng.standard_normal(logK.shape)
    logK[:, WATER_INDEX] += 0.55        # sorbents take up water strongly
    return logK


# Carbonyl capture on the amine-functionalised spots proceeds through
# Schiff-base condensation, which is far less readily reversible than the
# physisorption that governs the alkanes and aromatics.  The resulting spread of
# desorption time constants is what makes a plasmonic sensorgram a genuinely
# two-timescale signal.
SLOW_FACTOR_ALDEHYDE = 11.0
SLOW_FACTOR_KETONE = 3.4


def base_tau(n_ch: int, n_v: int, rng: np.random.Generator) -> np.ndarray:
    """
    Desorption time constants of the shared coating recipe.

    Physisorbed volatiles release within a single exhalation, whereas carbonyls
    captured chemically on the amine-functionalised spots release over a period
    comparable with the whole record, so both regimes appear in every
    sensorgram.
    """
    tau = 10.0 ** rng.uniform(np.log10(2.5), np.log10(28.0), size=(n_ch, n_v))
    amine = GROUP_GAIN[:n_ch, 0] > 1.0
    ald = _GROUP[:, 0] > 0.5
    ket = np.array([n in ("Acetone", "2-Pentanone", "2-Butanone") for n in VOC_NAMES])
    boost = np.ones((n_ch, n_v))
    boost[np.ix_(amine, ald)] = SLOW_FACTOR_ALDEHYDE
    boost[np.ix_(amine, ket)] = SLOW_FACTOR_KETONE
    return tau * boost


def rate_constants(logK: np.ndarray, rng: np.random.Generator):
    """
    Association and dissociation rate constants.

    Physisorbed volatiles equilibrate within a single exhalation, whereas
    carbonyls captured chemically on the amine-functionalised spots release over
    a period comparable with the whole record.  Both regimes are therefore
    present in every sensorgram.
    """
    n_ch, n_v = logK.shape
    tau = 10.0 ** rng.uniform(np.log10(2.5), np.log10(28.0), size=(n_ch, n_v))
    amine = GROUP_GAIN[:n_ch, 0] > 1.0
    ald = _GROUP[:, 0] > 0.5
    ket = np.array([n in ("Acetone", "2-Pentanone", "2-Butanone")
                    for n in VOC_NAMES])
    boost = np.ones((n_ch, n_v))
    boost[np.ix_(amine, ald)] = SLOW_FACTOR_ALDEHYDE
    boost[np.ix_(amine, ket)] = SLOW_FACTOR_KETONE
    tau = tau * boost * np.exp(rng.normal(0.0, 0.10, size=tau.shape))
    k_d = 1.0 / tau
    K = 10.0 ** logK
    k_a = K * k_d
    return k_a, k_d


def heterogeneity(rng: np.random.Generator, n_v: int) -> np.ndarray:
    """Langmuir-Freundlich exponent n_j in (0, 1]."""
    return np.clip(rng.normal(0.86, 0.07, size=n_v), 0.55, 1.0)


def integrate_coverage(conc, k_a, k_d, n_exp, q_scale, dt, substeps=8):
    """
    Integrate competitive Langmuir-Freundlich coverage.

        d(theta_kj)/dt = k_a[k,j] * c_j(t)^n_j * (1 - sum_i theta_ki)
                         - k_d[k,j] * theta_kj

    Parameters
    ----------
    conc : (T, J) concentration trace in ppb
    k_a, k_d : (K, J) rate constants
    n_exp : (J,) heterogeneity exponents
    q_scale : (K,) relative site density of each coating
    dt : sampling interval of `conc` in seconds

    Returns
    -------
    theta : (T, K, J) fractional coverage
    """
    T = conc.shape[0]
    n_ch, n_v = k_a.shape
    h = dt / substeps
    th = np.zeros((n_ch, n_v))
    out = np.empty((T, n_ch, n_v), dtype=np.float32)
    ka = k_a * q_scale[:, None]
    for t in range(T):
        c_pow = np.power(np.maximum(conc[t], 0.0), n_exp)
        drive = ka * c_pow[None, :]
        for _ in range(substeps):
            free = np.clip(1.0 - th.sum(axis=1, keepdims=True), 0.0, 1.0)
            dth = drive * free - k_d * th
            th = np.clip(th + h * dth, 0.0, 1.0)
            s = th.sum(axis=1, keepdims=True)
            over = s > 1.0
            if over.any():
                th = np.where(over, th / np.maximum(s, 1e-9), th)
        out[t] = th
    return out
