"""
Electromagnetic forward model of the plasmonic transducer.

The array is modelled in the Kretschmann attenuated-total-reflection geometry:

    SF10 prism | Cr (2 nm) | Au (47 nm) | sorbent overlayer | ambient gas

Reflectance is obtained from the Abeles characteristic-matrix formulation for
p-polarised light, gold is described by the Rakic Lorentz-Drude model and the
prism by the Schott SF10 Sellmeier equation.  The module exposes the two
quantities the downstream signal model needs:

    * S_bulk[k] = d(lambda_res)/d(n_ambient)          [nm / RIU]
    * S_surf[k] = d(lambda_res)/d(n_eff of adlayer)   [nm / RIU]
    * l_d[k]    = evanescent decay length in the dielectric   [nm]

Because S_bulk and S_surf scale differently with the resonance wavelength and
with the overlayer thickness, the array response to a bulk index excursion and
the array response to surface binding are not collinear.  That non-collinearity
is the physical basis of the dispersion projection used by the model.
"""
from __future__ import annotations
import numpy as np

HC_EV_NM = 1239.841984  # h*c in eV*nm

# Rakic et al., Lorentz-Drude parameters for gold (energies in eV)
_AU_WP = 9.03
_AU_F = np.array([0.760, 0.024, 0.010, 0.071, 0.601, 4.384])
_AU_G = np.array([0.053, 0.241, 0.345, 0.870, 2.494, 2.214])
_AU_W = np.array([0.000, 0.415, 0.830, 2.969, 4.304, 13.32])

# Schott SF10 Sellmeier coefficients (wavelength in micrometres)
_SF10_B = np.array([1.62153902, 0.256287842, 1.64447552])
_SF10_C = np.array([0.0122241457, 0.0595736775, 147.468793])


def sf10_index(lam_nm: np.ndarray) -> np.ndarray:
    """Refractive index of SF10 from the Sellmeier equation."""
    x2 = (np.asarray(lam_nm, dtype=float) / 1000.0) ** 2
    s = 1.0
    for b, c in zip(_SF10_B, _SF10_C):
        s = s + b * x2 / (x2 - c)
    return np.sqrt(s)


def gold_permittivity(lam_nm: np.ndarray) -> np.ndarray:
    """Complex relative permittivity of gold (Lorentz-Drude)."""
    w = HC_EV_NM / np.asarray(lam_nm, dtype=float)
    w = w[..., None]
    omp = np.sqrt(_AU_F[0]) * _AU_WP
    eps = 1.0 - omp ** 2 / (w[..., 0] ** 2 + 1j * _AU_G[0] * w[..., 0])
    for m in range(1, 6):
        eps = eps + _AU_F[m] * _AU_WP ** 2 / (
            (_AU_W[m] ** 2 - w[..., 0] ** 2) - 1j * _AU_G[m] * w[..., 0])
    return eps


def chromium_permittivity(lam_nm: np.ndarray) -> np.ndarray:
    """Chromium adhesion layer, weak dispersion over the scan band."""
    lam = np.asarray(lam_nm, dtype=float)
    n = 3.13 + 0.0011 * (lam - 633.0)
    k = 3.33 + 0.0021 * (lam - 633.0)
    return (n + 1j * k) ** 2


def _tmm_reflectance(lam_nm, theta_rad, eps_list, d_list):
    """
    p-polarised reflectance of a stratified stack.

    eps_list : sequence of complex permittivity arrays, incident medium first,
               substrate (semi-infinite ambient) last.
    d_list   : thickness (nm) of the intermediate layers only.
    """
    lam = np.asarray(lam_nm, dtype=float)
    eps0 = eps_list[0]
    kx2 = eps0 * np.sin(theta_rad) ** 2

    def _q(eps):
        kz = np.sqrt(eps - kx2 + 0j)
        kz = np.where(kz.imag < 0, -kz, kz)
        return kz / eps, kz

    q0, _ = _q(eps0)
    qs, _ = _q(eps_list[-1])

    m11 = np.ones_like(lam, dtype=complex)
    m12 = np.zeros_like(lam, dtype=complex)
    m21 = np.zeros_like(lam, dtype=complex)
    m22 = np.ones_like(lam, dtype=complex)
    for eps, d in zip(eps_list[1:-1], d_list):
        qj, kzj = _q(eps)
        beta = 2.0 * np.pi * d / lam * kzj
        c, s = np.cos(beta), np.sin(beta)
        a11, a12 = c, -1j * s / qj
        a21, a22 = -1j * qj * s, c
        m11, m12, m21, m22 = (m11 * a11 + m12 * a21, m11 * a12 + m12 * a22,
                              m21 * a11 + m22 * a21, m21 * a12 + m22 * a22)

    num = (m11 + m12 * qs) * q0 - (m21 + m22 * qs)
    den = (m11 + m12 * qs) * q0 + (m21 + m22 * qs)
    r = num / den
    return np.abs(r) ** 2


class PlasmonicChannel:
    """One functionalised sensing spot of the plasmonic array."""

    def __init__(self, d_au_nm, d_coat_nm, n_coat, theta_deg,
                 lam_range, lam_steps, d_cr_nm=2.0):
        self.d_au = float(d_au_nm)
        self.d_coat = float(d_coat_nm)
        self.n_coat = float(n_coat)
        self.theta = np.deg2rad(float(theta_deg))
        self.lam = np.linspace(lam_range[0], lam_range[1], lam_steps)
        self.d_cr = float(d_cr_nm)
        self._eps_prism = sf10_index(self.lam) ** 2
        self._eps_cr = chromium_permittivity(self.lam)
        self._eps_au = gold_permittivity(self.lam)

    def reflectance(self, n_amb, coat_index_shift=0.0):
        eps_coat = np.full_like(self.lam, (self.n_coat + coat_index_shift) ** 2, dtype=complex)
        eps_amb = np.full_like(self.lam, n_amb ** 2, dtype=complex)
        return _tmm_reflectance(
            self.lam, self.theta,
            [self._eps_prism, self._eps_cr, self._eps_au, eps_coat, eps_amb],
            [self.d_cr, self.d_au, self.d_coat])

    def resonance_wavelength(self, n_amb, coat_index_shift=0.0):
        """Sub-grid resonance wavelength by parabolic interpolation of the dip."""
        R = self.reflectance(n_amb, coat_index_shift)
        i = int(np.argmin(R))
        i = min(max(i, 1), len(R) - 2)
        y0, y1, y2 = R[i - 1], R[i], R[i + 1]
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.0 if abs(denom) < 1e-18 else 0.5 * (y0 - y2) / denom
        step = self.lam[1] - self.lam[0]
        return float(self.lam[i] + delta * step)

    # ---- sensitivities -------------------------------------------------
    def bulk_sensitivity(self, n_amb=1.00027, dn=2.0e-4):
        a = self.resonance_wavelength(n_amb - dn)
        b = self.resonance_wavelength(n_amb + dn)
        return (b - a) / (2.0 * dn)

    def decay_length(self, n_amb=1.00027):
        """1/e decay length of the evanescent field into the sensing overlayer."""
        lam0 = self.resonance_wavelength(n_amb)
        eps_m = gold_permittivity(np.array([lam0]))[0]
        eps_d = self.n_coat ** 2
        return float(lam0 / (2.0 * np.pi) * np.sqrt(abs(eps_m.real + eps_d)) / eps_d)

    def surface_sensitivity(self, n_amb=1.00027, dn_coat=2.0e-4):
        """
        d(lambda_res) / d(effective index of the sensing overlayer).

        Adsorbed analyte raises the effective index of the sorbent film rather
        than that of the bulk gas, so this is the coefficient that converts
        surface coverage into an observable resonance shift.
        """
        a = self.resonance_wavelength(n_amb, -dn_coat)
        b = self.resonance_wavelength(n_amb, +dn_coat)
        return (b - a) / (2.0 * dn_coat)


def build_array(d_au_nm, coat_thickness, coat_index, theta_deg,
                lam_range, lam_steps):
    """Instantiate the full array and pre-compute its response coefficients."""
    chans = [PlasmonicChannel(d_au_nm[k], coat_thickness[k], coat_index[k],
                              theta_deg, lam_range, lam_steps)
             for k in range(len(coat_thickness))]
    lam0 = np.array([c.resonance_wavelength(1.00027) for c in chans])
    s_bulk = np.array([c.bulk_sensitivity() for c in chans])
    s_surf = np.array([c.surface_sensitivity() for c in chans])
    l_d = np.array([c.decay_length() for c in chans])
    return dict(channels=chans, lambda0=lam0, s_bulk=s_bulk,
                s_surf=s_surf, l_d=l_d)


# ----------------------------------------------------------------------
# Nuisance response directions
# ----------------------------------------------------------------------
# Thermo-optic and ageing coefficients used to derive the physical directions
# along which the array responds to transduction nuisances rather than to
# analyte binding.
DN_COAT_DT = -4.5e-4     # 1/K, thermo-optic coefficient of the polymer/MOF sorbent
DN_AMB_DT = -9.3e-7      # 1/K, air thermo-optic coefficient
DGAMMA_DT = 1.2e-4       # eV/K, electron-phonon damping of gold
DD_COAT_DT = 3.1e-4      # 1/K, linear thermal expansion of the sorbent film
DN_COAT_DMONTH = 4.0e-4  # RIU per month, densification of the sorbent film
DD_COAT_DMONTH = -1.5e-3 # relative thickness loss per month


def _res_with_perturbation(ch: "PlasmonicChannel", n_amb, dn_coat=0.0,
                           dgamma=0.0, d_scale=1.0):
    """Resonance wavelength after perturbing coating index, Au damping, thickness."""
    lam = ch.lam
    eps_au = ch._eps_au
    if dgamma != 0.0:
        w = HC_EV_NM / lam
        omp = np.sqrt(_AU_F[0]) * _AU_WP
        eps = 1.0 - omp ** 2 / (w ** 2 + 1j * (_AU_G[0] + dgamma) * w)
        for m in range(1, 6):
            eps = eps + _AU_F[m] * _AU_WP ** 2 / (
                (_AU_W[m] ** 2 - w ** 2) - 1j * _AU_G[m] * w)
        eps_au = eps
    eps_coat = np.full_like(lam, (ch.n_coat + dn_coat) ** 2, dtype=complex)
    eps_amb = np.full_like(lam, n_amb ** 2, dtype=complex)
    R = _tmm_reflectance(lam, ch.theta,
                         [ch._eps_prism, ch._eps_cr, eps_au, eps_coat, eps_amb],
                         [ch.d_cr, ch.d_au, ch.d_coat * d_scale])
    i = int(np.argmin(R)); i = min(max(i, 1), len(R) - 2)
    y0, y1, y2 = R[i - 1], R[i], R[i + 1]
    den = (y0 - 2.0 * y1 + y2)
    delta = 0.0 if abs(den) < 1e-18 else 0.5 * (y0 - y2) / den
    return float(lam[i] + delta * (lam[1] - lam[0]))


def thermal_direction(arr, n_amb=1.00027, dT=1.0):
    """d(lambda_res)/dT per channel [nm/K]."""
    out = []
    for ch in arr["channels"]:
        a = _res_with_perturbation(ch, n_amb - DN_AMB_DT * dT,
                                   -DN_COAT_DT * dT, -DGAMMA_DT * dT,
                                   1.0 - DD_COAT_DT * dT)
        b = _res_with_perturbation(ch, n_amb + DN_AMB_DT * dT,
                                   +DN_COAT_DT * dT, +DGAMMA_DT * dT,
                                   1.0 + DD_COAT_DT * dT)
        out.append((b - a) / (2.0 * dT))
    return np.array(out)


def ageing_direction(arr, n_amb=1.00027, dm=1.0):
    """d(lambda_res)/d(month of film ageing) per channel [nm/month]."""
    out = []
    for ch in arr["channels"]:
        a = _res_with_perturbation(ch, n_amb, -DN_COAT_DMONTH * dm,
                                   0.0, 1.0 - DD_COAT_DMONTH * dm)
        b = _res_with_perturbation(ch, n_amb, +DN_COAT_DMONTH * dm,
                                   0.0, 1.0 + DD_COAT_DMONTH * dm)
        out.append((b - a) / (2.0 * dm))
    return np.array(out)


def fabrication_directions(arr, n_amb=1.00027):
    """
    Response of the array to fabrication tolerance.

    Chip-to-chip variation enters through metal thickness, overlayer thickness
    and overlayer index.  Each perturbation moves the twelve resonances along a
    direction fixed by the same stack model that fixes the thermal and ageing
    directions, so the geometry that separates transduction from binding also
    covers the difference between one chip and the next.
    """
    d_au, d_co, n_co = [], [], []
    for ch in arr["channels"]:
        base = ch.d_au
        ch.d_au = base * 1.01
        a = _res_with_perturbation(ch, n_amb)
        ch.d_au = base * 0.99
        b = _res_with_perturbation(ch, n_amb)
        ch.d_au = base
        d_au.append((a - b) / (0.02 * base))
        p = _res_with_perturbation(ch, n_amb, 0.0, 0.0, 1.01)
        q = _res_with_perturbation(ch, n_amb, 0.0, 0.0, 0.99)
        d_co.append((p - q) / (0.02 * ch.d_coat))
        u = _res_with_perturbation(ch, n_amb, +2e-3)
        v = _res_with_perturbation(ch, n_amb, -2e-3)
        n_co.append((u - v) / 4e-3)
    return np.stack([np.array(d_au), np.array(d_co), np.array(n_co)], axis=1)


def spectrograph_basis(arr):
    """
    Read-out drift of the wavelength scale.

    A grating spectrograph drifts as an offset plus a dispersive term that is
    linear in the recorded wavelength, so the array responds along a rank-two
    subspace that is fixed by the resonance positions alone and is independent
    of anything happening at the sensing surface.
    """
    lam0 = arr["lambda0"]
    e0 = np.ones_like(lam0)
    e1 = (lam0 - lam0.mean()) / 10.0
    M = np.stack([e0, e1], axis=1)
    return M / np.linalg.norm(M, axis=0, keepdims=True)
