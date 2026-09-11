"""
KINDRA: a kinetics-informed dual-rate architecture for plasmonic sensorgrams.

Three mechanisms distinguish the model from a conventional recurrent classifier.

1.  Dispersion-disentangling projection.  A bulk index excursion, a thermal
    excursion of the sensor stack, film ageing and read-out drift each move the
    twelve resonances along a direction that is fixed by the optics and is not
    collinear with any surface-binding direction.  A low-rank basis is therefore
    initialised from those physically derived directions, softly removed from
    the input, and kept explanatory of the nuisance by a co-operative head while
    the retained representation is pushed away from it by a gradient-reversal
    head.

2.  Langmuir kinetic state cell.  The recurrent state of the slow branch is a
    fractional-coverage vector that is advanced by an exponential integrator of
    the competitive Langmuir equation.  The forget term is not a free gate but
    exp(-(k_a u + k_d) dt), so the cell inherits the adsorption dynamics rather
    than having to learn them.

3.  Dual-rate coupling.  The fast branch is a gated recurrent unit that consumes
    only the part of the trace the kinetic branch cannot explain, and whose
    update gate is modulated by the instantaneous binding velocity of the slow
    branch.  The two branches therefore run on the two timescales that a
    plasmonic sensorgram actually contains instead of sharing one update rate.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)


class DispersionProjection(nn.Module):
    """Soft removal of a learned low-rank nuisance subspace."""

    def __init__(self, n_ch, rank, init_basis=None):
        super().__init__()
        if init_basis is not None:
            B = torch.as_tensor(init_basis, dtype=torch.float32)[:, :rank]
        else:
            B = torch.randn(n_ch, rank) / math.sqrt(n_ch)
        self.U = nn.Parameter(B.clone())
        # The gate starts high, so the model begins close to the hard physical
        # projection and has to be given a reason by the data to keep any part
        # of a nuisance direction.
        self.gate_logit = nn.Parameter(torch.full((rank,), 2.0))

    def orthonormal(self):
        Q, _ = torch.linalg.qr(self.U)
        return Q

    def forward(self, x):
        """x : (B, T, K) -> retained (B, T, K), nuisance coefficients (B, T, r)"""
        Q = self.orthonormal()
        a = x @ Q                                   # (B, T, r)
        g = torch.sigmoid(self.gate_logit)
        z = x - (a * g) @ Q.transpose(0, 1)
        return z, a


class LangmuirCell(nn.Module):
    """
    Coverage state of the whole array, advanced by an exponential integrator of
    the competitive Langmuir equation.

    Equation (6) couples a spot to the concentrations of every volatile in the
    sample, so the cell separates the two roles the equation assigns.  A shared
    encoder maps the projected array response to a nonnegative vector of latent
    pseudo-concentrations, which is where information from different spots is
    allowed to mix, exactly as concentration is the quantity common to all spots
    in the physical system.  Each spot then carries its own coverage of each
    latent component, with its own affinity and desorption rate, and the
    site-competition term couples the components within a spot and not across
    spots.  Coverage is therefore never mixed between spots, which is what makes
    the state a faithful coverage variable rather than a general hidden vector.

        theta_inf = k_a u (1 - s) / (k_a u + k_d)
        theta_new = theta_inf + (theta - theta_inf) exp(-(k_a u + k_d) dt)
    """

    def __init__(self, n_ch, n_lat, dt=2.0):
        super().__init__()
        self.n_ch, self.n_lat, self.dt = n_ch, n_lat, dt
        self.encode = nn.Linear(n_ch, n_lat)
        nn.init.constant_(self.encode.bias, -1.0)
        # Affinities are spread over four decades at initialisation.  A real
        # sorbent film has a distribution of site energies - the reason the
        # Freundlich exponent of (6) is below unity - and a spread of affinities
        # keeps the population responsive across the three orders of magnitude
        # of concentration a breath sample spans.
        lk = torch.linspace(-2.0, 2.0, n_lat)
        self.log_K = nn.Parameter(lk.unsqueeze(0).repeat(n_ch, 1).clone()
                                  + 0.25 * torch.randn(n_ch, n_lat))
        base = torch.linspace(math.log(1.0 / 240.0), math.log(1.0 / 4.0), n_lat)
        self.log_kd = nn.Parameter(base.unsqueeze(0).repeat(n_ch, 1).clone()
                                   + 0.10 * torch.randn(n_ch, n_lat))
        self.occ_w = nn.Parameter(torch.zeros(n_ch, n_lat))

    def forward(self, z):
        """z : (B, T, K) -> theta, dtheta each (B, T, K, J)"""
        B, T, K = z.shape
        u = F.softplus(self.encode(z))                    # (B, T, J)
        kd = torch.exp(self.log_kd).clamp(2e-3, 1.0)
        ka = torch.exp(self.log_K).clamp(0.02, 50.0) * kd
        w = torch.sigmoid(self.occ_w) / self.n_lat
        th = z.new_zeros(B, K, self.n_lat)
        out, dout = [], []
        for t in range(T):
            a = ka * u[:, t].unsqueeze(1)                 # (B, K, J)
            s = torch.clamp((th * w).sum(-1, keepdim=True), 0.0, 0.98)
            r = a + kd
            th_inf = a * (1.0 - s) / (r + 1e-6)
            new = (th_inf + (th - th_inf) * torch.exp(-r * self.dt)).clamp(0.0, 1.0)
            dout.append(new - th)
            th = new
            out.append(th)
        return torch.stack(out, 1), torch.stack(dout, 1)


class VelocityGRUCell(nn.Module):
    """
    GRU whose update gate is modulated by the binding velocity of the slow branch.

    The gate opens when coverage is changing quickly and closes when the surface
    is near equilibrium, so the fast branch spends its capacity on the transport
    and mixing transients that occur while binding is in progress instead of
    re-encoding a stationary plateau.
    """

    def __init__(self, n_in, n_hid, n_vel):
        super().__init__()
        self.x2h = nn.Linear(n_in, 3 * n_hid)
        self.h2h = nn.Linear(n_hid, 3 * n_hid, bias=False)
        self.vel = nn.Linear(n_vel, n_hid)
        self.n_hid = n_hid

    def forward(self, x, h, v):
        gx = self.x2h(x)
        gh = self.h2h(h)
        xr, xz, xn = gx.chunk(3, -1)
        hr, hz, hn = gh.chunk(3, -1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz + self.vel(v))
        n = torch.tanh(xn + r * hn)
        return (1.0 - z) * n + z * h


def phase_pool(z, exposure):
    """
    Per-spot descriptors of the drift-free trace.

    In the thin-film limit the shift of a spot is affine in its occupancy by
    (5), so simple statistics of the projected channel trace are direct
    observations of the coverage state of that spot: the amplitude extremes, the
    exhalation and purge levels, the contrast between the first exposure and the
    purge that follows it, the net excursion across the record and the extremes
    of the first difference.  The latent sites of the kinetic branch are a
    learned and saturating summary of the same quantity, so keeping the linear
    descriptors alongside them preserves the amplitude resolution that
    saturation would otherwise compress.  The identical descriptors are used by
    the classical baselines of Section IV-C, which makes the comparison between
    the learned and the engineered read-out transparent.
    """
    e = exposure.to(z.dtype).to(z.device).view(1, -1, 1)
    T = z.shape[1]
    idx = torch.nonzero(e.view(-1) > 0.5).flatten()
    a = int(idx[0].item()) if idx.numel() else 0
    n = max(int(e.sum().item()) // 3, 1)
    seg = z[:, a:a + n]
    off = z[:, a + n:a + 2 * n]
    ne = e.sum().clamp_min(1.0)
    npg = (1.0 - e).sum().clamp_min(1.0)
    dz = z[:, 1:] - z[:, :-1]
    return torch.cat([
        z.amax(1), z.amin(1), z.mean(1), z.std(1),
        (z * e).sum(1) / ne, (z * (1.0 - e)).sum(1) / npg,
        seg.mean(1), off.mean(1), seg.mean(1) - off.mean(1),
        z[:, -6:].mean(1) - z[:, :4].mean(1),
        dz.amax(1), dz.amin(1)], dim=-1)


class KINDRA(nn.Module):
    def __init__(self, n_ch, n_classes, cfg, init_basis=None, n_nuisance=2,
                 exposure=None):
        super().__init__()
        if exposure is None:
            exposure = torch.ones(1)
        self.register_buffer("exposure", torch.as_tensor(exposure).float())
        P, H, r = cfg.latent_analytes, cfg.hidden_fast, cfg.proj_rank
        self.n_ch, self.n_sites = n_ch, P
        self.proj = DispersionProjection(n_ch, r, init_basis)
        self.slow = LangmuirCell(n_ch, P)
        self.v_recon = nn.Parameter(torch.empty(n_ch, P).uniform_(0.5, 1.5))
        self.b_recon = nn.Parameter(torch.zeros(n_ch))
        self.fast_cell = VelocityGRUCell(2 * n_ch, H, n_ch * P)
        self.n_hid = H
        feat = 5 * n_ch * P + 2 * H + 12 * n_ch
        # Batch normalisation rather than layer normalisation: the descriptor is
        # dominated by coverage amplitudes, and normalising across features
        # within a sample would remove exactly the amplitude scale that carries
        # the concentration information.
        self.norm = nn.BatchNorm1d(feat)
        self.head = nn.Sequential(nn.Dropout(cfg.head_dropout),
                                  nn.Linear(feat, cfg.head_width), nn.GELU(),
                                  nn.Dropout(cfg.head_dropout),
                                  nn.Linear(cfg.head_width, n_classes))
        self.adv = nn.Sequential(nn.Linear(feat, 48), nn.GELU(), nn.Linear(48, n_nuisance))
        self.coop = nn.Sequential(nn.Linear(3 * r, 32), nn.GELU(), nn.Linear(32, n_nuisance))

    def _run(self, z, theta, dtheta):
        xhat = (theta * self.v_recon).sum(-1) + self.b_recon
        resid = z - xhat
        dz = torch.cat([z[:, :1] * 0.0, z[:, 1:] - z[:, :-1]], 1)
        fin = torch.cat([resid, dz], dim=-1)
        B, T, _ = resid.shape
        v = dtheta.reshape(B, T, -1)
        h = resid.new_zeros(B, self.n_hid)
        hs = []
        for t in range(T):
            h = self.fast_cell(fin[:, t], h, v[:, t])
            hs.append(h)
        hs = torch.stack(hs, 1)
        th = theta.reshape(B, T, -1)
        dth = dtheta.reshape(B, T, -1)
        feat = torch.cat([th[:, -1], th.mean(1), th.amax(1),
                          th.amax(1) - th[:, -1], dth.abs().amax(1),
                          hs[:, -1], hs.mean(1),
                          phase_pool(z, self.exposure)], dim=-1)
        return xhat, self.norm(feat)

    def forward(self, x, adv_lambda=0.0):
        z, a = self.proj(x)
        theta, dtheta = self.slow(z)
        xhat, feat = self._run(z, theta, dtheta)
        logits = self.head(feat)
        evidence = F.softplus(logits).clamp(max=1e6)
        acoef = torch.cat([a[:, -1], a.mean(1),
                           a.std(1) if a.shape[1] > 1 else a[:, 0] * 0.0], dim=-1)
        out = dict(evidence=evidence, alpha=evidence + 1.0, recon=xhat, z=z,
                   theta=theta, dtheta=dtheta, feat=feat,
                   nuis_coop=self.coop(acoef))
        if adv_lambda > 0:
            out["nuis_adv"] = self.adv(grad_reverse(feat, adv_lambda))
        return out


# ----------------------------------------------------------------------
# Ablation variants
# ----------------------------------------------------------------------
class KindraAblation(KINDRA):
    """
    Component switches used by the ablation study.

    use_proj : dispersion-disentangling projection
    use_kin  : Langmuir kinetic state cell (replaced by a plain gated cell)
    use_vel  : binding-velocity modulation of the fast branch
    use_evi  : evidential Dirichlet head (replaced by a softmax head)
    """

    def __init__(self, n_ch, n_classes, cfg, init_basis=None, n_nuisance=2,
                 exposure=None, use_proj=True, use_kin=True, use_vel=True,
                 use_evi=True):
        super().__init__(n_ch, n_classes, cfg, init_basis, n_nuisance, exposure)
        self.use_proj, self.use_kin = use_proj, use_kin
        self.use_vel, self.use_evi = use_vel, use_evi
        if not use_kin:
            self.lstm = nn.LSTM(n_ch, n_ch * cfg.latent_analytes, batch_first=True)

    def forward(self, x, adv_lambda=0.0):
        if self.use_proj:
            z, a = self.proj(x)
        else:
            z, a = x, x @ self.proj.orthonormal()
        B, T, K = z.shape
        if self.use_kin:
            theta, dtheta = self.slow(z)
        else:
            o, _ = self.lstm(z)
            theta = torch.sigmoid(o).reshape(B, T, K, self.n_sites)
            dtheta = torch.cat([theta[:, :1] * 0.0,
                                theta[:, 1:] - theta[:, :-1]], 1)
        if not self.use_vel:
            dtheta_in = dtheta * 0.0
        else:
            dtheta_in = dtheta
        xhat, feat = self._run(z, theta, dtheta_in)
        logits = self.head(feat)
        acoef = torch.cat([a[:, -1], a.mean(1),
                           a.std(1) if a.shape[1] > 1 else a[:, 0] * 0.0], dim=-1)
        out = dict(recon=xhat, z=z, theta=theta, dtheta=dtheta, feat=feat,
                   nuis_coop=self.coop(acoef), logits=logits)
        if self.use_evi:
            ev = F.softplus(logits).clamp(max=1e6)
            out["evidence"] = ev
            out["alpha"] = ev + 1.0
        if adv_lambda > 0:
            out["nuis_adv"] = self.adv(grad_reverse(feat, adv_lambda))
        return out
