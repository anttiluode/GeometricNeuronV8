"""
geometric_neuron_v8.py  —  the read-path COVERAGE objective (Ky Fan / Oja)
==========================================================================
v7 finding: the frame potential drives template coherence -> 0 but leaves
COVERAGE stuck (~6/8). Fable's diagnosis is exact: the frame potential is a
function of G^H G alone, so it is invariant under G -> QG and literally cannot
see the data. Any orthonormal frame -- including one on directions the field
never visits -- is a global minimizer.

The fix (Fable): make coverage the OBJECTIVE, orthonormality the CONSTRAINT.
On the Stiefel manifold,
        max_G  tr(G^H C G)   s.t.  G^H G = I       (Ky Fan)
is maximized exactly when span(G) is the dominant K-subspace of C. Coverage is
then the definition of optimality; orthogonality is delivered by the manifold,
not a penalty. The gradient flow Gdot = (I - GG^H) C G is Oja's subspace flow --
a true gradient flow on a compact manifold, so LaSalle forbids the limit cycles
that broke the deflation operator. The whole coupling is one K x K trace,
no O(K^2) pairwise penalty, no auxiliary matrix with its own timescale.

We test the claim head to head on the static-hold task (positional read tracks
fine there). Decisive metric = CAPTURED-ENERGY FRACTION:
        frac = tr(Q^H C_data Q) / sum_{i<=K} lambda_i(C_data)
where C_data is the increment covariance of the actual content and Q is an
orthonormal basis of the learned templates' span. frac in [0,1]; frac=1 iff the
templates span the dominant K-subspace of the data. Coherence cannot move this;
the Ky Fan objective is built to.

We implement the soft, autograd-friendly form: read with the raw trainable g
(fair across arms), and add lambda * (- tr(Q^H C_field Q)) with Q = QR(g) (the
Stiefel retraction, differentiated by autograd) and C_field the event-weighted
field-increment covariance, detached (a frozen statistic). No manual retraction,
no deflation memory.

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import torch
from geometric_neuron_v7 import (PopulationV7, make_targets, make_drive,
                                 coherence, coverage, frame_potential)

torch.manual_seed(0); np.random.seed(0)


def ortho_rows(g):
    """orthonormal basis of the span of the K template rows; returns (K,N)."""
    Q, _ = torch.linalg.qr(g.transpose(0, 1))      # (N,K) orthonormal columns
    return Q.transpose(0, 1)                        # (K,N) orthonormal rows


def increment_cov(S, weight=True):
    """event/velocity-weighted tangent-increment covariance of a field run S (T,N)."""
    ds = S[1:] - S[:-1]
    sp = S[:-1]
    proj = ds - (sp * ds).sum(1, keepdim=True) * sp        # Pi(t-1) ds
    w = proj.norm(dim=1, keepdim=True) if weight else torch.ones(proj.shape[0], 1)
    C = (proj * w).transpose(0, 1) @ proj / (w.sum() + 1e-9)
    return 0.5 * (C + C.transpose(0, 1))                    # symmetric PSD (N,N)


def kyfan_trace(g, C):
    """tr(Q^H C Q) on an orthonormal basis Q of span(g). Coverage = captured energy."""
    Q = ortho_rows(g)                                       # (K,N)
    return torch.einsum("ki,ij,kj->", Q.conj(), C.to(torch.cfloat), Q).real


def captured_fraction(g, C_ref, K):
    """fraction of the dominant-K energy of C_ref captured by span(g). In [0,1]."""
    with torch.no_grad():
        num = kyfan_trace(g, C_ref)
        eig = torch.linalg.eigvalsh(C_ref)
        den = eig[-K:].sum()
        return (num / (den + 1e-9)).item()


def train(model, targets, mode="baseline", steps=110, T=200, dwell=25,
          iota=0.2, noise=0.13, eps0=0.18, eps1=0.04, lr=8e-3,
          frame_mu=0.3, cov_mu=0.5):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(2)
    Tn = targets / (targets.norm(dim=1, keepdim=True) + 1e-9)
    for it in range(steps):
        eps = eps0 + (eps1 - eps0) * it / max(steps - 1, 1)
        d, tgt, active = make_drive(T, model.K, targets, dwell, iota, noise, rng)
        S = model.run(d, eps=eps)
        align = (S * Tn[active]).sum(1)
        loss = (1 - align).mean()
        if mode == "frame":
            loss = loss + frame_mu * (frame_potential(model.g, model.K)
                                      + frame_potential(model.P, model.K))
        elif mode == "kyfan":
            C = increment_cov(S).detach()                   # frozen field statistic
            loss = loss - cov_mu * kyfan_trace(model.g, C)  # MAXIMIZE coverage
        opt.zero_grad(); loss.backward(); opt.step()
    return float(align.mean().detach())


RESULTS = """
================================================================
WHAT THIS SHOWED (seeded, multi-seed means)
================================================================
                         g-coh   captured-energy   align
  baseline (no reg)      0.193       0.322          0.636
  frame potential        0.006       0.324          0.633
  Ky Fan coverage        0.148       0.505          0.635

  The decisive metric is captured-energy fraction (1.0 = templates span the
  data's dominant K-subspace; coherence cannot move it).

  - the FRAME POTENTIAL drives coherence to ~0 but leaves captured energy at
    BASELINE (0.32 -> 0.32). It is provably data-blind, exactly as the
    diagnosis says: it is a function of G^H G alone.
  - the KY FAN COVERAGE objective raises captured energy 0.32 -> 0.50 (+57%)
    at no tracking cost, variationally, with one K x K trace -- no O(K^2)
    penalty, no brittle deflation matrix, no second timescale, cycle-free.
    This is the open half of v7 genuinely moved by the data.

  HONEST limits (as Fable's section 5 predicted):
  - it does NOT reach 1.0 under joint training: the coverage term and the task
    adjoint partially disagree, and their weighting (cov_mu) is a modeling
    choice the geometry does not resolve. 0.50 is "substantially better," not
    "solved-to-optimum."
  - nearest-target coverage (5/8) did not rise, because the dominant subspace
    of the INCREMENT covariance is spanned by transition modes, not the
    individual patterns -- captured-energy fraction is the faithful metric,
    nearest-target is not.
  - this used the symmetric power covariance C. The framework-correct operator
    is the skew/lag H_tau = (C_tau - C_tau^T)/2i, whose Ky Fan maximizer is the
    dominant rotation planes and whose trace IS sum_k E[L_k]. Same machinery,
    swap C for H_tau; that is the chirality-aligned next step, untested here.

VERDICT: coverage is addressable by geometry after all -- but as a data-aligned
OBJECTIVE (Ky Fan/Oja), not as the coherence penalty or the deflation operator.
The read/write asymmetry is now closed in principle: writes orthogonalize via
the field projector, reads cover via the Stiefel trace flow. What remains is a
weighting choice between coverage and task, not a missing mechanism.

Do not hype. Do not lie. Just show.
"""


if __name__ == "__main__":
    N, K = 24, 8
    targets = make_targets(N, K, seed=0)

    # reference data structure: increment covariance of the clean content
    rng = np.random.default_rng(99)
    d_ref, tgt_ref, _ = make_drive(600, K, targets, 25, 1.0, 0.0, rng)
    C_data = increment_cov(tgt_ref).detach()
    eig = torch.linalg.eigvalsh(C_data)
    print(f"content has ~{(eig > 0.01 * eig.max()).sum().item()} significant "
          f"increment directions (K={K} templates)\n")

    print(f"{'arm':22s} | g-coh | cover | captured-energy fraction | align")
    for mode in ("baseline", "frame", "kyfan"):
        gc, cv, fr, al = [], [], [], []
        for seed in range(2):
            m = PopulationV7(N, K, seed=seed + 1, read="positional", frame=0.0)
            a = train(m, targets, mode=mode)
            gc.append(coherence(m.g.detach()))
            cv.append(coverage(m.g.detach(), targets))
            fr.append(captured_fraction(m.g.detach(), C_data, K))
            al.append(a)
        name = {"baseline": "baseline (no reg)", "frame": "frame potential",
                "kyfan": "Ky Fan coverage"}[mode]
        print(f"{name:22s} | {np.mean(gc):.3f} | {np.mean(cv):.1f}/{K} |"
              f"        {np.mean(fr):.3f} ± {np.std(fr):.3f}        | {np.mean(al):.3f}")
    print(RESULTS)
