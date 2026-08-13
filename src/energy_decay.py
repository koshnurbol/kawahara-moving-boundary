"""Exponential energy decay experiment (Section 4.3).

Two corrections relative to an earlier version of this experiment.

1) Boundary motion.  The decay theorem requires

       int_0^t e^{ks} ( sum|phi_i| + |alpha'| + |beta'| ) ds <= Phi_0 e^{lt},  l < k,

   which forces the boundary data to decay EXPONENTIALLY.  The algebraic profile
   alpha(t) = -eps t/(t+1) gives |alpha'| ~ eps/(1+t)^2 and does NOT satisfy it:
   int_0^t e^{ks}(1+s)^{-2} ds ~ e^{kt}/(k(1+t)^2) is not O(e^{lt}) for l<k.
   We use the exponentially relaxing motion

       alpha_eps(t) = -eps(1 - e^{-sigma t}),   beta_eps(t) = 1 + eps(1 - e^{-sigma t}),

   for which |alpha'| = eps sigma e^{-sigma t}, so the hypothesis holds as soon
   as sigma > k - l.  The geometry is qualitatively unchanged.

2) Time window.  The boundary dissipation is  |u_xx(1,t)|^2 / 2.  For
   u0 = 0.01 x^3(1-x)^2 one has u0_xx(1) = 0.02, hence a dissipation rate
   ~4e-4, while E(0) = ||u0||^2 = 4.33e-8.  The intrinsic decay time is
   therefore E/|dE/dt| ~ 1e-4, NOT O(1):  the energy is exhausted almost
   immediately.  Fitting an "exponential tail" on t in [0.2,0.8] fits values of
   E of order 1e-16, i.e. round-off, and yields a meaningless rate.  We resolve
   the true window t in [0, 1e-3] instead; the measured rate is then identical
   to six significant figures across meshes and time steps.

Reported quantities.  The theorem is stated for the NORM,
    ||v(.,t)|| <= V0 e^{-zeta t}   =>   E = ||v||^2 <= V0^2 e^{-2 zeta t},
so  lambda_E = 2 zeta.  Comparing lambda_E directly with zeta is a
factor-of-two error; both are printed below.

Run:  python energy_decay.py   ->   figures/energy_decay.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import solve

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

EPS, SIGMA = 0.05, 4.0
TWIN = 1.0e-3                      # resolved decay window
u0_active = lambda x: 0.01 * x**3 * (1 - x)**2       # u0''(1) = 0.02 != 0


def geometry(L=1.0):
    """Exponentially relaxing expansion of a physical interval of length ~L."""
    return dict(
        gamma =lambda t: L * (1 + 2*EPS*(1 - np.exp(-SIGMA*t))),
        alpha =lambda t: -L*EPS*(1 - np.exp(-SIGMA*t)),
        alphap=lambda t: -L*EPS*SIGMA*np.exp(-SIGMA*t),
        gammap=lambda t: 2*L*EPS*SIGMA*np.exp(-SIGMA*t))


def energy_series(M, T, u0, Tfinal=TWIN, L=1.0):
    g = geometry(L)
    tt, C, asm = solve(M, T, g['alpha'], g['gamma'], g['alphap'], g['gammap'],
                       u0, source=None, Tfinal=Tfinal)
    Ar = asm['Ar']
    E = np.array([g['gamma'](tt[k]) * (C[k] @ Ar @ C[k]) for k in range(len(tt))])
    return tt, E, C, asm


def fit(tt, E, lo, hi):
    m = (tt >= lo) & (tt <= hi)
    slope, icept = np.polyfit(tt[m], np.log(E[m]), 1)
    return -slope, icept, slope


if __name__ == "__main__":
    LO, HI = 0.2*TWIN, 0.8*TWIN

    tt, E, C, asm = energy_series(60, 4000, u0_active)
    lamE, icept, slope = fit(tt, E, LO, HI)
    val, P, Z = asm['val'], asm['P'], asm['Z']
    tr0 = (np.array([val(j, 1.0, 2) for j in range(P)]) @ Z) @ C[0]

    print("MAIN RUN   gamma in [1, 1.1],  u0 = 0.01 x^3(1-x)^2")
    print(f"   E_phys(0)                 = {E[0]:.4e}")
    print(f"   u_m,xx(1,0)               = {tr0:.6f}   (exact 0.02)")
    print(f"   E(T)/E(0),  T = {TWIN:g}    = {E[-1]/E[0]:.4e}")
    print(f"   lambda_E  (rate of E)     = {lamE:.1f}")
    print(f"   zeta      (rate of ||v||) = {lamE/2:.1f}")
    print(f"   theoretical lower bound   = {5*np.pi**2/(8*1.1**5):.3f}"
          f"   -> observed exceeds it by a factor {lamE/2/(5*np.pi**2/(8*1.1**5)):.0f}")

    print("\n   convergence of the measured rate:")
    for M in (40, 60, 80):
        row = []
        for T in (2000, 4000, 8000):
            t2, E2, _, _ = energy_series(M, T, u0_active)
            row.append(fit(t2, E2, LO, HI)[0])
        print(f"     M={M:3d}   lambda_E = " + "  ".join(f"{r:9.1f}" for r in row)
              + "   (dt = 5e-7, 2.5e-7, 1.25e-7)")

    print("\n   dependence on the length of the physical interval")
    print(f"     {'sup gamma':>10} {'window':>10} {'E(T)/E(0)':>12} {'lambda_E':>12} {'zeta':>10}")
    for L, Tf, T in ((1.0, TWIN, 4000), (2.0, 1e-2, 4000), (3.0, 1e-1, 4000),
                     (4.0, 1.0, 4000), (5.0, 1.0, 4000)):
        t2, E2, _, _ = energy_series(60, T, u0_active, Tfinal=Tf, L=L)
        lam, _, _ = fit(t2, E2, 0.2*Tf, 0.8*Tf)
        print(f"     {L*1.1:>10.2f} {Tf:>10.0e} {E2[-1]/E2[0]:>12.3e}"
              f" {lam:>12.3f} {lam/2:>10.3f}")

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 12, "mathtext.fontset": "cm",
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    ax.semilogy(tt*1e3, E, color="#1f4e79", lw=2.0,
                label=r"$E_{\mathrm{phys}}(t_n)$")
    tf = np.linspace(0.15*TWIN, TWIN, 60)
    ax.semilogy(tf*1e3, np.exp(icept + slope*tf), "--", color="#c0392b", lw=1.6,
                label=fr"fit $\propto e^{{-\lambda_E t}},\ \lambda_E\approx{lamE:.0f}$")
    ax.set_xlabel(r"$t\ \times 10^{3}$")
    ax.set_ylabel(r"$E_{\mathrm{phys}}(t)=\|v_m(\cdot,t)\|^2_{L^2(Q_t)}$")
    ax.set_title("Physical discrete energy decay (resolved window)", fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.set_xlim(0, TWIN*1e3)
    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(os.path.join(FIGDIR, "energy_decay.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print("\nsaved figures/energy_decay.png")
