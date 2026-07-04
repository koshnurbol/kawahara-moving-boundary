"""Exponential energy decay experiment (Section 4.3) for the corrected scheme.

Unforced problem, mild moving boundary (eps=0.05), initial datum
u0 = 0.01 x^3 (1-x)^2 (with u_xx(1,0) != 0 so the boundary dissipation is active).
Produces a semi-log plot of the physical discrete energy and estimates the
asymptotic decay rate lambda_E from the slope of the exponential tail.

Run:  python energy_decay.py   ->   figures/energy_decay.png
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import solve

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

EPS = 0.05
gamma  = lambda t: 1 + 2*EPS*t/(t+1)
alpha  = lambda t: -EPS*t/(t+1)
alphap = lambda t: -EPS/(t+1)**2
gammap = lambda t: 2*EPS/(t+1)**2
u0     = lambda x: 0.01*x**3*(1-x)**2


def energy_series(M, T, Tfinal=1.0):
    tt, C, asm = solve(M, T, alpha, gamma, alphap, gammap, u0,
                       source=None, Tfinal=Tfinal)
    Ar = asm['Ar']
    Ephys = np.array([gamma(tt[k]) * (C[k] @ Ar @ C[k]) for k in range(len(tt))])
    return tt, Ephys


if __name__ == "__main__":
    tt, E = energy_series(40, 2000)
    mask = (tt >= 0.2) & (tt <= 0.8)                    # exponential-tail window
    slope, intercept = np.polyfit(tt[mask], np.log(E[mask]), 1)
    lamE = -slope
    print(f"E_phys(0) = {E[0]:.4e}")
    print(f"asymptotic decay rate  lambda_E ~ {lamE:.2f}")

    plt.rcParams.update({"font.size": 12, "mathtext.fontset": "cm",
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    ax.semilogy(tt, E, color="#1f4e79", lw=2.0, label=r"$E_{\mathrm{phys}}(t_n)$")
    tf = np.linspace(0.15, 1.0, 50)
    ax.semilogy(tf, np.exp(intercept + slope*tf), "--", color="#c0392b", lw=1.6,
                label=fr"$\propto e^{{-\lambda_E t}},\ \lambda_E\approx{lamE:.1f}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$E_{\mathrm{phys}}(t)$ (log scale)")
    ax.set_title("Physical discrete energy decay", fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(os.path.join(FIGDIR, "energy_decay.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print("saved figures/energy_decay.png")
