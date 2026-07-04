"""Unforced wave dynamics in deforming physical domains (Section 4.2):
reconstructs the physical solution v_m(y,t) on the moving interval and produces
the 3-D surface figures for the two geometric scenarios.

Run:  python moving_domain.py   ->   figures/ex1_moving.png, figures/ex2_moving.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from solver import solve, reconstruct

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def physical_surface(M, T, alpha, gamma, alphap, gammap, u0, fname, title,
                     nx=161, stride_t=2):
    tt, C, asm = solve(M, T, alpha, gamma, alphap, gammap, u0)
    xf = np.linspace(0, 1, nx)
    U = np.array([reconstruct(asm, C[k], xf) for k in range(len(tt))])  # (nt,nx)

    ts, Us = tt[::stride_t], U[::stride_t]
    Tg = np.repeat(ts[:, None], nx, axis=1)
    Yg = np.array([alpha(ts[k]) + xf * gamma(ts[k]) for k in range(len(ts))])

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm",
                         "axes.linewidth": 0.6})
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.viridis
    norm = plt.Normalize(Us.min(), Us.max())
    rgb = ls.shade(Us, cmap=cmap, norm=norm, vert_exag=0.12, blend_mode="soft")

    fig = plt.figure(figsize=(7.6, 5.6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Yg, Tg, Us, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, rasterized=True)
    ax.set_xlabel(r"$y$", labelpad=8)
    ax.set_ylabel(r"$t$", labelpad=8)
    ax.set_zlabel(r"$v_m(y,t)$", labelpad=10)
    ax.view_init(elev=26, azim=-58)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.04)
    ax.grid(True, alpha=0.25)
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm); m.set_array(Us)
    cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.11, aspect=16)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=12, pad=4)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(os.path.join(FIGDIR, fname), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"{fname}: v_m range [{U.min():.3e}, {U.max():.3e}]")


if __name__ == "__main__":
    # Example 1: monotonically expanding domain
    physical_surface(
        48, 600,
        alpha=lambda s: -s/(s+1), gamma=lambda s: (3*s+1)/(s+1),
        alphap=lambda s: -1/(s+1)**2, gammap=lambda s: 2/(s+1)**2,
        u0=lambda x: x**3*(1-x)**3,
        fname="ex1_moving.png", title="Example 1: monotonically expanding domain")

    # Example 2: oscillating domain (small-amplitude, quasi-linear regime)
    A = 0.01
    physical_surface(
        48, 600,
        alpha=lambda s: 0.5*np.cos(2*np.pi*s) - 0.5,
        gamma=lambda s: 2 - np.cos(2*np.pi*s),
        alphap=lambda s: -np.pi*np.sin(2*np.pi*s),
        gammap=lambda s: 2*np.pi*np.sin(2*np.pi*s),
        u0=lambda x: A*np.sin(np.pi*x)**3,
        fname="ex2_moving.png", title="Example 2: oscillating domain")
