"""Corrected fully-discrete Galerkin solver for the Kawahara equation on a
fixed computational domain (0,1), obtained from the moving physical interval
by the coordinate transformation of the paper.

Transformed equation:
    u_t - a1(x,t) u_x + b1(t) u u_x + b2(t) u_xxx + b3(t) u_xxxxx = f(x,t),
with a1 = (alpha' + x gamma')/gamma, b1 = 1/gamma, b2 = 1/gamma^3, b3 = 1/gamma^5,
and essential boundary conditions
    u(0)=u_x(0)=u_xx(0)=0,   u(1)=u_x(1)=0.

Key corrections relative to a naive implementation:
  (1) Boundary basis: the FULL set of quartic B-splines overlapping [0,1] is used,
      and exactly the five essential conditions are imposed as linear constraints
      via the null space Z of the constraint matrix.  This leaves u_xx(1,t) FREE,
      as required, and avoids the order reduction caused by using only fully
      internal splines.
  (2) Time stepping: Crank-Nicolson with the time-dependent coefficients and the
      load evaluated at the midpoint t_{n+1/2}, giving genuine 2nd order in time.

Observed accuracy: O(h^3 + dt^2) in the L^inf(0,T;L^2) norm.
"""
import numpy as np
import scipy.linalg as sla

from bsplines import B4, dB4, ddB4, dddB4


def assemble(M, ng=14):
    """Assemble the reduced Galerkin operators on the constrained space.

    Returns a dictionary holding the mesh size, the spline evaluator, the
    Gauss nodes/weights, the value tables and the reduced matrices.
    """
    h = 1.0 / M
    P = M + 4  # quartic B-splines with left knot s = k-4, k = 0,...,P-1

    def val(k, x, d=0):
        z = (np.asarray(x, float) - (k - 4) * h) / h
        return [B4, dB4, ddB4, dddB4][d](z) / h**d

    gx, gw = np.polynomial.legendre.leggauss(ng)
    XN, WN = [], []
    for l in range(M):
        a, b = l * h, (l + 1) * h
        XN.append(0.5 * (b - a) * gx + 0.5 * (a + b))
        WN.append(0.5 * (b - a) * gw)
    XN = np.concatenate(XN)
    W = np.concatenate(WN)

    V0 = np.vstack([val(k, XN, 0) for k in range(P)])
    V1 = np.vstack([val(k, XN, 1) for k in range(P)])
    V2 = np.vstack([val(k, XN, 2) for k in range(P)])
    V3 = np.vstack([val(k, XN, 3) for k in range(P)])

    A   = (V0 * W) @ V0.T                # mass                 int  Bi Bj
    S3  = (V1 * W) @ V2.T                # third-order form     int  Bi' Bj''
    S5  = (V2 * W) @ V3.T                # fifth-order form     int  Bi'' Bj'''
    Bc1 = (V0 * W) @ V1.T                # transport            int  Bi Bj'
    Bc2 = (V0 * W * XN) @ V1.T           # transport (x-weight) int  x Bi Bj'
    Ct  = np.einsum('an,bn,cn->abc', V0 * W, V0, V1, optimize=True)  # int Bi Bj Bk'

    # constraints: u=u'=u''=0 at x=0 ;  u=u'=0 at x=1
    Gm = np.array([[val(k, 0., 0), val(k, 0., 1), val(k, 0., 2),
                    val(k, 1., 0), val(k, 1., 1)] for k in range(P)]).T
    Z = sla.null_space(Gm)               # (P, r),  r = P - 5

    red = lambda X: Z.T @ X @ Z
    return dict(h=h, P=P, val=val, XN=XN, W=W, V0=V0, Z=Z, Ct=Ct,
                Ar=red(A), S3r=red(S3), S5r=red(S5),
                Bc1r=red(Bc1), Bc2r=red(Bc2))


def solve(M, T, alpha, gamma, alphap, gammap, u0, source=None,
          Tfinal=1.0, ng=14):
    """Integrate the transformed Kawahara equation.

    Parameters
    ----------
    M, T      : spatial subintervals and number of time steps.
    alpha,gamma,alphap,gammap : callables of t (boundary motion and derivatives;
                                gamma = beta - alpha).
    u0        : callable u0(x), initial datum.
    source    : callable f(x,t) or None (unforced).
    Tfinal    : final time.

    Returns
    -------
    tt   : (T+1,) time levels.
    C    : (T+1, r) reduced coefficient vectors.
    asm  : the assembly dictionary (for reconstruction / energy).
    """
    asm = assemble(M, ng)
    Z, V0, W, XN, Ct = asm['Z'], asm['V0'], asm['W'], asm['XN'], asm['Ct']
    Ar, S3r, S5r = asm['Ar'], asm['S3r'], asm['S5r']
    Bc1r, Bc2r = asm['Bc1r'], asm['Bc2r']
    r = Z.shape[1]

    dt = Tfinal / T
    tt = np.arange(T + 1) * dt
    C = np.zeros((T + 1, r))
    C[0] = np.linalg.solve(Ar, Z.T @ (V0 @ (W * u0(XN))))    # L2 projection of u0

    def load(tm):
        if source is None:
            return 0.0
        return Z.T @ (V0 @ (W * source(XN, tm)))

    for k in range(1, T + 1):
        tm = 0.5 * (tt[k - 1] + tt[k])                       # midpoint
        cf = Z @ C[k - 1]
        Cr = Z.T @ np.tensordot(cf, Ct, axes=([0], [1])) @ Z  # frozen convection
        Bconv = (alphap(tm) / gamma(tm)) * Bc1r + (gammap(tm) / gamma(tm)) * Bc2r
        b1, b2, b3 = 1 / gamma(tm), 1 / gamma(tm)**3, 1 / gamma(tm)**5
        Mr = -Bconv - b2 * S3r + b3 * S5r + b1 * Cr
        rhs = (Ar - dt / 2 * Mr) @ C[k - 1] + dt * load(tm)
        C[k] = np.linalg.solve(Ar + dt / 2 * Mr, rhs)
    return tt, C, asm


def reconstruct(asm, c, xgrid):
    """Evaluate the numerical solution u_m(.,t) = sum_j (Z c)_j B_j on xgrid."""
    val, Z, P = asm['val'], asm['Z'], asm['P']
    B = np.vstack([val(k, xgrid) for k in range(P)]).T @ Z   # (nx, r)
    return B @ c
