"""Method of Manufactured Solutions: spatial AND temporal convergence tables.

Reproduces the tables of Section 4.1.

Manufactured solutions
----------------------
  Ex.1  expanding    u_ex = x^3 (1-x)^3 (t+1)          u_xx(1,t) = 0
  Ex.2  oscillating  u_ex = sin^3(pi x) e^{-t}         u_xx(1,t) = 0
  Ex.3  expanding    u_ex = x^3 (1-x)^2 (1+x) (t+1)    u_xx(1,t) = 8(t+1)   [NEW]

Examples 1 and 2 both satisfy u_xx(1,t) = 0 and therefore do NOT exercise the
free boundary trace that carries the dissipation of the fifth-order operator.
Example 3 has u_xx(1,t) != 0 and tests exactly the feature the boundary-adapted
spline space is built for, while still exhibiting the generic O(h^3) rate.

  Ex.4  expanding    u_ex = x^3 (1-x)^2 (t+1)          [superconvergence probe]

Ex.4 is a quintic polynomial; for it the observed rate is ~5 rather than 3.
The O(h^3) estimate of the convergence theorem is therefore an upper bound that
is attained for generic data but is NOT sharp for every manufactured solution.

  Ex.5  expanding    u_ex = x^3 (1-x)^2 cos(4 pi t)    [temporal probe]

Ex.5 has a negligible spatial error and a strong time dependence, so the
temporal order can be read off without contamination by the spatial error.

Run:  python convergence.py
"""
import numpy as np
import sympy as sp

from solver import solve, assemble

xs, ts = sp.symbols('x t', real=True)

GEOMETRIES = {
    'expanding':   dict(alpha=-ts / (ts + 1), gamma=(3 * ts + 1) / (ts + 1)),
    'oscillating': dict(alpha=sp.Rational(1, 2) * sp.cos(2 * sp.pi * ts) - sp.Rational(1, 2),
                        gamma=2 - sp.cos(2 * sp.pi * ts)),
}

EXAMPLES = {
    1: dict(geom='expanding',   uex=xs**3 * (1 - xs)**3 * (ts + 1),
            label='Ex.1  expanding domain,   u_ex = x^3(1-x)^3(t+1)'),
    2: dict(geom='oscillating', uex=sp.sin(sp.pi * xs)**3 * sp.exp(-ts),
            label='Ex.2  oscillating domain, u_ex = sin^3(pi x) e^{-t}'),
    3: dict(geom='expanding',   uex=xs**3 * (1 - xs)**2 * (1 + xs) * (ts + 1),
            label='Ex.3  expanding domain,   u_ex = x^3(1-x)^2(1+x)(t+1)'),
    4: dict(geom='expanding',   uex=xs**3 * (1 - xs)**2 * (ts + 1),
            label='Ex.4  expanding domain,   u_ex = x^3(1-x)^2(t+1)  [quintic]'),
    5: dict(geom='expanding',   uex=xs**3 * (1 - xs)**2 * sp.cos(4 * sp.pi * ts),
            label='Ex.5  expanding domain,   u_ex = x^3(1-x)^2 cos(4 pi t)'),
}


def build(example):
    """Return callables (alpha, gamma, alphap, gammap, u0, uex, f) for an example."""
    spec = EXAMPLES[example]
    g = GEOMETRIES[spec['geom']]
    al, ga = g['alpha'], g['gamma']
    alp, gap = sp.diff(al, ts), sp.diff(ga, ts)
    u = spec['uex']

    a1 = (alp + xs * gap) / ga
    b1, b2, b3 = 1 / ga, 1 / ga**3, 1 / ga**5
    f = (sp.diff(u, ts) - a1 * sp.diff(u, xs) + b1 * u * sp.diff(u, xs)
         + b2 * sp.diff(u, xs, 3) + b3 * sp.diff(u, xs, 5))

    lam = lambda e, args: sp.lambdify(args, sp.simplify(e), 'numpy')
    fn, un = lam(f, (xs, ts)), lam(u, (xs, ts))
    fill = lambda v, x: np.broadcast_to(np.asarray(v, float), np.shape(x)).copy()
    return dict(
        alpha=lam(al, ts), gamma=lam(ga, ts),
        alphap=lam(alp, ts), gammap=lam(gap, ts),
        u0=lambda x: fill(un(x, 0.0), x),
        uex=lambda x, t: fill(un(x, t), x),
        f=lambda x, t: fill(fn(x, t), x),
        uxx1=sp.simplify(sp.diff(u, xs, 2).subs(xs, 1)),
    )


def global_error(M, T, prob, Tfinal=1.0, linearisation='extrapolated'):
    """Discrete L^inf(0,T;L^2(0,1)) error, evaluated by Gauss quadrature."""
    tt, C, asm = solve(M, T, prob['alpha'], prob['gamma'], prob['alphap'],
                       prob['gammap'], prob['u0'], source=prob['f'],
                       Tfinal=Tfinal, linearisation=linearisation)
    XN, W, V0, Z = asm['XN'], asm['W'], asm['V0'], asm['Z']
    BZ = V0.T @ Z
    return max(np.sqrt(np.sum(W * (prob['uex'](XN, tt[k]) - BZ @ C[k])**2))
               for k in range(1, len(tt)))


def table(rows, keyname):
    print(f"    {keyname:>10s} {'global error E':>16s} {'order':>8s}")
    print("    " + "-" * 36)
    prev = None
    for key, e in rows:
        order = "  --" if prev is None else f"{np.log(prev[1]/e)/np.log(prev[0]/key):5.2f}"
        print(f"    {key:>10.6g} {e:>16.3e} {order:>8s}")
        prev = (key, e)


if __name__ == "__main__":
    DT = 5e-4
    TS = int(round(1.0 / DT))

    print("=" * 70)
    print(f"SPATIAL CONVERGENCE   (dt = {DT:g} fixed; temporal error negligible)")
    print("=" * 70)
    for ex in (1, 2, 3, 4):
        prob = build(ex)
        print(f"\n  {EXAMPLES[ex]['label']}")
        print(f"  free trace  u_xx(1,t) = {prob['uxx1']}")
        table([(1.0 / M, global_error(M, TS, prob)) for M in (10, 20, 40, 80)], "h")

    print()
    print("=" * 70)
    print("TEMPORAL CONVERGENCE  (Ex.5, h = 1/80 fixed; spatial error negligible)")
    print("=" * 70)
    prob = build(5)
    for lin in ('frozen', 'extrapolated'):
        print(f"\n  linearisation = '{lin}'"
              + ("   (advecting field at t_n)" if lin == 'frozen'
                 else "   (3/2 u^n - 1/2 u^{n-1})"))
        table([(1.0 / T, global_error(80, T, prob, linearisation=lin))
               for T in (20, 40, 80, 160)], "dt")
