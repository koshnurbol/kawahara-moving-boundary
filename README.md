# Analysis and Numerical Simulation of the Kawahara Equation on a Moving Domain

Reference implementation for the paper
**"Analysis and Numerical Simulation of the Kawahara Equation"** (N. Koshkarbayev).

The code solves the fifth-order Kawahara equation on a time-dependent interval by
mapping it to a fixed computational domain and discretising with a **quartic
B-spline Galerkin method** in space and a **semi-implicit Crank–Nicolson** scheme
in time. It reproduces the convergence tables, the moving-domain wave figures,
and the energy-decay experiment.

## Method summary

- Full quartic (degree-4) B-spline space on `[0,1]`; the five essential boundary
  conditions `u=u'=u''=0` at `x=0` and `u=u'=0` at `x=1` are imposed exactly as
  linear constraints via the null space of the constraint matrix. This leaves
  `u_xx(1,t)` free (needed for the boundary dissipation) and avoids the order
  reduction caused by using only fully internal splines.
- Crank–Nicolson with time-dependent coefficients and load evaluated at the
  midpoint `t_{n+1/2}`; the convective term is linearised by **extrapolating**
  the advecting field to the midpoint, `u_hat = 1.5 u^n - 0.5 u^{n-1}`
  (first step: predictor–corrector).  Freezing the advecting field at `t_n`
  instead — `solve(..., linearisation='frozen')` — reduces the scheme to FIRST
  order in time; see `convergence.py`.
- Observed accuracy in the `L^inf(0,T;L^2)` norm: **`O(h^3 + dt^2)`**. The
  third spatial order reflects the `H^2` (energy-norm) approximation rate of
  quartic B-splines controlled by the fifth-order operator.

## Repository layout

```
src/
  bsplines.py       quartic B-spline and its derivatives
  solver.py         assembly (constrained basis) + Crank–Nicolson time stepping
  convergence.py    Method of Manufactured Solutions: spatial tables
                    (Examples 1-4) and the temporal-order study
  moving_domain.py  reconstructed physical solution v_m(y,t), 3-D surfaces
  energy_decay.py   physical energy decay + asymptotic rate lambda_E
figures/            generated PNGs
```

## Installation

```bash
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```

## Usage

```bash
cd src
python convergence.py      # prints the L2 convergence tables (order ~3)
python moving_domain.py    # writes figures/ex1_moving.png, ex2_moving.png
python energy_decay.py     # writes figures/energy_decay.png, prints lambda_E
```

## What each script reproduces

| Script             | Output                                            |
|--------------------|---------------------------------------------------|
| `convergence.py`   | Tables of Section 4.1 (Examples 1-4) + temporal    |
| `moving_domain.py` | Figures 1–2 (unforced `v_m(y,t)`, Section 4.2)    |
| `energy_decay.py`  | Figure 3 and the decay rate `lambda_E` (Sec. 4.3) |

## Notes on reproducibility

- Convergence is measured in the discrete `L^inf(0,T;L^2)` norm; the observed
  order is `~3`, in agreement with the corrected convergence theorem.
- In the energy-decay experiment the decay must be measured on its intrinsic
  time scale.  The boundary dissipation is `|u_xx(1,t)|^2/2 ~ 4e-4` while
  `E(0) = 4.33e-8`, so the decay time is `~1e-4`, NOT `O(1)`: fitting a tail on
  `t in [0.2,0.8]` fits round-off.  `energy_decay.py` integrates over
  `t in [0,1e-3]` and returns `lambda_E = 1.12301e4`, identical to six
  significant figures for `M = 40,60,80` and three time steps.
- The boundary motion is exponentially relaxing, `alpha ~ -eps(1-e^{-sigma t})`,
  so that the hypothesis of the decay theorem is actually satisfied; the
  algebraic profile `-eps t/(t+1)` does not satisfy it.

## License

MIT — see [LICENSE](LICENSE).
