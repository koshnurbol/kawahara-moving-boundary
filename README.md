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
  midpoint `t_{n+1/2}` (second order in time); the convective term is linearised
  by freezing the advecting field at the previous level.
- Observed accuracy in the `L^inf(0,T;L^2)` norm: **`O(h^3 + dt^2)`**. The
  third spatial order reflects the `H^2` (energy-norm) approximation rate of
  quartic B-splines controlled by the fifth-order operator.

## Repository layout

```
src/
  bsplines.py       quartic B-spline and its derivatives
  solver.py         assembly (constrained basis) + Crank–Nicolson time stepping
  convergence.py    Method of Manufactured Solutions, Examples 1 & 2 (tables)
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
| `convergence.py`   | Tables of Section 4.1 (Examples 1 and 2)          |
| `moving_domain.py` | Figures 1–2 (unforced `v_m(y,t)`, Section 4.2)    |
| `energy_decay.py`  | Figure 3 and the decay rate `lambda_E` (Sec. 4.3) |

## Notes on reproducibility

- Convergence is measured in the discrete `L^inf(0,T;L^2)` norm; the observed
  order is `~3`, in agreement with the corrected convergence theorem.
- In the energy-decay experiment the physical energy decreases monotonically;
  the rate `lambda_E ~ 4` is read from the **exponential tail** (after a short
  initial transient) and is essentially mesh-independent.

## License

MIT — see [LICENSE](LICENSE).
