# MMS convergence tests

Method of manufactured solutions for the compressible (`NS`), incompressible (`iNS`) and
multiphase (`MU`) solvers. A smooth periodic solution is imposed with a source term, and
the L2 error at the final time gives the order of convergence under h-refinement.

## Run

Build the solvers first (`make ns ins mu` in `Solver`), then:

```
python3 run.py NS                                  # P = 2 3 on meshes 4 8
python3 run.py NS --P 2 3 4 5 --meshes 4 8 16      # full study
python3 run.py MU --check                          # compare with MU/reference.csv
python3 plot.py NS                                 # NS/convergence.png
```

`run.py` generates `<case>/SETUP/ProblemFile.f90`, builds it with the HORSES3D build
settings (`COMPILER`, `ENABLE_THREADS`, ... from the environment and `Solver/make.inc`),
and runs every mesh and P with the same `dt` and final time. Results go to
`<case>/errors.csv`; pairs already run with the same binary, problem file and control
file are skipped. `--check` fails if a run fails or an error differs from
`<case>/reference.csv` by more than 1e-3. It needs Python 3 with `sympy` (and
`matplotlib` for the plots).

## Files

* `generate.py`: manufactured solutions and equations of each case. The source term is
  the residual of the solution, emitted without symbolic simplification. The physical
  parameters are read from the solver at run time.
* `ProblemFile.f90.in`: problem file template. It sets the initial condition, adds the
  source term and writes `mms_l2_error.dat` with the total and per-variable L2 errors,
  normalised by the mesh volume (integral of 1).
* `<case>/control_template.control`: control file of each case.
* `<case>/reference.csv`: errors on upstream master f751d86f5 (gfortran, RELEASE).
* Meshes: `Solver/test/Euler/outdated/Benchmark/MESH/Cube{2,4,8,16}.mesh`.

## Cases

| Case | Solution | Discretization | dt, t_final |
|---|---|---|---|
| NS  | rho = 1, divergence-free u ~ sin(t), p = 1 + 0.1 sin sin sin cos(t). Ma 0.3, Re 1e5, Sutherland | GL, Split-form Chandrasekar, Matrix dissipation, BR1, RK3 | 1e-4, 0.1 |
| iNS | same u and p, rho = 1, mu = 1e-3, rho0 c0^2 = 1000 | GL, Split-form Skew-Symmetric 1, Exact, BR1 | 1e-4, 0.1 |
| MU  | c, u ~ cos(t), p ~ sin(t). rho1 = 1, rho2 = 2, mu = 0.1, c0^2 = 1000 | GL, Exact, BR1, RK3 | 1e-7, 1e-4 |

Orders of the total error between meshes 8 and 16 (theory P+1):

| P | NS | iNS | MU |
|---|---|---|---|
| 2 | 2.65 | 2.17 | 2.00 |
| 3 | 3.90 | 3.35 | 3.04 |
| 4 | 4.87 | 4.41 | 4.06 |
| 5 | 5.69 | | |

NS is still approaching P+1 on these meshes; between 16 and 32 elements per side (a mesh
not in the repository) it gives 2.95, 3.89, 4.72 and 5.54. iNS and MU converge closer to
P because the artificial compressibility multiplies the divergence error by c0^2:
lowering c0^2 brings iNS at P = 3 from 3.35 to 3.83. In MU at P = 2 the momentum and the
concentration converge more slowly than the pressure, which dominates the total.

## Notes

* NS assumes the default Sutherland constants, so do not set `sutherland temperature` or
  `sutherland reference temperature`. MU assumes per-fluid sound speeds, not
  `artificial sound speed square`. Gravity is not included.
* At Re = 1e5 the viscous terms are a small part of the NS source term.
* Keep the final time: with u ~ sin(t) a short run leaves the flow nearly at rest, the
  source is dominated by du/dt and the measured order drops.
* The error sums are rank-local, so run without MPI (OpenMP is fine).
* The legacy MU solver is not bitwise reproducible with more than one OpenMP thread
  (differences around 1e-7); use `OMP_NUM_THREADS=1` to compare runs exactly.
