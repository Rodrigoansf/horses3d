#!/usr/bin/env python3
"""
MMS convergence test for HORSES3D.

    python3 run.py NS|iNS|MU [--P 2 3] [--meshes 4 8 16] [--check]

Generates and builds <case>/SETUP/ProblemFile.f90, runs the solver for every
(mesh, P) pair with the same dt and final time, and writes the L2 errors to
<case>/errors.csv. Pairs already run with the same solver binary, problem
file and control file are skipped. Then prints the order of convergence
between consecutive meshes. With --check it exits with an error if a run
failed or an error differs from <case>/reference.csv by more than 1e-3.

The problem file library is built with the HORSES3D build settings taken
from the environment (e.g. COMPILER=gfortran ENABLE_THREADS=NO) and from
Solver/make.inc if present.
"""
import argparse
import csv
import hashlib
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import generate

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE.parent / "Euler" / "outdated" / "Benchmark" / "MESH"   # Cube{N}.mesh on [-1,1]x[0,2]x[-1,1]
RTOL = 1e-3

CASES = {
    "NS":  dict(solver="ns",  dt=1e-4, t_final=1e-1, variables=["rho", "rhou", "rhov", "rhow", "rhoe"]),
    "iNS": dict(solver="ins", dt=1e-4, t_final=1e-1, variables=["rho", "rhou", "rhov", "rhow", "p"]),
    "MU":  dict(solver="mu",  dt=1e-7, t_final=1e-4, variables=["c", "sqrhou", "sqrhov", "sqrhow", "p"]),
}


def build(case_dir, solver, home):
    setup = case_dir / "SETUP"
    generate.generate(case_dir.name)
    makefile = (home.parent / "utils" / "ProblemFile" / "Makefile").read_text()
    (setup / "Makefile").write_text(makefile.replace("@HORSES_HOME_DIR", str(home)))
    with open(setup / "make.log", "w") as log:
        if subprocess.run(["make", f"libproblemfile_{solver}.so"], cwd=setup, stdout=log, stderr=subprocess.STDOUT).returncode:
            sys.exit(f"Building the problem file failed, see {setup / 'make.log'}")


def fingerprint(*files):
    h = hashlib.sha256()
    for f in files:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:12]


def run(case_dir, binary, N, P, dt, t_final, keep):
    """Run one case and return [NDOF, volume, L2 total, L2 per variable...], or None if it failed."""
    name = f"N{N}_P{P}"
    steps = round(t_final / dt)
    control = (case_dir / "control_template.control").read_text()
    for token, value in [("{MESH_FILE}", os.path.relpath(MESH_DIR / f"Cube{N}.mesh", case_dir)),
                         ("{SOLUTION_FILE}", f"RESULTS/{name}.hsol"), ("{P}", str(P)),
                         ("{DT}", f"{dt:.6e}"), ("{N_STEPS}", str(steps))]:
        control = control.replace(token, value)
    for d in ("CONTROL", "RESULTS", "MESH"):        # MU writes the boundary mesh to ./MESH
        (case_dir / d).mkdir(exist_ok=True)
    # The control file reader truncates lines at 132 characters
    (case_dir / "CONTROL" / f"{name}.control").write_text("\n".join(l.strip() for l in control.splitlines()) + "\n")

    dat = case_dir / "mms_l2_error.dat"
    dat.unlink(missing_ok=True)
    with open(case_dir / "CONTROL" / f"{name}.log", "w") as log:
        rc = subprocess.run([str(binary), f"CONTROL/{name}.control"], cwd=case_dir, stdout=log, stderr=subprocess.STDOUT).returncode
    if not keep:
        for f in (case_dir / "RESULTS").glob(f"{name}*"):
            f.unlink()
    try:
        # nelems, P, NDOF, volume, total, 5 variables, time
        v = [float(s) for s in dat.read_text().split(",")]
    except (OSError, ValueError):
        return None
    if (rc != 0 or len(v) != 11 or not all(map(math.isfinite, v)) or v[0] != N**3 or v[1] != P
            or abs(v[3] - 8.0) > 1e-8 or abs(v[-1] - steps * dt) > 0.5 * dt):
        return None
    return [int(v[2])] + v[3:10]


def orders(rows, P):
    """Rows of one P sorted by mesh, each with the orders against the previous mesh."""
    rows = sorted((r for r in rows if r["P"] == P), key=lambda r: r["N"])
    out = []
    for prev, r in zip([None] + rows, rows):
        eoc = {}
        if prev:
            for k in r:
                if k.startswith("L2_"):
                    eoc[k] = math.log(prev[k] / r[k]) / math.log(r["N"] / prev["N"])
        out.append((r, eoc))
    return out


def report(case, rows, ps, dt, t_final):
    variables = ["total"] + CASES[case]["variables"]
    lines = [f"MMS {case}: dt = {dt:g}, t_final = {t_final:g}", "",
             "| P | N | NDOF | L2 total | " + " | ".join(f"order {v}" for v in variables) + " |",
             "|---" * (4 + len(variables)) + "|"]
    for P in ps:
        for r, eoc in orders(rows, P):
            cells = [f"{eoc[f'L2_{v}']:.2f}" if eoc else "" for v in variables]
            lines.append(f"| {P} | {r['N']} | {r['NDOF']} | {r['L2_total']:.4e} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")


def check(case_dir, rows, failed):
    ok = not failed
    for N, P in failed:
        print(f"FAIL: run N={N} P={P} failed")
    ref = {}
    if (case_dir / "reference.csv").exists():
        ref = {(int(r["N"]), int(r["P"])): r for r in csv.DictReader(open(case_dir / "reference.csv"))}
    for r in rows:
        if (r["N"], r["P"]) not in ref:
            print(f"N={r['N']} P={r['P']}: no reference")
            continue
        diff = max(abs(r[k] / float(v) - 1) for k, v in ref[(r["N"], r["P"])].items() if k.startswith("L2_"))
        if diff > RTOL:
            ok = False
            print(f"FAIL: N={r['N']} P={r['P']} differs from the reference by {diff:.1e}")
    print("PASS" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=CASES)
    ap.add_argument("--P", type=int, nargs="+", default=[2, 3], help="polynomial orders (default 2 3)")
    ap.add_argument("--meshes", type=int, nargs="+", default=[4, 8], choices=[2, 4, 8, 16],
                    help="elements per side of the Cube meshes (default 4 8)")
    ap.add_argument("--dt", type=float, help="time step (default per case)")
    ap.add_argument("--t-final", type=float, help="final time (default per case)")
    ap.add_argument("--home", type=Path, default=HERE.parents[1], help="HORSES3D Solver directory")
    ap.add_argument("--check", action="store_true", help=f"fail on a failed run or a difference above {RTOL:g} with reference.csv")
    ap.add_argument("--keep", action="store_true", help="keep the solution files")
    args = ap.parse_args()

    cfg = CASES[args.case]
    dt, t_final = args.dt or cfg["dt"], args.t_final or cfg["t_final"]
    case_dir = HERE / args.case
    binary = args.home.resolve() / "bin" / f"horses3d.{cfg['solver']}"
    if not binary.exists():
        sys.exit(f"{binary} not found: build HORSES3D first")
    build(case_dir, cfg["solver"], args.home.resolve())
    build_id = fingerprint(binary, case_dir / "SETUP" / "ProblemFile.f90", case_dir / "control_template.control")

    header = ["build", "N", "P", "dt", "t_final", "NDOF", "volume", "L2_total"] + [f"L2_{v}" for v in cfg["variables"]]
    errors_csv = case_dir / "errors.csv"
    rows = []
    if errors_csv.exists():
        with open(errors_csv) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == header:
                rows = [{k: v if k == "build" else int(v) if k in ("N", "P", "NDOF") else float(v) for k, v in r.items()}
                        for r in reader]
    # Only the mesh may differ within a convergence series
    def same(r):
        return (r["build"], r["dt"], r["t_final"]) == (build_id, dt, t_final)

    failed = []
    for P in args.P:
        for N in args.meshes:
            if any(same(r) and (r["N"], r["P"]) == (N, P) for r in rows):
                continue
            start = time.time()
            result = run(case_dir, binary, N, P, dt, t_final, args.keep)
            if result is None:
                failed.append((N, P))
                print(f"{args.case} N={N} P={P}: FAILED, see {case_dir}/CONTROL/N{N}_P{P}.log", flush=True)
                continue
            rows.append(dict(zip(header, [build_id, N, P, dt, t_final] + result)))
            with open(errors_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(rows)
            print(f"{args.case} N={N} P={P}: L2 = {rows[-1]['L2_total']:.4e} ({time.time() - start:.0f} s)", flush=True)

    rows = [r for r in rows if same(r) and r["P"] in args.P and r["N"] in args.meshes]
    report(args.case, rows, args.P, dt, t_final)
    if args.check and not check(case_dir, rows, failed):
        sys.exit(1)


if __name__ == "__main__":
    main()
