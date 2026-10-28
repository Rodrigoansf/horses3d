#!/usr/bin/env python3
"""
Write <case>/SETUP/ProblemFile.f90 for an MMS case.

    python3 generate.py NS|iNS|MU

Each case defines a manufactured solution, periodic on [-1,1]^3, and the
equations the solver discretises written as residuals. The residual of the
manufactured solution is the source term. The expressions are emitted as
they are, without symbolic simplification. The physical parameters are read
from the solver at run time, except that NS assumes the default Sutherland
constants and MU per-fluid sound speeds ("fluid 1/2 sound speed square").
"""
import sys
from pathlib import Path
import sympy as sp

t, x, y, z = sp.symbols("t x y z")
pi = sp.pi


def grad(f):
    return sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)


def div(fx, fy, fz):
    return sp.diff(fx, x) + sp.diff(fy, y) + sp.diff(fz, z)


def lap(f):
    return div(*grad(f))


def navier_stokes():
    """Compressible NS with Sutherland's law, which the solver always applies."""
    gamma, eta, kappa, gammaM2, s_suth = sp.symbols("gamma eta kappa gammaM2 s_suth")

    rho = sp.Integer(1)
    u = sp.sin(pi*x)*sp.cos(pi*y)*sp.cos(pi*z)*sp.sin(t)
    v = sp.cos(pi*x)*sp.sin(pi*y)*sp.cos(pi*z)*sp.sin(t)
    w = -2*sp.cos(pi*x)*sp.cos(pi*y)*sp.sin(pi*z)*sp.sin(t)
    p = 1 + 0.1*sp.sin(pi*x)*sp.sin(pi*y)*sp.sin(pi*z)*sp.cos(t)
    E = p/(gamma - 1) + sp.Rational(1, 2)*rho*(u**2 + v**2 + w**2)

    # mu(T) = SutherlandsLaw(T)/Re with the solver's temperature T = gamma M^2 p/rho
    T = gammaM2*p/rho
    suth = (1 + s_suth)/(T + s_suth)*T**sp.Rational(3, 2)
    mu, k = eta*suth, kappa*suth

    ux, uy, uz = grad(u)
    vx, vy, vz = grad(v)
    wx, wy, wz = grad(w)
    divu = ux + vy + wz
    txx = mu*(2*ux - sp.Rational(2, 3)*divu)
    tyy = mu*(2*vy - sp.Rational(2, 3)*divu)
    tzz = mu*(2*wz - sp.Rational(2, 3)*divu)
    txy, txz, tyz = mu*(uy + vx), mu*(uz + wx), mu*(vz + wy)
    # Heat flux with T = p/rho: the missing gamma M^2 is absorbed in kappa below
    qx, qy, qz = (-k*d for d in grad(p/rho))

    source = [
        sp.diff(rho, t) + div(rho*u, rho*v, rho*w),
        sp.diff(rho*u, t) + div(rho*u*u + p, rho*u*v, rho*u*w) - div(txx, txy, txz),
        sp.diff(rho*v, t) + div(rho*v*u, rho*v*v + p, rho*v*w) - div(txy, tyy, tyz),
        sp.diff(rho*w, t) + div(rho*w*u, rho*w*v, rho*w*w + p) - div(txz, tyz, tzz),
        sp.diff(E, t) + div((E + p)*u, (E + p)*v, (E + p)*w)
        - div(txx*u + txy*v + txz*w, txy*u + tyy*v + tyz*w, txz*u + tyz*v + tzz*w)
        + div(qx, qy, qz),
    ]
    exact = [rho, rho*u, rho*v, rho*w, E]
    parameters = """\
         real(kind=RP) :: gamma, eta, kappa, gammaM2, s_suth
         gamma   = thermodynamics_ % gamma
         eta     = dimensionless_ % mu
         kappa   = dimensionless_ % mu * gamma / ((gamma - 1.0_RP) * dimensionless_ % Pr)
         gammaM2 = dimensionless_ % gammaM2
         ! Default Sutherland constants (PhysicsStorage_NS): S = 110.4 K, TRef = refValues % T
         s_suth  = 198.6_RP*5.0_RP/9.0_RP / refValues_ % T"""
    return "NAVIERSTOKES", ["IRHO", "IRHOU", "IRHOV", "IRHOW", "IRHOE"], exact, source, parameters


def incompressible():
    """Incompressible NS with artificial compressibility: dp/dt + rho0 c0^2 div(u) = 0."""
    mu, c0_sq = sp.symbols("mu c0_sq")

    rho = sp.Integer(1)
    u = sp.sin(pi*x)*sp.cos(pi*y)*sp.cos(pi*z)*sp.sin(t)
    v = sp.cos(pi*x)*sp.sin(pi*y)*sp.cos(pi*z)*sp.sin(t)
    w = -2*sp.cos(pi*x)*sp.cos(pi*y)*sp.sin(pi*z)*sp.sin(t)
    p = 1 + 0.1*sp.sin(pi*x)*sp.sin(pi*y)*sp.sin(pi*z)*sp.cos(t)

    ux, uy, uz = grad(u)
    vx, vy, vz = grad(v)
    wx, wy, wz = grad(w)
    txx, tyy, tzz = mu*2*ux, mu*2*vy, mu*2*wz
    txy, txz, tyz = mu*(uy + vx), mu*(uz + wx), mu*(vz + wy)

    source = [
        sp.diff(rho, t) + div(rho*u, rho*v, rho*w),
        sp.diff(rho*u, t) + div(rho*u*u + p, rho*u*v, rho*u*w) - div(txx, txy, txz),
        sp.diff(rho*v, t) + div(rho*v*u, rho*v*v + p, rho*v*w) - div(txy, tyy, tyz),
        sp.diff(rho*w, t) + div(rho*w*u, rho*w*v, rho*w*w + p) - div(txz, tyz, tzz),
        sp.diff(p, t) + c0_sq*(ux + vy + wz),
    ]
    exact = [rho, rho*u, rho*v, rho*w, p]
    parameters = """\
         real(kind=RP) :: mu, c0_sq
         mu    = dimensionless_ % mu(1)
         c0_sq = thermodynamics_ % rho0c02"""
    return "INCNS", ["INSRHO", "INSRHOU", "INSRHOV", "INSRHOW", "INSP"], exact, source, parameters


def multiphase():
    """Cahn-Hilliard + incompressible NS with artificial compressibility, two fluids."""
    rho_1, rho_2, eta_1, eta_2, cs_1, cs_2 = sp.symbols("rho_1 rho_2 eta_1 eta_2 cs_1 cs_2")
    M0, sigma, eps = sp.symbols("M0 sigma eps")

    # cos(t) so the field starts at full amplitude: the explicit time step is
    # limited by Cahn-Hilliard (~h^4) and the runs only reach t ~ 1e-4.
    c = sp.Rational(1, 2)*(1 + sp.cos(pi*x)*sp.cos(pi*y)*sp.cos(pi*z)*sp.cos(t))
    u = sp.sin(pi*x)*sp.cos(pi*y)*sp.cos(pi*z)*sp.cos(t)
    v = sp.cos(pi*x)*sp.sin(pi*y)*sp.cos(pi*z)*sp.cos(t)
    w = -2*sp.cos(pi*x)*sp.cos(pi*y)*sp.sin(pi*z)*sp.cos(t)
    p = sp.sin(pi*x)*sp.sin(pi*y)*sp.sin(pi*z)*sp.sin(t)

    # Mixture density, viscosity and sound speed as in the solver
    rho = c*rho_1 + (1 - c)*rho_2
    sqrho = sp.sqrt(rho)
    eta = c*eta_1 + (1 - c)*eta_2
    cs = c*cs_1 + (1 - c)*cs_2
    # Chemical potential
    mu_c = (sigma/eps)*(12*c**2*(2*c - 2) + 24*c*(1 - c)**2) - sp.Rational(3, 2)*sigma*eps*lap(c)

    ux, uy, uz = grad(u)
    vx, vy, vz = grad(v)
    wx, wy, wz = grad(w)
    tx = div(eta*2*ux, eta*(uy + vx), eta*(uz + wx))
    ty = div(eta*(vx + uy), eta*2*vy, eta*(vz + wy))
    tz = div(eta*(wx + uz), eta*(wy + vz), eta*2*wz)

    # Skew-symmetric convection: half divergence form, half advective form,
    # as split in the solver (Physics_MU.f90)
    half = sp.Rational(1, 2)
    source = [
        sp.diff(c, t) + div(c*u, c*v, c*w) - M0*lap(mu_c),
        sqrho*sp.diff(sqrho*u, t) + half*div(rho*u*u, rho*v*u, rho*w*u)
        + half*rho*(u*ux + v*uy + w*uz) + c*sp.diff(mu_c, x) + sp.diff(p, x) - tx,
        sqrho*sp.diff(sqrho*v, t) + half*div(rho*u*v, rho*v*v, rho*w*v)
        + half*rho*(u*vx + v*vy + w*vz) + c*sp.diff(mu_c, y) + sp.diff(p, y) - ty,
        sqrho*sp.diff(sqrho*w, t) + half*div(rho*u*w, rho*v*w, rho*w*w)
        + half*rho*(u*wx + v*wy + w*wz) + c*sp.diff(mu_c, z) + sp.diff(p, z) - tz,
        sp.diff(p, t) + cs**2*rho*(ux + vy + wz),
    ]
    exact = [c, sqrho*u, sqrho*v, sqrho*w, p]
    parameters = """\
         real(kind=RP) :: rho_1, rho_2, eta_1, eta_2, cs_1, cs_2, M0, sigma, eps
         rho_1 = dimensionless_ % rho(1)
         rho_2 = dimensionless_ % rho(2)
         eta_1 = dimensionless_ % mu(1)
         eta_2 = dimensionless_ % mu(2)
         cs_1  = sqrt(dimensionless_ % invMa2(1) / dimensionless_ % rho(1))
         cs_2  = sqrt(dimensionless_ % invMa2(2) / dimensionless_ % rho(2))
         M0    = multiphase_ % M0
         sigma = multiphase_ % sigma
         eps   = multiphase_ % eps"""
    return "MULTIPHASE", ["IMC", "IMSQRHOU", "IMSQRHOV", "IMSQRHOW", "IMP"], exact, source, parameters


CASES = {"NS": navier_stokes, "iNS": incompressible, "MU": multiphase}


def fortran(lhs, expr):
    expr = sp.sympify(expr).subs({pi: sp.Symbol("PI"), x: sp.Symbol("x(1)"),
                                  y: sp.Symbol("x(2)"), z: sp.Symbol("x(3)")})
    code = sp.fcode(expr, assign_to=lhs, source_format="free", standard=95, contract=False)
    return "\n".join("         " + line for line in code.splitlines())


def generate(case):
    macro, names, exact, source, parameters = CASES[case]()
    here = Path(__file__).resolve().parent
    pf = (here / "ProblemFile.f90.in").read_text()
    pf = pf.replace("@SOLVER@", case).replace("@MACRO@", macro).replace("@PARAMETERS@", parameters)
    pf = pf.replace("@EXACT@", "\n".join(fortran(f"Q({n})", e) for n, e in zip(names, exact)))
    pf = pf.replace("@SOURCE@", "\n".join(fortran(f"S({n})", e) for n, e in zip(names, source)))
    out = here / case / "SETUP" / "ProblemFile.f90"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pf)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        sys.exit(f"usage: {sys.argv[0]} {'|'.join(CASES)}")
    print(f"Written {generate(sys.argv[1])}")
