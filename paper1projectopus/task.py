# task:
# Consider a two-level system driven by a monochromatic field described by
# H(t)= epsilon0 |0><0| + epsilon1 |1><1| + 2 Omega cos(omega t)(|0><1| + |1><0|)
# Simulate the Bloch-vector trajectory for different values of ω.
# Describe the most relevant geometric features and explain their physical origin.

"""
Bloch-vector trajectories of a two-level system under a linearly polarised
monochromatic drive, solved exactly (no rotating-wave approximation).

Conventions (hbar = 1)
----------------------
    omega0 = epsilon1 - epsilon0          (bare transition frequency)
    H(t)   = (epsilon0 + epsilon1)/2 * 1 + 1/2 B(t) . sigma
    B(t)   = (4 Omega cos(omega t), 0, epsilon0 - epsilon1)
    r      = (<sx>, <sy>, <sz>),   z = +1 <-> |0>,   z = -1 <-> |1>
The identity term only adds a global phase, so r obeys the classical
precession equation  dr/dt = B(t) x r : |r| = 1 forever (pure state,
unitary evolution) and the motion is a rotation on the unit sphere.

The linear drive 4 Omega cos(omega t) x_hat is the sum of two circular
components of amplitude 2 Omega rotating at +omega and -omega about z.
The free spin precesses clockwise (seen from +z) at omega0, so the
clockwise ("co-rotating") component is quasi-resonant and the other one
("counter-rotating") is detuned by omega + omega0.

Geometric features (see the generated figures)
----------------------------------------------
1. Lab frame - spherical spiral ("ball of yarn").  The vector precesses
   fast around z at ~omega while its polar angle changes slowly (nutation /
   Rabi oscillation).  Projected on XY this gives a spiral that winds out
   to radius sqrt(1 - z^2) and back in (both hemispheres overlap, so a
   rosette appears); XZ and YZ show fast oscillations inside the envelope
   +-sqrt(1 - z^2).  Number of turns per Rabi cycle ~ omega / Omega_R.

2. Rotating frame (rotating with the drive at omega) - a circle.  Within
   the RWA the co-rotating field is static, B_eff = (2 Omega, 0, -Delta),
   Delta = omega0 - omega, so r precesses on a cone around the tilted axis
   n = B_eff/|B_eff| with generalised Rabi frequency
   Omega_R = sqrt(4 Omega^2 + Delta^2).  Starting from the pole the circle
   passes through the north pole; its lowest point is
   z_min = (Delta^2 - 4 Omega^2) / Omega_R^2.
     - resonance (Delta = 0): n lies on the equator, the circle is a great
       circle through both poles -> complete population inversion;
     - detuned: the axis tilts towards the pole, the circle shrinks to a
       polar cap -> partial inversion, faster but smaller oscillation.

3. Beyond the RWA - ripples and Bloch-Siegert shift.  The counter-rotating
   component appears in the rotating frame as a field turning at 2 omega.
   (a) It decorates the RWA circle with fast trochoid-like ripples (cusps
       or small loops, 2 omega micromotion, amplitude ~ Omega/(omega +
       omega0)); they grow as omega moves away from omega0 because the
       RWA circle shrinks while the ripple amplitude does not.  In the
       lab frame the same effect makes z(t) a staircase: dz/dt =
       4 Omega cos(omega t) y, so the inversion only advances while the
       linear field is large and pauses at its zeros (twice per period);
   (b) at second order it renormalises the effective splitting, so the
       true resonance (minimum effective Rabi frequency, i.e. symmetric
       avoided crossing of the Floquet quasi-energies) occurs at
       omega_res ~ omega0 + Omega^2/omega0 (Bloch-Siegert shift), not at
       omega0.  Driving exactly at omega0 leaves a residual detuning: the
       rotating-frame circle is slightly tilted and the exact trajectory
       drifts away from the RWA one cycle after cycle.
   The resonance position is obtained non-perturbatively here from the
   Floquet quasi-energies of the one-period propagator (minimum gap).
   Note: the depth of inversion reached from |0> is not a clean resonance
   marker at this order, because the micromotion also tilts the effective
   starting point by ~Omega/omega in a way that depends on the drive
   phase at t = 0; the effective Rabi frequency is phase independent.

4. Far off resonance.  For omega << omega0 the field direction swings
   slowly in the XZ plane and r makes small precession circles around the
   instantaneous field (quasi-adiabatic following), so the trajectory
   stays near the pole and is stretched along x.  For omega >> omega0 the
   drive averages out: r only wobbles slightly around the pole while
   precessing at a (Bessel-renormalised) frequency.  Even multiples
   omega ~ omega0/2 are not resonant: the cos-type sigma_x coupling only
   allows odd-photon (omega0 ~ (2k+1) omega) resonances.
"""

import argparse
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))

SX = np.array([[0, 1], [1, 0]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)

RTOL = 1e-10
ATOL = 1e-12


# ---------------------------------------------------------------- physics
def hamiltonian(t, eps0, eps1, Omega, omega):
    return eps0 * P0 + eps1 * P1 + 2.0 * Omega * np.cos(omega * t) * SX


def evolve_states(psi0, t_eval, eps0, eps1, Omega, omega):
    """Integrate i d|psi>/dt = H(t)|psi> exactly; returns array (len(t), 2)."""
    def rhs(t, psi):
        return -1j * (hamiltonian(t, eps0, eps1, Omega, omega) @ psi)

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), np.asarray(psi0, dtype=complex),
                    t_eval=t_eval, method="DOP853", rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def bloch_vector(psi):
    """Bloch vector(s) of state(s) psi with shape (..., 2)."""
    c0, c1 = psi[..., 0], psi[..., 1]
    rho01 = np.conj(c0) * c1
    return np.stack([2 * rho01.real, 2 * rho01.imag,
                     np.abs(c0) ** 2 - np.abs(c1) ** 2], axis=-1)


def bloch_field(t, eps0, eps1, Omega, omega):
    return np.array([4.0 * Omega * np.cos(omega * t), 0.0, eps0 - eps1])


def bloch_equation_trajectory(r0, t_eval, eps0, eps1, Omega, omega):
    """Independent integration of the classical equation dr/dt = B(t) x r."""
    def rhs(t, r):
        return np.cross(bloch_field(t, eps0, eps1, Omega, omega), r)

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), np.asarray(r0, dtype=float),
                    t_eval=t_eval, method="DOP853", rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def to_rotating_frame(r, t, omega):
    """Coordinates in the frame co-rotating with the drive (clockwise about z)."""
    w = (r[:, 0] + 1j * r[:, 1]) * np.exp(1j * omega * t)
    return np.column_stack([w.real, w.imag, r[:, 2]])


def rwa_trajectory_rotating(r0, t, eps0, eps1, Omega, omega):
    """Analytic RWA solution in the rotating frame: rotation about B_eff."""
    delta = (eps1 - eps0) - omega
    b_eff = np.array([2.0 * Omega, 0.0, -delta])
    omega_r = np.linalg.norm(b_eff)
    r0 = np.asarray(r0, dtype=float)
    if omega_r == 0.0:
        return np.tile(r0, (len(t), 1))
    n = b_eff / omega_r
    c = np.cos(omega_r * t)[:, None]
    s = np.sin(omega_r * t)[:, None]
    return r0 * c + np.cross(n, r0) * s + n * np.dot(n, r0) * (1 - c)


def rwa_summary(eps0, eps1, Omega, omega):
    delta = (eps1 - eps0) - omega
    omega_r = np.hypot(2.0 * Omega, delta)
    z_min = (delta ** 2 - 4.0 * Omega ** 2) / omega_r ** 2
    return delta, omega_r, z_min


def one_period_propagator(eps0, eps1, Omega, omega):
    """Monodromy matrix U(T, 0), built column by column from the basis states."""
    t_eval = np.array([0.0, 2.0 * np.pi / omega])
    return np.column_stack([evolve_states(e, t_eval, eps0, eps1, Omega, omega)[-1]
                            for e in np.eye(2)])


def floquet_quasienergies(eps0, eps1, Omega, omega):
    period = 2.0 * np.pi / omega
    U = one_period_propagator(eps0, eps1, Omega, omega)
    return -np.angle(np.linalg.eigvals(U)) / period


def stroboscopic_max_transfer(eps0, eps1, Omega, omega):
    """sup_n |<1|U(T)^n|0>|^2, from U(T) = e^{i phi}(cos a - i sin a n.sigma).

    Sampled at t = nT the Bloch vector rotates about the Floquet axis n, so the
    best transfer from the north pole is 1 - n_z^2.  Written with matrix
    elements this needs no eigenvectors (ill-conditioned near crossings).
    """
    U = one_period_propagator(eps0, eps1, Omega, omega)
    off = abs(U[1, 0]) ** 2
    diag = abs(U[1, 1] - U[0, 0]) ** 2 / 4.0
    return off / (off + diag) if off + diag > 0 else 0.0


def floquet_gap(eps0, eps1, Omega, omega):
    """Quasi-energy splitting folded into the first Brillouin zone [0, omega/2]."""
    e = floquet_quasienergies(eps0, eps1, Omega, omega)
    d = (e[0] - e[1]) % omega
    return min(d, omega - d)


def find_resonance(eps0, eps1, Omega):
    """Exact (non-perturbative) one-photon resonance: minimum Floquet gap."""
    omega0 = eps1 - eps0
    window = 2.0 * Omega
    res = minimize_scalar(lambda w: floquet_gap(eps0, eps1, Omega, w),
                          bounds=(omega0 - window, omega0 + window),
                          method="bounded", options={"xatol": 1e-8})
    return res.x, res.fun


def simulate(omegas, eps0, eps1, Omega, t_max, dt, psi0=(1.0, 0.0)):
    t = np.arange(0.0, t_max + dt / 2, dt)
    r0 = bloch_vector(np.asarray(psi0, dtype=complex))
    results = []
    for omega in omegas:
        r_lab = bloch_vector(evolve_states(psi0, t, eps0, eps1, Omega, omega))
        results.append({
            "omega": omega,
            "t": t,
            "lab": r_lab,
            "rot": to_rotating_frame(r_lab, t, omega),
            "rwa_rot": rwa_trajectory_rotating(r0, t, eps0, eps1, Omega, omega),
        })
    return results


# ---------------------------------------------------------------- plots
def _label(res):
    return r"$\omega$ = %.5g" % res["omega"]


def _draw_sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="lightgray", linewidth=0.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def plot_3d(results, frame, title, filename, plt, with_rwa=False):
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    for k, res in enumerate(results):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        _draw_sphere(ax)
        r = res[frame]
        ax.plot(r[:, 0], r[:, 1], r[:, 2], lw=0.8, color="tab:blue", label="exact")
        if with_rwa:
            q = res["rwa_rot"]
            ax.plot(q[:, 0], q[:, 1], q[:, 2], "--", lw=1.2, color="tab:red", label="RWA")
        ax.scatter(*r[0], color="k", s=20, label="start")
        ax.set_title(_label(res))
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    return fig


def plot_projections(results, frame, title, filename, plt, with_rwa=False):
    from matplotlib.collections import LineCollection

    planes = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n), squeeze=False,
                             constrained_layout=True)
    for i, res in enumerate(results):
        r, t = res[frame], res["t"]
        for j, (name, a, b) in enumerate(planes):
            ax = axes[i, j]
            phi = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(phi), np.sin(phi), color="lightgray", lw=0.8)
            pts = np.column_stack([r[:, a], r[:, b]]).reshape(-1, 1, 2)
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                                cmap="viridis", linewidth=0.8)
            lc.set_array(t[:-1])
            ax.add_collection(lc)
            if with_rwa:
                q = res["rwa_rot"]
                ax.plot(q[:, a], q[:, b], "--", color="tab:red", lw=1.0, label="RWA")
            ax.plot(r[0, a], r[0, b], "ko", ms=4)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.set_xlabel(name[0].lower())
            ax.set_ylabel(name[1].lower())
            ax.set_title("%s plane, %s" % (name, _label(res)))
            if with_rwa and i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(lc, ax=axes[i, :].tolist(), label="t", shrink=0.9)
    fig.suptitle(title)
    fig.savefig(filename, dpi=110)
    return fig


def plot_z_vs_time(results, title, filename, plt):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True, squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        ax.plot(res["t"], res["lab"][:, 2], lw=0.9, label="exact")
        ax.plot(res["t"], res["rwa_rot"][:, 2], "--", lw=1.0, label="RWA")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("z")
        ax.set_title(_label(res), fontsize=9)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[-1, 0].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    return fig


def plot_bloch_siegert(eps0, eps1, Omega, omega_res, filename, plt):
    omega0 = eps1 - eps0
    w = np.linspace(omega0 - 2 * Omega, omega0 + 2 * Omega, 161)
    gap = [floquet_gap(eps0, eps1, Omega, wi) for wi in w]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(w, gap, label="exact Floquet gap")
    ax.plot(w, np.hypot(2 * Omega, w - omega0), "--",
            label=r"RWA $\sqrt{4\Omega^2+(\omega-\omega_0)^2}$")
    ax.axvline(omega0, color="gray", ls=":", label=r"$\omega_0$")
    ax.axvline(omega0 + Omega ** 2 / omega0, color="tab:red", ls=":",
               label=r"$\omega_0+\Omega^2/\omega_0$ (Bloch-Siegert)")
    ax.plot(omega_res, floquet_gap(eps0, eps1, Omega, omega_res), "k*", ms=10,
            label=r"exact resonance %.5f" % omega_res)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("effective Rabi frequency (quasi-energy gap)")
    ax.set_title(r"Resonance position, $\Omega$ = %g, $\omega_0$ = %g" % (Omega, omega0))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    return fig


def print_summary(results, eps0, eps1, Omega):
    print("\n omega    Delta    Omega_R(RWA)  z_min(RWA)  z_min(exact)  "
          "max|r_rot - r_RWA|  max||r|-1|")
    for res in results:
        delta, omega_r, z_min = rwa_summary(eps0, eps1, Omega, res["omega"])
        dev = np.max(np.linalg.norm(res["rot"] - res["rwa_rot"], axis=1))
        norm_err = np.max(np.abs(np.linalg.norm(res["lab"], axis=1) - 1))
        print("%7.4f  %7.4f  %10.4f   %10.4f  %11.4f   %14.4f   %10.1e"
              % (res["omega"], delta, omega_r, z_min, res["lab"][:, 2].min(),
                 dev, norm_err))


# ---------------------------------------------------------------- tests
def test_bloch_vector_stays_on_unit_sphere():
    t = np.linspace(0, 50, 2001)
    for omega in (0.5, 1.0, 2.0):
        r = bloch_vector(evolve_states([1, 0], t, 0.0, 1.0, 0.3, omega))
        assert np.max(np.abs(np.linalg.norm(r, axis=1) - 1)) < 1e-8


def test_free_precession():
    # Omega = 0, start on the equator: x = cos(omega0 t), y = -sin(omega0 t), z = 0
    t = np.linspace(0, 20, 501)
    omega0 = 1.3
    r = bloch_vector(evolve_states(np.array([1, 1]) / np.sqrt(2), t, 0.2, 0.2 + omega0, 0.0, 1.0))
    expected = np.column_stack([np.cos(omega0 * t), -np.sin(omega0 * t), 0 * t])
    assert np.allclose(r, expected, atol=1e-8)


def test_schrodinger_vs_bloch_equation():
    t = np.linspace(0, 40, 1001)
    psi0 = np.array([np.cos(0.4), np.exp(0.7j) * np.sin(0.4)])
    for omega in (0.5, 1.0, 3.0):
        r_q = bloch_vector(evolve_states(psi0, t, 0.0, 1.0, 0.2, omega))
        r_c = bloch_equation_trajectory(bloch_vector(psi0), t, 0.0, 1.0, 0.2, omega)
        assert np.allclose(r_q, r_c, atol=1e-7)


def test_global_energy_shift_invariance():
    t = np.linspace(0, 30, 601)
    r1 = bloch_vector(evolve_states([1, 0], t, 0.0, 1.0, 0.1, 0.9))
    r2 = bloch_vector(evolve_states([1, 0], t, 5.0, 6.0, 0.1, 0.9))
    assert np.allclose(r1, r2, atol=1e-7)


def test_rwa_limit():
    # Deviation from the RWA over one Rabi period scales linearly with Omega/omega0
    errors = []
    for Omega in (0.02, 0.01):
        res = simulate([1.0], 0.0, 1.0, Omega, np.pi / Omega, 0.05)[0]
        errors.append(np.max(np.linalg.norm(res["rot"] - res["rwa_rot"], axis=1)))
    assert errors[0] < 0.1
    assert 0.35 < errors[1] / errors[0] < 0.65


def test_bloch_siegert_shift():
    eps0, eps1, Omega = 0.0, 1.0, 0.1
    omega_res, gap = find_resonance(eps0, eps1, Omega)
    shift = Omega ** 2 / (eps1 - eps0)
    assert abs(omega_res - (1.0 + shift)) < 0.1 * shift
    assert abs(gap - 2 * Omega) < 0.02 * 2 * Omega
    # the Floquet gap is the Rabi frequency actually seen by the exact dynamics:
    # stroboscopic samples z(nT) are exactly a + b cos(gap nT) + c sin(gap nT)
    period = 2 * np.pi / omega_res
    t = period * np.arange(60)
    z = bloch_vector(evolve_states([1, 0], t, eps0, eps1, Omega, omega_res))[:, 2]
    basis = np.column_stack([np.ones_like(t), np.cos(gap * t), np.sin(gap * t)])
    coef = np.linalg.lstsq(basis, z, rcond=None)[0]
    assert np.max(np.abs(basis @ coef - z)) < 1e-6
    assert np.hypot(coef[1], coef[2]) > 0.9


def test_micromotion_amplitude():
    # In the rotating frame the counter-rotating field drives circles of radius
    # rho = 2 Omega / (omega + omega0); the exact curve starts on the RWA one,
    # so its maximum distance from it is the diameter 2 rho (short window, so
    # that the O(Omega^2) Bloch-Siegert drift is negligible).
    Omega = 0.01
    for omega in (0.8, 1.0, 1.5):
        res = simulate([omega], 0.0, 1.0, Omega, 2 * np.pi, 0.002)[0]
        dev = np.max(np.linalg.norm(res["rot"] - res["rwa_rot"], axis=1))
        expected = 4 * Omega / (omega + 1.0)
        assert abs(dev - expected) < 0.02 * expected


def test_stroboscopic_max_transfer_formula():
    U_args = (0.0, 1.0, 0.1, 0.9)
    U = one_period_propagator(*U_args)
    psi = np.array([1.0, 0.0], dtype=complex)
    best = 0.0
    for _ in range(20000):
        psi = U @ psi
        best = max(best, abs(psi[1]) ** 2)
    assert abs(stroboscopic_max_transfer(*U_args) - best) < 1e-4


def test_three_photon_resonance():
    # Odd k-photon resonances are avoided crossings of the Floquet levels with
    # full population transfer.  Second-order level shifts (both one-photon
    # couplings are off resonant) give omega_3 ~ omega0/3 + 3 Omega^2/(2 omega0).
    eps0, eps1, Omega = 0.0, 1.0, 0.1
    gap = minimize_scalar(lambda w: floquet_gap(eps0, eps1, Omega, w),
                          bounds=(0.33, 0.37), method="bounded",
                          options={"xatol": 1e-9})
    shift = 1.5 * Omega ** 2
    assert abs(gap.x - (1 / 3 + shift)) < 0.1 * shift
    assert 1e-3 < gap.fun < 1e-2
    peak = minimize_scalar(lambda w: -stroboscopic_max_transfer(eps0, eps1, Omega, w),
                           bounds=(gap.x - 3e-3, gap.x + 3e-3), method="bounded",
                           options={"xatol": 1e-9})
    assert -peak.fun > 0.999


def test_no_two_photon_resonance():
    # Generalised parity sigma_z H(t + T/2) sigma_z = H(t) forbids even
    # k-photon couplings: the levels cross exactly (no gap) near
    # omega0/2 + 8 Omega^2/(3 omega0), and no transfer peak appears there.
    eps0, eps1, Omega = 0.0, 1.0, 0.1
    cross = minimize_scalar(lambda w: floquet_gap(eps0, eps1, Omega, w),
                            bounds=(0.45, 0.60), method="bounded",
                            options={"xatol": 1e-10})
    shift = 8 * Omega ** 2 / 3
    assert abs(cross.x - (0.5 + shift)) < 0.1 * shift
    assert cross.fun < 1e-6
    ws = np.append(np.linspace(cross.x - 0.02, cross.x + 0.02, 21), cross.x)
    transfer = [stroboscopic_max_transfer(eps0, eps1, Omega, w) for w in ws]
    assert max(transfer) < 0.5


def run_tests():
    tests = [test_bloch_vector_stays_on_unit_sphere, test_free_precession,
             test_schrodinger_vs_bloch_equation, test_global_energy_shift_invariance,
             test_rwa_limit, test_bloch_siegert_shift, test_micromotion_amplitude,
             test_stroboscopic_max_transfer_formula, test_three_photon_resonance,
             test_no_two_photon_resonance]
    for test in tests:
        test()
        print("[PASS] %s" % test.__name__)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--eps0", type=float, default=0.0)
    parser.add_argument("--eps1", type=float, default=1.0)
    parser.add_argument("--Omega", type=float, default=0.1)
    parser.add_argument("--omegas", type=float, nargs="+", default=None,
                        help="drive frequencies (default: set around omega0 "
                             "including the exact Bloch-Siegert resonance)")
    parser.add_argument("--t-max", type=float, default=None,
                        help="default: one resonant Rabi period pi/Omega")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.no_tests:
        run_tests()

    eps0, eps1, Omega = args.eps0, args.eps1, args.Omega
    omega0 = eps1 - eps0
    omega_res, gap = find_resonance(eps0, eps1, Omega)
    print("\nomega0 = %g, Omega = %g" % (omega0, Omega))
    print("exact resonance (Floquet)   omega_res = %.6f, Rabi freq = %.6f"
          % (omega_res, gap))
    print("Bloch-Siegert estimate      omega0 + Omega^2/omega0 = %.6f"
          % (omega0 + Omega ** 2 / omega0))

    omegas = args.omegas or [0.5 * omega0, 0.9 * omega0, omega0, omega_res,
                             1.1 * omega0, 2.0 * omega0]
    t_max = args.t_max or np.pi / Omega
    results = simulate(omegas, eps0, eps1, Omega, t_max, args.dt)
    print_summary(results, eps0, eps1, Omega)

    sub = r"($\omega_0$ = %g, $\Omega$ = %g, $|\psi_0\rangle=|0\rangle$)" % (omega0, Omega)
    plot_3d(results, "lab", "Bloch trajectories, lab frame " + sub,
            os.path.join(HERE, "bloch_3d_lab.png"), plt)
    plot_projections(results, "lab", "Projections, lab frame " + sub,
                     os.path.join(HERE, "bloch_projections_lab.png"), plt)
    plot_3d(results, "rot", "Bloch trajectories, frame rotating at $\\omega$ " + sub,
            os.path.join(HERE, "bloch_3d_rotating.png"), plt, with_rwa=True)
    plot_projections(results, "rot", "Projections, frame rotating at $\\omega$ " + sub,
                     os.path.join(HERE, "bloch_projections_rotating.png"), plt, with_rwa=True)
    plot_z_vs_time(results, "Population inversion z(t) " + sub,
                   os.path.join(HERE, "bloch_z_vs_time.png"), plt)
    plot_bloch_siegert(eps0, eps1, Omega, omega_res,
                       os.path.join(HERE, "bloch_siegert.png"), plt)
    print("\nFigures saved in %s" % HERE)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()