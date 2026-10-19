# task
# Simulate a two-level quantum system with Hamiltonian H = ω/2 σz + Ω σx. 
# Verify numerically that the Bloch vector remains on the unit sphere during the evolution, and explain why.

# Opus 5.5 notes:
# Review of the previous version of this script (all points below are now implemented).
# 1) Make the check meaningful. For ANY vector psi, |b| = ||psi||^2, so checking |b| = 1 on
#    the Schrodinger solution only re-checks that expm() returned a unitary matrix (the same
#    thing as the state-norm assert). The Bloch equation db/dt = B x b is now integrated
#    independently (solve_ivp) and compared with the Schrodinger result.
# 2) Show the role of unitarity. A non-unitary integrator (explicit Euler) drifts off the
#    sphere (|b| ~ 1.13 at t = 10 with dt = 0.01), while a unitary one (Crank-Nicolson /
#    Cayley) stays on it to machine precision even with a coarse step.
# 3) Verify the precession picture numerically. Writing H = 1/2 B.sigma with
#    B = (2*Omega, 0, omega): the projection b.n (n = B/|B|) and the energy
#    <H> = 1/2 B.b are conserved, and the motion is periodic with period
#    2*pi/|B| = pi/R, R = sqrt((omega/2)^2 + Omega^2).
# 4) Generic initial states. Random pure states are tested besides |0>, and mixed states
#    rho(t) = U rho U^dagger as well: their |b| < 1 is also preserved. The conserved quantity
#    is |b|; purity only fixes its value to 1.
# 5) Code quality. H is diagonalised once (eigh) instead of calling expm at every time;
#    tolerances are explicit and consistent; checks are explicit instead of assert (which
#    python -O disables); computation is separated from printing; the printed table rows do
#    not depend on num_times; the trajectory is plotted on the Bloch sphere.
# 6) Explanation fixed. hbar instead of h; explicit derivation d|b|^2/dt = 2 b.(B x b) = 0;
#    SU(2) -> SO(3) rotation argument; precession frequency |B| = 2R. The omega = 0 case is
#    no longer called "resonant": with omega = energy splitting it means degenerate levels
#    (it is "resonant" only if omega is read as the detuning in a rotating frame).

import os

import numpy as np
from scipy.integrate import solve_ivp

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (SIGMA_X, SIGMA_Y, SIGMA_Z)
IDENTITY = np.eye(2, dtype=complex)

KET_0 = np.array([1.0, 0.0], dtype=complex)

# Exact propagation (eigendecomposition) gives errors ~1e-15.
TOL = 1e-10
# Independent ODE integration of the Bloch equation (rtol = atol = 1e-12).
ODE_TOL = 1e-8
# Minimum drift expected from the explicit Euler demonstration.
EULER_MIN_DRIFT = 1e-3


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def hamiltonian(omega, Omega):
    """H = ω/2 σz + Ω σx  (ħ = 1)."""
    return (omega / 2) * SIGMA_Z + Omega * SIGMA_X


def field_vector(omega, Omega):
    """Effective field B such that H = 1/2 B·σ, i.e. B = (2Ω, 0, ω)."""
    return np.array([2.0 * Omega, 0.0, omega])


def rabi_frequency(omega, Omega):
    """R = sqrt((ω/2)^2 + Ω^2): eigenvalues of H are ±R and |B| = 2R."""
    return np.sqrt((omega / 2) ** 2 + Omega ** 2)


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------

def propagators(H, times):
    """
    U(t) = exp(-iHt) for every t in `times`, from a single diagonalisation
    H = V diag(E) V^†  ->  U(t) = V diag(exp(-iEt)) V^†.
    Returns an array of shape (len(times), 2, 2).
    """
    energies, V = np.linalg.eigh(H)
    phases = np.exp(-1j * np.outer(np.atleast_1d(times), energies))
    return np.einsum("ij,tj,kj->tik", V, phases, V.conj())


def evolve_pure(psi0, H, times):
    """|ψ(t)> = U(t)|ψ(0)>, shape (len(times), 2)."""
    return propagators(H, times) @ psi0


def evolve_mixed(rho0, H, times):
    """ρ(t) = U(t) ρ(0) U(t)^†, shape (len(times), 2, 2)."""
    U = propagators(H, times)
    return U @ rho0 @ U.conj().transpose(0, 2, 1)


def evolve_euler(psi0, H, dt, n_steps):
    """Explicit Euler, ψ_{n+1} = (1 - iH dt) ψ_n: NOT unitary."""
    return _iterate(psi0, IDENTITY - 1j * dt * H, n_steps)


def evolve_cayley(psi0, H, dt, n_steps):
    """Crank-Nicolson / Cayley, ψ_{n+1} = (1 + iH dt/2)^-1 (1 - iH dt/2) ψ_n: unitary."""
    step = np.linalg.solve(IDENTITY + 0.5j * dt * H, IDENTITY - 0.5j * dt * H)
    return _iterate(psi0, step, n_steps)


def _iterate(psi0, step, n_steps):
    states = np.empty((n_steps + 1, 2), dtype=complex)
    states[0] = psi0
    for n in range(n_steps):
        states[n + 1] = step @ states[n]
    return states


def evolve_bloch_ode(b0, B, times):
    """Integrate the Bloch equation db/dt = B x b, independently of the Schrödinger equation."""
    sol = solve_ivp(lambda t, b: np.cross(B, b), (times[0], times[-1]), b0,
                    t_eval=times, method="DOP853", rtol=1e-12, atol=1e-12)
    if not sol.success:
        raise RuntimeError("Bloch ODE integration failed: " + sol.message)
    return sol.y.T


# ---------------------------------------------------------------------------
# Bloch vectors and states
# ---------------------------------------------------------------------------

def bloch_vectors(psis):
    """b = (<σx>, <σy>, <σz>) for one state (shape (2,)) or many (shape (n, 2))."""
    psis = np.atleast_2d(psis)
    return np.stack([np.real(np.einsum("ti,ij,tj->t", psis.conj(), s, psis))
                     for s in PAULI], axis=-1)


def bloch_vectors_rho(rhos):
    """b_i = Tr(ρ σ_i) for one density matrix or many."""
    rhos = np.asarray(rhos)
    return np.stack([np.real(np.einsum("...ij,ji->...", rhos, s)) for s in PAULI], axis=-1)


def density_matrix(b):
    """ρ = (I + b·σ)/2."""
    return 0.5 * (IDENTITY + sum(bi * s for bi, s in zip(b, PAULI)))


def random_pure_state(rng):
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    return psi / np.linalg.norm(psi)


def random_mixed_bloch_vector(rng):
    direction = rng.normal(size=3)
    return rng.uniform(0.1, 0.9) * direction / np.linalg.norm(direction)


# ---------------------------------------------------------------------------
# Analyses (computation only, no printing)
# ---------------------------------------------------------------------------

def analyse_pure(omega, Omega, psi0, times):
    """Evolve a pure state and return (metrics, trajectory). Metrics are max abs deviations."""
    H = hamiltonian(omega, Omega)
    B = field_vector(omega, Omega)
    R = rabi_frequency(omega, Omega)

    psis = evolve_pure(psi0, H, times)
    b = bloch_vectors(psis)
    b_norm = np.linalg.norm(b, axis=1)
    energy = np.real(np.einsum("ti,ij,tj->t", psis.conj(), H, psis))

    metrics = {
        "state norm ||psi|| = 1": np.max(np.abs(np.linalg.norm(psis, axis=1) - 1.0)),
        "Bloch norm |b| = 1": np.max(np.abs(b_norm - 1.0)),
        "Schrodinger vs Bloch ODE": np.max(np.abs(b - evolve_bloch_ode(b[0], B, times))),
        "energy <H> conserved": np.max(np.abs(energy - energy[0])),
        "<H> = B.b / 2": np.max(np.abs(energy - 0.5 * b @ B)),
    }
    if R > 0:
        n = B / np.linalg.norm(B)
        b_period = bloch_vectors(evolve_pure(psi0, H, [np.pi / R]))[0]
        metrics["b.n conserved"] = np.max(np.abs(b @ n - b[0] @ n))
        metrics["periodic, b(pi/R) = b(0)"] = np.max(np.abs(b_period - b[0]))
    return metrics, {"times": times, "b": b, "b_norm": b_norm}


def analyse_mixed(omega, Omega, b0, times):
    """Evolve ρ0 = (I + b0·σ)/2 with |b0| < 1; |b| must stay equal to |b0|."""
    H = hamiltonian(omega, Omega)
    rhos = evolve_mixed(density_matrix(b0), H, times)
    b = bloch_vectors_rho(rhos)
    b_norm = np.linalg.norm(b, axis=1)
    purity = np.real(np.einsum("tij,tji->t", rhos, rhos))
    return {
        "mixed: |b(t)| = |b(0)| < 1": np.max(np.abs(b_norm - np.linalg.norm(b0))),
        "mixed: Tr(rho^2) = (1+|b|^2)/2": np.max(np.abs(purity - 0.5 * (1 + b_norm ** 2))),
    }


def analyse_integrators(omega, Omega, psi0, t_max, dt_euler=0.01, dt_cayley=0.1):
    """Bloch-norm deviation for a non-unitary (Euler) and a unitary (Cayley) integrator."""
    H = hamiltonian(omega, Omega)
    n_euler = int(round(t_max / dt_euler))
    n_cayley = int(round(t_max / dt_cayley))
    b_euler = np.linalg.norm(bloch_vectors(evolve_euler(psi0, H, dt_euler, n_euler)), axis=1)
    b_cayley = np.linalg.norm(bloch_vectors(evolve_cayley(psi0, H, dt_cayley, n_cayley)), axis=1)
    return {
        "euler": (np.arange(n_euler + 1) * dt_euler, b_euler, dt_euler),
        "cayley": (np.arange(n_cayley + 1) * dt_cayley, b_cayley, dt_cayley),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class Report:
    """Collects PASS/FAIL results (explicit checks, unaffected by python -O)."""

    def __init__(self):
        self.failures = []

    def at_most(self, name, value, tol):
        self._record(name, value <= tol, "{:.2e} <= {:.0e}".format(value, tol))

    def at_least(self, name, value, threshold):
        self._record(name, value >= threshold, "{:.2e} >= {:.0e}".format(value, threshold))

    def _record(self, name, ok, detail):
        print("  [{}] {:<34} {}".format("PASS" if ok else "FAIL", name, detail))
        if not ok:
            self.failures.append(name)

    @property
    def passed(self):
        return not self.failures


def print_norm_table(trajectory, n_rows=6):
    times, b_norm = trajectory["times"], trajectory["b_norm"]
    print("  {:>8} {:>14} {:>12}".format("Time", "Bloch norm", "Deviation"))
    for i in np.unique(np.linspace(0, len(times) - 1, n_rows).astype(int)):
        print("  {:8.3f} {:14.12f} {:12.2e}".format(times[i], b_norm[i], abs(b_norm[i] - 1.0)))


def max_per_metric(metric_dicts):
    return {key: max(m[key] for m in metric_dicts) for key in metric_dicts[0]}


def run_case(report, omega, Omega, label, rng, times, n_random=3):
    R = rabi_frequency(omega, Omega)
    print("CASE: {}".format(label))
    print("  omega = {}, Omega = {}, B = {}, |B| = 2R = {:.4f}, period pi/R = {}".format(
        omega, Omega, field_vector(omega, Omega), 2 * R,
        "{:.4f}".format(np.pi / R) if R > 0 else "inf"))

    initial_states = [KET_0] + [random_pure_state(rng) for _ in range(n_random)]
    results = [analyse_pure(omega, Omega, psi0, times) for psi0 in initial_states]
    print("  Initial state |0>:")
    print_norm_table(results[0][1])
    print("  Checks (max over |0> and {} random pure states, t in [0, {}]):".format(
        n_random, times[-1]))
    for name, value in max_per_metric([m for m, _ in results]).items():
        report.at_most(name, value, ODE_TOL if "ODE" in name else TOL)

    mixed = [analyse_mixed(omega, Omega, random_mixed_bloch_vector(rng), times)
             for _ in range(n_random)]
    for name, value in max_per_metric(mixed).items():
        report.at_most(name, value, TOL)
    print()
    return results[0][1]


def save_plot(trajectories, integrators, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="lightgray", linewidth=0.4)
    for (omega, Omega, traj), color in zip(trajectories, ("C0", "C1", "C2")):
        b = traj["b"]
        n = field_vector(omega, Omega) / np.linalg.norm(field_vector(omega, Omega))
        ax.plot(b[:, 0], b[:, 1], b[:, 2], color=color,
                label="omega={}, Omega={}".format(omega, Omega))
        ax.quiver(0, 0, 0, *n, color=color, linestyle="--", arrow_length_ratio=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Bloch trajectories from |0> (arrows: axis B/|B|)")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    floor = 1e-17
    exact = trajectories[0][2]
    ax2.semilogy(exact["times"], np.abs(exact["b_norm"] - 1) + floor, label="exact U(t)")
    t_e, n_e, dt_e = integrators["euler"]
    ax2.semilogy(t_e, np.abs(n_e - 1) + floor, label="explicit Euler, dt={}".format(dt_e))
    t_c, n_c, dt_c = integrators["cayley"]
    ax2.semilogy(t_c, np.abs(n_c - 1) + floor, label="Cayley (unitary), dt={}".format(dt_c))
    ax2.set_xlabel("t")
    ax2.set_ylabel("| |b| - 1 |")
    ax2.set_title("Distance from the unit sphere")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def explain_bloch_sphere_preservation():
    """
    Explain why the Bloch vector remains on the unit sphere.
    """
    return """
EXPLANATION: Why the Bloch vector remains on the unit sphere  (hbar = 1)
=======================================================================

1. Length of the Bloch vector = purity.
   Any qubit state can be written rho = (I + b.sigma)/2, with b_i = Tr(rho sigma_i).
   Then Tr(rho^2) = (1 + |b|^2)/2, so |b| = 1 <=> Tr(rho^2) = 1 <=> the state is pure.
   For a state vector, |b| = <psi|psi> = ||psi||^2, so a normalised |psi> has |b| = 1.

2. The evolution is a rotation of b.
   Write H = 1/2 B.sigma with B = (2*Omega, 0, omega). From the Heisenberg equation
   d<sigma_i>/dt = (i/hbar) <[H, sigma_i]> and [sigma_j, sigma_k] = 2i eps_jkl sigma_l:
       db/dt = B x b           (Larmor precession)
   hence
       d|b|^2/dt = 2 b.(B x b) = 0.
   Equivalently: since H is Hermitian, U(t) = exp(-iHt) is unitary (in SU(2) up to a
   phase), and U sigma_i U^dagger = sum_j R_ij sigma_j with R in SO(3). So b(t) = R(t) b(0):
   a rotation about n = B/|B| by the angle |B| t, with |B| = 2R, R = sqrt((omega/2)^2 + Omega^2).
   Rotations preserve length, so |b(t)| = |b(0)| for ANY state, pure or mixed.
   They also preserve b.n, i.e. the energy <H> = B.b/2, and b returns to b(0) after pi/R.

3. Conclusion.
   Unitarity keeps |b| constant; purity of the initial state fixes the constant to 1.
   A pure state therefore stays on the unit sphere, and a mixed state stays on a sphere
   of radius |b(0)| < 1.

Numerical remarks:
   - Checking |b| = 1 on the Schrodinger solution is equivalent to checking ||psi|| = 1,
     so the Bloch equation is also integrated independently and compared.
   - Exact unitary propagation keeps |b| = 1 to ~1e-15. A non-unitary integrator such as
     explicit Euler does NOT (|psi| grows by (1 + R^2 dt^2)^(1/2) per step), whereas a unitary
     scheme (Crank-Nicolson / Cayley) stays on the sphere even with a coarse step.
"""


if __name__ == "__main__":
    print("=" * 78)
    print("BLOCH VECTOR UNIT SPHERE VERIFICATION")
    print("=" * 78)
    print()

    rng = np.random.default_rng(12345)
    times = np.linspace(0.0, 10.0, 201)
    report = Report()

    cases = [
        (1.0, 1.0, "omega = Omega = 1"),
        (0.0, 1.0, "omega = 0: degenerate levels (zero detuning in a rotating-frame reading)"),
        (2.0, 0.5, "omega = 2, Omega = 0.5: weak drive, precession mostly about z"),
    ]
    trajectories = [(omega, Omega, run_case(report, omega, Omega, label, rng, times))
                    for omega, Omega, label in cases]

    print("INTEGRATOR COMPARISON (omega = Omega = 1, initial state |0>, t in [0, 10])")
    integrators = analyse_integrators(1.0, 1.0, KET_0, times[-1])
    euler_drift = np.max(np.abs(integrators["euler"][1] - 1.0))
    cayley_drift = np.max(np.abs(integrators["cayley"][1] - 1.0))
    print("  explicit Euler: |b(t=10)| = {:.6f}".format(integrators["euler"][1][-1]))
    report.at_least("Euler leaves the sphere (expected)", euler_drift, EULER_MIN_DRIFT)
    report.at_most("Cayley stays on the sphere", cayley_drift, TOL)
    print()

    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task3b_bloch.png")
    save_plot(trajectories, integrators, plot_path)
    print("Plot saved to {}".format(plot_path))

    print(explain_bloch_sphere_preservation())

    print("=" * 78)
    if report.passed:
        print("VERIFICATION SUCCESSFUL: the Bloch vector stays on the unit sphere in all cases")
    else:
        print("VERIFICATION FAILED: " + ", ".join(report.failures))
    print("=" * 78)
    raise SystemExit(0 if report.passed else 1)
