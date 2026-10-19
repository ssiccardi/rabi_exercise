# task:
# Simulate a qubit initialized in |0⟩ under the Hamiltonian H = Ω σx and verify that 
# the target state |1⟩ is reached at time π/Ω.


# Units: ħ = 1. Basis convention: |0⟩ = (1, 0)ᵀ, |1⟩ = (0, 1)ᵀ.
#
# Analytic reference: since σx² = I,
#   U(t) = exp(-iΩσx t) = cos(Ωt) I - i sin(Ωt) σx
#   |ψ(t)⟩ = U(t)|0⟩ = cos(Ωt)|0⟩ - i sin(Ωt)|1⟩,   P1(t) = sin²(Ωt)
# so |1⟩ is first reached at t = π/(2Ω) (up to the global phase -i), while at t = π/Ω
# the state is -|0⟩, i.e. back to the initial state: P1(π/Ω) = 0.
#
# The statement "|1⟩ is reached at π/Ω" is correct for the other common convention
# H = (Ω/2) σx, where Ω is the Rabi frequency: then P1(t) = sin²(Ωt/2) and the π-pulse
# lasts π/Ω. The program checks the claim as stated and also this alternative convention.

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)


def hamiltonian(Omega, prefactor=1.0):
    """H = prefactor · Ω σx (prefactor = 1 is the task's Hamiltonian, 1/2 the Rabi-frequency convention)."""
    return prefactor * Omega * SIGMA_X


def evolve_eigen(H, psi0, times):
    """Exact evolution: U(t) = V exp(-iEt) V†, from the eigendecomposition H = V E V†."""
    energies, V = np.linalg.eigh(H)
    coeffs = V.conj().T @ psi0
    phases = np.exp(-1j * np.outer(np.atleast_1d(times), energies))
    return (phases * coeffs) @ V.T


def evolve_rk4(H, psi0, times, substeps=50):
    """Independent check: integrate dψ/dt = -iHψ with 4th-order Runge-Kutta."""
    def f(psi):
        return -1j * (H @ psi)

    states = np.empty((len(times), len(psi0)), dtype=complex)
    psi = psi0.astype(complex)
    states[0] = psi
    for k in range(1, len(times)):
        dt = (times[k] - times[k - 1]) / substeps
        for _ in range(substeps):
            k1 = f(psi)
            k2 = f(psi + 0.5 * dt * k1)
            k3 = f(psi + 0.5 * dt * k2)
            k4 = f(psi + dt * k3)
            psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        states[k] = psi
    return states


def fidelity(states, target):
    """Pure-state fidelity F = |⟨target|ψ⟩|² / (⟨ψ|ψ⟩⟨target|target⟩), row by row."""
    states = np.asarray(states, dtype=complex)
    target = np.asarray(target, dtype=complex)
    overlap = np.sum(target.conj() * states, axis=-1)
    norms = np.sum(np.abs(states) ** 2, axis=-1) * np.sum(np.abs(target) ** 2, axis=-1)
    return np.abs(overlap) ** 2 / norms


def analytic_states(times, Omega, prefactor=1.0):
    """Closed-form |ψ(t)⟩ = cos(aΩt)|0⟩ - i sin(aΩt)|1⟩ with a = prefactor."""
    x = prefactor * Omega * np.asarray(times)
    return np.stack([np.cos(x), -1j * np.sin(x)], axis=-1)


def first_time_reaching(H, psi0, target, t_max, n=200001):
    """First time in [0, t_max] where F(ψ(t), target) is maximal, found on a fine grid."""
    t = np.linspace(0.0, t_max, n)
    F = fidelity(evolve_eigen(H, psi0, t), target)
    k = np.argmax(F > F.max() - 1e-12)
    return t[k], F[k]


def format_state(psi):
    return " ".join(f"{c.real:+.4f}{c.imag:+.4f}i" for c in psi)


def check_claim(Omega, prefactor=1.0, tol=1e-9):
    """Simulate and return a dict of checks; the claim itself is reported, not asserted."""
    H = hamiltonian(Omega, prefactor)
    t_claim = np.pi / Omega
    psi_claim = evolve_eigen(H, KET_0, [t_claim])[0]
    F_claim = fidelity(psi_claim, KET_1)
    t_first, F_first = first_time_reaching(H, KET_0, KET_1, t_max=t_claim)
    t_expected = np.pi / (2 * prefactor * Omega)
    return {
        "H": H,
        "t_claim": t_claim,
        "psi_claim": psi_claim,
        "F_claim": F_claim,
        "claim_holds": abs(F_claim - 1) < tol,
        "t_first": t_first,
        "F_first": F_first,
        "t_expected": t_expected,
    }


def verify_simulation(Omega, times, states_eig, states_rk4, tol=1e-8):
    """Sanity checks on the simulation itself; raise AssertionError if any fails."""
    H = hamiltonian(Omega)
    states_th = analytic_states(times, Omega)
    checks = {
        "H is Hermitian": np.allclose(H, H.conj().T),
        "eigenvalues are +-Omega": np.allclose(np.linalg.eigvalsh(H), [-Omega, Omega]),
        "norm conserved (eigen)": np.allclose(np.linalg.norm(states_eig, axis=1), 1, atol=tol),
        "norm conserved (RK4)": np.allclose(np.linalg.norm(states_rk4, axis=1), 1, atol=tol),
        "eigen states match closed form": np.allclose(states_eig, states_th, atol=tol),
        "RK4 states match closed form": np.allclose(states_rk4, states_th, atol=1e-6),
        "P1 = sin^2(Omega t)": np.allclose(fidelity(states_eig, KET_1), np.sin(Omega * times) ** 2, atol=tol),
        "propagator = cos(Ot) I - i sin(Ot) sx": np.allclose(
            evolve_eigen(H, KET_1, [0.7])[0],
            (np.cos(0.7 * Omega) * IDENTITY - 1j * np.sin(0.7 * Omega) * SIGMA_X) @ KET_1, atol=tol),
    }
    print(f"Simulation checks (H = Omega sx, Omega = {Omega}):")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"Simulation checks failed: {failed}"


def report(Omega, prefactor, label):
    r = check_claim(Omega, prefactor)
    print(f"  {label}, Omega = {Omega}:")
    print(f"    t = pi/Omega = {r['t_claim']:.6f}: psi = [{format_state(r['psi_claim'])}], "
          f"F(psi, |1>) = {r['F_claim']:.3e}  -> claim {'HOLDS' if r['claim_holds'] else 'FAILS'}")
    print(f"    |1> first reached at t = {r['t_first']:.6f} (F = {r['F_first']:.12f}); "
          f"t*Omega/pi = {r['t_first'] * Omega / np.pi:.6f}, "
          f"expected pi/(2*{prefactor}*Omega) = {r['t_expected']:.6f}")
    return r


def main():
    Omega = 1.0
    t_max = 2 * np.pi / Omega
    times = np.linspace(0.0, t_max, 1001)

    H = hamiltonian(Omega)
    states_eig = evolve_eigen(H, KET_0, times)
    states_rk4 = evolve_rk4(H, KET_0, times)
    verify_simulation(Omega, times, states_eig, states_rk4)

    print("\nClaim: starting from |0>, the target |1> is reached at t = pi/Omega.")
    results = {}
    for Om in (0.5, 1.0, 2.0):
        results[("task", Om)] = report(Om, 1.0, "H = Omega sx  (as in the task)")
    for Om in (0.5, 1.0, 2.0):
        results[("rabi", Om)] = report(Om, 0.5, "H = (Omega/2) sx  (Rabi-frequency convention)")

    task_ok = all(r["claim_holds"] for (conv, _), r in results.items() if conv == "task")
    rabi_ok = all(r["claim_holds"] for (conv, _), r in results.items() if conv == "rabi")
    times_ok = all(np.isclose(r["t_first"], r["t_expected"], atol=1e-4) for r in results.values())
    print("\nConclusion:")
    print(f"  [{'PASS' if times_ok else 'FAIL'}] first arrival time at |1> equals pi/(2 a Omega) for H = a Omega sx")
    print(f"  Claim with H = Omega sx:     {'TRUE' if task_ok else 'FALSE'} "
          f"(|1> is reached at pi/(2 Omega); at pi/Omega the state is -|0>)")
    print(f"  Claim with H = (Omega/2) sx: {'TRUE' if rabi_ok else 'FALSE'}")

    # Plot P1(t) for both conventions, marking pi/(2Ω) and pi/Ω.
    states_rabi = evolve_eigen(hamiltonian(Omega, 0.5), KET_0, times)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, np.sin(Omega * times) ** 2, "k-", lw=1, label="analytic sin²(Ωt)")
    ax.plot(times, fidelity(states_eig, KET_1), "b--", lw=2, label="H = Ω σx, eigen-decomposition")
    ax.plot(times[::25], fidelity(states_rk4, KET_1)[::25], "ro", ms=4, label="H = Ω σx, RK4")
    ax.plot(times, fidelity(states_rabi, KET_1), color="gray", alpha=0.6, lw=2,
            label="H = (Ω/2) σx (reference)")
    ax.axvline(np.pi / (2 * Omega), color="g", ls=":", label="t = π/(2Ω)")
    ax.axvline(np.pi / Omega, color="m", ls=":", label="t = π/Ω (claimed)")
    ax.set_xlabel("time (units of 1/Ω, ħ = 1)")
    ax.set_ylabel("F(ψ(t), |1⟩) = P(|1⟩)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Qubit from |0⟩ under H = Ω σx (Ω = {Omega})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("task2_rabi.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
