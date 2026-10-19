# task 1
# Simulate a two-level quantum system with Hamiltonian H = ω/2 σz + Ω σx. 
# Plot the probability of measuring state |1⟩ and verify the correctness of the simulation.
# task 2
# Compute the fidelity between the evolved state and a target state.


# Units: ħ = 1, so energies and angular frequencies coincide and time is in units of 1/frequency.
# Basis convention: |0⟩ = (1, 0)ᵀ, |1⟩ = (0, 1)ᵀ, so σz|0⟩ = +|0⟩ and σz|1⟩ = -|1⟩.
#
# Analytic reference: writing H = R (n·σ) with R = sqrt((ω/2)² + Ω²), the propagator is
#   U(t) = exp(-iHt) = cos(Rt) I - i sin(Rt) H / R
# and, starting from |0⟩, the probability of finding the system in |1⟩ is (Rabi formula)
#   P1(t) = (Ω² / R²) sin²(R t)
# i.e. it oscillates with period π/R and maximum Ω²/R² (= 1 only at resonance, ω = 0).
# The closed form also gives the evolved state explicitly:
#   |ψ(t)⟩ = U(t)|0⟩ = (cos(Rt) - i (ω/2R) sin(Rt)) |0⟩ - i (Ω/R) sin(Rt) |1⟩
#
# Fidelity between pure states: F(ψ, φ) = |⟨φ|ψ⟩|², with 0 ≤ F ≤ 1 and F = 1 iff the states
# coincide up to a global phase. With target |1⟩ it reduces to P1(t).

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)


def ket_from_bloch(theta, phi):
    """Pure qubit state cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩ (polar angle θ, azimuth φ on the Bloch sphere)."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def hamiltonian(omega, Omega):
    return 0.5 * omega * SIGMA_Z + Omega * SIGMA_X


def is_hermitian(M, tol=1e-12):
    return np.allclose(M, M.conj().T, atol=tol)


def evolve_eigen(H, psi0, times):
    """Exact evolution: U(t) = V exp(-iEt) V†, from the eigendecomposition H = V E V†."""
    energies, V = np.linalg.eigh(H)
    coeffs = V.conj().T @ psi0                                  # psi0 in the eigenbasis
    phases = np.exp(-1j * np.outer(times, energies))            # shape (n_times, 2)
    return (phases * coeffs) @ V.T                              # rows are psi(t)


def evolve_rk4(H, psi0, times, substeps=20):
    """Numerical integration of the Schrödinger equation dψ/dt = -iHψ with 4th-order Runge-Kutta."""
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


def probability(states, ket):
    """Born rule: P(t) = |⟨ket|ψ(t)⟩|² for every row ψ(t) of `states`."""
    return np.abs(states @ ket.conj()) ** 2


def fidelity(states, target):
    """Pure-state fidelity F = |⟨target|ψ⟩|² / (⟨ψ|ψ⟩⟨target|target⟩).

    `states` can be a single state vector or an array whose rows are states ψ(t);
    `target` can likewise be a single state or rows matching `states` one to one.
    """
    states = np.asarray(states, dtype=complex)
    target = np.asarray(target, dtype=complex)
    overlap = np.sum(target.conj() * states, axis=-1)
    norms = np.sum(np.abs(states) ** 2, axis=-1) * np.sum(np.abs(target) ** 2, axis=-1)
    return np.abs(overlap) ** 2 / norms


def rabi_frequency(omega, Omega):
    return np.sqrt((omega / 2) ** 2 + Omega ** 2)


def analytic_p1(times, omega, Omega):
    R = rabi_frequency(omega, Omega)
    return (Omega / R) ** 2 * np.sin(R * times) ** 2


def analytic_states(times, omega, Omega):
    """Closed-form |ψ(t)⟩ = U(t)|0⟩, one row per time."""
    R = rabi_frequency(omega, Omega)
    c, s = np.cos(R * np.asarray(times)), np.sin(R * np.asarray(times))
    return np.stack([c - 1j * (omega / (2 * R)) * s, -1j * (Omega / R) * s], axis=-1)


def verify(omega, Omega, times, states_eig, states_rk4, targets, tol=1e-6):
    """Run the correctness checks; raise AssertionError if any of them fails."""
    H = hamiltonian(omega, Omega)
    R = rabi_frequency(omega, Omega)
    p1_eig = probability(states_eig, KET_1)
    p1_rk4 = probability(states_rk4, KET_1)
    p1_th = analytic_p1(times, omega, Omega)
    checks = {}

    # 1. The Hamiltonian is Hermitian and its eigenvalues are ±R.
    checks["H is Hermitian"] = is_hermitian(H)
    checks["eigenvalues are +-R"] = np.allclose(np.linalg.eigvalsh(H), [-R, R])

    # 2. The evolution is unitary: the norm of ψ(t) stays 1.
    checks["norm conserved (eigen)"] = np.max(np.abs(np.linalg.norm(states_eig, axis=1) - 1)) < tol
    checks["norm conserved (RK4)"] = np.max(np.abs(np.linalg.norm(states_rk4, axis=1) - 1)) < tol

    # 3. Probabilities add up to one.
    checks["P0 + P1 = 1"] = np.allclose(probability(states_eig, KET_0) + p1_eig, 1)

    # 4. Both simulations agree with the analytic Rabi formula and with each other.
    err_eig = np.max(np.abs(p1_eig - p1_th))
    err_rk4 = np.max(np.abs(p1_rk4 - p1_th))
    checks[f"eigen vs analytic (max err {err_eig:.1e})"] = err_eig < tol
    checks[f"RK4 vs analytic (max err {err_rk4:.1e})"] = err_rk4 < tol

    # 5. The eigen propagator matches the closed form cos(Rt) I - i sin(Rt) H/R.
    t = times[len(times) // 3]
    U_closed = np.cos(R * t) * IDENTITY - 1j * np.sin(R * t) * H / R
    checks["closed-form propagator"] = np.allclose(evolve_eigen(H, KET_0, [t])[0], U_closed @ KET_0)

    # 6. Physical limits: full inversion at resonance, no transitions without drive.
    t_pi = np.pi / (2 * Omega) if Omega else 1.0
    p_res = probability(evolve_eigen(hamiltonian(0.0, Omega), KET_0, [t_pi]), KET_1)[0]
    p_free = probability(evolve_eigen(hamiltonian(omega, 0.0), KET_0, times), KET_1)
    checks["resonant pi-pulse gives P1 = 1"] = Omega == 0 or np.isclose(p_res, 1)
    checks["no drive (Omega = 0) gives P1 = 0"] = np.allclose(p_free, 0)

    # 7. Basic properties of the fidelity function.
    rng = np.random.default_rng(0)
    a = rng.normal(size=2) + 1j * rng.normal(size=2)
    b = rng.normal(size=2) + 1j * rng.normal(size=2)
    F_ab = fidelity(a, b)
    checks["fidelity: F(psi, psi) = 1"] = np.isclose(fidelity(a, a), 1)
    checks["fidelity: orthogonal states give 0"] = np.isclose(fidelity(KET_0, KET_1), 0)
    checks["fidelity: symmetric and in [0, 1]"] = np.isclose(F_ab, fidelity(b, a)) and 0 <= F_ab <= 1
    checks["fidelity: global phase invariant"] = np.isclose(fidelity(np.exp(0.7j) * a, b), F_ab)
    checks["fidelity with |1> equals P1"] = np.allclose(fidelity(states_eig, KET_1), p1_eig)

    # 8. Fidelity with each target agrees with the closed-form evolved state.
    states_th = analytic_states(times, omega, Omega)
    checks["closed-form states match eigen evolution"] = np.allclose(states_th, states_eig)
    for name, target in targets.items():
        F_th = fidelity(states_th, target)
        err = max(np.max(np.abs(fidelity(states_eig, target) - F_th)),
                  np.max(np.abs(fidelity(states_rk4, target) - F_th)))
        checks[f"fidelity with {name} vs analytic (max err {err:.1e})"] = err < tol

    # 9. RK4 and exact evolved states coincide (fidelity between them is 1).
    infid = np.max(1 - fidelity(states_rk4, states_eig))
    checks[f"F(RK4 state, exact state) = 1 (max 1-F {infid:.1e})"] = infid < tol

    print(f"Verification (omega = {omega}, Omega = {Omega}, R = {R:.4f}):")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  Expected max P1 = Omega^2/R^2 = {(Omega / R) ** 2:.4f}, simulated = {p1_eig.max():.4f}")
    print(f"  Expected period = pi/R = {np.pi / R:.4f}")

    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"Verification failed: {failed}"


def main():
    omega = 1.0     # level splitting (detuning) ω
    Omega = 1.0     # drive strength Ω
    t_max = 20.0
    times = np.linspace(0.0, t_max, 801)

    # Target states for the fidelity (ASCII names so they print on any console).
    targets = {
        "|1>": KET_1,
        "|+>": ket_from_bloch(np.pi / 2, 0.0),           # (|0⟩ + |1⟩)/√2
        "|-i>": ket_from_bloch(np.pi / 2, -np.pi / 2),   # (|0⟩ - i|1⟩)/√2
        "not normalized": np.array([1.0, 1.0], dtype=complex), 
    }

    H = hamiltonian(omega, Omega)
    states_eig = evolve_eigen(H, KET_0, times)
    states_rk4 = evolve_rk4(H, KET_0, times)

    verify(omega, Omega, times, states_eig, states_rk4, targets)

    fidelities = {name: fidelity(states_eig, target) for name, target in targets.items()}

    # The motion is periodic with period π/R, so the best fidelity over all times is reached
    # within the first period; scan it on a fine grid to get the first optimal time.
    R = rabi_frequency(omega, Omega)
    t_fine = np.linspace(0.0, np.pi / R, 20001)
    states_fine = evolve_eigen(H, KET_0, t_fine)
    best = {}
    print("Fidelity with target states:")
    for name, target in targets.items():
        F_fine = fidelity(states_fine, target)
        k = np.argmax(F_fine)
        best[name] = (t_fine[k], F_fine[k])
        print(f"  {name:5s} max F = {F_fine[k]:.4f} first reached at t = {t_fine[k]:.4f}, "
              f"F(t_max) = {fidelities[name][-1]:.4f}")

    # Plot: the detuned case together with the resonant case (ω = 0) for comparison.
    states_res = evolve_eigen(hamiltonian(0.0, Omega), KET_0, times)
    fig, (ax, ax_fid, ax_err) = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 2, 1]})

    ax.plot(times, analytic_p1(times, omega, Omega), "k-", lw=1, label="analytic (Rabi formula)")
    ax.plot(times, probability(states_eig, KET_1), "b--", lw=2, label="eigen-decomposition")
    ax.plot(times[::16], probability(states_rk4, KET_1)[::16], "ro", ms=4, label="RK4 integration")
    ax.plot(times, probability(states_res, KET_1), color="gray", alpha=0.5,
            label="resonant case ω = 0 (reference)")
    ax.axhline((Omega / rabi_frequency(omega, Omega)) ** 2, color="g", ls=":", label="Ω²/R²")
    ax.set_ylabel("P(|1⟩)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Two-level system, H = ω/2 σz + Ω σx  (ω = {omega}, Ω = {Omega}), initial state |0⟩")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, F) in enumerate(fidelities.items()):
        color = colors[i % len(colors)]
        label = name.replace(">", "⟩")
        t_best, F_best = best[name]
        ax_fid.plot(times, F, color=color, lw=2, label=f"target {label}")
        ax_fid.plot(times[::16], fidelity(states_rk4[::16], targets[name]), "o", color=color, ms=3)
        ax_fid.plot(t_best, F_best, "*", color=color, ms=12, mec="k")
    ax_fid.set_ylabel("fidelity F = |⟨target|ψ(t)⟩|²")
    ax_fid.set_ylim(-0.02, 1.05)
    ax_fid.legend(loc="upper right", fontsize=9, title="lines: eigen, dots: RK4, ★: first maximum", title_fontsize=8)
    ax_fid.grid(alpha=0.3)

    p1_th = analytic_p1(times, omega, Omega)
    ax_err.semilogy(times, np.abs(probability(states_eig, KET_1) - p1_th) + 1e-18, "b", label="P1 error, eigen")
    ax_err.semilogy(times, np.abs(probability(states_rk4, KET_1) - p1_th) + 1e-18, "r", label="P1 error, RK4")
    ax_err.semilogy(times, np.abs(1 - fidelity(states_rk4, states_eig)) + 1e-18, "m",
                    label="1 - F(RK4, exact)")
    ax_err.set_xlabel("time (units of 1/ω, ħ = 1)")
    ax_err.set_ylabel("|error|")
    ax_err.legend(fontsize=9)
    ax_err.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("task1_rabi.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
