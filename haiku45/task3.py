# task
# Simulate a two-level quantum system with Hamiltonian H = ω/2 σz + Ω σx. 
# Verify numerically that the Bloch vector remains on the unit sphere during the evolution, and explain why.


# Units: ħ = 1. Basis convention: |0⟩ = (1, 0)ᵀ, |1⟩ = (0, 1)ᵀ.
#
# Why the Bloch vector stays on the unit sphere:
# The Bloch vector b = ⟨σ⟩ = (⟨σx⟩, ⟨σy⟩, ⟨σz⟩) for a pure state |ψ⟩ satisfies
#   |b|² = Σ ⟨σi⟩² = 1  (for pure states)
# because the evolution U(t) = exp(-iHt) is unitary: it preserves the trace and maintains purity.
# More directly: if H is Hermitian (which it is), then d|b|²/dt = 0 along the evolution,
# so |b| stays constant. Since it equals 1 at t=0 for any pure state, it remains 1 forever.
#
# The Bloch vector precesses around an effective magnetic field proportional to H itself.

from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)


def hamiltonian(omega, Omega):
    """H = ω/2 σz + Ω σx."""
    return 0.5 * omega * SIGMA_Z + Omega * SIGMA_X


def evolve_eigen(H, psi0, times):
    """Exact evolution: U(t) = V exp(-iEt) V†, from the eigendecomposition H = V E V†."""
    energies, V = np.linalg.eigh(H)
    coeffs = V.conj().T @ psi0
    phases = np.exp(-1j * np.outer(np.atleast_1d(times), energies))
    return (phases * coeffs) @ V.T


def evolve_rk4(H, psi0, times, substeps=50):
    """Integrate dψ/dt = -iHψ with 4th-order Runge-Kutta."""
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


def bloch_vector(states):
    """Compute Bloch vector ⟨σ⟩ = (⟨σx⟩, ⟨σy⟩, ⟨σz⟩) for each row of states."""
    b = np.empty((len(states), 3), dtype=float)
    for i, psi in enumerate(states):
        b[i, 0] = np.real(np.vdot(psi, SIGMA_X @ psi))
        b[i, 1] = np.real(np.vdot(psi, SIGMA_Y @ psi))
        b[i, 2] = np.real(np.vdot(psi, SIGMA_Z @ psi))
    return b


def bloch_norm(b):
    """Return the norm of each Bloch vector (row)."""
    return np.linalg.norm(b, axis=1)


def verify_bloch_sphere(omega, Omega, times, states_eig, states_rk4, tol=1e-8):
    """Check that the Bloch vector stays on the unit sphere."""
    b_eig = bloch_vector(states_eig)
    b_rk4 = bloch_vector(states_rk4)
    norm_eig = bloch_norm(b_eig)
    norm_rk4 = bloch_norm(b_rk4)

    checks = {
        "Bloch norm = 1 (eigen)": np.allclose(norm_eig, 1.0, atol=tol),
        "Bloch norm = 1 (RK4)": np.allclose(norm_rk4, 1.0, atol=tol),
        "Bloch vectors match (eigen vs RK4)": np.allclose(b_eig, b_rk4, atol=1e-6),
    }
    
    print(f"Bloch sphere verification (ω = {omega}, Ω = {Omega}):")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    
    if "Bloch norm = 1 (eigen)" in checks and checks["Bloch norm = 1 (eigen)"]:
        err_eigen = np.max(np.abs(norm_eig - 1.0))
        print(f"  Max deviation from unit sphere (eigen): {err_eigen:.3e}")
    if "Bloch norm = 1 (RK4)" in checks and checks["Bloch norm = 1 (RK4)"]:
        err_rk4 = np.max(np.abs(norm_rk4 - 1.0))
        print(f"  Max deviation from unit sphere (RK4):  {err_rk4:.3e}")
    
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"Bloch sphere checks failed: {failed}"
    
    return b_eig, norm_eig


def main():
    omega = 1.0
    Omega = 1.0
    t_max = 10.0
    times = np.linspace(0.0, t_max, 501)

    H = hamiltonian(omega, Omega)
    
    # Evolve from |0⟩
    states_eig = evolve_eigen(H, KET_0, times)
    states_rk4 = evolve_rk4(H, KET_0, times)
    
    print("Starting from |0⟩:")
    b_eig_0, norm_eig_0 = verify_bloch_sphere(omega, Omega, times, states_eig, states_rk4)
    
    # Evolve from |1⟩
    states_eig_1 = evolve_eigen(H, KET_1, times)
    states_rk4_1 = evolve_rk4(H, KET_1, times)
    print("\nStarting from |1⟩:")
    b_eig_1, norm_eig_1 = verify_bloch_sphere(omega, Omega, times, states_eig_1, states_rk4_1)

    # Plot 1: Bloch vector norms over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(times, norm_eig_0, "b-", lw=2, label="|0⟩ initial state (eigen)")
    ax1.plot(times[::10], bloch_norm(bloch_vector(states_rk4))[::10], "bo", ms=3, label="|0⟩ (RK4)")
    ax1.plot(times, norm_eig_1, "r-", lw=2, label="|1⟩ initial state (eigen)")
    ax1.plot(times[::10], bloch_norm(bloch_vector(states_rk4_1))[::10], "ro", ms=3, label="|1⟩ (RK4)")
    ax1.axhline(1.0, color="k", ls=":", alpha=0.5, label="unit sphere")
    ax1.set_xlabel("time")
    ax1.set_ylabel("|b| = √(⟨σx⟩² + ⟨σy⟩² + ⟨σz⟩²)")
    ax1.set_title(f"Bloch Vector Norm (ω = {omega}, Ω = {Omega})")
    ax1.set_ylim(0.99, 1.01)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Log scale of deviation from unit sphere
    dev_eig_0 = np.abs(norm_eig_0 - 1.0)
    dev_eig_1 = np.abs(norm_eig_1 - 1.0)
    ax2.semilogy(times, dev_eig_0 + 1e-17, "b-", lw=2, label="|0⟩ initial (eigen)")
    ax2.semilogy(times, dev_eig_1 + 1e-17, "r-", lw=2, label="|1⟩ initial (eigen)")
    ax2.set_xlabel("time")
    ax2.set_ylabel("|1 - |b||")
    ax2.set_title("Deviation from Unit Sphere")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")
    
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("task3_bloch_norm.png"), dpi=150)
    plt.show()

    # Plot 3: 3D Bloch sphere trajectory for |0⟩
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Draw the unit sphere as a wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color="gray")
    
    # Plot the trajectory
    ax.plot(b_eig_0[:, 0], b_eig_0[:, 1], b_eig_0[:, 2], "b-", lw=2, label="trajectory from |0⟩")
    ax.scatter(b_eig_0[0, 0], b_eig_0[0, 1], b_eig_0[0, 2], color="g", s=100, marker="o", label="start |0⟩")
    ax.scatter(b_eig_0[-1, 0], b_eig_0[-1, 1], b_eig_0[-1, 2], color="r", s=100, marker="s", label="end")
    
    ax.set_xlabel("⟨σx⟩")
    ax.set_ylabel("⟨σy⟩")
    ax.set_zlabel("⟨σz⟩")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title(f"Bloch Sphere Trajectory (ω = {omega}, Ω = {Omega})")
    ax.legend(fontsize=9)
    
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("task3_bloch_trajectory.png"), dpi=150)
    plt.show()

    print("\n" + "="*70)
    print("EXPLANATION: Why the Bloch vector stays on the unit sphere")
    print("="*70)
    print("""
The Bloch vector b = ⟨σ⟩ for a pure quantum state satisfies |b| = 1 always.

1. MATHEMATICAL REASON (Purity):
   For a pure state |ψ⟩, the density matrix ρ = |ψ⟩⟨ψ| satisfies ρ² = ρ.
   The Bloch vector components are b_i = Tr(ρ σ_i) = ⟨ψ|σ_i|ψ⟩.
   For any 2×2 matrix ρ representing a pure state,
      |b|² = Σ_i ⟨σ_i⟩² = 1  (this is an identity for Pauli matrices)

2. DYNAMICAL REASON (Unitarity):
   Schrödinger evolution U(t) = exp(-iHt) is unitary: U†U = I.
   It preserves the norm ⟨ψ(t)|ψ(t)⟩ = 1 and the purity Tr(ρ²) = 1.
   Therefore, |b(t)| = 1 for all t if |b(0)| = 1.

3. CONSERVATION LAW:
   d|b|²/dt = 0 for unitary evolution (follows from the cyclicity of trace and
   the fact that [H, σ] terms rearrange due to the commutation relations of Pauli matrices).

In summary: pure-state evolution cannot change |b| from 1 because the evolution
operator is unitary and pure states stay pure. The Bloch vector traces a path
on the unit sphere, never leaving it. This is a fundamental property of two-level
quantum systems under any Hermitian Hamiltonian.
    """)


if __name__ == "__main__":
    main()
