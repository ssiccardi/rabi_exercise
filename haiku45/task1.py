# 1st task
# Simulate a two-level quantum system with Hamiltonian H = ω/2 σz + Ω σx. 
# Plot the probability of measuring state |1⟩ and verify the correctness of the simulation.
# 2nd task
# Compute the fidelity between the evolved state and a target state.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define Pauli matrices manually
def pauli_x():
    """Pauli X matrix"""
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_z():
    """Pauli Z matrix"""
    return np.array([[1, 0], [0, -1]], dtype=complex)

def identity():
    """Identity matrix"""
    return np.eye(2, dtype=complex)

def ket_from_bloch(theta, phi):
    """Pure qubit state cos(theta/2)|0> + e^{i*phi} sin(theta/2)|1> on the Bloch sphere."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)

def fidelity(state, target):
    """Fidelity F = |<target|state>|^2 / (norm(state)^2 * norm(target)^2).
    
    Handles both single states and arrays of states (rows).
    """
    state = np.asarray(state, dtype=complex)
    target = np.asarray(target, dtype=complex)
    overlap = np.sum(target.conj() * state, axis=-1)
    state_norm = np.sum(np.abs(state) ** 2, axis=-1)
    target_norm = np.sum(np.abs(target) ** 2, axis=-1)
    return np.abs(overlap) ** 2 / (state_norm * target_norm)

# Parameters
omega = 1.0  # Detuning frequency (in units of 2π GHz)
Omega = 1.0  # Rabi frequency (in units of 2π GHz)
t_max = 10.0  # Maximum time (in units of 1/GHz)
n_points = 1000  # Number of time points

# Define Hamiltonian: H = ω/2 σz + Ω σx
sigma_x = pauli_x()
sigma_z = pauli_z()
H = (omega / 2) * sigma_z + Omega * sigma_x

# Initial state: |0⟩ (ground state)
psi0 = np.array([1, 0], dtype=complex)

# Time evolution using matrix exponential
times = np.linspace(0, t_max, n_points)
states = []
P_1 = []

for t in times:
    # Time evolution operator: U(t) = exp(-i * H * t)
    U = expm(-1j * H * t)
    
    # Evolved state: |ψ(t)⟩ = U(t) |ψ(0)⟩
    psi_t = U @ psi0
    states.append(psi_t)
    
    # Probability of measuring state |1⟩: P(|1⟩) = |⟨1|ψ(t)⟩|²
    state_1 = np.array([0, 1], dtype=complex)
    prob = np.abs(np.dot(state_1.conj(), psi_t)) ** 2
    P_1.append(prob)

states = np.array(states)
P_1 = np.array(P_1)

# Define target states for fidelity examples
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = ket_from_bloch(np.pi / 2, 0.0)           # (|0⟩ + |1⟩)/√2
ket_minus_i = ket_from_bloch(np.pi / 2, -np.pi / 2) # (|0⟩ - i|1⟩)/√2

targets = {
    "|0>": ket_0,
    "|1>": ket_1,
    "|+>": ket_plus,
    "|-i>": ket_minus_i,
    "not normalized": np.array([1.0, 1.0], dtype=complex),
}

# Compute fidelities
print("Fidelity with target states:")
fidelities_dict = {}
for name, target in targets.items():
    F = np.array([fidelity(s, target) for s in states])
    fidelities_dict[name] = F
    max_idx = np.argmax(F)
    print(f"  {name:5s} max F = {F[max_idx]:.4f} at t = {times[max_idx]:.3f}, final F = {F[-1]:.4f}")

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Probability of measuring |1⟩
ax1.plot(times, P_1, 'b-', linewidth=2, label='P(|1⟩)')
ax1.set_ylabel('Probability')
ax1.set_title(f'Two-Level Quantum System: Rabi Oscillations\n(omega={omega}, Omega={Omega})')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0, 1])

# Plot 2: Fidelity with different target states
colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
for (name, F), color in zip(fidelities_dict.items(), colors):
    ax2.plot(times, F, linewidth=2, color=color, label=f"target {name}")
    max_idx = np.argmax(F)
    ax2.plot(times[max_idx], F[max_idx], 'o', color=color, markersize=8)

ax2.set_xlabel('Time (1/GHz)')
ax2.set_ylabel('Fidelity F = |<target|psi(t)>|^2')
ax2.set_title('Fidelity with Various Target States')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('rabi_oscillations_with_fidelity.png', dpi=150)
plt.show()

# Verification: Check Rabi frequency
Omega_rabi_theory = np.sqrt(Omega**2 + (omega/2)**2)
print(f"\nTheoretical Rabi frequency: {Omega_rabi_theory:.4f}")

T_rabi_theory = 2 * np.pi / Omega_rabi_theory
max_prob_idx = np.argmax(P_1)
T_max_simulated = times[max_prob_idx]
print(f"Theoretical time to max P(|1>): {T_rabi_theory/2:.4f}")
print(f"Simulated time to max P(|1>): {T_max_simulated:.4f}")
print(f"Maximum P(|1>): {P_1[max_prob_idx]:.4f}")

# Check time to first maximum (manually modified)
first_max_prob = np.argmax(P_1[:n_points//4])
t_first_max_prob = times[first_max_prob]
print(f"Time to first maximum probability: {t_first_max_prob:.4f}")
try:
    assert np.isclose(t_first_max_prob, T_rabi_theory/4, atol=0.1), "Simulated Rabi period does not match theoretical value"
except AssertionError as e:
    print(f"Verification failed: {e}")
else:
    print("Verification passed: Simulated Rabi period matches theoretical value")