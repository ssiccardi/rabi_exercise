import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def Hamiltonian(omega, Omega):
    # Construct the Hamiltonian matrix for a two-level system
    # in the form  H = \omega / 2 \sigma_z + \Omega \sigma_x
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    H = (omega / 2) * sigma_z + Omega * sigma_x
    #print("Hamiltonian:\n", H)
    return H

def time_evolution(H, t):
    # Compute the time evolution operator U(t) = exp(-i H t)
    U = scipy.linalg.expm(-1j * H * t)
    return U

def evolve_states(psi0, H, t):
    return time_evolution(H, t) @ psi0

def probabilities(psi):
    return abs(psi)**2

# planned tests
# Test that the evolved state remains normalized
psi0 = np.array([1.0, 0.0])
omega = 1.0
Omega = 1.0
H = Hamiltonian(omega, Omega)
t = 1.0
psi = evolve_states(psi0, H, t)
#print(probabilities(psi), psi, psi0)
assert np.isclose(
    np.linalg.norm(psi),
    1.0
)
probabilities_list = []
for t in np.linspace(0, 10, 100):
    psi = evolve_states(psi0, H, t)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    probabilities_list.append(probabilities(psi))

# plot the probabilities over time
probabilities_array = np.array(probabilities_list)
plt.plot(np.linspace(0, 10, 100), probabilities_array[:, 0], label='|0>')
plt.plot(np.linspace(0, 10, 100), probabilities_array[:, 1], label='|1>')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.show()

# test that time evolution is unitary
omega = 1.0
Omega = 1.0
H = Hamiltonian(omega, Omega)
t = 1.0
U = time_evolution(H, t)
assert np.allclose(U.conj().T @ U, np.eye(2))

# test that if Omega =0, the state does not evolve
Omega = 0.0
omega = 1.0
H = Hamiltonian(omega, Omega)
psi0 = np.array([1.0, 0.0])
t = 1.0
psi = evolve_states(psi0, H, t)
assert probabilities(psi)[0] == probabilities(psi0)[0]
assert probabilities(psi)[1] == probabilities(psi0)[1]

