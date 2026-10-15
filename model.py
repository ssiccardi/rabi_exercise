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

def analytical_probabilities(omega, Omega, t):
    # Analytical solution for the probabilities of a two-level system
    # with Hamiltonian H = omega/2 * sigma_z + Omega * sigma_x
    delta = np.sqrt((omega / 2)**2 + Omega**2)
    p0 = 1 - (Omega**2 / delta**2) * np.sin(delta * t)**2
    p1 = 1 - p0
    return np.array([p0, p1])

def evolve_states(psi0, H, t):
    return time_evolution(H, t) @ psi0

def probabilities(psi):
    return abs(psi)**2

# planned tests
# Test that the evolved state remains normalized
def test_normalization():
    psi0 = np.array([1.0, 0.0])
    omega = 1.0
    Omega = 1.0
    H = Hamiltonian(omega, Omega)
    t = 1.0
    psi = evolve_states(psi0, H, t)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    print("Test normalization passed.")
#psi0 = np.array([1.0, 0.0])
#omega = 1.0
#Omega = 1.0
#H = Hamiltonian(omega, Omega)
##t = 1.0
#psi = evolve_states(psi0, H, t)
##print(probabilities(psi), psi, psi0)
#assert np.isclose(
#    np.linalg.norm(psi),
#    1.0
#)
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

def test_unitary():
    # test that time evolution is unitary
    omega = 1.0
    Omega = 1.0
    H = Hamiltonian(omega, Omega)
    t = 1.0
    U = time_evolution(H, t)
    assert np.allclose(U.conj().T @ U, np.eye(2))
    print("Test unitary passed.")

def test_no_evolution_when_Omega_zero():
    # test that if Omega =0, the state does not evolve
    Omega = 0.0
    omega = 1.0
    H = Hamiltonian(omega, Omega)
    psi0 = np.array([1.0, 0.0])
    t = 1.0
    psi = evolve_states(psi0, H, t)
    assert np.isclose(probabilities(psi)[0], probabilities(psi0)[0])
    assert np.isclose(probabilities(psi)[1], probabilities(psi0)[1])
    print("Test no evolution when Omega zero passed.")

def test_analytical_vs_numerical():
    # test that the analytical probabilities match the numerical ones
    psi0 = np.array([1.0, 0.0])
    omega = 1.0
    Omega = 1.0
    H = Hamiltonian(omega, Omega)
    for t in np.linspace(0, 10, 100):
        psi = evolve_states(psi0, H, t)
        p_numerical = probabilities(psi)
        p_analytical = analytical_probabilities(omega, Omega, t)
        assert np.allclose(p_numerical, p_analytical, atol=1e-10)
    print("Test analytical vs numerical passed.")

def test_fidelity():
    # test that the fidelity between the initial and evolved state is as expected
    psi0 = np.array([1.0, 0.0])
    omega = 1.0
    Omega = 1.0
    H = Hamiltonian(omega, Omega)
    t = 1.0
    psi = evolve_states(psi0, H, t)
    fidelity = np.abs(np.vdot(psi0, psi))**2
    assert 0.0 <= fidelity <= 1.0
    print("Test fidelity passed.")
    psi_target = np.array([0.0, 1.0])
    fidelity_target = np.abs(np.vdot(psi_target, psi))**2
    fidelities = []
    for t in np.linspace(0, 10, 100):
        psi = evolve_states(psi0, H, t)
        fidelity_target = np.abs(np.vdot(psi_target, psi))**2
        fidelities.append(fidelity_target)
        assert 0.0 <= fidelity_target <= 1.0
    # plot the fidelity over time
    plt.plot(np.linspace(0, 10, 100), fidelities, label='Fidelity with target state')
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    plt.legend()
    plt.show()

def test_resonant():
    # test the system with resonant driving (omega = 0)
    psi0 = np.array([1.0, 0.0])
    psi_target = np.array([0.0, 1.0])
    omega = 0.0
    Omega = 1.0
    H = Hamiltonian(omega, Omega)
    t=np.pi/(2*Omega)
    psi = evolve_states(psi0, H, t)
    F = np.abs(np.vdot(psi_target, psi))**2
    assert np.isclose(F, 1.0)
    print("Test resonant passed.")

if __name__ == "__main__":
    test_normalization()
    test_unitary()
    test_no_evolution_when_Omega_zero()
    test_analytical_vs_numerical()
    test_fidelity()
    test_resonant()