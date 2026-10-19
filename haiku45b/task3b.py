# task
# Simulate a two-level quantum system with Hamiltonian H = ω/2 σz + Ω σx. 
# Verify numerically that the Bloch vector remains on the unit sphere during the evolution, and explain why.

import numpy as np
import scipy.linalg

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def hamiltonian(omega, Omega):
    """
    Construct the Hamiltonian matrix for a two-level system.
    H = ω/2 σz + Ω σx
    
    Parameters:
    -----------
    omega : float
        Energy splitting parameter
    Omega : float
        Driving amplitude parameter
        
    Returns:
    --------
    H : 2x2 complex array
        The Hamiltonian matrix
    """
    H = (omega / 2) * SIGMA_Z + Omega * SIGMA_X
    return H


def time_evolution(H, t):
    """
    Compute the time evolution operator U(t) = exp(-i H t)
    
    Parameters:
    -----------
    H : 2x2 complex array
        Hamiltonian matrix
    t : float
        Time
        
    Returns:
    --------
    U : 2x2 complex array
        Time evolution operator
    """
    U = scipy.linalg.expm(-1j * H * t)
    return U


def evolve_state(psi0, H, t):
    """
    Evolve a quantum state under the Hamiltonian.
    |ψ(t)⟩ = U(t)|ψ(0)⟩
    
    Parameters:
    -----------
    psi0 : 1D array of length 2
        Initial state
    H : 2x2 complex array
        Hamiltonian matrix
    t : float
        Evolution time
        
    Returns:
    --------
    psi : 1D array of length 2
        Evolved state
    """
    U = time_evolution(H, t)
    psi = U @ psi0
    return psi


def bloch_vector(psi):
    """
    Compute the Bloch vector from a quantum state.
    b = (⟨σx⟩, ⟨σy⟩, ⟨σz⟩) where ⟨σi⟩ = ⟨ψ|σi|ψ⟩
    
    Parameters:
    -----------
    psi : 1D array of length 2
        Quantum state
        
    Returns:
    --------
    b : 1D array of length 3
        Bloch vector components (x, y, z)
    """
    bx = np.real(np.vdot(psi, SIGMA_X @ psi))
    by = np.real(np.vdot(psi, SIGMA_Y @ psi))
    bz = np.real(np.vdot(psi, SIGMA_Z @ psi))
    return np.array([bx, by, bz])


def verify_bloch_on_unit_sphere(omega, Omega, num_times=100, time_max=10.0):
    """
    Verify that the Bloch vector remains on the unit sphere during evolution.
    
    Parameters:
    -----------
    omega : float
        Energy splitting parameter
    Omega : float
        Driving amplitude parameter
    num_times : int
        Number of time points to check
    time_max : float
        Maximum evolution time
        
    Returns:
    --------
    on_sphere : bool
        True if all Bloch vectors have norm ~= 1.0
    norms : list
        List of Bloch vector norms at each time
    """
    # Initialize state to |0>
    psi0 = np.array([1.0, 0.0], dtype=complex)
    
    # Construct Hamiltonian
    H = hamiltonian(omega, Omega)
    
    # Evolution times
    times = np.linspace(0, time_max, num_times)
    
    # Track norms of Bloch vectors
    bloch_norms = []
    max_deviation = 0.0
    
    print("Verifying Bloch vector remains on unit sphere:")
    print("Parameters: omega = {}, Omega = {}".format(omega, Omega))
    print("{:>8} {:>12} {:>15}".format("Time", "Bloch norm", "Deviation"))
    print("-" * 40)
    
    for i, t in enumerate(times):
        # Evolve state
        psi = evolve_state(psi0, H, t)
        
        # Verify normalization
        state_norm = np.linalg.norm(psi)
        assert np.isclose(state_norm, 1.0), "State not normalized at t={}".format(t)
        
        # Compute Bloch vector
        bloch = bloch_vector(psi)
        
        # Compute norm of Bloch vector
        bloch_norm = np.linalg.norm(bloch)
        bloch_norms.append(bloch_norm)
        
        # Deviation from 1.0
        deviation = abs(bloch_norm - 1.0)
        max_deviation = max(max_deviation, deviation)
        
        # Print every 20th step for readability
        if i % 20 == 0 or i == len(times) - 1:
            print("{:8.3f} {:12.10f} {:15.2e}".format(t, bloch_norm, deviation))
    
    print("-" * 40)
    print("Maximum deviation from unit sphere: {:.2e}".format(max_deviation))
    
    # Check that all norms are close to 1.0
    tolerance = 1e-10
    on_sphere = np.allclose(bloch_norms, 1.0, atol=tolerance)
    
    print("All Bloch vectors on unit sphere (tol={}): {}\n".format(tolerance, on_sphere))
    
    return on_sphere, bloch_norms


def explain_bloch_sphere_preservation():
    """
    Explain why the Bloch vector remains on the unit sphere.
    """
    explanation = """
EXPLANATION: Why the Bloch vector remains on the unit sphere
=============================================================

The Bloch vector b = <sigma> = (<sigma_x>, <sigma_y>, <sigma_z>) for a pure 
quantum state |psi> always lies on the unit sphere in the Bloch sphere representation.

Key physical reasons:

1. **Purity of the state**:
   For a pure state |psi>, the Bloch vector norm is exactly:
     |b|^2 = <sigma_x>^2 + <sigma_y>^2 + <sigma_z>^2 = 1
   This is a fundamental mathematical property of pure states in a two-level system.

2. **Unitarity of time evolution**:
   The time evolution operator U(t) = exp(-iHt) is unitary (since H is Hermitian).
   Unitary evolution preserves:
   - The norm of the state: ||psi(t)|| = ||psi(0)|| = 1
   - The purity of the state: pure states remain pure throughout evolution

3. **Conservation of Bloch vector norm**:
   During unitary evolution, the Bloch vector norm |b|^2 is a conserved quantity:
     d|b|^2/dt = 0
   This follows from the Heisenberg equation of motion:
     d<sigma_i>/dt = (i/h)<[H, sigma_i]>
   The Pauli matrices satisfy special commutation relations that ensure the sum
   <sigma_x>^2 + <sigma_y>^2 + <sigma_z>^2 remains constant.

4. **Geometric interpretation**:
   The Hamiltonian H defines an effective magnetic field:
     B_eff = (Omega, 0, omega/2)
   The Bloch vector precesses around this effective field, like a classical spinning
   top in a magnetic field, maintaining constant length (unit sphere radius).

CONCLUSION:
-----------
Purity is conserved under unitary evolution, and for 2-level systems, purity is 
equivalent to the Bloch vector having unit norm. Therefore, the Bloch vector remains 
on the unit sphere throughout the evolution.
"""
    return explanation


if __name__ == "__main__":
    print("=" * 70)
    print("BLOCH VECTOR UNIT SPHERE VERIFICATION")
    print("=" * 70)
    print()
    
    # Test Case 1: Standard parameters
    print("TEST CASE 1: Standard parameters (omega=1.0, Omega=1.0)")
    print("-" * 70)
    on_sphere_1, _ = verify_bloch_on_unit_sphere(omega=1.0, Omega=1.0)
    
    # Test Case 2: Resonant case
    print("TEST CASE 2: Resonant case (omega=0.0, Omega=1.0)")
    print("-" * 70)
    on_sphere_2, _ = verify_bloch_on_unit_sphere(omega=0.0, Omega=1.0)
    
    # Test Case 3: Different parameters
    print("TEST CASE 3: Different parameters (omega=2.0, Omega=0.5)")
    print("-" * 70)
    on_sphere_3, _ = verify_bloch_on_unit_sphere(omega=2.0, Omega=0.5)
    
    # Print explanation
    print("=" * 70)
    explanation = explain_bloch_sphere_preservation()
    print(explanation)
    
    # Final verdict
    print("=" * 70)
    if on_sphere_1 and on_sphere_2 and on_sphere_3:
        print("VERIFICATION SUCCESSFUL")
        print("Bloch vector remains on unit sphere for all test cases")
    else:
        print("VERIFICATION FAILED")
        print("Some test cases show deviation from unit sphere")
    print("=" * 70)
