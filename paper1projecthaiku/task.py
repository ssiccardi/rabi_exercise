# task:
# Consider a two-level system driven by a monochromatic field described by
# H(t)= epsilon0 |0><0| + epsilon1 |1><1| + 2 Omega cos(omega t)(|0><1| + |1><0|)
# Simulate the Bloch-vector trajectory for different values of ω.
# Describe the most relevant geometric features and explain their physical origin.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

def rabi_hamiltonian(t, omega_d, Omega, epsilon0, epsilon1):
    """
    Construct the time-dependent Hamiltonian matrix.
    H(t) = epsilon0 |0><0| + epsilon1 |1><1| + 2*Omega*cos(omega_d*t) (|0><1| + |1><0|)
    
    In matrix form:
    H(t) = [[epsilon0, Omega*cos(omega_d*t)],
            [Omega*cos(omega_d*t), epsilon1]]
    """
    sigma_x_coupling = 2 * Omega * np.cos(omega_d * t)
    H = np.array([
        [epsilon0, sigma_x_coupling],
        [sigma_x_coupling, epsilon1]
    ], dtype=complex)
    return H

def bloch_vector_from_state(psi):
    """
    Compute the Bloch vector components from a normalized state |psi>.
    Bloch vector: r = (<sigma_x>, <sigma_y>, <sigma_z>)
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    x = np.real(np.vdot(psi, sigma_x @ psi))
    y = np.real(np.vdot(psi, sigma_y @ psi))
    z = np.real(np.vdot(psi, sigma_z @ psi))
    
    return np.array([x, y, z])

def psi_to_vector(psi):
    """Convert quantum state psi to a 4-element real vector for ODE solver."""
    # Store as [Re(psi[0]), Im(psi[0]), Re(psi[1]), Im(psi[1])]
    return np.array([psi[0].real, psi[0].imag, psi[1].real, psi[1].imag])

def vector_to_psi(v):
    """Convert 4-element real vector back to quantum state."""
    return np.array([v[0] + 1j*v[1], v[2] + 1j*v[3]], dtype=complex)

def time_dependent_schrodinger(y, t, omega_d, Omega, epsilon0, epsilon1):
    """
    Solve the time-dependent Schrödinger equation: i*d|psi>/dt = H(t)|psi>
    Converted to real ODE for numerical integration.
    dy/dt = -Im(H)|psi> and dy/dt = -Re(H)|psi> (in appropriate form)
    """
    psi = vector_to_psi(y)
    H = rabi_hamiltonian(t, omega_d, Omega, epsilon0, epsilon1)
    
    # i * dpsi/dt = H * psi  =>  dpsi/dt = -i * H * psi
    dpsi_dt = -1j * H @ psi
    
    return psi_to_vector(dpsi_dt)

def simulate_bloch_trajectory(omega_d, Omega, epsilon0=0.0, epsilon1=1.0, 
                              psi0=None, t_max=10.0, num_points=500):
    """
    Simulate the time evolution of a two-level system under time-dependent driving.
    
    Parameters:
    -----------
    omega_d : float
        Driving frequency
    Omega : float
        Rabi frequency (coupling strength)
    epsilon0, epsilon1 : float
        Energy level offsets
    psi0 : complex array
        Initial state (default: |0>)
    t_max : float
        Maximum simulation time
    num_points : int
        Number of time points
    
    Returns:
    --------
    times : array
        Time points
    bloch_trajectory : array (N, 3)
        Bloch vectors at each time
    states : array (N, 2)
        Quantum states at each time
    """
    epsilon1 = omega_d   # for coherence with the paper
    if psi0 is None:
        psi0 = np.array([1.0, 0.0], dtype=complex)  # |0> state
    
    # Normalize initial state
    psi0 = psi0 / np.linalg.norm(psi0)
    
    # Initial condition for ODE solver
    y0 = psi_to_vector(psi0)
    
    # Time points
    times = np.linspace(0, t_max, num_points)
    
    # Solve ODE
    solution = odeint(time_dependent_schrodinger, y0, times,
                      args=(omega_d, Omega, epsilon0, epsilon1),
                      rtol=1e-8, atol=1e-10)
    
    # Convert solutions back to quantum states and Bloch vectors
    states = np.array([vector_to_psi(v) for v in solution])
    bloch_trajectory = np.array([bloch_vector_from_state(psi) for psi in states])
    
    return times, bloch_trajectory, states

def analyze_bloch_trajectory(bloch_traj, times, omega_d, Omega):
    """
    Analyze and print geometric features of the Bloch trajectory.
    """
    print(f"\n{'='*70}")
    print(f"Analysis for omega_d = {omega_d:.3f}, Omega = {Omega:.3f}")
    print(f"{'='*70}")
    
    # Radius of Bloch vector (should be 1 for pure states)
    radii = np.linalg.norm(bloch_traj, axis=1)
    print(f"Bloch vector norm (should be ~1.0 for pure states):")
    print(f"  Min: {radii.min():.6f}, Max: {radii.max():.6f}, Mean: {radii.mean():.6f}")
    
    # Analyze trajectory in Bloch space
    # Look for rotation patterns
    velocities = np.diff(bloch_traj, axis=0)
    speed = np.linalg.norm(velocities, axis=1)
    print(f"\nTrajectory speed (|dr/dt|):")
    print(f"  Min: {speed.min():.6f}, Max: {speed.max():.6f}")
    
    # Detuning from resonance
    detuning = abs(omega_d - 1.0)  # Assuming epsilon0=0, epsilon1=1 => transition freq = 1
    print(f"\nDetuning from resonance: Delta = |omega_d - (eps1-eps0)| = {detuning:.3f}")
    
    # Key regimes
    if detuning < 0.1 * Omega:
        regime = "On-resonance (omega_d = transition frequency)"
    elif detuning < 3 * Omega:
        regime = "Intermediate detuning"
    else:
        regime = "Far off-resonance"
    print(f"Regime: {regime}")
    
    return radii, speed

def plot_trajectories_comparison(omega_list, Omega=1.0, t_max=20.0):
    """
    Plot Bloch trajectories for different driving frequencies.
    Illustrates the effect of detuning on the trajectory shape.
    """
    fig = plt.figure(figsize=(16, 10))
    
    for idx, omega_d in enumerate(omega_list):
        times, bloch_traj, states = simulate_bloch_trajectory(
            omega_d=omega_d, 
            Omega=Omega, 
            t_max=t_max, 
            num_points=400
        )
        
        # 3D subplot
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        # Plot trajectory
        ax.plot(bloch_traj[:, 0], bloch_traj[:, 1], bloch_traj[:, 2], 
                'b-', linewidth=1.5, alpha=0.7)
        
        # Mark start and end
        ax.scatter(*bloch_traj[0], color='green', s=100, marker='o', label='Start')
        ax.scatter(*bloch_traj[-1], color='red', s=100, marker='s', label='End')
        
        # Draw unit sphere (wireframe)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'omega_d = {omega_d:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('bloch_trajectories_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: bloch_trajectories_comparison.png")
    plt.show()

def plot_detailed_analysis(omega_d, Omega=1.0, t_max=20.0):
    """
    Detailed plot of Bloch trajectory with additional information.
    """
    times, bloch_traj, states = simulate_bloch_trajectory(
        omega_d=omega_d, 
        Omega=Omega, 
        t_max=t_max, 
        num_points=500
    )
    
    # Analyze trajectory
    radii, speed = analyze_bloch_trajectory(bloch_traj, times, omega_d, Omega)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(bloch_traj[:, 0], bloch_traj[:, 1], bloch_traj[:, 2], 
            'b-', linewidth=2, alpha=0.8, label='Trajectory')
    ax1.scatter(*bloch_traj[0], color='green', s=100, marker='o', label='t=0')
    ax1.scatter(*bloch_traj[-1], color='red', s=100, marker='s', label=f't={t_max}')
    
    # Unit sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel('sigma_x')
    ax1.set_ylabel('sigma_y')
    ax1.set_zlabel('sigma_z')
    ax1.set_title(f'Bloch Sphere Trajectory\nomega_d = {omega_d:.3f}, Omega = {Omega:.3f}')
    ax1.legend()
    
    # Bloch vector components vs time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, bloch_traj[:, 0], label='sigma_x', linewidth=2)
    ax2.plot(times, bloch_traj[:, 1], label='sigma_y', linewidth=2)
    ax2.plot(times, bloch_traj[:, 2], label='sigma_z', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Bloch Vector Component')
    ax2.set_title('Bloch Vector Components vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Trajectory radius vs time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times, radii, 'k-', linewidth=2)
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Pure state (r=1)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Bloch Vector Norm |r|')
    ax3.set_title('Purity: Bloch Vector Norm vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0.95, 1.05])
    
    # Speed vs time
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(times[:-1], speed, 'g-', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Trajectory Speed |dr/dt|')
    ax4.set_title('Trajectory Speed vs Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'bloch_detailed_analysis_omega{omega_d:.2f}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: bloch_detailed_analysis_omega{omega_d:.2f}.png")
    plt.show()

def plot_projections(omega_d, Omega=1.0, t_max=20.0):
    """
    Plot 2D projections of Bloch trajectory on XY, XZ, and YZ planes.
    """
    times, bloch_traj, states = simulate_bloch_trajectory(
        omega_d=omega_d, 
        Omega=Omega, 
        t_max=t_max, 
        num_points=500
    )
    
    fig = plt.figure(figsize=(15, 5))
    
    # Color the trajectory by time for better visualization
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    # XY plane projection
    ax1 = fig.add_subplot(1, 3, 1)
    for i in range(len(times) - 1):
        ax1.plot(bloch_traj[i:i+2, 0], bloch_traj[i:i+2, 1], 
                color=colors[i], linewidth=2)
    ax1.scatter(bloch_traj[0, 0], bloch_traj[0, 1], color='green', 
               s=150, marker='o', label='Start', zorder=5)
    ax1.scatter(bloch_traj[-1, 0], bloch_traj[-1, 1], color='red', 
               s=150, marker='s', label='End', zorder=5)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1.5)
    ax1.set_xlim([-1.3, 1.3])
    ax1.set_ylim([-1.3, 1.3])
    ax1.set_aspect('equal')
    ax1.set_xlabel('sigma_x')
    ax1.set_ylabel('sigma_y')
    ax1.set_title(f'XY Projection (omega_d = {omega_d:.2f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # XZ plane projection
    ax2 = fig.add_subplot(1, 3, 2)
    for i in range(len(times) - 1):
        ax2.plot(bloch_traj[i:i+2, 0], bloch_traj[i:i+2, 2], 
                color=colors[i], linewidth=2)
    ax2.scatter(bloch_traj[0, 0], bloch_traj[0, 2], color='green', 
               s=150, marker='o', label='Start', zorder=5)
    ax2.scatter(bloch_traj[-1, 0], bloch_traj[-1, 2], color='red', 
               s=150, marker='s', label='End', zorder=5)
    
    # Draw unit circle
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1.5)
    ax2.set_xlim([-1.3, 1.3])
    ax2.set_ylim([-1.3, 1.3])
    ax2.set_aspect('equal')
    ax2.set_xlabel('sigma_x')
    ax2.set_ylabel('sigma_z')
    ax2.set_title(f'XZ Projection (omega_d = {omega_d:.2f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # YZ plane projection
    ax3 = fig.add_subplot(1, 3, 3)
    for i in range(len(times) - 1):
        ax3.plot(bloch_traj[i:i+2, 1], bloch_traj[i:i+2, 2], 
                color=colors[i], linewidth=2)
    ax3.scatter(bloch_traj[0, 1], bloch_traj[0, 2], color='green', 
               s=150, marker='o', label='Start', zorder=5)
    ax3.scatter(bloch_traj[-1, 1], bloch_traj[-1, 2], color='red', 
               s=150, marker='s', label='End', zorder=5)
    
    # Draw unit circle
    ax3.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1.5)
    ax3.set_xlim([-1.3, 1.3])
    ax3.set_ylim([-1.3, 1.3])
    ax3.set_aspect('equal')
    ax3.set_xlabel('sigma_y')
    ax3.set_ylabel('sigma_z')
    ax3.set_title(f'YZ Projection (omega_d = {omega_d:.2f})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'bloch_projections_omega{omega_d:.2f}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: bloch_projections_omega{omega_d:.2f}.png")
    plt.show()

# ============================================================================
# PHYSICAL INTERPRETATION AND GEOMETRIC FEATURES
# ============================================================================
"""
GEOMETRIC FEATURES OF BLOCH TRAJECTORIES:

1. RESONANCE EFFECTS (omega_d approximately equal to transition frequency):
   - When driving frequency matches the transition energy gap (eps1 - eps0), 
     the system undergoes Rabi oscillations
   - Trajectory forms large ellipses or circles that drive the system between states
   - Strong coupling -> rapid state transitions
   - The trajectory precesses around the Z-axis in a regular pattern

2. FAR OFF-RESONANCE (omega_d >> gap or omega_d << gap):
   - Rapid oscillations at the driving frequency
   - The envelope of oscillations is small (weak effective coupling)
   - Trajectory appears as tight spirals or small loops
   - The system barely leaves its initial state
   - This is related to the Rotating Wave Approximation (RWA)

3. INTERMEDIATE DETUNING:
   - Complex trajectories with both slow and fast oscillations
   - Combination of Rabi oscillations and AC Stark shift effects
   - Non-trivial geometric features: cusps, loops, or complex winding patterns
   - Physical origin: interplay between resonant and non-resonant driving terms

4. PURITY PRESERVATION:
   - All pure states lie on the unit Bloch sphere (|r| = 1)
   - The trajectory MUST remain on the unit sphere due to unitarity of evolution
   - Any deviation indicates numerical errors

5. ROTATING WAVE APPROXIMATION (RWA) CONNECTION:
   - In the rotating frame at omega_d, one can transform away the fast oscillations
   - In this frame: H_RWA ≈ detuning*sigma_z/2 + Omega*sigma_x
   - RWA is valid when Omega << omega_d (weak coupling limit)
   - Beyond RWA: counter-rotating terms become important, causing additional features
"""

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DRIVEN TWO-LEVEL SYSTEM: BLOCH TRAJECTORY ANALYSIS")
    print("="*70)
    
    # Parameters
    Omega = 1.0  # Rabi frequency (coupling strength)
    t_max =  2*np.pi  #20.0  # Simulation time
    
    # Different driving frequencies to explore
    # Including resonance, near-resonance, and off-resonance cases
#    omega_frequencies = [0.5, 1.0, 1.5, 2.5, 4.0]
    omega_frequencies = [2.5, 5.0, 20.0]
    
    print("\nSimulating Bloch trajectories for different driving frequencies...")
    print("Parameters: Omega = {:.2f}, transition frequency = 1.0, t_max = {:.1f}".format(Omega, t_max))
    
    # Comparison plot
    plot_trajectories_comparison(omega_frequencies, Omega=Omega, t_max=t_max)
    
    # Detailed analysis for key regimes
    print("\nDetailed analysis for key regimes:")
    
    # On-resonance
    print("\n--- On-Resonance Case (omega_d = 2.5) ---")
    plot_detailed_analysis(omega_d=2.5, Omega=Omega, t_max=t_max)
    plot_projections(omega_d=2.5, Omega=Omega, t_max=t_max)
    
    # Far off-resonance
    print("\n--- Far Off-Resonance Case (omega_d = 20.0) ---")
    plot_detailed_analysis(omega_d=20.0, Omega=Omega, t_max=t_max)
    plot_projections(omega_d=20.0, Omega=Omega, t_max=t_max)
    
    print("\n" + "="*70)
    print("SUMMARY OF PHYSICAL INSIGHTS")
    print("="*70)
    print("""
The Bloch vector trajectories reveal fundamental quantum dynamics:

• RESONANCE (omega_d = 1.0): Large-amplitude oscillations, strong state transfer capability.
  The trajectory drives the system efficiently between |0> and |1> states.
  
• OFF-RESONANCE (omega_d >> Omega): Tiny loops, virtual transitions dominate.
  The AC Stark shift causes a small rotation of the Bloch vector around an effective axis.
  
• INTERMEDIATE DETUNING: Complex trajectories revealing the breakdown of RWA.
  Counter-rotating terms create geometric features not predicted by simple models.
  
• GEOMETRIC SIGNIFICANCE: All trajectories lie on the unit Bloch sphere.
  This manifests PURITY: the evolving state remains pure throughout (no decoherence).
  
• PHYSICAL ORIGIN: The shape is determined by the interplay between:
  - The detuning (omega_d - (eps1 - eps0))
  - The coupling strength Omega
  - The ratio Omega/detuning (dimensionless parameter controlling the regime)
    """)
    print("="*70)
