We use a 2 level quantum system to test the ability of a simple and a powerful model to:

* write correct code
* implement suitable tests to check that the code is correct
* manage vague or incorrect prompts



\*\* prompts to check \*\*

\*\* Simulate a two-level quantum system with Hamiltonian H = ω/2 σz + Ω σx. 

Plot the probability of measuring state |1⟩ and verify the correctness of the simulation. \*\*

* haiku only checks the time to achieve maximal probability of |1> and such time is not correct due to a bug
* opus checks:
* Verification (omega = 1.0, Omega = 1.0, R = 1.1180):
* &#x20; \[PASS] H is Hermitian
* &#x20; \[PASS] eigenvalues are +-R
* &#x20; \[PASS] norm conserved (eigen)
* &#x20; \[PASS] norm conserved (RK4)
* &#x20; \[PASS] P0 + P1 = 1
* &#x20; \[PASS] eigen vs analytic (max err 6.7e-16)
* &#x20; \[PASS] RK4 vs analytic (max err 5.4e-13)
* &#x20; \[PASS] closed-form propagator
* &#x20; \[PASS] resonant pi-pulse gives P1 = 1
* &#x20; \[PASS] no drive (Omega = 0) gives P1 = 0
* &#x20; Expected max P1 = Omega^2/R^2 = 0.8000, simulated = 0.8000
* &#x20; Expected period = pi/R = 2.8099



\*\* Compute the fidelity between the evolved state and a target state. \*\*

* normalization of states ok for both



\*\* Simulate a qubit initialized in |0⟩ under the Hamiltonian H = Ω σx and verify that the target state |1⟩ is reached at time π/Ω. \*\*

* the prompt is wrong (the correct time is one half), both models correctly identified the mistake



\*\* Verify numerically that the Bloch vector remains on the unit sphere during the evolution, and explain why. \*\*

* haiku succeeds and gives a reasonable explanation
* I request opus to check haiku's work, and it adds some observations





A second example uses the situation of https://arxiv.org/pdf/1305.1336



\*\* Prompt:

Consider a two-level system driven by a monochromatic field described by

H(t)= epsilon0 |0><0| + epsilon1 |1><1| + 2 Omega cos(omega t)(|0><1| + |1><0|)

Simulate the Bloch-vector trajectory for different values of ω.

Describe the most relevant geometric features and explain their physical origin. \*\*

* opus considers both the laboratory frame and a rotating frame, haiku only the laboratory
* when requested, opus can also write a detailed document describing both the physics and the implementation (explanations.md)







In all the repo: haiku means Haiku 4.5 and opus means Opus 5.5

