# Digital Simulation Platforms for Quantum Algorithms: Benchmarking Suite

This repository contains the isolated benchmarking suite and reproducible source code for the IEEE research paper: **"A Comparative Study on Digital Simulation Platforms for Quantum Algorithms."**

It evaluates the computational boundaries, memory scaling, and structural vulnerabilities of five prominent classical quantum simulation architectures: dense statevectors (**Google qsim**, **Qiskit Aer**), tensor networks/MPS (**Quimb**), decision diagrams (**MQT DDSIM**), and adaptive subsystems (**PyQrack**).

## 🗂️ Repository Structure

The benchmarking suite is divided into four strictly controlled multi-qubit experiments designed to decouple the effects of entanglement entropy and phase rotation complexity.

*   `benchmark_exp1_parity.py` — **Experiment 1: Classical Parity Permutation (Zero Entanglement)**
    Evaluates baseline memory overhead and execution latency for separable states ($\chi = 1$) using deterministic classical XOR logic (Pauli-$X$ and CNOT cascades) without superpositions.
*   `benchmark_exp2_chaos.py` — **Experiment 2: Deep Volume-Law Chaotic Stress Test**
    Evaluates backends under maximal computational hardness, combining volume-law bipartite entanglement (staggered CNOT brickwork) with continuous fractional non-Clifford phase shifts ($H, T, R_z$).
*   `benchmark_exp3_clifford.py` — **Experiment 3: Clifford-Only Entanglement Ablation**
    Isolates entanglement entropy by creating a massively entangled global GHZ state utilizing only Clifford group operations, allowing structural evaluation of graph and tensor symmetries.
*   `benchmark_exp4_rotations.py` — **Experiment 4: Non-Clifford Rotation (Zero Entanglement) Ablation**
    Evaluates the impact of continuous phase diversity ($H, T, R_z(\theta)$) on simulator compression structures in the complete absence of two-qubit entangling gates.

## ⚙️ Environment & Installation

### Dependency Pinning (NumPy 2.0 Compatibility)
Several backend libraries (such as `cirq` and `quimb`) have fragile C-extension dependencies. Recent updates to Kaggle and colab environments default to NumPy 2.0, which utilizes a 96-byte `dtype` C-header that is fundamentally incompatible with the 88-byte structure expected by older versions of Qiskit. 

To prevent `ValueError: numpy.dtype size changed` and `libgomp` TLS memory exhaustion, you **must** strictly pin the environment to the 1.26.x NumPy branch and compatible Qiskit builds.

Run the following installation command in a fresh environment:

```bash
pip uninstall -y numpy qiskit qiskit-aer
pip install "numpy<2.0" "qiskit==1.0.2" "qiskit-aer==0.13.3" mqt.ddsim cirq-core qsimcirq quimb cotengra pyqrack psutil pandas
