"""
Pilot 1: Symmetric Equal Superposition (Memory Baseline)
=========================================================
Goal: Measures incremental memory footprint without entanglement.
Circuit: Applies a Hadamard (H) gate to all wires across N in [10, 24].
Entanglement: Exactly zero two-qubit gates (no CNOTs).
Measurement: Isolated multiprocessing 'spawn' context using
             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss.
"""

import time
import math
import traceback
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt

try:
    import resource
    def _peak_mem_mb():
        """Peak RSS of this process so far, in MB. Kernel-tracked, no polling gap."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    import psutil
    def _peak_mem_mb():
        """Fallback for environments without Unix resource module."""
        return psutil.Process().memory_info().peak_wset / (1024.0 * 1024.0)


# ==========================================================
# WORKER FUNCTIONS (ISOLATED EXECUTION IN SPAWNED SUBPROCESS)
# ==========================================================

def run_qiskit(n, q):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method='statevector')

        # Warm-up: trigger one-time backend init before baseline snapshot
        warm = QuantumCircuit(1)
        warm.h(0)
        warm.save_statevector()
        backend.run(warm).result()

        baseline = _peak_mem_mb()

        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.save_statevector()
        result = backend.run(qc).result()
        _ = result.get_statevector()

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


def run_qsim(n, q):
    try:
        import cirq
        import qsimcirq

        sim = qsimcirq.QSimSimulator()

        warm_qubits = cirq.LineQubit.range(1)
        sim.simulate(cirq.Circuit(cirq.H(warm_qubits[0])))

        baseline = _peak_mem_mb()

        qubits = cirq.LineQubit.range(n)
        circuit = cirq.Circuit(cirq.H(qb) for qb in qubits)
        _ = sim.simulate(circuit)

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


def run_quimb(n, q):
    try:
        import quimb.tensor as qtn

        warm = qtn.Circuit(1)
        warm.apply_gate('H', 0)
        _ = warm.psi.contract()

        baseline = _peak_mem_mb()

        circ = qtn.Circuit(n)
        for i in range(n):
            circ.apply_gate('H', i)
        
        _ = circ.psi.contract()

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


def run_ddsim(n, q):
    try:
        from qiskit import QuantumCircuit
        from mqt.ddsim import DDSIMProvider

        provider = DDSIMProvider()
        backend = provider.get_backend('qasm_simulator')

        warm = QuantumCircuit(1)
        warm.h(0)
        warm.measure_all()
        backend.run(warm, shots=1).result()

        baseline = _peak_mem_mb()

        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.measure_all()
        _ = backend.run(qc, shots=1).result()

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


def run_pyqrack(n, q):
    try:
        from pyqrack import QrackSimulator

        warm = QrackSimulator(1)
        warm.h(0)
        warm.m(0)
        del warm

        baseline = _peak_mem_mb()

        sim = QrackSimulator(n)
        for i in range(n):
            sim.h(i)
            
        sim.m(0)

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


# ==========================================================
# BENCHMARK ORCHESTRATOR
# ==========================================================

SIMULATORS = {
    'Qiskit (Statevector)': run_qiskit,
    'Google qsim (Statevector)': run_qsim,
    'Quimb (Tensor Network)': run_quimb,
    'MQT DDSIM (Decision Diagrams)': run_ddsim,
    'PyQrack (GPU-capable Statevector)': run_pyqrack,
}


def measure_in_subprocess(target, n, timeout=60):
    """Run target(n, queue) in a clean subprocess via 'spawn'."""
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=target, args=(n, queue))
    start = time.time()
    p.start()
    p.join(timeout)
    elapsed = time.time() - start

    if p.is_alive():
        p.terminate()
        p.join()
        return None, f"TIMEOUT after {elapsed:.1f}s"

    if p.exitcode != 0:
        return None, f"exited with code {p.exitcode} after {elapsed:.1f}s"

    if not queue.empty():
        payload = queue.get()
        if payload['error'] is not None:
            return None, payload['error']
        return payload['mem'], f"ok ({elapsed:.1f}s)"

    return None, f"no result returned after {elapsed:.1f}s"


def main():
    qubit_counts = [10, 14, 16, 18, 20, 22, 24]
    results = []

    print("=" * 70)
    print("Pilot 1: Symmetric Equal Superposition (Memory Baseline)")
    print("Zero two-qubit gates | Isolated spawn context | ru_maxrss")
    print("=" * 70 + "\n")

    for n in qubit_counts:
        print(f"Testing N = {n} qubits...")
        current_data = {'Qubits': n}

        for name, fn in SIMULATORS.items():
            mem, status = measure_in_subprocess(fn, n)
            current_data[name] = mem
            if mem is None:
                print(f"  -> {name}: FAILED — {status}")
            else:
                print(f"  -> {name}: {mem:.3f} MB [{status}]")

        results.append(current_data)

    df = pd.DataFrame(results).set_index('Qubits')

    print("\n" + "=" * 70)
    print("Pilot 1 Memory Summary Table (Incremental Peak MB)")
    print("=" * 70)
    print(df.round(3))

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#f1c40f', '#2ecc71']
    markers = ['o', 's', '^', 'D', 'v']

    plot_cols = [c for c in df.columns if df[c].notna().any() and (df[c].fillna(0) > 0).any()]
    for i, col in enumerate(plot_cols):
        plt.plot(df.index, df[col], marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], label=col, linewidth=2)

    plt.title('Pilot 1: Memory Footprint under Symmetric Equal Superposition (Zero Entanglement)',
              fontsize=12, fontweight='bold')
    plt.xlabel('Number of Qubits (N)', fontsize=11)
    plt.ylabel('Incremental Peak Memory (MB)', fontsize=11)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if plot_cols:
        plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig('pilot1_memory_baseline.png', dpi=150)
    print("\nPlot saved to 'pilot1_memory_baseline.png'")
    plt.show()

    return df


if __name__ == '__main__':
    main()
