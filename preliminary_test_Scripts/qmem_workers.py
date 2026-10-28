"""
Worker functions for the quantum simulator memory benchmark.
Features deep volume-law entanglement (10 layers) and forces all simulators
to evaluate their mathematical states to prevent lazy-evaluation "cheating".
"""

import traceback
import math

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


def run_qiskit(n, q):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method='statevector')

        # Warm-up
        warm = QuantumCircuit(2)
        warm.h(0)
        warm.rz(math.pi / 7, 0)
        warm.cx(0, 1)
        warm.save_statevector()
        backend.run(warm).result()

        baseline = _peak_mem_mb()

        qc = QuantumCircuit(n)
        qc.h(range(n))

        # Deep volume-law entanglement (10 layers = 5 loops x 2 brickwork stages)
        for layer in range(5):
            for i in range(n):
                qc.rz(math.pi / (i + 3), i)
            # Even-odd brickwork
            for i in range(0, n - 1, 2):
                qc.cx(i, i + 1)
            # Odd-even brickwork
            for i in range(1, n - 1, 2):
                qc.cx(i, i + 1)

        qc.save_statevector()
        result = backend.run(qc).result()
        _ = result.get_statevector()  # Anti-cheating: forces state array materialization

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


def run_qsim(n, q):
    try:
        import cirq
        import qsimcirq

        sim = qsimcirq.QSimSimulator()

        # Warm-up
        warm_qubits = cirq.LineQubit.range(2)
        warm_qc = cirq.Circuit(
            cirq.H(warm_qubits[0]),
            cirq.rz(rads=math.pi / 7)(warm_qubits[0]),
            cirq.CX(warm_qubits[0], warm_qubits[1])
        )
        sim.simulate(warm_qc)

        baseline = _peak_mem_mb()

        qubits = cirq.LineQubit.range(n)
        circuit = cirq.Circuit()

        for qb in qubits:
            circuit.append(cirq.H(qb))

        # Deep volume-law entanglement (10 layers = 5 loops x 2 brickwork stages)
        for layer in range(5):
            for i, qb in enumerate(qubits):
                circuit.append(cirq.rz(rads=math.pi / (i + 3))(qb))
            for i in range(0, n - 1, 2):
                circuit.append(cirq.CX(qubits[i], qubits[i + 1]))
            for i in range(1, n - 1, 2):
                circuit.append(cirq.CX(qubits[i], qubits[i + 1]))

        _ = sim.simulate(circuit)  # Anti-cheating: forces full simulation

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


def run_quimb(n, q):
    try:
        import quimb.tensor as qtn

        # Warm-up
        warm = qtn.Circuit(2)
        warm.apply_gate('H', 0)
        warm.apply_gate('RZ', math.pi / 7, 0)
        warm.apply_gate('CX', 0, 1)
        _ = warm.psi.contract()

        baseline = _peak_mem_mb()

        circ = qtn.Circuit(n)
        for i in range(n):
            circ.apply_gate('H', i)

        # Deep volume-law entanglement (10 layers = 5 loops x 2 brickwork stages)
        for layer in range(5):
            for i in range(n):
                circ.apply_gate('RZ', math.pi / (i + 3), i)
            for i in range(0, n - 1, 2):
                circ.apply_gate('CX', i, i + 1)
            for i in range(1, n - 1, 2):
                circ.apply_gate('CX', i, i + 1)

        # Anti-cheating: .contract() forces tensor network multiplication
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

        # Warm-up
        warm = QuantumCircuit(2)
        warm.h(0)
        warm.rz(math.pi / 7, 0)
        warm.cx(0, 1)
        warm.measure_all()
        backend.run(warm, shots=1).result()

        baseline = _peak_mem_mb()

        qc = QuantumCircuit(n)
        qc.h(range(n))

        # Deep volume-law entanglement (10 layers = 5 loops x 2 brickwork stages)
        for layer in range(5):
            for i in range(n):
                qc.rz(math.pi / (i + 3), i)
            for i in range(0, n - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, n - 1, 2):
                qc.cx(i, i + 1)

        qc.measure_all()
        _ = backend.run(qc, shots=1).result()  # Anti-cheating: forces DD sampling

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})


def run_pyqrack(n, q):
    try:
        from pyqrack import QrackSimulator

        # Warm-up
        warm = QrackSimulator(2)
        warm.h(0)
        warm.r(1, math.pi / 7, 0)
        warm.mcx([0], 1)
        warm.m(0)
        del warm

        baseline = _peak_mem_mb()

        sim = QrackSimulator(n)
        for i in range(n):
            sim.h(i)

        # Deep volume-law entanglement (10 layers = 5 loops x 2 brickwork stages)
        for layer in range(5):
            for i in range(n):
                sim.r(1, math.pi / (i + 3), i)
            for i in range(0, n - 1, 2):
                sim.mcx([i], i + 1)
            for i in range(1, n - 1, 2):
                sim.mcx([i], i + 1)

        # Anti-cheating: forces state calculation via measurement
        sim.m(0)

        q.put({'mem': max(0.0, _peak_mem_mb() - baseline), 'error': None})
    except Exception:
        q.put({'mem': None, 'error': traceback.format_exc()})

