"""
Worker functions for the computational-throughput (timing) benchmark.

Design notes:
1. WARM-UP: Each backend is constructed and run once on a 1-qubit circuit before timing.
2. TIMED REGION: Only the simulation/evaluation is timed.
3. REPEATS: Multiple repeats, reports minimum time.
4. PyQrack: isSchmidtDecompose=False for apples-to-apples statevector comparison.
5. Quimb: .to_dense() forces full state representation materialization.
"""

import time
import traceback
from qmem_circuits import build_gate_sequence


def run_qiskit_speed(n, depth, repeats, q):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method='statevector')

        warm = QuantumCircuit(1)
        warm.h(0)
        warm.save_statevector()
        backend.run(warm).result()

        ops = build_gate_sequence(n, 'nonclifford', depth=depth)
        qc = QuantumCircuit(n)
        for op in ops:
            if op[0] == 'h':
                qc.h(op[1])
            elif op[0] == 't':
                qc.t(op[1])
            elif op[0] == 'cz':
                qc.cz(op[1], op[2])
        qc.save_statevector()

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = backend.run(qc).result()
            _ = result.get_statevector()
            times.append(time.perf_counter() - t0)

        q.put({'times': times, 'error': None, 'warning': None})
    except Exception:
        q.put({'times': None, 'error': traceback.format_exc(), 'warning': None})


def run_qsim_speed(n, depth, repeats, q):
    try:
        import cirq
        import qsimcirq

        sim = qsimcirq.QSimSimulator()
        warm_qubits = cirq.LineQubit.range(1)
        sim.simulate(cirq.Circuit(cirq.H(warm_qubits[0])))

        qubits = cirq.LineQubit.range(n)
        ops = build_gate_sequence(n, 'nonclifford', depth=depth)
        gates = []
        for op in ops:
            if op[0] == 'h':
                gates.append(cirq.H(qubits[op[1]]))
            elif op[0] == 't':
                gates.append(cirq.T(qubits[op[1]]))
            elif op[0] == 'cz':
                gates.append(cirq.CZ(qubits[op[1]], qubits[op[2]]))
        circuit = cirq.Circuit(gates)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = sim.simulate(circuit)
            _ = result.state_vector()
            times.append(time.perf_counter() - t0)

        q.put({'times': times, 'error': None, 'warning': None})
    except Exception:
        q.put({'times': None, 'error': traceback.format_exc(), 'warning': None})


def run_ddsim_speed(n, depth, repeats, q):
    try:
        from qiskit import QuantumCircuit
        from mqt.ddsim import DDSIMProvider

        provider = DDSIMProvider()
        backend = provider.get_backend('statevector_simulator')

        warm = QuantumCircuit(1)
        warm.h(0)
        backend.run(warm).result()

        ops = build_gate_sequence(n, 'nonclifford', depth=depth)
        qc = QuantumCircuit(n)
        for op in ops:
            if op[0] == 'h':
                qc.h(op[1])
            elif op[0] == 't':
                qc.t(op[1])
            elif op[0] == 'cz':
                qc.cz(op[1], op[2])

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = backend.run(qc).result()
            _ = result.get_statevector()
            times.append(time.perf_counter() - t0)

        q.put({'times': times, 'error': None, 'warning': None})
    except Exception:
        q.put({'times': None, 'error': traceback.format_exc(), 'warning': None})


def run_quimb_speed(n, depth, repeats, q):
    try:
        import quimb.tensor as qtn

        warm = qtn.CircuitMPS(1)
        warm.apply_gate('H', 0)
        _ = warm.psi

        ops = build_gate_sequence(n, 'nonclifford', depth=depth)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            circ = qtn.CircuitMPS(n)
            for op in ops:
                if op[0] == 'h':
                    circ.apply_gate('H', op[1])
                elif op[0] == 't':
                    circ.apply_gate('T', op[1])
                elif op[0] == 'cz':
                    circ.apply_gate('CZ', op[1], op[2])
            _ = circ.psi.to_dense()
            times.append(time.perf_counter() - t0)

        q.put({'times': times, 'error': None, 'warning': None})
    except Exception:
        q.put({'times': None, 'error': traceback.format_exc(), 'warning': None})


def _make_qrack_sim(n):
    from pyqrack import QrackSimulator
    try:
        return QrackSimulator(n, isSchmidtDecompose=False), True
    except TypeError:
        return QrackSimulator(n), False


def run_pyqrack_speed(n, depth, repeats, q):
    try:
        warm, _schmidt_disabled = _make_qrack_sim(1)
        warm.h(0)
        del warm

        ops = build_gate_sequence(n, 'nonclifford', depth=depth)

        times = []
        schmidt_disabled = True
        for _ in range(repeats):
            t0 = time.perf_counter()
            sim, schmidt_disabled = _make_qrack_sim(n)
            for op in ops:
                if op[0] == 'h':
                    sim.h(op[1])
                elif op[0] == 't':
                    sim.t(op[1])
                elif op[0] == 'cz':
                    sim.mcz([op[1]], op[2])
            try:
                _ = sim.out_ket()
            except AttributeError:
                _ = [sim.prob(i) for i in range(n)]
            times.append(time.perf_counter() - t0)
            del sim

        warning = None if schmidt_disabled else (
            "installed pyqrack version has no isSchmidtDecompose option; "
            "ran with default engine."
        )
        q.put({'times': times, 'error': None, 'warning': warning})
    except Exception:
        q.put({'times': None, 'error': traceback.format_exc(), 'warning': None})

