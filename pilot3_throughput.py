"""
Pilot 3: Computational Throughput Scaling (Execution Latency)
==============================================================
Goal: Measures execution time (wall-clock / CPU time) rather than memory.
Circuit: Fixed depth of d=5 layers, Universal Non-Clifford gate set (H, T, CZ).
Qubit range: N in {10, 12, 14, 16, 18, 20}.
Constraints: Explicit timeout limit of 180 seconds.
Measurement: time.perf_counter() around execution phase, with early-stopping.
"""

import time
import math
import traceback
import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# GATE SEQUENCE GENERATOR (UNIVERSAL NON-CLIFFORD d=5)
# ==========================================================

def build_gate_sequence(n, depth=5):
    """
    Constructs a depth-d Universal Non-Clifford circuit:
    - Layer-wise H and T on every qubit (T introduces non-Clifford hardness).
    - Entangling CZ ladder between adjacent qubits.
    """
    ops = []
    for _ in range(depth):
        for qi in range(n):
            ops.append(('h', qi))
            ops.append(('t', qi))
        for qi in range(n - 1):
            ops.append(('cz', qi, qi + 1))
    return ops


# ==========================================================
# WORKER FUNCTIONS (ISOLATED TIMING IN SPAWNED SUBPROCESS)
# ==========================================================

def run_qiskit_speed(n, depth, repeats, q):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method='statevector')

        # 1-qubit warm-up to absorb initialization/thread pool costs
        warm = QuantumCircuit(1)
        warm.h(0)
        warm.save_statevector()
        backend.run(warm).result()

        ops = build_gate_sequence(n, depth=depth)
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
        ops = build_gate_sequence(n, depth=depth)
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

        ops = build_gate_sequence(n, depth=depth)
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

        ops = build_gate_sequence(n, depth=depth)

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

        ops = build_gate_sequence(n, depth=depth)

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


# ==========================================================
# BENCHMARK ORCHESTRATOR
# ==========================================================

SIMULATORS = {
    'Qiskit (Standard)': run_qiskit_speed,
    'Google qsim': run_qsim_speed,
    'MQT DDSIM': run_ddsim_speed,
    'Quimb (MPS)': run_quimb_speed,
    'PyQrack': run_pyqrack_speed,
}


def measure_in_subprocess(target, n, depth, repeats, timeout=180):
    """Run target(n, depth, repeats, queue) in a fresh subprocess with 180s timeout."""
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=target, args=(n, depth, repeats, queue))
    start = time.time()
    p.start()
    p.join(timeout)
    elapsed = time.time() - start

    if p.is_alive():
        p.terminate()
        p.join()
        return None, f"TIMEOUT after {elapsed:.1f}s", None

    if p.exitcode != 0:
        return None, f"exited with code {p.exitcode} after {elapsed:.1f}s", None

    if not queue.empty():
        payload = queue.get()
        if payload['error'] is not None:
            return None, payload['error'], None
        return payload['times'], f"ok ({elapsed:.1f}s total)", payload.get('warning')

    return None, f"no result returned after {elapsed:.1f}s", None


def main():
    QUBIT_COUNTS = [10, 12, 14, 16, 18, 20]
    DEPTH = 5
    REPEATS = 3

    print("=" * 70)
    print("Pilot 3: Computational Throughput Scaling (Execution Latency)")
    print(f"Universal Non-Clifford (H, T, CZ) | Depth d={DEPTH} | Timeout=180s")
    print("=" * 70 + "\n")

    results = {name: [] for name in SIMULATORS.keys()}
    active_simulators = list(SIMULATORS.keys())

    for n in QUBIT_COUNTS:
        print(f"\n================ Testing N = {n} Qubits ================")

        for name in list(active_simulators):
            print(f"Running {name}...", flush=True)
            fn = SIMULATORS[name]

            times, status, warning = measure_in_subprocess(fn, n, DEPTH, REPEATS, timeout=180)

            if times is None:
                print(f"  -> FAILED / {status}")
                print(f"  -> Early-stopping: dropping {name} from higher qubit tests.")
                results[name].append(np.nan)
                active_simulators.remove(name)
                continue

            best = min(times)
            print(f"  -> min: {best:.4f}s   [{status}]")
            results[name].append(best)

    df = pd.DataFrame(results, index=QUBIT_COUNTS)
    df.index.name = 'Qubits'

    print("\n" + "=" * 70)
    print("Pilot 3 Execution Latency Summary Table (Min Time in Seconds)")
    print("=" * 70)
    print(df.round(4))

    # --- Plotting the Scaling Graph ---
    plt.figure(figsize=(9, 6))

    styles = {
        'Qiskit (Standard)': {'color': '#e74c3c', 'marker': 'o', 'linestyle': '-'},
        'Google qsim': {'color': '#e67e22', 'marker': 's', 'linestyle': '--'},
        'MQT DDSIM': {'color': '#9b59b6', 'marker': 'D', 'linestyle': ':'},
        'Quimb (MPS)': {'color': '#2ecc71', 'marker': '^', 'linestyle': '-.'},
        'PyQrack': {'color': '#3498db', 'marker': 'v', 'linestyle': '-'}
    }

    for name, data_points in results.items():
        valid_n = [QUBIT_COUNTS[i] for i, val in enumerate(data_points) if not np.isnan(val)]
        valid_times = [val for val in data_points if not np.isnan(val)]

        if valid_times:
            plt.plot(valid_n, valid_times,
                     label=name,
                     color=styles[name]['color'],
                     marker=styles[name]['marker'],
                     linestyle=styles[name]['linestyle'],
                     linewidth=2, markersize=6)

    plt.title(f'Pilot 3: Computational Throughput Scaling (Depth $d={DEPTH}$)',
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Number of Qubits ($N$)', fontsize=11)
    plt.ylabel('Execution Time (seconds) [log scale]', fontsize=11)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xticks(QUBIT_COUNTS)
    plt.legend(fontsize=10, loc='upper left', framealpha=0.9)
    plt.tight_layout()

    plt.savefig('pilot3_throughput.pdf', format='pdf', dpi=300)
    plt.savefig('pilot3_throughput.png', format='png', dpi=300)
    print("\nGraph successfully saved as 'pilot3_throughput.pdf' and 'pilot3_throughput.png'")
    plt.show()

    return df


if __name__ == '__main__':
    main()

