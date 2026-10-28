import multiprocessing as mp
import os
import resource
import time
import warnings
import numpy as np
import pandas as pd
import psutil

# Suppress harmless third-party notices
warnings.filterwarnings("ignore")

# ==========================================================
# IMPORT BACKENDS AT THE TOP (INHERITED BY FORKED PROCESSES)
# ==========================================================
import cirq
import qsimcirq
from mqt.ddsim import DDSIMProvider
from pyqrack import QrackSimulator
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import quimb.tensor as qtn

# Sweep configuration for Entangled benchmark
QUBIT_COUNTS = [12, 16, 18, 20, 22, 24]
TRIALS = 5            # 5 trials per N provides mean ± std
TIMEOUT_SEC = 120.0   # Safely kill simulator if bond dimension or state explodes

def get_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_trial(sim_name, n, conn):
    expected = "0"
    process = psutil.Process()

    try:
        # ==========================================================
        # PHASE 1: MEASURE SETUP / INITIALIZATION TIME
        # ==========================================================
        t_setup_start = time.perf_counter()

        if sim_name == "Google qsim (SV)":
            qubits = cirq.LineQubit.range(n)
            circuit = cirq.Circuit()
            # 1. Create GHZ Entanglement
            circuit.append(cirq.H(qubits[0]))
            for i in range(n - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
            # 2. Reverse Entanglement Cascade
            for i in reversed(range(n - 1)):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
            circuit.append(cirq.H(qubits[0]))
            circuit.append(cirq.measure(qubits[n - 1], key="m"))
            simulator = qsimcirq.QSimSimulator()

        elif sim_name == "MQT DDSIM (QDD)":
            qc = QuantumCircuit(n, 1)
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            for i in reversed(range(n - 1)):
                qc.cx(i, i + 1)
            qc.h(0)
            qc.measure(n - 1, 0)
            backend = DDSIMProvider().get_backend("qasm_simulator")

        elif sim_name == "PyQrack (Hybrid)":
            sim = QrackSimulator(n)

        elif sim_name == "Qiskit Aer (SV)":
            qc = QuantumCircuit(n, 1)
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            for i in reversed(range(n - 1)):
                qc.cx(i, i + 1)
            qc.h(0)
            qc.measure(n - 1, 0)
            # optimization_level=0 prevents transpiler gate cancellation
            sim = AerSimulator(method="statevector")

        elif sim_name == "Quimb (MPS)":
            circ = qtn.Circuit(n)
            circ.apply_gate("H", 0)
            for i in range(n - 1):
                circ.apply_gate("CNOT", i, i + 1)
            for i in reversed(range(n - 1)):
                circ.apply_gate("CNOT", i, i + 1)
            circ.apply_gate("H", 0)

        t_setup_ms = (time.perf_counter() - t_setup_start) * 1000.0
        mem_after_setup = get_mem_mb()

        # ==========================================================
        # PHASE 2: MEASURE CIRCUIT EXECUTION & THREADS
        # ==========================================================
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()

        if sim_name == "Google qsim (SV)":
            result = simulator.run(circuit, repetitions=1)
            measured_bit = str(result.measurements["m"][0][0])

        elif sim_name == "MQT DDSIM (QDD)":
            job = backend.run(qc, shots=1)
            counts = job.result().get_counts()
            measured_bit = list(counts.keys())[0]

        elif sim_name == "PyQrack (Hybrid)":
            sim.h(0)
            for i in range(n - 1):
                sim.mcx([i], i + 1)
            for i in reversed(range(n - 1)):
                sim.mcx([i], i + 1)
            sim.h(0)
            measured_bit = str(int(sim.m(n - 1)))

        elif sim_name == "Qiskit Aer (SV)":
            # Disable transpiler optimization to force full linear algebra evaluation
            job = sim.run(qc, shots=1, optimization_level=0)
            counts = job.result().get_counts()
            measured_bit = list(counts.keys())[0]

        elif sim_name == "Quimb (MPS)":
            sample_val = next(circ.sample(1))
            if isinstance(sample_val, (int, np.integer)):
                measured_bit = str(sample_val)
            elif isinstance(sample_val, str):
                measured_bit = sample_val[-1]
            else:
                measured_bit = str(sample_val[-1])

        t_wall_ms = (time.perf_counter() - t_start_wall) * 1000.0
        t_cpu_ms = (time.process_time() - t_start_cpu) * 1000.0

        active_threads = process.num_threads()
        mem_final = get_mem_mb()

        state_ram_mb = max(0.0, mem_final - mem_after_setup)
        peak_ram_mb = mem_final

        conn.send({
            "status": "SUCCESS",
            "setup_ms": t_setup_ms,
            "elapsed_ms": t_wall_ms,
            "cpu_ms": t_cpu_ms,
            "threads": active_threads,
            "peak_ram": peak_ram_mb,
            "state_ram": state_ram_mb,
            "correct": (measured_bit == expected)
        })
    except Exception as e:
        conn.send({"status": "FAILED", "error": str(e)})
    finally:
        conn.close()

def evaluate_simulator_at_n(sim_name, n):
    records = []

    # 1. Warmup run to prime JIT compiler and path optimizers
    warmup_parent, warmup_child = mp.Pipe()
    p_warm = mp.Process(target=run_trial, args=(sim_name, n, warmup_child))
    p_warm.start()
    p_warm.join(timeout=TIMEOUT_SEC)
    if warmup_parent.poll():
        _ = warmup_parent.recv()

    # 2. Statistical benchmark trials
    for _ in range(TRIALS):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=run_trial, args=(sim_name, n, child_conn))
        p.start()
        p.join(timeout=TIMEOUT_SEC)

        if p.is_alive():
            p.terminate()
            p.join()
            records.append({"status": "TIMEOUT"})
            continue

        if p.exitcode != 0:
            records.append({"status": f"OOM/Crash ({p.exitcode})"})
            continue

        if parent_conn.poll():
            res = parent_conn.recv()
            records.append(res)
        else:
            records.append({"status": "FAILED"})

    successes = [r for r in records if r.get("status") == "SUCCESS"]
    if not successes:
        failure_mode = records[-1].get("status", "FAILED")
        return {
            "N": n,
            "Simulator": sim_name,
            "Setup (ms)": failure_mode,
            "Elapsed (ms)": failure_mode,
            "CPU (ms)": failure_mode,
            "Threads": "-",
            "Peak RAM (MB)": failure_mode,
            "State RAM (MB)": failure_mode,
            "Success": "0.0%"
        }

    df_valid = pd.DataFrame(successes)
    return {
        "N": n,
        "Simulator": sim_name,
        "Setup (ms)": f"{df_valid['setup_ms'].mean():.2f} ± {df_valid['setup_ms'].std():.2f}",
        "Elapsed (ms)": f"{df_valid['elapsed_ms'].mean():.2f} ± {df_valid['elapsed_ms'].std():.2f}",
        "CPU (ms)": f"{df_valid['cpu_ms'].mean():.2f} ± {df_valid['cpu_ms'].std():.2f}",
        "Threads": int(round(df_valid['threads'].mean())),
        "Peak RAM (MB)": f"{df_valid['peak_ram'].mean():.2f} ± {df_valid['peak_ram'].std():.2f}",
        "State RAM (MB)": f"{df_valid['state_ram'].mean():.2f} ± {df_valid['state_ram'].std():.2f}",
        "Success": f"{(df_valid['correct'].mean() * 100):.1f}%"
    }

if __name__ == "__main__":
    simulators = [
        "Google qsim (SV)",
        "MQT DDSIM (QDD)",
        "PyQrack (Hybrid)",
        "Qiskit Aer (SV)",
        "Quimb (MPS)"
    ]

    all_results = []
    print(f"Beginning Entanglement Scaling sweep across N = {QUBIT_COUNTS}...")

    for n in QUBIT_COUNTS:
        print(f"\n--- Benchmarking N = {n} Qubits ---")
        for sim in simulators:
            print(f"  Running {sim}...", end="", flush=True)
            res = evaluate_simulator_at_n(sim, n)
            all_results.append(res)
            print(f" Done -> Elapsed: {res['Elapsed (ms)']}, State RAM: {res['State RAM (MB)']}")

    df_all = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("ENTANGLEMENT SCALING BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(df_all.to_string(index=False))