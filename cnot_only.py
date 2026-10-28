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

# Force joblib into sequential mode if loaded anywhere
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

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

# Sweep configuration for CNOT-Only Entanglement Sweep
QUBIT_COUNTS = [10, 12, 14, 16, 18, 20]
TRIALS = 5            # 5 recorded trials per N
TIMEOUT_SEC = 120.0   # 2-minute safety ceiling

def get_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_trial(sim_name, n, conn):
    process = psutil.Process()
    depth_cycles = n

    try:
        # ==========================================================
        # PHASE 1: MEASURE SETUP / INITIALIZATION TIME
        # ==========================================================
        t_setup_start = time.perf_counter()

        if sim_name == "Google qsim (SV)":
            qubits = cirq.LineQubit.range(n)
            circuit = cirq.Circuit()
            # Initial equal superposition layer
            for q in qubits:
                circuit.append(cirq.H(q))
            # CNOT-only brickwork entanglement over D = N cycles
            for _ in range(depth_cycles):
                for i in range(0, n - 1, 2):
                    circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
                for i in range(1, n - 1, 2):
                    circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
            # Full register measurement
            circuit.append(cirq.measure(*qubits, key="m"))
            simulator = qsimcirq.QSimSimulator()

        elif sim_name == "MQT DDSIM (QDD)":
            qc = QuantumCircuit(n, n)
            for i in range(n):
                qc.h(i)
            for _ in range(depth_cycles):
                for i in range(0, n - 1, 2):
                    qc.cx(i, i + 1)
                for i in range(1, n - 1, 2):
                    qc.cx(i, i + 1)
            qc.measure(range(n), range(n))
            backend = DDSIMProvider().get_backend("qasm_simulator")

        elif sim_name == "PyQrack (Hybrid)":
            sim = QrackSimulator(n)

        elif sim_name == "Qiskit Aer (SV)":
            qc = QuantumCircuit(n, n)
            for i in range(n):
                qc.h(i)
            for _ in range(depth_cycles):
                for i in range(0, n - 1, 2):
                    qc.cx(i, i + 1)
                for i in range(1, n - 1, 2):
                    qc.cx(i, i + 1)
            qc.measure(range(n), range(n))
            sim = AerSimulator(method="statevector")

        elif sim_name == "Quimb (MPS)":
            circ = qtn.Circuit(n)
            for i in range(n):
                circ.apply_gate("H", i)
            for _ in range(depth_cycles):
                for i in range(0, n - 1, 2):
                    circ.apply_gate("CNOT", i, i + 1)
                for i in range(1, n - 1, 2):
                    circ.apply_gate("CNOT", i, i + 1)

        t_setup_ms = (time.perf_counter() - t_setup_start) * 1000.0
        mem_after_setup = get_mem_mb()

        # ==========================================================
        # PHASE 2: MEASURE CIRCUIT EXECUTION & THREADS
        # ==========================================================
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()

        if sim_name == "Google qsim (SV)":
            result = simulator.run(circuit, repetitions=1)
            _ = result.measurements["m"][0]

        elif sim_name == "MQT DDSIM (QDD)":
            job = backend.run(qc, shots=1)
            _ = job.result().get_counts()

        elif sim_name == "PyQrack (Hybrid)":
            for i in range(n):
                sim.h(i)
            for _ in range(depth_cycles):
                for i in range(0, n - 1, 2):
                    sim.mcx([i], i + 1)
                for i in range(1, n - 1, 2):
                    sim.mcx([i], i + 1)
            _ = sim.m_all()

        elif sim_name == "Qiskit Aer (SV)":
            job = sim.run(qc, shots=1, optimization_level=0)
            _ = job.result().get_counts()

        elif sim_name == "Quimb (MPS)":
            dense_state = circ.psi.to_dense()
            _ = dense_state[0]

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
            "state_ram": state_ram_mb
        })
    except Exception as e:
        conn.send({"status": "FAILED", "error": str(e)})
    finally:
        conn.close()

def evaluate_simulator_at_n(sim_name, n):
    records = []

    # 1. Warmup trial
    warmup_parent, warmup_child = mp.Pipe()
    p_warm = mp.Process(target=run_trial, args=(sim_name, n, warmup_child))
    p_warm.start()
    p_warm.join(timeout=TIMEOUT_SEC)
    if warmup_parent.poll():
        _ = warmup_parent.recv()

    # 2. Benchmark trials
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
        "Success": "100.0%"
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
    print(f"Beginning CNOT-Only Brickwork Entanglement Sweep across N = {QUBIT_COUNTS}...")

    for n in QUBIT_COUNTS:
        print(f"\n--- Benchmarking N = {n} Qubits (Depth D = {n}) ---")
        for sim in simulators:
            print(f"  Running {sim}...", end="", flush=True)
            res = evaluate_simulator_at_n(sim, n)
            all_results.append(res)
            print(f" Done -> Elapsed: {res['Elapsed (ms)']}, State RAM: {res['State RAM (MB)']}")

    df_all = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("CNOT-ONLY BRICKWORK ENTANGLEMENT ABLATION SUMMARY TABLE")
    print("="*80)
    print(df_all.to_string(index=False))