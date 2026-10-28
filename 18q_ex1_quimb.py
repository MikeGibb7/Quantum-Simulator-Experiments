import multiprocessing as mp
import os
import resource
import time
import warnings
import numpy as np
import pandas as pd
import psutil

# Suppress cotengra/kahypar and numba warnings
warnings.filterwarnings("ignore")

# Pre-import at top level to match the new harness methodology
import quimb.tensor as qtn

N_QUBITS = 18
TRIALS = 5
TIMEOUT_SEC = 120.0

def get_parity_inputs(n):
    return [1 if i % 2 == 0 else 0 for i in range(n)]

def get_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_quimb_trial(n, inputs, conn):
    process = psutil.Process()
    expected = str(sum(inputs) % 2)

    try:
        # Phase 1: Setup
        t_setup_start = time.perf_counter()
        circ = qtn.Circuit(n)
        for i, bit in enumerate(inputs):
            if bit == 1:
                circ.apply_gate("X", i)
        for i in range(n - 1):
            circ.apply_gate("CNOT", i, i + 1)

        t_setup_ms = (time.perf_counter() - t_setup_start) * 1000.0
        mem_after_setup = get_mem_mb()

        # Phase 2: Execution / Sampling
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()

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

        conn.send({
            "status": "SUCCESS",
            "setup_ms": t_setup_ms,
            "elapsed_ms": t_wall_ms,
            "cpu_ms": t_cpu_ms,
            "threads": active_threads,
            "peak_ram": mem_final,
            "state_ram": max(0.0, mem_final - mem_after_setup),
            "correct": (measured_bit == expected)
        })
    except Exception as e:
        conn.send({"status": "FAILED", "error": str(e)})
    finally:
        conn.close()

def benchmark_quimb_n18():
    inputs = get_parity_inputs(N_QUBITS)

    # 1. Warmup Run (discards first-time JIT and contraction path optimizer latency)
    print("Performing initial warmup run to prime JIT compiler...")
    p_conn, c_conn = mp.Pipe()
    p_warm = mp.Process(target=run_quimb_trial, args=(N_QUBITS, inputs, c_conn))
    p_warm.start()
    p_warm.join(timeout=TIMEOUT_SEC)
    if p_conn.poll():
        _ = p_conn.recv()
    print("Warmup complete. Starting steady-state trials...")

    # 2. Steady-State Timed Trials
    records = []
    for trial_idx in range(TRIALS):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=run_quimb_trial, args=(N_QUBITS, inputs, child_conn))
        p.start()
        p.join(timeout=TIMEOUT_SEC)

        if p.is_alive():
            p.terminate()
            p.join()
            records.append({"status": "TIMEOUT"})
            continue

        if parent_conn.poll():
            res = parent_conn.recv()
            records.append(res)
            print(f"  Trial {trial_idx + 1}: Elapsed = {res.get('elapsed_ms', 0):.2f} ms, State RAM = {res.get('state_ram', 0):.2f} MB")
        else:
            records.append({"status": "FAILED"})

    successes = [r for r in records if r.get("status") == "SUCCESS"]
    if not successes:
        print("All trials failed.")
        return

    df = pd.DataFrame(successes)
    summary = {
        "N": N_QUBITS,
        "Simulator": "Quimb (MPS)",
        "Setup (ms)": f"{df['setup_ms'].mean():.2f} ± {df['setup_ms'].std():.2f}",
        "Elapsed (ms)": f"{df['elapsed_ms'].mean():.2f} ± {df['elapsed_ms'].std():.2f}",
        "CPU (ms)": f"{df['cpu_ms'].mean():.2f} ± {df['cpu_ms'].std():.2f}",
        "Threads": int(round(df['threads'].mean())),
        "Peak RAM (MB)": f"{df['peak_ram'].mean():.2f} ± {df['peak_ram'].std():.2f}",
        "State RAM (MB)": f"{df['state_ram'].mean():.2f} ± {df['state_ram'].std():.2f}",
        "Success": f"{(df['correct'].mean() * 100):.1f}%"
    }

    print("\n" + "=" * 80)
    print("QUIMB N = 18 CLEAN BENCHMARK RESULT")
    print("=" * 80)
    print(pd.DataFrame([summary]).to_string(index=False))

if __name__ == "__main__":
    benchmark_quimb_n18()