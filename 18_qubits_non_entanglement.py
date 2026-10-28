import multiprocessing as mp
import os
import resource
import time
import numpy as np
import pandas as pd
import psutil

N_QUBITS = 18
TRIALS = 20

def get_parity_inputs(n):
    return [1 if i % 2 == 0 else 0 for i in range(n)]

def get_mem_mb():
    # Linux max resident set size in megabytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_trial(sim_name, n, inputs, conn):
    mem_initial = get_mem_mb()
    expected = str(sum(inputs) % 2)
    process = psutil.Process()

    # ==========================================================
    # PHASE 1: MEASURE SETUP / INITIALIZATION TIME
    # ==========================================================
    t_setup_start = time.perf_counter()

    if sim_name == "Google qsim (SV)":
        import cirq
        import qsimcirq
        qubits = cirq.LineQubit.range(n)
        circuit = cirq.Circuit()
        for i, bit in enumerate(inputs):
            if bit == 1:
                circuit.append(cirq.X(qubits[i]))
        for i in range(n - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        circuit.append(cirq.measure(qubits[n - 1], key="m"))
        simulator = qsimcirq.QSimSimulator()

    elif sim_name == "MQT DDSIM (QDD)":
        from mqt.ddsim import DDSIMProvider
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n, 1)
        for i, bit in enumerate(inputs):
            if bit == 1:
                qc.x(i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(n - 1, 0)
        backend = DDSIMProvider().get_backend("qasm_simulator")

    elif sim_name == "PyQrack (Hybrid)":
        from pyqrack import QrackSimulator
        sim = QrackSimulator(n)

    elif sim_name == "Qiskit Aer (SV)":
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(n, 1)
        for i, bit in enumerate(inputs):
            if bit == 1:
                qc.x(i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(n - 1, 0)
        sim = AerSimulator(method="statevector")

    elif sim_name == "Quimb (MPS)":
        import quimb.tensor as qtn
        circ = qtn.Circuit(n)
        for i, bit in enumerate(inputs):
            if bit == 1:
                circ.apply_gate("X", i)
        for i in range(n - 1):
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
        measured_bit = str(result.measurements["m"][0][0])

    elif sim_name == "MQT DDSIM (QDD)":
        job = backend.run(qc, shots=1)
        counts = job.result().get_counts()
        measured_bit = list(counts.keys())[0]

    elif sim_name == "PyQrack (Hybrid)":
        for i, bit in enumerate(inputs):
            if bit == 1:
                sim.x(i)
        for i in range(n - 1):
            sim.mcx([i], i + 1)
        measured_bit = str(int(sim.m(n - 1)))

    elif sim_name == "Qiskit Aer (SV)":
        job = sim.run(qc, shots=1)
        counts = job.result().get_counts()
        measured_bit = list(counts.keys())[0]

    elif sim_name == "Quimb (MPS)":
        # Convert generator to a list or consume the first item
        sample_val = next(circ.sample(1))
        # sample_val is either an int (e.g. 0/1), a string ("010101"), or a tuple/list of bits
        if isinstance(sample_val, (int, np.integer)):
            measured_bit = str(sample_val)
        elif isinstance(sample_val, str):
            # Target qubit q_{N-1} is at the end of the bitstring
            measured_bit = sample_val[-1]
        else:
            # Tuple or list of bits across qubits
            measured_bit = str(sample_val[-1])

    t_wall_ms = (time.perf_counter() - t_start_wall) * 1000.0
    t_cpu_ms = (time.process_time() - t_start_cpu) * 1000.0

    # Capture threads and memory footprint
    active_threads = process.num_threads()
    mem_final = get_mem_mb()

    state_ram_mb = max(0.0, mem_final - mem_after_setup)
    peak_ram_mb = mem_final

    conn.send({
        "setup_ms": t_setup_ms,
        "elapsed_ms": t_wall_ms,
        "cpu_ms": t_cpu_ms,
        "threads": active_threads,
        "peak_ram": peak_ram_mb,
        "state_ram": state_ram_mb,
        "correct": (measured_bit == expected)
    })
    conn.close()

def evaluate_simulator(sim_name):
    inputs = get_parity_inputs(N_QUBITS)
    records = []

    for _ in range(TRIALS):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=run_trial, args=(sim_name, N_QUBITS, inputs, child_conn))
        p.start()
        res = parent_conn.recv()
        p.join()
        records.append(res)

    df_trials = pd.DataFrame(records)
    return {
        "Simulator": sim_name,
        "Setup (ms)": f"{df_trials['setup_ms'].mean():.2f} ± {df_trials['setup_ms'].std():.2f}",
        "Elapsed (ms)": f"{df_trials['elapsed_ms'].mean():.2f} ± {df_trials['elapsed_ms'].std():.2f}",
        "CPU (ms)": f"{df_trials['cpu_ms'].mean():.2f} ± {df_trials['cpu_ms'].std():.2f}",
        "Threads": int(round(df_trials['threads'].mean())),
        "Peak RAM (MB)": f"{df_trials['peak_ram'].mean():.2f} ± {df_trials['peak_ram'].std():.2f}",
        "State RAM (MB)": f"{df_trials['state_ram'].mean():.2f} ± {df_trials['state_ram'].std():.2f}",
        "Success": f"{(df_trials['correct'].mean() * 100):.1f}%"
    }

if __name__ == "__main__":
    simulators = [
        "Google qsim (SV)",
        "MQT DDSIM (QDD)",
        "PyQrack (Hybrid)",
        "Qiskit Aer (SV)",
        "Quimb (MPS)"
    ]
    summary = [evaluate_simulator(sim) for sim in simulators]
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))