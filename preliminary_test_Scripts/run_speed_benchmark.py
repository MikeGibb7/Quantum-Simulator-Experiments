"""
Computational Throughput Scaling Benchmark
==========================================
Iterates over qubit counts {10, 12, 14, 16, 18, 20} with depth d=5.
Includes early-stopping to prevent waiting for guaranteed timeouts on
larger circuits once a simulator fails or exceeds 180s.
"""

import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import qmem_speed_workers as w

SIMULATORS = {
    'Qiskit (Standard)': w.run_qiskit_speed,
    'Google qsim': w.run_qsim_speed,
    'MQT DDSIM': w.run_ddsim_speed,
    'Quimb (MPS)': w.run_quimb_speed,
    'PyQrack': w.run_pyqrack_speed,
}


def measure_in_subprocess(target, n, depth, repeats, timeout=180):
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
        if p.exitcode is not None and p.exitcode < 0:
            return None, f"killed by signal {-p.exitcode} after {elapsed:.1f}s", None
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

    print("Starting Computational Throughput Scaling Benchmark...\n")

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
                print(f"  -> Dropping {name} from higher qubit tests.")
                results[name].append(np.nan)
                active_simulators.remove(name)
                continue

            best = min(times)
            print(f"  -> min: {best:.4f}s   [{status}]")
            results[name].append(best)

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

    plt.title(f'Computational Throughput Scaling (Depth $d={DEPTH}$)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Qubits ($N$)', fontsize=12)
    plt.ylabel('Run time (seconds) [log scale]', fontsize=12)

    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xticks(QUBIT_COUNTS)

    plt.legend(fontsize=10, loc='upper left', framealpha=0.9)
    plt.tight_layout()

    plt.savefig('throughput_scaling.pdf', format='pdf', dpi=300)
    plt.savefig('throughput_scaling.png', format='png', dpi=300)
    print("\nGraph successfully saved as 'throughput_scaling.pdf' and '.png'")
    plt.show()

    df = pd.DataFrame(results, index=QUBIT_COUNTS)
    return df


if __name__ == '__main__':
    main()

