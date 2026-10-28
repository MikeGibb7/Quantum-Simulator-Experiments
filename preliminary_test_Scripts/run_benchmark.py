"""
Precision Architecture Benchmark — memory usage of quantum simulators
-----------------------------------------------------------------------
Run this as an actual script:  python run_benchmark.py
Uses multiprocessing 'spawn' and worker functions in qmem_workers.py.

Fixes vs original notebook script:
1. RACE CONDITION: Replaced 5ms background psutil polling thread with
   kernel-level resource.getrusage(RUSAGE_SELF).ru_maxrss.
2. ISOLATION: Each simulator runs in its own clean subprocess.
3. DEADLOCK PREVENTION: Uses 'spawn' context instead of 'fork' to prevent
   OpenMP/BLAS thread pool deadlocks.
"""

import time
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt

import qmem_workers as w

SIMULATORS = {
    'Qiskit (Statevector)': w.run_qiskit,
    'Google qsim (Statevector)': w.run_qsim,
    'Quimb (Tensor Network)': w.run_quimb,
    'MQT DDSIM (Decision Diagrams)': w.run_ddsim,
    'PyQrack (GPU-capable Statevector)': w.run_pyqrack,
}


def measure_in_subprocess(target, n, timeout=60):
    """Run `target(n, queue)` in a fresh subprocess; return (mem_mb|None, status_str)."""
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
        return None, f"TIMEOUT after {elapsed:.1f}s (likely hung, not OOM)"

    if p.exitcode != 0:
        if p.exitcode is not None and p.exitcode < 0:
            sig = -p.exitcode
            if sig == 9 and queue.empty():
                return None, f"killed by SIGKILL after {elapsed:.1f}s (real OOM)"
            return None, f"killed by signal {sig} after {elapsed:.1f}s"
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

    print("Starting Precision Architecture Benchmark...\n")

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

    plt.figure(figsize=(12, 7))
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#f1c40f', '#2ecc71']
    markers = ['o', 's', '^', 'D', 'v']

    plot_cols = [c for c in df.columns if df[c].notna().any() and (df[c].fillna(0) > 0).any()]
    for i, col in enumerate(plot_cols):
        plt.plot(df.index, df[col], marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], label=col, linewidth=2)

    plt.title('Architecture Scaling: True Peak Memory vs Qubit Count', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Qubits (N)', fontsize=12)
    plt.ylabel('Incremental Peak Memory (MB)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if plot_cols:
        plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('memory_benchmark.png', dpi=150)
    plt.show()

    print("\nRaw Data Table (MB):")
    print(df.round(3))

    return df


if __name__ == '__main__':
    df = main()

