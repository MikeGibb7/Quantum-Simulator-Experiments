"""
Shared circuit definitions for the quantum simulator memory benchmark.

Building the circuit ONCE as a backend-agnostic list of gate ops, then
translating it into each library's own circuit object, guarantees every
simulator runs the exact same logical circuit. That way the only thing
varying between simulators is how their internal representation scales
with it - not accidental differences in what circuit each one got.
"""

import math
import random


def build_gate_sequence(n, circuit_type='ghz', depth=None, seed=42):
    """
    Returns a list of gate ops as tuples:
        ('h', qubit)
        ('cx', control, target)
        ('ry', qubit, theta)

    circuit_type:
      'product' - H on every qubit, NO entanglement (the original benchmark
                  from earlier). Tensor-network / decision-diagram /
                  Schmidt-decomposition methods represent this in O(n) space
                  trivially - this is the best-case scenario for them.

      'ghz'     - one H + a linear chain of CNOTs -> a single maximally
                  entangled GHZ state. Genuinely entangled, but still cheap
                  for MPS/DD/Qrack: the required bond dimension stays at 2
                  regardless of N. Good contrast case: dense statevector
                  simulators (Qiskit, qsim) still blow up exponentially here,
                  while the entanglement-aware methods barely notice.

      'random'  - alternating layers of random single-qubit RY rotations and
                  brick-wall-pattern CNOTs. This is the real stress test:
                  entanglement (and therefore bond dimension / DD size) grows
                  with circuit depth. With enough depth, even MPS/DD/Qrack
                  approach exponential memory cost, same as a dense
                  statevector - this is where you'd expect to see the
                  "efficient" methods stop being free.

    depth: number of entangling layers for 'random' (ignored otherwise).
           Defaults to n if not given - roughly enough depth for a brick-wall
           circuit to generate volume-law entanglement across all n qubits.

    seed: fixes the random angles so results are reproducible run-to-run.
    """
    ops = []

    if circuit_type == 'product':
        for q in range(n):
            ops.append(('h', q))

    elif circuit_type == 'ghz':
        ops.append(('h', 0))
        for i in range(n - 1):
            ops.append(('cx', i, i + 1))

    elif circuit_type == 'random':
        depth = depth if depth is not None else n
        rng = random.Random(seed)
        for layer in range(depth):
            for q in range(n):
                theta = rng.uniform(0, 2 * math.pi)
                ops.append(('ry', q, theta))
            # brick-wall pattern: alternate which pairs get an entangling gate
            start = 0 if layer % 2 == 0 else 1
            for q in range(start, n - 1, 2):
                ops.append(('cx', q, q + 1))

    elif circuit_type == 'nonclifford':
        # H + T + a linear chain of CZ per layer. T is the standard
        # "makes this genuinely hard to simulate classically" gate -
        # a Clifford+CZ-only circuit can be efficiently tracked by some
        # methods, but T breaks that, so this is a fair computational
        # throughput stress test across all simulator architectures.
        depth = depth if depth is not None else n
        for _ in range(depth):
            for qi in range(n):
                ops.append(('h', qi))
                ops.append(('t', qi))
            for qi in range(n - 1):
                ops.append(('cz', qi, qi + 1))

    else:
        raise ValueError(f"unknown circuit_type: {circuit_type!r}")

    return ops

