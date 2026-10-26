# Adversarial Test Suggestions: simple_tensor

## Contracts
src/**/*.e; full clause text in evidence/phase6-contracts-after.txt.

## Current Tests
testing/lib_tests.e (31 tests), testing/training_twin_worker.e, testing/gc_probe_worker.e.

## Suggest Tests For:
1. Boundary values (rank 1, size 1, Max_elements, empty corpus)
2. Capacity limits (storage budget near zero; tensors near Max_elements)
3. SCOOP concurrent access (several trainers; tensors must never cross processors)
4. State transitions in unexpected order (backward twice; release then use; optimizer with a missing gradient)
5. Resource exhaustion (C-heap allocation failure: malloc returning NULL in kernels)

## Open for later phases
- Kernel scratch malloc failure (the transpose-B path falls back to the slow loop; no test forces NULL).
- GPU backend (M3) must pass this same suite through twin parity.
