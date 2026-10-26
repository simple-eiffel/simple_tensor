# Test Coverage Review: simple_tensor

## Contracts
src/**/*.e (46 classes). Full clause text: evidence/phase4-contracts-after.txt.

## Tests
testing/lib_tests.e (24 tests, run by testing/test_app.e), testing/gradient_checker.e, testing/twin_checker.e,
testing/gc_probe_worker.e, testing/test_scoop_consumer.e.

## Check For:
- Postconditions without corresponding tests
- Edge cases not tested
- Precondition boundary tests missing

## Coverage sweep done in Phase 5 (2026-10-26)
Every public feature of SIMPLE_TENSOR, TENSOR, GRAD_STORE, the optimizers, TENSOR_RANDOM, TOKEN_BATCHER and
SAFETENSORS_FILE was grepped against testing/. Untested before this phase, now each exercised (contracts on):
zeros_like, ones_like, to_backend, scale_all, clip_global_norm (both branches), merge (union and accumulate),
SGD_OPTIMIZER (hand trace, both backends), draw_index, fill_keep_mask, storage_shared.
Precondition boundaries: 10 misuse cases each stop at a PRECONDITION_VIOLATION, with a control call that does not.

## Known gaps (input for /eiffel.harden)
- TENSOR_BACKEND kernel commands (sgd_update, adamw_update, unary, binary, ...) carry preconditions only; their
  effect is pinned by tests (twin parity, hand traces), not by postconditions. Postconditions over every element
  would cost O(n) per call in assertion-on builds; decide per kernel in hardening.
- fill_normal has no postcondition (a Gaussian has no range bound); covered by determinism and the XOR test only.
