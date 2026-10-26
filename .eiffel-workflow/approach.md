# Approach: simple_tensor (M1)

Position paper for the Phase 2 review. It covers how the Phase 1 contracts (46 classes in `src/`, 5 in `testing/`) will be implemented in Phase 4.

## Architecture

```
SIMPLE_TENSOR (facade, one explicit backend)
   └─ creates TENSOR (shape + storage + backend + id + requires_grad + origin)
          ├─ TENSOR_SHAPE (immutable dims, cached count)
          ├─ TENSOR_STORAGE ── CPU_STORAGE (MANAGED_POINTER, C heap; STORAGE_BUDGET per thread)
          ├─ TENSOR_BACKEND ── REFERENCE_BACKEND (plain Eiffel, REAL_64 accumulation, gc_point)
          │                        └─ CPU_BACKEND (13 hot kernels redefined as C blocking inline)
          └─ origin: TENSOR_OP (18 classes; inputs + saved values; backward)
GRAD_STORE (gradients by tensor id; accumulate/discard/scale/clip/merge/norm)
TENSOR_OPTIMIZER ── SGD_OPTIMIZER, ADAMW_OPTIMIZER (in-place update kernels)
LR_SCHEDULE ── CONSTANT_SCHEDULE, WARMUP_COSINE_SCHEDULE
TOKEN_BATCHER → TOKEN_BATCH (INDEX_TENSOR inputs/targets);  TENSOR_RANDOM (simple_randomizer)
SAFETENSORS_FILE (+ TENSOR_DTYPE_CODEC)
```

## Data flow of one training step
1. `batcher.forth`; `batch := batcher.batch` (two INDEX_TENSORs, batch × block).
2. **Forward.** Each TENSOR op:
   - allocates `out := backend.new_storage (result count)`;
   - runs the backend kernel (C blocking on CPU);
   - if any input `is_tracked`, creates the TENSOR_OP (inputs and saved values) and passes it as `origin` to `make_result`.
3. `loss := logits.cross_entropy (targets, -1)`, a scalar with an origin.
4. **`grads := loss.backward`:**
   - an iterative DFS over `origin.inputs`, visiting each id once, with a GC point every 256 nodes, emitting post-order;
   - seed the loss gradient with ones;
   - walk reverse post-order: for each tensor with an origin and a gradient, run `origin.backward (g, store)`, then `store.discard (t)` and `release` that gradient's storage.
   - Backward formulas use TENSOR ops on untracked tensors, so no graph is built during backward.
5. `grads.clip_global_norm (1.0)` uses `backend.sum_of_squares` per gradient, then `scale_all`.
6. `optimizer.step (grads)`: per parameter with a gradient, `backend.adamw_update` in place; the m and v storages are created lazily per parameter id.
7. Garbage: activations and intermediate gradients become unreachable; CPU_STORAGE's per-thread STORAGE_BUDGET forces `MEMORY.full_collect` every 256 MB allocated (spike S-4).

## Implementation order (Phase 4)
1. TENSOR_SHAPE, CPU_STORAGE/STORAGE_BUDGET, INDEX_TENSOR, GC_POINT.
2. REFERENCE_BACKEND, all kernels in plain Eiffel (the oracle first).
3. TENSOR forward ops + TENSOR_OP classes + backward; GRAD_STORE.
4. GRADIENT_CHECKER, then gradient checks on every primitive.
5. CPU_BACKEND fast kernels one at a time, each gated by TWIN_CHECKER.
6. Optimizers and schedules (hand traces); TENSOR_RANDOM (after gap G-2); TOKEN_BATCHER.
7. SAFETENSORS_FILE (+ codec); memory and GC-probe tests.

## Key design decisions (with evidence)
| Decision | Evidence |
|---|---|
| C-heap storage + `C blocking` kernels | S-3: 7 ms vs 2,744 ms worst root allocation |
| GC point in long Eiffel loops | S-3: 3,129 ms → 6 ms |
| Byte-budget forced collection + explicit release | S-4: 8,040 MB → 21 MB |
| C inline hot kernels | S-1: 26.6 vs 4.03 GFLOP/s (lean) |
| Graph in tensors (no global tape) | candle backprop.rs pattern; SCOOP-safe |
| Contiguous tensors + matmul transpose flags | llm.c/ggml practice; fewer kernel layouts |
| `once` = per thread = per SCOOP processor (id counter, storage budget) | Claim under test: see review spike R-1 |

## Dependencies
- base: MANAGED_POINTER, MEMORY, HASH_TABLE, ARRAYED_LIST, CELL.
- simple_mml: MML_SET/MML_SEQUENCE.
- simple_math: exp, log, sqrt, cos, pi; tanh and erf are **gap G-3**.
- simple_randomizer: make_with_seed, random_real; Gaussian is **gap G-2**.
- simple_json, simple_file: safetensors.
- simple_testing.

## Risk areas
- Correctness of the broadcast backward (`reduced_to`) and of batched matmul with a shared B.
- Gradient-check precision in REAL_32 (intent Q9).
- `release` of storages still referenced by a saved op value (use-after-release).
- Determinism of CPU kernels once threaded (M2).
- Memory: saved values in ops keep activations alive until backward finishes (expected; bounded by the graph).
