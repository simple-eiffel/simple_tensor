# Implementation Tasks: simple_tensor (M1)

Inputs:
- the contracts in `src/` (43 classes after the Phase 2 conditions);
- `approach.md` (implementation order);
- `synopsis.md` and `evidence/phase2-claude-response.md` (ruled findings; the "land with its class" items are attached to the task that owns the class).

Gates for every task:
- `ec.sh check -target simple_tensor` (library, all classes) **and** `-target simple_tensor_tests` read 0 errors / 0 warnings in the LOG.
- The tests the task fills pass.
- No contract was weakened without a recorded reason (Phase 4 contract-freeze rule).
- One compile at a time.

---

## T0: Gap fills in the owning libraries (separate PRs; FILL GAPS, never fall back)
**Files:** `D:\prod\simple_randomizer`, `D:\prod\simple_math`, `D:\prod\simple_file`
**Features:**
- **G-2** `SIMPLE_RANDOMIZER.random_gaussian (mean, std): REAL_64` (Box–Muller).
- **G-3** `SIMPLE_MATH.tanh`, `erf` (and `sigmoid`).
- **G-5** `SIMPLE_FILE` bulk binary read into and write from a MANAGED_POINTER (offset, count as INTEGER_64). ISE `RAW_FILE.read_to_managed_pointer` exists (file.e:1528).
- **G-6** a long-period generator (e.g. PCG32 or xoshiro128**) in simple_randomizer. Lehmer 16807's single 2^31−2 cycle is spent in ~9 RixGPT dropout iterations (R3-03).
### Acceptance Criteria
- [ ] Each fill has its own tests:
  - G-2: mean/std of 1e6 draws within 3σ/√n;
  - G-3: erf and tanh against reference values to 1e-12;
  - G-5: byte-exact round trip of a 64 MB block;
  - G-6: published reference outputs for the chosen seed.
- [ ] Each fill has a CHANGELOG entry and a version bump.
- [ ] Each library's downstream dependents are rebuilt clean (fleet rule).
### Dependencies
None. G-2/G-3 block T11/T4 (gelu); G-5 blocks T12; G-6 blocks M2 dropout, not M1.

## T1: TENSOR_SHAPE bodies
**Files:** src/tensor_shape.e
**Features:** dims_model, to_array, broadcasts_with, is_equal, is_permutation, broadcast_with, with_dim, without_dim, appended, permuted
### Acceptance Criteria
- [ ] R1-09: broadcasts_with symmetric; broadcast_with ensure per-dim (trailing-aligned max).
- [ ] dims_model ensures `Result [i] = dim (i)` (R1-19).
- [ ] test_shape_basics and test_shape_broadcast pass.
### Dependencies
None.

## T2: Storage, budget, indices
**Files:** src/storage/*.e, src/index_tensor.e
**Features:** STORAGE_BUDGET.note_allocation (full_collect when over); CPU_STORAGE.release (`block.resize (1)`); INDEX_TENSOR make_from_array, all_in_range, all_in_range_or, reshaped.
### Acceptance Criteria
- [ ] R2-03: INDEX_TENSOR `address` exported `{TENSOR_BACKEND}`; `put` restricted.
- [ ] test_storage_budget_collects passes (collections count rises at the budget).
### Dependencies
T1.

## T3: Split kind constants (R2-09)
**Files:** src/backend/tensor_kinds.e → TENSOR_OP_KINDS + BACKEND_KINDS
### Acceptance Criteria
- [ ] Both stay public (VAPE, E-1).
- [ ] TENSOR inherits only the op kinds; SIMPLE_TENSOR uses BACKEND_KINDS.
- [ ] 0/0 on both targets.
### Dependencies
None (do it early: it touches many files).

## T4: REFERENCE kernels (the oracle) + HOST_BACKEND
**Files:** src/backend/*.e (+ new host_backend.e)
**Features:** all 25 kernels in plain Eiffel; REAL_64 accumulation; `gc_point` every outer loop (S-3 law).
### Acceptance Criteria
- [ ] R2-05: HOST_BACKEND (effective) holds the cold kernels; REFERENCE_BACKEND and CPU_BACKEND are siblings.
- [ ] R2-12: a kernel roster constant, with its count asserted.
- [ ] R1-17: `distinct_output` on the 7 kernels.
- [ ] R2-07: derivative table; relu'(0) = 0; `param_unused`; gelu waits on G-3.
- [ ] R3-05: `address` note.
### Dependencies
T1, T2, T3; G-3 for gelu/tanh.

## T5: TENSOR forward ops + facade factories
**Files:** src/tensor.e, src/simple_tensor.e, src/ops/*.e (constructors)
**Features:** the 33 primitives' forward paths (allocate, kernel, op and origin when tracked), zeros_like, ones_like, to_backend, detached, item, to_array; SIMPLE_TENSOR factories, causal_mask.
### Acceptance Criteria
- [ ] R1-10: exact result shapes (matmul and transposed with_dim/permuted); maximum scalar_from_vector.
- [ ] R1-11: `Result.storage /= storage` on data ops; not_a_parameter.
- [ ] R1-12: detached `live`.
- [ ] R1-15: at least one non-ignored target.
- [ ] R2-04: dropout over a supplied mask.
- [ ] R2-10: `reshaped`, `permuted`, `sliced`.
- [ ] R2-11: `create {TENSOR}` on ops.
### Dependencies
T4.

## T6: Autograd: TENSOR.backward, op backwards, GRAD_STORE
**Files:** src/tensor.e (backward, grad_leaf_ids_model), src/ops/*.e (backward bodies), src/grad_store.e
### Acceptance Criteria
- [ ] Every backward computes on `i.detached` (G-C).
- [ ] Iterative DFS with a GC point every 256 nodes; the sweep discards intermediates.
- [ ] accumulate copies on first store (G-A).
- [ ] R1-07: has_id, merge preconditions.
- [ ] R1-08: clip requires finite and never scales up.
- [ ] test_tied_weight_accumulates and test_frozen_leaf_gets_no_gradient pass.
### Dependencies
T5.

## T7: GRADIENT_CHECKER + gradient checks for every primitive
**Files:** testing/gradient_checker.e, testing/lib_tests.e
### Acceptance Criteria
- [ ] The roster's primitives all pass on REFERENCE: step 1e-2·max(1,|x|), tolerance 1e-2 abs + 1e-3 rel, inputs away from kinks.
- [ ] Any failure is re-run in a REAL_64 storage variant before it counts (intent Q9).
### Dependencies
T6.

## T8: CPU_BACKEND hot kernels (C inline, `C blocking`, prefix `stensor_`)
**Files:** src/backend/cpu_backend.e, testing/twin_checker.e
**Features:** the 13 hot kernels (binary, unary, unary_derivative, reduce, sum_of_squares, has_non_finite, matmul, softmax, gather_rows, scatter_add_rows, cross_entropy, cross_entropy_grad, adamw_update).
### Acceptance Criteria
- [ ] Each kernel matches REFERENCE within 1e-5 relative (TWIN_CHECKER, scope = roster).
- [ ] Kernels touch only C-heap memory (CL-5).
- [ ] matmul is deterministic (row partition when threaded in M2).
- [ ] test_twin_parity_all_kernels passes.
### Dependencies
T4, T7.

## T9: Optimizers and schedules
**Files:** src/optim/*.e
### Acceptance Criteria
- [ ] AdamW decay-first (PyTorch 2.14 docs, P1).
- [ ] R1-14: distinct ids; frames for no-gradient parameters and the schedule.
- [ ] WARMUP_COSINE via SIMPLE_MATH equals rixgpt.py `lr_at` at steps 0, 75, 150, 2575, 5000.
- [ ] test_adamw_hand_trace and test_warmup_cosine_matches_rixgpt pass.
### Dependencies
T6 (GRAD_STORE), T8 (adamw_update kernel; the REFERENCE version suffices first).

## T10: TENSOR_RANDOM and TOKEN_BATCHER
**Files:** src/tensor_random.e, src/data/*.e
### Acceptance Criteria
- [ ] R2-04: `draw_index` command + `last_index`.
- [ ] fill_normal via G-2; fill_keep_mask.
- [ ] Batcher: forth/batch; targets are inputs shifted by one.
- [ ] test_token_batcher_shifted_targets and test_determinism_same_seed (REFERENCE and CPU) pass.
### Dependencies
T0 (G-2), T5.

## T11: Checkpoints
**Files:** src/io/*.e
### Acceptance Criteria
- [ ] F32/F16/BF16 load (edge values: ±0, subnormals, ±inf, NaN, max); F32 save.
- [ ] `__metadata__` key tolerated.
- [ ] R1-16 ensures.
- [ ] test_safetensors_round_trip is byte-identical.
- [ ] The HF safetensors format spec is fetched raw first (R3: P3 for the format).
### Dependencies
T0 (G-5), T5.

## T12: Memory, GC-law and non-finite tests
**Files:** testing/*.e (+ a SCOOP probe target in the ECF)
### Acceptance Criteria
- [ ] AC-7: test_gc_probe_under_10ms, a training step on a separate processor; root worst allocation < 10 ms.
- [ ] AC-8: test_memory_stays_bounded, measuring PeakPagefileUsage in a fresh process; peak ≤ live set + budget + margin.
- [ ] AC-10: test_non_finite_detected passes.
### Dependencies
T6, T8, T9.

## T13: Lean benchmark target (AC-9)
**Files:** simple_tensor.ecf (+ testing/bench_app.e)
### Acceptance Criteria
- [ ] Target `simple_tensor_bench` built with `ec.sh release`; run `_lean.exe`.
- [ ] CPU matmul 256×768×768 ≥ 20 GFLOP/s single thread.
- [ ] Figure recorded in evidence.
### Dependencies
T8.

## T14: Release hygiene
**Files:** README.md, CHANGELOG.md, docs/
### Acceptance Criteria
- [ ] README and /docs before any push (fleet rule).
- [ ] All 17+ tests filled and passing (R1-19: no "ready" claim with empty tests).
- [ ] /eiffel.verify and /eiffel.harden follow.
### Dependencies
All.

---
**Order:** T0 ‖ (T1 → T2 → T3) → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11 → T12 → T13 → T14.
T0 runs in parallel; its fills gate T4 (gelu/tanh), T10 and T11.
