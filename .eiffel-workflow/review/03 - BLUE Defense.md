# BLUE Defense (ROE General v1.1)
S6: I defend my own design. I re-read tensor_op.e:34-44 and tensor.e:565-582 (P1): RED's quotes are accurate. No finding was defeatable by raw read or run, so CONTEST = 0. No compiler run.

Same defect, answered together: **G-A** R1-02, R1-03, R2-02 (shared storage, unguarded release) · **G-B** R1-04, R1-18, R2-08 (budget blind to release) · **G-C** R1-01, R1-05, R1-06 (backward unconstrained) · **G-D** R2-03, R2-05, R2-06 (backend seam).

| ID | Verdict | Fix accepted |
|---|---|---|
| R1-01 | CONCEDE | tensor_op.e:38 guards only `g`. Invariant `across inputs as i all not i.is_tracked end`; ops save `a.detached`, keep tracked parents separately |
| R1-02 | CONCEDE | accumulate ensure `gradient (t).storage /= g.storage` (copy on first store) |
| R1-03 | CONCEDE | `storage_shared` (owner count, O(1)); release require `not storage_shared`; backward require inputs not released |
| R1-04 | CONCEDE | INDEX_TENSOR: `note_allocation`, `release`, `is_released`, live preconditions |
| R1-05 | NARROW | Boolean stubs are legal Phase 1, but vacuous. Add `grad_leaf_ids_model`; backward ensure `Result.ids_model \|=\| grad_leaf_ids_model`, `origin = old origin` |
| R1-06 | CONCEDE | accumulate require `t.is_tracked`; backward ensure untracked inputs absent from store |
| R1-07 | NARROW | Key-only model is deliberate (F1, O(1)). Accept: `ids_model` ensure `across Result as k all has_id (k.item) end`; merge require `o /= Current`, `common_ids_shaped`. No value model |
| R1-08 | CONCEDE | clip_global_norm require `not has_non_finite`; ensure `old global_norm <= a_max implies global_norm = old global_norm` |
| R1-09 | CONCEDE | is_equal ensure `Result implies rank = o.rank and count = o.count`; broadcasts_with symmetric; broadcast_with per-dim max |
| R1-10 | CONCEDE | RED's `with_dim`/`permuted` ensures; maximum `scalar_from_vector`; dropout tracking clause; reword CL-1 (batch dims were free) |
| R1-11 | NARROW | `Result /= Current` is weak: add `Result.storage /= storage` on data-producing ops. `not_a_parameter` only where data is produced (detached/reshape share storage by design). Add backend to zeros_like/ones_like, shape to detached |
| R1-12 | CONCEDE | `live: not is_released` on detached |
| R1-13 | CONCEDE | TENSOR_SHAPE.make, CPU_STORAGE.make require INTEGER_64 product `<= Max_elements` |
| R1-14 | CONCEDE | make require distinct ids; step require `a_grads.all_shapes_match`; frame: no-gradient params and schedule unchanged |
| R1-15 | CONCEDE | cross_entropy require at least one non-ignored target |
| R1-16 | NARROW | fill_* range/mask ensures; load error ensures `names_model.is_empty`; save require distinct names. Not "file exists" (environmental) |
| R1-17 | NARROW | Add `distinct_output` to the 7 kernels; frame ensure on `a_out`. Value ensures belong to TWIN_CHECKER in Phase 4 |
| R1-18 | CONCEDE | `note_release`, `live_bytes`, `peak_bytes`; invariant `allocated_since_collect <= budget + largest_request`; frame ensures; CPU_STORAGE.release calls note_release |
| R1-19 | CONCEDE | Phase 4 gate: fill and run the 13 empty tests before any "ready" claim (E-2, E-3) |
| R1-20 | CONCEDE | RED reports no defect |
| R2-01 | CONCEDE | add `test_gc_probe_under_10ms` (S-3 pattern) and a benchmark target |
| R2-02 | CONCEDE | release require `not storage_shared` (replace `not is_tracked` by `not is_parameter`); export `{TENSOR_OP, TENSOR_OPTIMIZER}` |
| R2-03 | NARROW | M1 is CPU-only. Now: restrict `address`/`put` export to the CPU backends. Record deferred `INDEX_STORAGE` (from `backend.new_index_storage`) as an M3 obligation in spec 04 |
| R2-04 | CONCEDE | `draw_index` command plus `last_index`; dropout takes a supplied mask (`masked (a_mask)`, pure), filled by `fill_keep_mask` |
| R2-05 | CONCEDE | effective `HOST_BACKEND` holds cold kernels; REFERENCE and CPU are siblings; TWIN_CHECKER scope from a roster. CL-7 IS-A claim withdrawn (`is_compatible` contradicts it) |
| R2-06 | NARROW | Fill on host then `upload`; restrict `put` export; batched reduction is an M3 obligation |
| R2-07 | CONCEDE | derivative table; relu'(0)=0; GELU = exact erf, gated on G-3; `param_unused` for the 9 kinds |
| R2-08 | CONCEDE | as R1-18 |
| R2-09 | CONCEDE | split `TENSOR_OP_KINDS` and `BACKEND_KINDS`; keep public (VAPE, E-1) |
| R2-10 | CONCEDE | `reshaped`, `permuted`, `sliced` |
| R2-11 | CONCEDE | `create {TENSOR}` on op classes |
| R2-12 | CONCEDE | roster constant; test asserts its count; reconcile 33/34/18 in spec |
| R2-13 | CONCEDE | note: only basic values cross processors |
| R3-01 | CONCEDE | TENSOR_RANDOM.make require `a_seed > 0 and a_seed < 2_147_483_647` |
| R3-02 | CONCEDE | declare gap G-5 (bulk binary read into MANAGED_POINTER); fallback RAW_FILE.read_to_managed_pointer; edit approach.md:63 |
| R3-03 | CONCEDE | note Lehmer period 2^31-2 (derived); reseed per iteration from (seed, iteration); AC-6 claims determinism only; gap G-6 (better generator) |
| R3-04 | CONCEDE | corrected in E-5a; also copy the log into evidence/ and fix the E-5 heading |
| R3-05 | CONCEDE | `address` note: no kernel holds an address across `release` |
| R3-06 | CONCEDE | RED reports no defect; parity test keeps `total > warmup` |

**Counts:** CONCEDE 32, NARROW 7, CONTEST 0 (39 findings).
**Order:** G-C and G-A first (class shape), then G-B and G-D, then the remaining contract edits.
