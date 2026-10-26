# RED-2 Design and Seams

Lens: CQS, IS-A, exports, backend seam, traceability. All evidence P1-raw (source read 2026-10-25, no compile). Paths under `D:\prod\simple_tensor\`.

## Findings (most severe first)

| ID | Sev | Location | Finding and evidence | Fix |
|---|---|---|---|---|
| R2-01 | HIGH | AC-7, AC-9 | No test or target for the GC probe or for 20 GFLOP/s. `lib_tests.e:14-113` has 14 tests, none about either. `test_scoop_consumer.e:10-28` only creates objects. Probe and throughput code exist only under `spikes\`. | Add `test_gc_probe_under_10ms` (as S-3 `kernel_runner.e`) and a benchmark target. |
| R2-02 | HIGH | `TENSOR.release` tensor.e:573-580 | Q7 frees "activations after backward", but `release` requires `not is_tracked` and every activation is tracked. `release` is unexported, yet Q7 says "selectively exported". `reshape` and `detached` alias storage (:328, :521), so releasing one frees the others, and no precondition guards it. | Export to `{TENSOR_OP, TENSOR_OPTIMIZER, trainer}`. Rework the precondition. Add an alias or owner count. |
| R2-03 | HIGH | `INDEX_TENSOR` index_tensor.e:54,76; tensor_backend.e:210,267,278,291,301 | The M3 seam leaks the CPU. Public `address: POINTER`, public unrestricted `put` (unlike `TENSOR_STORAGE.put`). Five backend kernels take a concrete C-heap INDEX_TENSOR, so a Vulkan backend must re-upload indices on every call. Spec 04:29 gives VULKAN its own storage but gives indices none. | Deferred `INDEX_STORAGE` from `backend.new_index_storage`. INDEX_TENSOR stays host-side with a device cache. Restrict `address` and `put`. |
| R2-04 | MEDIUM | `TENSOR.dropout` tensor.e:460; `TENSOR_RANDOM.next_index` tensor_random.e:33 | Real CQS violations. Both are functions that advance the generator, so the same call twice differs. TOKEN_BATCHER (`forth` / `batch`, :65, :43) is the right pattern. Allocation plus id bump in TENSOR ops is acceptable: no existing object changes observably. | `draw_index` command plus `last_index`. Move dropout to M2 NN_DROPOUT (the doc already calls it a composition). |
| R2-05 | MEDIUM | `CPU_BACKEND` cpu_backend.e:19-26 (CL-7) | The motive is reuse, not IS-A. (a) Inherited cold kernels are identical in oracle and candidate, so twin parity cannot see an oracle bug in them. (b) The 13-name `redefine` list is the real twin scope, kept by hand, with no link to TWIN_CHECKER. (c) `is_compatible` (tensor_backend.e:31) says the two are not interchangeable, which breaks behavioral subtyping. (d) Spec 04:164 draws them as siblings. (e) VULKAN cannot copy the pattern. | Effective `HOST_BACKEND` with the cold kernels. REFERENCE and CPU become siblings under it. |
| R2-06 | MEDIUM | `TENSOR_STORAGE.put/item` tensor_storage.e:33; tensor_random.e:45-69; tensor_backend.e:164-180 | Per-element access is in the abstract interface. Export gives `put` to TENSOR_RANDOM, SAFETENSORS_FILE and SIMPLE_TENSOR, so init and load would be per-element writes on a GPU. `sum_of_squares` and `has_non_finite` return a host scalar, so one device sync per parameter. | Fill on the host, then `upload`. Add a batched reduction. |
| R2-07 | MEDIUM | CL-8, tensor_backend.e:120-129; tensor_op_unary.e:21-36, 53-56 | The math holds for all 11: neg -1; scale c; add_scalar 1; exp y; log 1/a; sqrt 1/(2y); tanh 1-y^2; sigmoid y(1-y); relu [a>0]; gelu and silu use `a`. But `unary_derivative` has no postcondition on `a_out`. relu'(0) and the GELU variant (erf or tanh) are unspecified. gelu needs `erf`, which is absent (execution record E-6, gap G-3). `param` is unconstrained for the 9 kinds that ignore it. | Add a derivative table, relu'(0) = 0, and the pinned GELU variant. Add `param_unused`. Gate gelu on G-3. |
| R2-08 | MEDIUM | `STORAGE_BUDGET` storage_budget.e:44-69; cpu_storage.e:21-32, 65-69 | Q7 asks for live and allocated bytes and an invariant `allocated_since_collect <= budget + largest_request`. The class has neither: the invariant is only `non_negative`, and `CPU_STORAGE.release` never informs the budget. Freed bytes still count toward the next forced collection, and AC-8 cannot be observed. `make` runs `full_collect` as a side effect inside a creation procedure. | Add `note_release`, `live_bytes`, `peak_bytes`, and the Q7 invariant. |
| R2-09 | LOW | `inherit TENSOR_KINDS` tensor.e:18; simple_tensor.e:22 | Constants by inheritance put about 30 constants and the `is_*_kind` predicates on TENSOR's public API. SIMPLE_TENSOR cannot see them and writes `backend.Cpu_backend_kind`. Op kinds and backend kinds are unrelated. `{NONE}` inheritance would break the VAPE-exposed preconditions (execution record E-1). | Split `TENSOR_OP_KINDS` and `BACKEND_KINDS`. |
| R2-10 | LOW | Naming tensor.e:318,333,357 | `reshape`, `permute`, `slice` read as commands but are queries. Siblings are participles (`transposed`, `concatenated`). `exponential` and `square_root` mix with `sigmoid` and `relu`. `plus`, `minus`, `product` and `quotient` are fine. | `reshaped`, `permuted`, `sliced`. |
| R2-11 | LOW | Exports tensor.e:21; tensor_op_unary.e:16 | TENSOR's creation list is coherent for M1, and NN modules go through the facade. `TENSOR_OPTIMIZER` shows no reason to create tensors. `create make` on every TENSOR_OP has no `{...}` clause, so clients can forge ops. | `create {TENSOR, TENSOR_OP}`. |
| R2-12 | LOW | CL-10, "34 primitives" | The roster is nowhere. src has 18 op classes, spec says 33. My count of distinct backward rules is 30. Test "all primitives" cannot assert "every". | A roster constant, and assert its count. |
| R2-13 | INFO | SCOOP (UC-004) | Per-processor `once` is verified (execution record E-4). No contract says only basic values cross the boundary. | One sentence in the SIMPLE_TENSOR note. |

## Traceability

| AC | Home | Status |
|---|---|---|
| AC-1 gradient checks | `test_gradient_checks_all_primitives`, GRADIENT_CHECKER | test exists (empty); roster unclear (R2-12) |
| AC-2 twin parity | `test_twin_parity_all_kernels`, TWIN_CHECKER | exists (empty); scope per R2-05 |
| AC-3 AdamW trace | `test_adamw_hand_trace`; ADAMW ensure `state_for_updated` | exists (empty) |
| AC-4 schedule | `test_warmup_cosine_matches_rixgpt`; `rate` ensure | exists (empty) |
| AC-5 safetensors | `test_safetensors_round_trip` | exists (empty) |
| AC-6 determinism | `test_determinism_same_seed` | exists (empty); threaded CPU case not separate |
| AC-7 GC probe | none | **GAP** |
| AC-8 memory | `test_memory_stays_bounded` | exists (empty); no peak counter (R2-08) |
| AC-9 CPU 20 GFLOP/s | none | **GAP** |
| AC-10 non-finite | `test_non_finite_detected`; `has_non_finite`; optimizer `finite` precondition | exists (empty) |
| AC-11 M2 gate | own milestone | not applicable |

13 of 15 tests are empty (execution record E-2), so this table proves existence, not coverage.

Count: HIGH 3, MEDIUM 5, LOW 4, INFO 1 (13).
