# Synopsis: simple_tensor Phase 2 review (2026-10-25)

**Overall: PASS WITH CONDITIONS.**
- 39 findings + 1 from the adjudicator: 23 sustained, 16 narrowed, 0 overruled.
- Final severities: 3 HIGH, 19 MEDIUM (incl. A-01), 14 LOW, 4 INFO.
- Condition grade **C3**; **C2** for CL-2, CL-6 and E-1 under the executed-evidence rule (gated 2026-10-11).

Sources:
- Verdict: `review/04 - VERDICT.md`.
- Ruled findings: `evidence/phase2-claude-response.md`.

## Critical: must fix before Phase 3 tasks are cut
1. **G-A, memory safety of `release` (HIGH: R1-02, R1-03, R2-02).**
   - `accumulate` copies on first store (`gradient (t).storage /= g.storage`).
   - Add `storage_shared` (a monotone owner count). `release` requires `not storage_shared and not requires_grad`.
   - TENSOR_OP gets `saved_live`, covering inputs **and** saved storages such as TENSOR_OP_UNARY.output_values; `backward` requires `saved_live`.
   - `release` stays public. Any restricted export must include TENSOR and GRAD_STORE.
   - Explicit activation release becomes optional.
2. **G-C, backward must not build a graph (R1-01 MEDIUM; R1-05, R1-06).**
   - Keep `inputs` tracked (the DFS walk needs them). Compute backward on `i.detached`.
   - `accumulate` requires `t.is_tracked`.
   - `backward` ensures `Result.ids_model |=| grad_leaf_ids_model` and `origin = old origin`.
3. **G-B, budget (R1-04, R2-08, R1-18).**
   - The Q7 invariant `allocated_since_collect <= budget + largest_request`.
   - INDEX_TENSOR notes its allocations through a shared per-thread budget accessor.
   - Frames on `note_allocation`/`set_budget`.
4. **R1-13:** an INTEGER_64 element-count bound (`<= {INTEGER_32}.max_value // 4`). **R3-01:** seed precondition `0 < seed < 2_147_483_647`. **R1-09:** an exact `is_equal` postcondition.
5. **Register and intent hygiene.**
   - Reword CL-1, CL-4, CL-5, CL-7 and CL-10.
   - Give AC-7 (GC probe) and AC-9 (CPU throughput) a test or target.
   - Annotate intent-v2 Q7 (A-01) and set the AC-8 measuring method (PeakPagefileUsage, fresh process).
6. **Recompile both targets to 0 errors / 0 warnings.**

## Important: fix with their class in Phase 4
- R1-07 (GRAD_STORE `has_id`; merge `o /= Current`), R1-08 (clip requires finite), R1-10 (exact matmul/transposed shapes), R1-11, R1-14, R1-15.
- R2-04: CQS. `draw_index` + `last_index`; dropout over a supplied mask.
- R2-05: TWIN_CHECKER scope from a roster; HOST_BACKEND **preferred**, not required.
- R2-03: INDEX_TENSOR `address` exported to {TENSOR_BACKEND}; INDEX_STORAGE is an M3 obligation.
- R2-01: GC-probe test and benchmark target as Phase 3 tasks.
- R3-02: gap **G-5**, bulk binary file read into MANAGED_POINTER (simple_file; ISE `RAW_FILE.read_to_managed_pointer` exists).
- R3-03: gap **G-6**, a better RNG before recipe-scale dropout (Lehmer 16807's single 2^31−2 cycle is spent in about 9 RixGPT iterations).

## Minor
- Naming (`reshaped`, `permuted`, `sliced`).
- Split kind constants (TENSOR_OP_KINDS / BACKEND_KINDS).
- `create {TENSOR}` on op classes.
- Op roster constant.
- Kernel `distinct_output` preconditions.
- Fill/load/save ensures.
- `address` note on realloc (R3-05).
- `__metadata__` key in safetensors.

## A-01 (adjudicator's own finding), now measured
S-4 arm B's "peak 21 MB" was a post-collection floor. Re-measured every iteration before collecting (E-5b): **221 MB** true peak (8,040 MB with no forced collection; 40 MB with explicit release). Peak under a byte budget is about the live set plus the budget, so AC-8 needs its measuring method fixed. The memory notes and the oracle are corrected.

## Recommended before Phase 3
Land items 1–6 above as contract edits, recompile to 0/0, then `/eiffel.tasks`.

## Escalations
None. HOST_BACKEND is cheap and API-neutral, so it's decided as preferred.

## Do not assume (from the verdict)
- full_collect every 50 keeps peak at 21 MB (it's 221).
- AC-8 holds at the 256 MB default budget.
- Every op fixes its result shape exactly.
- Every AC has a home.
- Reseeding cures dropout repetition.
- CPU_BACKEND IS-A REFERENCE_BACKEND.
- simple_file covers safetensors I/O.
- INDEX_TENSOR churn is budgeted.
- Any positive seed works.
