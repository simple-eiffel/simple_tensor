# Phase 2: Review Response (debate cycle)
# STATUS: COMPLETE (Larry gate APPROVED 2026-10-25)
# Date: 2026-10-25
# Method: separate-seat cycle (RED-1, RED-2, RED-3 by lens; BLUE; fresh Opus adjudicator); condition C3 (C2 with the executed-evidence rule for CL-2, CL-6, E-1)

Format: one block per finding. Full wording and evidence are in `review/02a`–`02c` (RED), `03` (BLUE) and `04 - VERDICT.md` (rulings).

### ISSUE 1: Stored gradient aliases a view, then gets released (R1-02)
- **LOCATION**: GRAD_STORE.accumulate (grad_store.e:~91); approach.md backward step 4
- **SEVERITY**: HIGH
- **RULING**: SUSTAINED (RED-1; BLUE concede)
- **DESCRIPTION**: accumulate stores g itself; reshape backward yields a view; the sweep releases that storage while it is still in the store.
- **EVIDENCE**: P1 raw read (tensor.e:328, grad_store.e)
- **REQUIRED FIX**: accumulate ensure `gradient (t).storage /= g.storage` (copy on first store).

### ISSUE 2: `release` frees shared storage (R1-03, R2-02)
- **LOCATION**: TENSOR.release (tensor.e:571); detached (:516), reshape (:328); TENSOR_OP.backward
- **SEVERITY**: HIGH
- **RULING**: R1-03 SUSTAINED; R2-02 NARROWED (release is public, not unexported)
- **DESCRIPTION**: `p.detached.release` frees a live parameter. Saved storages (TENSOR_OP_UNARY.output_values) can be released before backward.
- **EVIDENCE**: P1 raw reads; the adjudicator's spot-check
- **REQUIRED FIX**:
  - `storage_shared` (monotone owner count, O(1));
  - `release` require `not storage_shared and not requires_grad`;
  - TENSOR_OP `saved_live` over inputs and saved storages; `backward` require `saved_live`;
  - any restricted export of `release` must include TENSOR and GRAD_STORE;
  - activation release becomes optional.

### ISSUE 3: Backward builds a graph (R1-01)
- **LOCATION**: tensor_op.e:7-9 (doctrine), :38; approach.md step 4
- **SEVERITY**: MEDIUM (fails loudly at accumulate's `untracked_grad`)
- **RULING**: NARROWED (RED's invariant on `inputs` rejected: it breaks the DFS and empties `tracked_inputs_have_gradients`)
- **REQUIRED FIX**: keep `inputs` tracked; compute every backward formula on `i.detached`.

### ISSUE 4: backward postconditions vacuous (R1-05)
- **SEVERITY**: MEDIUM
- **RULING**: NARROWED
- **REQUIRED FIX**: `grad_leaf_ids_model`; backward ensure `Result.ids_model |=| grad_leaf_ids_model` and `origin = old origin`.

### ISSUE 5: Untracked leaf can receive a gradient (R1-06)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **REQUIRED FIX**: accumulate require `t.is_tracked`.

### ISSUE 6: INDEX_TENSOR churn unbudgeted (R1-04)
- **SEVERITY**: MEDIUM
- **RULING**: NARROWED (budget collections also reclaim dead index blocks; release optional)
- **REQUIRED FIX**: note_allocation through a shared per-thread budget accessor.

### ISSUE 7: STORAGE_BUDGET lacks the Q7 invariant (R2-08, R1-18)
- **SEVERITY**: MEDIUM / LOW
- **RULING**: NARROWED (overruled parts: "AC-8 unobservable", "side effect in creation")
- **REQUIRED FIX**: invariant `allocated_since_collect <= budget + largest_request`; frames on note_allocation/set_budget; note_release/live_bytes optional.

### ISSUE 8: Index seam leaks the CPU (R2-03); per-element storage API (R2-06)
- **SEVERITY**: MEDIUM / LOW
- **RULING**: NARROWED
- **REQUIRED FIX**: INDEX_TENSOR `address` exported to {TENSOR_BACKEND} now; INDEX_STORAGE and batched reductions are M3 obligations in spec 04.

### ISSUE 9: CPU_BACKEND inherits REFERENCE_BACKEND (R2-05)
- **SEVERITY**: MEDIUM
- **RULING**: NARROWED
- **REQUIRED FIX**: TWIN_CHECKER scope from a kernel roster (R2-12); strike CL-7's IS-A. HOST_BACKEND (siblings under a common effective parent) is PREFERRED, not required.

### ISSUE 10: GRAD_STORE model and merge (R1-07)
- **SEVERITY**: MEDIUM
- **RULING**: NARROWED
- **REQUIRED FIX**: `has_id`; `ids_model` ensure ties keys to `has_id`; merge require `o /= Current`, `common_ids_shaped`; no value model.

### ISSUE 11: clip_global_norm with NaN (R1-08)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **REQUIRED FIX**: require `not has_non_finite`; ensure `old global_norm <= a_max implies global_norm = old global_norm`.

### ISSUE 12: TENSOR_SHAPE equality unspecified (R1-09)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **REQUIRED FIX**: is_equal ensure `Result = (rank = o.rank and then across 1 |..| rank as i all dim (i) = o.dim (i) end)`; broadcasts_with symmetric; broadcast_with per-dim.

### ISSUE 13: Result shapes not exact (R1-10)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **REQUIRED FIX**:
  - matmul_transposed `Result.shape ~ shape.with_dim (rank-1, matmul_rows (a_ta)).with_dim (rank, o.matmul_cols (a_tb))`;
  - transposed via `permuted`;
  - maximum `scalar_from_vector`;
  - dropout tracking clause;
  - reword CL-1.

### ISSUE 14: Freshness and parameter clauses (R1-11)
- **SEVERITY**: MEDIUM
- **RULING**: NARROWED
- **REQUIRED FIX**: `Result.storage /= storage` on data-producing ops; not_a_parameter where data is produced; backend on zeros_like/ones_like; shape on detached.

### ISSUE 15: detached lacks `live` (R1-12)
- **SEVERITY**: LOW
- **RULING**: NARROWED
- **REQUIRED FIX**: add `live: not is_released`.

### ISSUE 16: INTEGER_32 overflow in counts (R1-13)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **REQUIRED FIX**: TENSOR_SHAPE.make and CPU_STORAGE.make require INTEGER_64 product `<= {INTEGER_32}.max_value // 4`. Real case: 64×256×50,257 logits = 823M elements.

### ISSUES 17–21: R1-14 optimizer frames; R1-15 all-ignored cross-entropy; R1-16 fill/load/save ensures; R1-17 kernel distinct_output; R1-19/R1-20
- **SEVERITY**: LOW / LOW / LOW / LOW / INFO
- **RULING**: R1-14 and R1-15 SUSTAINED; R1-16 and R1-17 NARROWED; R1-19 and R1-20 SUSTAINED (INFO)
- **REQUIRED FIX**: as BLUE (03): distinct ids and no-gradient frame; require one non-ignored target; range/mask ensures and distinct save names; `distinct_output` on 7 kernels; fill the 13 empty tests before any "ready" claim.

### ISSUE 22: AC-7 and AC-9 have no home (R2-01)
- **SEVERITY**: MEDIUM
- **RULING**: NARROWED
- **REQUIRED FIX**: GC-probe test and benchmark target as Phase 3 tasks.

### ISSUE 23: CQS violations (R2-04)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **REQUIRED FIX**: TENSOR_RANDOM `draw_index` command + `last_index`; dropout over a supplied mask filled by `fill_keep_mask`.

### ISSUE 24: Unary derivative contract (R2-07)
- **SEVERITY**: LOW
- **RULING**: NARROWED (exact GELU already pinned, tensor.e:234)
- **REQUIRED FIX**: derivative table; relu'(0) = 0; `param_unused` for the 9 kinds; gelu gated on G-3.

### ISSUES 25–29: R2-09 kinds split; R2-10 naming; R2-11 op creation export; R2-12 roster; R2-13 SCOOP note
- **SEVERITY**: LOW ×4, INFO
- **RULING**: SUSTAINED
- **REQUIRED FIX**: TENSOR_OP_KINDS / BACKEND_KINDS (public); `reshaped`/`permuted`/`sliced`; `create {TENSOR}` on ops; roster constant with a count assertion; note that only basic values cross processors.

### ISSUE 30: Seed 0 or 2^31−1 gives a constant stream (R3-01)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **EVIDENCE**: ISE random.e:72-90 (Lehmer 16807, increment 0); simple_randomizer.e:45-52
- **REQUIRED FIX**: TENSOR_RANDOM.make require `a_seed > 0 and a_seed < 2_147_483_647`.

### ISSUE 31: No bulk binary file read (R3-02)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED
- **EVIDENCE**: simple_file.e:440-470, 993-1020; ISE file.e:1528
- **REQUIRED FIX**: gap G-5 in simple_file (bulk read into MANAGED_POINTER). RAW_FILE.read_to_managed_pointer exists in base.

### ISSUE 32: RNG period spent in about 9 iterations of recipe dropout (R3-03)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED (BLUE's reseeding cure rejected: 16807 is a primitive root, so every seed is an offset in one cycle)
- **REQUIRED FIX**: gap G-6, a better generator, before M2 dropout.

### ISSUES 33–35: R3-04 S-4 citation; R3-05 address after resize; R3-06 schedule stricter than the oracle
- **SEVERITY**: LOW / LOW / INFO
- **RULING**: SUSTAINED (R3-04 cured by E-5a and E-5b)
- **REQUIRED FIX**: `address` note (no kernel holds an address across release); keep `total > warmup` in parity tests.

### ISSUE 36: S-4 "peak 21 MB" was a floor (A-01, adjudicator)
- **SEVERITY**: MEDIUM
- **RULING**: SUSTAINED, then confirmed by measurement (E-5b: true peak 221 MB)
- **REQUIRED FIX**: annotate intent-v2 Q7 (Larry's approval needed; it's an approved document); AC-8 measures PeakPagefileUsage in a fresh process; size the budget against the live set.
