# 04 - VERDICT: simple_tensor Phase 2 review
Fresh adjudicator, 2026-10-25. Read: register, E-1..E-6 + E-5a, approach, intent-v2, RED-1/2/3, BLUE; raw spot-checks of src, simple_randomizer, simple_file, ISE random.e / managed_pointer.e / file.e, both spike sources.

## 0. Declarations
- **ROE:** General Research ROE **v1.1 (2026-10-12)**, read in full. Theological ROE not opened.
- **Condition:** without the executed-evidence rule **C3** (same-family seats; checks = raw reads; C2 only for CL-9/CL-11 strands on external primary documents: PyTorch docs, ISE source). With it (gated 2026-10-11): **C2** for CL-2 (R-1), CL-6 (S-4) and E-1. Not C1 until Larry gates.
- **Interest (S6):** same model family as every seat and the author; the author designed the spec, wrote the contracts, ran the spikes. BLUE contested 0 of 39, so concessions were re-tested, not counted.
- **Pipeline log:** deleted `review/spikes/once_per_processor/EIFGENs` (exe/obj/lib, ec.lock; no compile running); re-check: 0 EIFGENs, 0 binaries under review/spikes. Left untouched, outside the cycle folder: `spikes/{gpu_matmul,matmul,memory_gc,memory_pressure}/EIFGENs`. R-1 output exists only as pasted in E-4 (no log file). No compiler run here; one Python arithmetic check. Error charged to pipeline: R2-02 "release is unexported" (it is public, tensor.e:571); BLUE missed it.

## 1. Rulings
HIGH = memory-unsafe or silent in lean builds; MEDIUM = contract gap Phase 4 would build on; LOW = hygiene.

| ID | Ruling | Sev | Required fix / reason |
|---|---|---|---|
| **G-A** R1-02 | SUSTAINED | HIGH | accumulate ensure `gradient (t).storage /= g.storage` (copy on first store). Lean: `resize (1)` then a kernel read = heap overrun |
| R1-03 | SUSTAINED | HIGH | `storage_shared` (monotone owner count); release require `not storage_shared and not requires_grad`; TENSOR_OP `saved_live` over inputs AND saved storages (TENSOR_OP_UNARY.output_values is not an input); backward require `saved_live` |
| R2-02 | NARROWED | HIGH | Same fix. `release` is public, not unexported. A restricted export must include TENSOR (the sweep's qualified calls) and GRAD_STORE; BLUE's `{TENSOR_OP, TENSOR_OPTIMIZER}` omits both. Activation release becomes optional |
| **G-B** R1-04 | NARROWED | MEDIUM | `note_allocation` through a shared per-thread budget accessor (CPU_STORAGE.budget is `{NONE}`). Not 8 GB: budget collections also reclaim dead index blocks in training. release optional |
| R2-08 | NARROWED | MEDIUM | Q7 invariant `allocated_since_collect <= budget + largest_request`. Overruled: "AC-8 unobservable" (OS measure) and "side effect in creation" (allocation) |
| R1-18 | NARROWED | LOW | frames on note_allocation/set_budget; note_release, live_bytes optional |
| **G-C** R1-01 | NARROWED | MEDIUM | Loud: `untracked_grad` (grad_store.e:244) rejects it. Defect is tensor_op.e:7-9 and approach step 4. Keep `inputs` tracked (DFS, keys), compute on `i.detached`. RED's invariant on `inputs` rejected: it breaks the DFS and empties `tracked_inputs_have_gradients` |
| R1-05 | NARROWED | MEDIUM | BLUE's `grad_leaf_ids_model` ensure + `origin = old origin` |
| R1-06 | SUSTAINED | MEDIUM | accumulate require `t.is_tracked` |
| **G-D** R2-03 | NARROWED | MEDIUM | `address` export `{TENSOR_BACKEND}` now; INDEX_STORAGE an M3 obligation in spec 04 |
| R2-05 | NARROWED | MEDIUM | Required: twin scope from a roster (R2-12); strike CL-7's IS-A. HOST_BACKEND preferred (deferred hot kernels = compile-time completeness), not required. (a) is not cured by it; (c) overruled: `is_compatible` governs operand mixing, no contract broken |
| R2-06 | NARROWED | LOW | M3 obligation |
| R1-07 | NARROWED | MEDIUM | add `has_id`; merge require `o /= Current`, `common_ids_shaped`; no value model |
| R1-08 | SUSTAINED | MEDIUM | as BLUE |
| R1-09 | SUSTAINED | MEDIUM | is_equal ensure `Result = (rank = o.rank and then across 1 \|..\| rank as i all dim (i) = o.dim (i) end)` |
| R1-10 | SUSTAINED | MEDIUM | as BLUE; reword CL-1 |
| R1-11 | NARROWED | MEDIUM | as BLUE |
| R1-12 | NARROWED | LOW | make_leaf already fails loudly; blame placement only |
| R1-13 | SUSTAINED | MEDIUM | INTEGER_64 product `<= {INTEGER_32}.max_value // 4` (MANAGED_POINTER.make takes INTEGER bytes). Real: 64x256x50,257 logits = 823M elements = 3.3 GB |
| R1-14, R1-15 | SUSTAINED | LOW | as BLUE |
| R1-16, R1-17 | NARROWED | LOW | as BLUE |
| R1-19, R1-20 | SUSTAINED | INFO | Phase 4 gate / none |
| R2-01 | NARROWED | MEDIUM | traceability, cheap: GC-probe test + benchmark target in the Phase 3 tasks |
| R2-04 | SUSTAINED | MEDIUM | as BLUE |
| R2-07 | NARROWED | LOW | variant already pinned (tensor.e:234 "Exact GELU"); add relu'(0)=0, table, `param_unused` |
| R2-09..R2-12 | SUSTAINED | LOW | as BLUE |
| R2-13 | SUSTAINED | INFO | note |
| R3-01 | SUSTAINED | MEDIUM | `a_seed > 0 and a_seed < 2_147_483_647` |
| R3-02 | SUSTAINED | MEDIUM | gap G-5; `RAW_FILE.read_to_managed_pointer` (file.e:1528) |
| R3-03 | SUSTAINED | MEDIUM | BLUE's reseeding does not cure it: 16807 is a primitive root mod 2^31-1 (checked), so every seed is an offset in ONE cycle; 233M draws/iteration = 9.2 iterations per cycle. G-6 before recipe-scale dropout (M2) |
| R3-04, R3-05 | SUSTAINED | LOW | cured by E-5a (see A-01) / note |
| R3-06 | SUSTAINED | INFO | none |

**A-01 (this seat, raw read, MEDIUM): S-4 arm B's "peak" is a post-collection floor.** GR-25 side by side:
1. Older, intent-v2.md Q7 table under "Peak private MB": "`MEMORY.full_collect` every 50 | 21".
2. New: s4_app.e reads `PagefileUsage` (current, not peak) only when `i \\ 100 = 0`, right after the `i \\ 50 = 0` collect of the same iteration. The true arm-B peak is about 16 + 49x4 = 212 MB (derived, unmeasured).
3. Frame: both are S-4 arm B, 2,000 x 4 MB, lean, 2026-10-25.
4. Respect: **PARTLY.** "Periodic full_collect bounds churn" stands; "peak 21 MB" conflicts.
5. Coverage: tested as framed (the measuring code). Remedy: annotate, don't overwrite.

Consequence: under the 256 MB default budget, peak is about the live set + 256 MB, so AC-8 (2x steady state) fails by design for live sets under ~256 MB. The AC-8 test must read PeakPagefileUsage in a fresh process, and the budget must be sized against the live set.

## 2. Scored claims (confidence / rootedness)
| CL | Conf | Root | Status | Basis |
|---|---|---|---|---|
| 1 | 0.15 | 0.40 | DISPROVEN | R1-10 |
| 2 | 0.90 | 0.90 | ESTABLISHED | R-1 run (E-4); only two `once` in src |
| 3 | 0.35 | 0.50 | UNSUPPORTED | stub Booleans (R1-05); R1-06 |
| 4 | 0.20 | 0.55 | DISPROVEN | two claims: g-guard holds (tensor_op.e:38); "no graph" fails as designed (R1-01) |
| 5 | 0.50 | 0.35 | UNTESTED | no kernel exists (E-3); as worded it omits cold kernels, which rely on GC points (F3) |
| 6 | 0.55 | 0.60 | PREFERRED | budget strand: S-4; release strand unsafe until G-A; R1-04; A-01 |
| 7 | 0.25 | 0.40 | UNSUPPORTED | IS-A withdrawn; nothing links redefine list to TWIN_CHECKER |
| 8 | 0.75 | 0.80 | PREFERRED | 11 derivatives correct; no derivative contract; gelu waits on G-3 |
| 9 | 0.70 | 0.85 | PREFERRED | notes match rixgpt.py and PyTorch 2.14 docs (P1); ensures encode the warmup branch only |
| 10 | 0.10 | 0.80 | DISPROVEN | AC-7, AC-9 have no home |
| 11 | 0.65 | 0.90 | PREFERRED | verified raw; exceptions R3-01, G-5, G-6 |

## 3. Deployment lines
**May rely on:**
- `once` is per SCOOP processor.
- C-heap + `C blocking` + GC points cure the stall (S-3).
- full_collect reclaims dead MANAGED_POINTER blocks (S-4 "after collect 21").
- The 11 derivative formulas, AdamW decay-first order, the lr_at formula.
- E-1: 0/0 on both targets (Phase 1 shells).

**Must not rely on:**
- Any contract having run (13/15 tests empty, 50 stubs).
- `release` as specified.
- Q7's "free activations after backward".
- Backward without detached compute.

## 4. Do not assume (GR-24)
| Statement | Reason |
|---|---|
| full_collect every 50 keeps peak at 21 MB | A-01 |
| AC-8 holds at the 256 MB default budget | A-01 |
| Every op fixes its result shape exactly | R1-10 |
| Every AC has a contract or test | R2-01 |
| Reseeding per iteration fixes dropout repetition | R3-03: one cycle |
| CPU_BACKEND IS-A REFERENCE; twin test covers exactly the redefined kernels | R2-05; withdrawn |
| simple_file covers safetensors I/O | R3-02 |
| INDEX_TENSOR churn is budgeted | R1-04 |
| Any positive INTEGER seed works | R3-01: 2^31-1 gives constant 0 |

## 5. Escalations
**None.**
- HOST_BACKEND is internal layout over stubs, cheap now or later, with no public API change: decided here as preferred, not required.
- The G-A ownership model starts conservative (monotone count). A precise view model later only weakens `release`'s precondition, which is backward compatible.

## 6. Overall: PASS WITH CONDITIONS
Must land before Phase 3 tasks are cut:
1. G-A contract text: `saved_live`, `storage_shared`, and an export set that includes TENSOR.
2. G-C: tracked inputs with detached compute; R1-05, R1-06.
3. G-B: the Q7 invariant; INDEX_TENSOR budget through a shared accessor.
4. R1-13 bound, R3-01 seed precondition, R1-09 is_equal.
5. Register rewording for CL-1/4/5/7/10; homes for AC-7/AC-9; A-01 annotation plus the AC-8 measuring method.
6. Recompile both targets to 0/0 (one compile at a time).

Other MEDIUM/LOW items land with their class in Phase 4. G-6 before M2 dropout.

## 7. Open items for Larry
1. Adopt this verdict as adjudicated?
2. Count executed runs as external confrontation here (C2) or not (C3)?
3. Regenerate `eiffel-debate/ROE-CARD.md` for v1.1 (void now)?
4. Approve annotating approved intent-v2 Q7: the 21 MB note (A-01), and explicit activation release narrowed to optional?
5. Delete the M0 spike EIFGENs outside the cycle folder?

## 8. Larry's gate (C1)
**2026-10-25, Larry: "yes to all five, keep going."**
1. Verdict ADOPTED as adjudicated.
2. Executed runs count as external confrontation: the cycle grades **C2** (CL-2, CL-6, E-1 discharged by runs).
3. ROE-CARD regeneration for ROE v1.1: approved.
4. Annotating intent-v2 Q7 (A-01; activation release optional): approved.
5. Deleting the M0 spike EIFGENs: approved.
