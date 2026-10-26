# Intent v2: simple_tensor

Status: refined after the adversarial self-review (2026-10-25). It contains everything in `intent.md`, plus the review below, the dependency audit, and spike S-4.

## What / Why / Users / Out of Scope
Unchanged from `intent.md`.

## Acceptance Criteria (revised)
- [ ] Every one of the 34 primitives passes a central-difference gradient check on REFERENCE_BACKEND.
- [ ] Every CPU kernel matches its REFERENCE twin within 1e-5 relative on seeded random inputs.
- [ ] AdamW reproduces a 3-step hand trace of torch.optim.AdamW's documented algorithm.
- [ ] WARMUP_COSINE_SCHEDULE equals rixgpt.py's `lr_at` at steps 0, 75, 150, 2575, 5000.
- [ ] safetensors: F32 round trip byte-identical; F16/BF16 load matches reference conversions on edge values (±0, subnormals, ±inf, NaN, max).
- [ ] Determinism: same seed → identical loss sequence over 20 steps, on REFERENCE and on CPU (threaded CPU too; see Q6).
- [ ] GC probe: worst allocation on another processor < 10 ms during a training step (UC-004).
- [ ] **Memory (new, from S-4):** 1,000 training steps of a small model keep peak private memory within 2× the steady-state working set.
- [ ] CPU matmul ≥ 20 GFLOP/s single thread (lean).
- [ ] Non-finite detection: a step that produces NaN/Inf gradients is reported by `GRAD_STORE.has_non_finite` (Q5).
- [ ] M2 gate (its own milestone): RixGPT full recipe reaches val 1.201 ± 0.05 at iter 500 (two seeds).

## Deep Review: questions and recommended answers

### Q1. Should simple_tensor know any model file format (RIXG, .cact, GGUF)?
- **Why it matters:** formats are scope creep that couples the library to applications.
- **Alternatives:** (a) built-in RIXG + .cact writers; (b) only safetensors, with model formats in their owning libraries; (c) a plugin registry of formats.
- **Recommended:** (b). safetensors is the generic named-tensor format; RIXG belongs to simple_rixgpt, .cact to simple_needle. Already applied in spec 03.

### Q2. Higher-order gradients (gradient of a gradient)?
- **Why it matters:** supporting them means backward must build graphs too, and that doubles the contract surface.
- **Alternatives:** (a) not supported; precondition `grad_untracked`; (b) full support; (c) later via a flag.
- **Recommended:** (a) for M1–M5. No target model needs it. The precondition makes misuse fail loudly.

### Q3. In-place ops (x += y) to save memory?
- **Why it matters:** in-place writes break autograd when the overwritten value is saved for backward (PyTorch's "version counter" problem).
- **Alternatives:** (a) none, except selectively exported optimizer writes; (b) in-place with version counters; (c) in-place only on untracked tensors.
- **Recommended:** (a). Memory is handled by Q7's design, not by in-place math. Revisit with (c) if a profile demands it.

### Q4. Contiguous-only tensors or strided views?
- **Why it matters:** strides multiply kernel code by layout cases across three backends.
- **Alternatives:** (a) contiguous + matmul transpose flags + copy permute; (b) full strides; (c) strides on CPU only.
- **Recommended:** (a) (spec 03 A-S01). Overturn trigger: permute/transpose copies above 10% of step time on RixGPT.

### Q5. What happens on NaN/Inf mid-training?
- **Why it matters:** one bad batch can poison AdamW state silently; PyTorch's AMP scaler skips such steps.
- **Alternatives:** (a) exception from the op; (b) a query `GRAD_STORE.has_non_finite` with the policy left to the trainer (skip the step, log, lower the LR); (c) ignore.
- **Recommended:** (b). It's a query, so CQS holds; the trainer decides. No exceptions for an environmental condition (fleet error pattern).

### Q6. Is threaded CPU training still deterministic?
- **Why it matters:** NFR-005, and the M2 curve comparison, need repeatable runs.
- **Alternatives:** (a) threads partition **output rows**, so each output element is summed by one thread in a fixed order (deterministic); (b) split-K with atomic adds (faster on narrow shapes, non-deterministic); (c) single thread only.
- **Recommended:** (a). The thread pool lives inside one `C blocking` call (GC-safe by construction). Settle pool vs SCOOP workers with a spike at M2 start (spec 08 open issue 2).

### Q7. Will C-heap tensors exhaust RAM? (**SETTLED by spike S-4, 2026-10-25**)
- **Why it matters:** the collector sees only the Eiffel wrapper, not the C-heap block.
- **Evidence** (`spikes/memory_pressure`, lean): 2,000 dead 4 MB MANAGED_POINTERs (8 GB churned):

  | Strategy | Peak private MB |
  |---|---|
  | GC alone | **8,040** (no collection happened) |
  | `MEMORY.full_collect` every 50 | 21 |
  | Explicit release (`resize (1)`) | 40 |

- **Alternatives:** (a) GC alone; (b) a byte-budget trigger; (c) explicit release; (d) a storage pool/arena.
- **Recommended:** (b) + (c).
  - CPU_STORAGE counts live and allocated bytes in a per-thread `STORAGE_BUDGET`. When bytes allocated since the last collection exceed a budget (default 256 MB, configurable), it calls `MEMORY.full_collect` before allocating.
  - `TENSOR.release` is selectively exported to backward and the trainer. It frees intermediates known dead: activations after backward, gradients of intermediates during the sweep.
  - A pool/arena (d) is a later optimization behind the same storage interface.
- **New contract:** STORAGE_BUDGET invariant `allocated_since_collect <= budget + largest_single_request`.

> **Annotation (Phase 2 review A-01, approved by Larry 2026-10-25; the original text above is kept as written):**
> - **Measurement.** The "21" in the table is a post-collection floor, not a peak. The spike sampled memory only right after a forced collection. Re-measured by sampling every iteration before collecting (`spikes/memory_pressure/s4_run_2026-10-25_per_iteration.log`): GC alone 8,040 MB; full_collect every 50: **221 MB peak**; explicit release 40 MB. The conclusion stands (a forced collection bounds the churn), but the peak under a byte budget is about the live set plus the budget.
> - **AC-8's measuring method.** The test reads PeakPagefileUsage in a fresh process, and the budget is sized against the live set. At the 256 MB default, a "2× steady state" bound fails by design for live sets under about 256 MB.
> - **Release.** Explicit release of activations after backward becomes **optional**. The required safety rules are `storage_shared`, `saved_live`, and a copy on the first gradient store (verdict G-A). `release` stays public, and any restricted export includes TENSOR and GRAD_STORE.

### Q8. Can a trainer run on a separate SCOOP processor cleanly?
- **Why it matters:** UC-004 (train beside a GUI). Tensors are processor-local objects.
- **Alternatives:**
  - (a) the trainer object is `separate`, and its API crosses the boundary only with basic or expanded values (step count, loss as REAL_64, a file path for checkpoints);
  - (b) separate tensors everywhere;
  - (c) no SCOOP support.
- **Recommended:** (a). All tensors live on the trainer's processor. Document the pattern; test it with the GC probe target.

### Q9. Precision of gradient checks with REAL_32 storage?
- **Why it matters:** false failures on correct ops waste days; false passes hide wrong backwards.
- **Alternatives:** (a) REAL_32 storage, REAL_64 compute in REFERENCE, relative step 1e-2, tolerance 1e-2 abs + 1e-3 rel; (b) REAL_64 storage variant for checks; (c) both, with (b) as escalation.
- **Recommended:** (c). Start with (a) (spec 03 A-S04). Any failure is re-run under a REAL_64 storage build of REFERENCE before it counts as a defect. That needs CPU_STORAGE's element width to be a constant chosen per build target: 4 bytes normally, 8 in a `gradcheck64` target. Cheap to add now, costly later.

### Q10. Naming: `TENSOR_OP_*`, `NN_*`, SIMPLE_TENSOR?
- **Why it matters:** collisions across the fleet are non-local failures (C-prefix collision lesson).
- **Alternatives:** (a) as specced; (b) `ST_` prefix on everything; (c) no prefixes.
- **Recommended:** (a). The grep found no collisions. C helper symbols in inline C get the prefix `stensor_` (C-prefix collision rule), never two letters.

### Q11. Is the PyTorch curve a fair oracle given different RNG and bf16 autocast?
- **Why it matters:** an unfair oracle either fails a correct trainer or passes a broken one.
- **Alternatives:** (a) iter-500 val within ± 0.05 over two seeds; (b) the full curve within ± 0.03; (c) only "loss decreases".
- **Recommended:** (a) for the CPU gate; the full 0.782 ± 0.03 at iter 5,000 is the M3 GPU gate. (c) is too weak to catch a wrong gradient that still trains slowly.

### Q12. Is anything premature in M1?
- **Why it matters:** M1 should prove the core.
- **Candidates:** TENSOR_TRAINER, NN_LORA_LINEAR, the SGD momentum variant.
- **Recommended:** move NN_LORA_LINEAR to M2/M5 (M1 keeps modules out entirely). Keep SGD without momentum. TENSOR_TRAINER stays a SHOULD in M2.

## Escalations
None to /eiffel.debate.
- Q7 was the only costly-to-reverse fork resting on unverified evidence, and it was settled by spike S-4 (executed run).
- Q6's pool-vs-SCOOP choice is deferred to an M2-start spike.
- Q9's 64-bit build variant is a cheap reversible addition.

## Dependency Audit (simple_* First)
| Need | Library | Status |
|------|---------|--------|
| MANAGED_POINTER, MEMORY, containers | base (ISE) | Allowed: fundamental types, no simple_* equivalent |
| exp, log, sqrt, cos, pi | simple_math (SIMPLE_MATH: exp 205, log 211, sqrt 241, cos 135, pi 115) | ✓ present |
| tanh, erf | simple_math | **GAP G-3**: not present |
| Uniform RNG, seeds | simple_randomizer (random_real, make_with_seed) | ✓ present |
| Gaussian RNG | simple_randomizer | **GAP G-2**: not present |
| JSON | simple_json (parse, SIMPLE_JSON_OBJECT) | ✓ (used in the simple_needle spike) |
| File I/O | simple_file (read_bytes, read_text_utf_8) | ✓ |
| MML | simple_mml 1.0.2 | ✓ |
| Tests | simple_testing (TEST_SET_BASE) | ✓ |
| GPU | simple_vulkan + simple_shaderc | ✓ works as-is (S-2); gaps G-4 (device-local memory, batching, fences, `blocking`, descriptor pool) |

**Spec correction:** the WARMUP_COSINE_SCHEDULE text in spec 07 uses DOUBLE_MATH. It becomes SIMPLE_MATH (simple_* first).

## Gaps Identified (fill in the owning libraries)
| Gap | Current Workaround | Fill |
|-----|-------------------|------|
| G-2 Gaussian draw | none | simple_randomizer `random_gaussian (mean, std)` (Box–Muller), tests, CHANGELOG, dependents checked |
| G-3 tanh, erf | none | simple_math `tanh`, `erf` (and `sigmoid` as a convenience), tests vs reference values |
| G-4 GPU training gaps | none needed until M3 | simple_vulkan: device-local + staging, recorded batches, fences, `C blocking` waits, descriptor pool sizing and freeing |
| (fleet) ec.sh's error scan misses MSVC C errors ("error C2065") | ec.sh still fails the build ("No .exe found") | Candidate ec.sh scan pattern addition. A fleet tool, so it needs Larry's OK |

## Approval
APPROVED by Larry, 2026-10-25 ("approved, keep going").
