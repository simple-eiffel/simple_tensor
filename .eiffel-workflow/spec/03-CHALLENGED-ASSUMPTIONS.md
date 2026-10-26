# CHALLENGED ASSUMPTIONS: simple_tensor

## Assumptions Challenged

### A-S01: Tensors need strided views (transpose without copying)
- **Challenge:** PyTorch-style strides make every kernel handle arbitrary layouts. That multiplies kernel code (C, GLSL, twin) and its tests.
- **Evidence for strides:** saves copies on transpose-heavy code (attention's K^T).
- **Evidence against:** llm.c and ggml run GPT training on contiguous layouts plus explicit permute copies. A matmul kernel can take a "B is transposed" flag, which covers attention without strides.
- **Verdict:** INVALID for M1–M3.
- **Action:**
  - All tensors are contiguous row-major. `reshape` is a free view (same storage).
  - `transpose`/`permute` are copy ops with a backward.
  - matmul takes `transpose_b` (and `transpose_a`) flags.
  - Strides can be revisited if a profile shows copies above 10% of step time.

### A-S02: The graph must live on a global tape
- **Challenge:** a global (per-thread `once`) tape collides with SCOOP (one per processor), needs explicit clearing, and leaks graphs between steps.
- **Evidence against the tape:** candle stores each tensor's Op inside the tensor; backward sorts from the loss; the graph dies with the tensors (GC).
- **Verdict:** INVALID. The research's word "tape" becomes "graph in tensors".
- **Action:** TENSOR.origin: detachable TENSOR_OP. No global state. `no_grad` needs no mode switch: untracked inputs make untracked results.

### A-S03: Plain-Eiffel twins are cheap enough to run in the normal test suite
- **Evidence:** S-1: 4 GFLOP/s lean (but tests run in the DbC build: 0.93). A 256×768×768 twin matmul costs ~0.3 s in DbC.
- **Verdict:** VALID with care.
- **Action:** twin tests use small shapes (≤ 64 per dim); a separate lean "parity sweep" target runs larger shapes.

### A-S04: REAL_32 gradient checks are reliable
- **Challenge:** central differences in REAL_32 lose about half the digits; relative error 1e-3 can fail on correct code (ops like log, softmax near saturation).
- **Verdict:** NEEDS_VALIDATION.
- **Action:** the gradient check runs on REFERENCE_BACKEND, whose kernels compute in REAL_64 internally and store REAL_32. Inputs are drawn away from kinks (relu at 0, log near 0). Step 1e-2·max(1,|x|), tolerance 1e-2 absolute + 1e-3 relative. A failing op is re-checked with a REAL_64 twin build flag (D-005) before it counts as a defect.

### A-S05: One TENSOR class can serve indices too
- **Challenge:** token ids and targets are integers, and REAL_32 is exact only to 2^24, which covers vocabularies but is a type lie.
- **Verdict:** INVALID.
- **Action:** a separate INDEX_TENSOR (INTEGER_32), never differentiable.

### A-S06: The backend is chosen globally
- **Challenge:** a global backend can't test twin parity (two backends in one process).
- **Verdict:** INVALID.
- **Action:** each storage belongs to a backend. Ops require `same_backend`, and `to (a_backend)` copies a tensor across. A default backend is passed explicitly to factories (SIMPLE_TENSOR.make_with_backend), never global.

### A-S07: simple_randomizer covers initialization
- **Evidence:** simple_randomizer.e has random_real (uniform) and no Gaussian draw (survey). It wraps ISE RANDOM.
- **Verdict:** gap.
- **Action:** **G-2**: add `random_gaussian (mean, std): REAL_64` (Box–Muller) to simple_randomizer, with tests, version bump and CHANGELOG. TENSOR_RANDOM wraps it.

### A-S08: Fused attention is needed in M1
- **Challenge:** attention composes from primitives (matmul, scale, causal mask, softmax, matmul). Fusing early duplicates backward code.
- **Verdict:** MODIFY.
- **Action:** M1 builds attention from primitives (correctness first). A fused OP_CAUSAL_ATTENTION is added in M2 only if the profile demands it, checked against the composed version (the composed version is its oracle).

## Requirements Questioned

### FR-008: RIXG writer belongs in simple_tensor?
- **Challenge:** RIXG is simple_rixgpt's format.
- **Verdict:** MODIFY. The RIXG writer lives in simple_rixgpt's training app; simple_tensor provides generic named-tensor export (safetensors) only. That keeps simple_tensor free of model formats, by the same logic that puts Needle's .cact in simple_needle.

### FR-012: training-loop helper
- **Verdict:** KEEP as SHOULD. It's a thin TRAINER class (M2), not a framework.

### NFR-003: CPU training under 1 h for RixGPT
**Recipe, read from `D:\rixgpt\rixgpt.py` and `train.log`, 2026-10-25:**
- 6 layers, 6 heads, n_embd 384, block 256, batch 64, dropout 0.2.
- LayerNorm with bias; MultiheadAttention (fused in_proj QKV + out_proj, with biases); MLP 384→1536 (GELU exact) →384.
- Token + position embeddings; head tied to the token embedding; init normal(0, 0.02), biases zero.
- AdamW lr 1e-3, betas (0.9, 0.99), weight decay 0.1 on ALL parameters; linear warmup 150, cosine to 0.1·lr at 5,000 iters; clip 1.0; bf16 autocast; seed 1337; 90/10 split.
- Eval: 50 batches every 500 iters.
- Logged val: iter 500 → 1.201, 1000 → 0.999, 3000 → 0.824, 4000 → 0.795, **5000 → 0.782**; 193 s total.

**Compute:** 5,000 × 64 × 256 = 82M tokens × 6 × 10.9M params ≈ 5.4e15 FLOP, plus attention. That's about 57 h single-thread CPU at the S-1 rate (26.6 GFLOP/s), ~3.7 h on the GPU as-is (S-2, ~400 GFLOP/s), and ~45 min at 2 TFLOP/s.

**Verdict:** INVALID as written.
**Action:**
- **M2 gate (CPU):** match the PyTorch curve at iter 500 (val 1.201 ± 0.05) on the full recipe. That's ~5.5 h on one thread; the target is under 1 h with CPU threading (a SHOULD becomes a MUST for M2).
- **The full 5,000-iter 0.782 match moves to the M3 GPU gate.**

**New op required by the recipe:** DROPOUT (seeded mask, saved for backward). Train/eval mode on modules (`set_training`).

## Missing Requirements Identified
| ID | Missing Requirement | How Discovered |
|----|---------------------|----------------|
| FR-NEW-001 | `to (backend)`: copy a tensor across backends | A-S06 (twin parity) |
| FR-NEW-002 | GC_POINT mixin, plus an S-3-style allocation probe in the test suite (UC-004) | S-3 law |
| FR-NEW-003 | `detach` (same data, untracked) | Needed for eval and for stopping gradients (e.g. targets) |
| FR-NEW-004 | Gaussian init (gap fill in simple_randomizer, G-2) | A-S07 |
| FR-NEW-005 | Parameter naming (dotted paths) for checkpoints | Module ↔ safetensors mapping |
| FR-NEW-006 | matmul `transpose_a/transpose_b` flags | A-S01 |

## Design Constraints Validated
| Constraint | Valid? | Notes |
|------------|--------|-------|
| simple_* first | YES | simple_json (safetensors header), simple_file, simple_randomizer (+G-2), simple_vulkan + simple_shaderc (M3), simple_testing |
| SCOOP-compatible | YES | No global state (A-S02, A-S06); C-heap storage; blocking kernels; GC points |
| Void-safe | YES | origin and gradients are detachable |
| Invariants O(1) | YES | Shape/storage counts are cached attributes; no data walks |
