# Intent: simple_tensor

## What
A pure-Eiffel library for building and **training** neural models:
- REAL_32 tensors on C-heap memory;
- reverse-mode automatic gradients (the graph lives in the tensors);
- 34 primitive operations;
- modules (linear, embedding, norms, attention, LoRA);
- SGD/AdamW with schedules and clipping;
- safetensors checkpoints and a seeded token batcher;
- three interchangeable backends: a plain-Eiffel reference (the oracle), a fast CPU backend (C inline, `blocking`), and later Vulkan (GLSL via simple_shaderc).

"Pure Eiffel" means D-004 option 2, gated by Larry 2026-10-25: Eiffel owns everything, hot loops are C inline inside Eiffel classes, GPU kernels are our own GLSL, there's no Python or third-party ML library at run time, and every fast kernel has a plain-Eiffel twin.

## Why
- Larry, 2026-10-25: "I would love to have a pure Eiffel AI and trainable model that I can use against anything I like."
- Today every model he trains goes through Python (RixGPT, RixQwen, a planned Needle fine-tune). No Eiffel tensor/autograd library exists anywhere (research/02).

## Users
- **Larry** (primary): train RixGPT from scratch (M2), fine-tune Needle 3 with LoRA (M5, in simple_needle), and later other models.
- **Fleet apps:** train or adapt a model inside an Eiffel app, on a separate SCOOP processor, without freezing the GUI or server.
- **The Eiffel community:** a contract-checked, first-of-its-kind library (EUG case-study material, Larry's call).

## Acceptance Criteria
- [ ] Every one of the 34 primitives passes a central-difference gradient check (REFERENCE backend).
- [ ] Every CPU kernel matches its REFERENCE twin within 1e-5 relative on seeded random inputs.
- [ ] AdamW reproduces a 3-step hand trace of torch.optim.AdamW's documented algorithm.
- [ ] WARMUP_COSINE_SCHEDULE equals rixgpt.py's `lr_at` at steps 0, 75, 150, 2575, 5000.
- [ ] safetensors round trip is byte-identical for F32; F16/BF16 load matches reference conversions.
- [ ] Same seed gives an identical loss sequence over 20 steps (CPU and REFERENCE).
- [ ] GC probe: worst allocation on another processor < 10 ms during a training step (UC-004).
- [ ] Memory: 1,000 training steps of a small model keep private memory under a set budget (S-4 law).
- [ ] CPU matmul ≥ 20 GFLOP/s (256×768×768, lean, single thread).
- [ ] M2 gate (separate milestone): RixGPT full recipe reaches val 1.201 ± 0.05 at iter 500.

## Out of Scope
- The Needle architecture, .cact, tokenizer (simple_needle).
- RIXG model file writing (simple_rixgpt's training app).
- Multi-GPU, CUDA, Python interop, lazy graphs, kernel fusion beyond hand-fused ops.
- Higher-order gradients.

## Dependencies (REQUIRED - simple_* First Policy)

| Need | Library | Justification |
|------|---------|---------------|
| Containers, MANAGED_POINTER, MEMORY | base | Fundamental types (no simple_* equivalent) |
| exp, log, sqrt, cos, pi (+ **tanh, erf: gap G-3**) | simple_math | Eiffel twin math; GELU needs erf |
| Seeded RNG (+ **Gaussian: gap G-2**) | simple_randomizer | Init, dropout masks, batching |
| safetensors header | simple_json | JSON header |
| File I/O | simple_file | Checkpoints |
| Model queries | simple_mml | Small frames only |
| Tests | simple_testing | TEST_SET_BASE |
| GPU (M3) | simple_vulkan, simple_shaderc | Fleet GPU stack (gaps filled there) |

## MML Decision (REQUIRED)

**Decision:** YES-Optional.
**Rationale:** MML frames where collections are small and checks are postconditions: GRAD_STORE ids, module parameter lists, shape dims. Never over tensor data, and never in invariants (O(1) law).
