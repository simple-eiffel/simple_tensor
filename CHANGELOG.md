# Changelog: simple_tensor

## Unreleased
- **GPU backend (M3b), `simple_tensor_vulkan.ecf`:** VULKAN_BACKEND and VULKAN_STORAGE on simple_vulkan 0.4.0.
  - Every kernel is GLSL with device-address push constants, recorded into batches, with buffers recycled through a pool.
  - Hot and cold twin parity against the oracle; 0 host fallbacks. `profile_report` gives the per-kernel GPU time.
  - Kernel highlights: 64×64 register-tiled matmul; deterministic CSR scatter for embedding gradients; Philox `fill_random` (masks bit-identical to the CPU's).
- **Modules (src/nn):** NN_MODULE, NN_LAYER (`accepts`), NN_LINEAR, NN_EMBEDDING, NN_LAYER_NORM, NN_DROPOUT, NN_MLP, NN_CAUSAL_SELF_ATTENTION.
  - PyTorch layouts and parameter names, so trained weights map one to one.
- **Fused ops (memory):** `causal_softmax`, `layer_norm`, `linear` (bias in place), `dropped` (mask regenerated in backward, never stored).
  - Each is held to its primitive composition by a test, values and gradients, and passes gradient checks.
- **RixGPT trained in Eiffel (M2 gate):** step 500 reaches train 1.510 / val 1.242, against PyTorch's 1.511 / 1.201 (simple_rixgpt `simple_rixgpt_train.ecf`).
- **M3 gate:** the full 5,000 steps end at train 0.727 / val 0.807, against PyTorch's 0.724 / 0.782 (`.eiffel-workflow/evidence/m3-gate-rixgpt-5000.txt`).
- **Matmul speed:** a RixGPT step went from about 600 ms to 334 ms (`.eiffel-workflow/evidence/m3-speed-splitk.txt`).
  - 128×128 register tiles (`matmul_128`) for products of at least 128×128: about 10% faster than 64×64.
  - Split-K for one long-K product (the weight gradients, K = tokens): K slices into partials, then `sum_slices`. Those products went from 0.8–2.5 to 6.4–8.9 TFLOP/s.
  - `set_wide_tiles` and `set_split_k` switch each route off for A/B runs; `profiled_ms` gives one kernel's GPU time.
  - New twins: 128-wide tiles and split-K with ragged slices, every transpose flag.
- **Fixes:**
  - VULKAN_STORAGE.dispose called a routine on another object, which may already be reclaimed; it now uses its own raw handles.
  - A failed GPU allocation used to go unnoticed; it now retries after a collection, then raises a clear error.
  - Three test helpers had side effects inside `check`.
- TENSOR_RANDOM now draws from Philox4x32-10 (simple_randomizer 1.2.0, gap G-6). The period is 2^64 per stream instead of 2^31.
  - Every value has a fixed stream address (`position`), so a GPU kernel can reproduce any element.
  - Seeds may now be any non-negative INTEGER.
  - A new test checks a keep mask element by element against an independent generator, and checks the mean and spread of a 100,000-value normal fill. 32 tests in all.

## 0.1.0 (2026-10-25): M1, CPU training
- TENSOR and SIMPLE_TENSOR: contiguous `REAL_32` tensors on the C heap; INDEX_TENSOR for token ids and masks.
- 34 primitive operations, each with its backward. Autograd keeps the graph in the tensors themselves (no global tape) and frees intermediates during `backward`.
- Backends: CPU_BACKEND (13 hot kernels in C `blocking` inline) and REFERENCE_BACKEND (the plain-Eiffel `REAL_64` oracle).
- SGD_OPTIMIZER, ADAMW_OPTIMIZER; CONSTANT_SCHEDULE, WARMUP_COSINE_SCHEDULE; GRAD_STORE with norm clipping and non-finite detection.
- TENSOR_RANDOM, TOKEN_BATCHER, SAFETENSORS_FILE (F32/F16/BF16 read, F32 write), STORAGE_BUDGET.
- 31 tests:
  - gradient checks on every primitive;
  - twin parity on every hot kernel;
  - AdamW against a PyTorch hand trace;
  - the schedule against rixgpt.py;
  - bounded memory over 1,000 steps;
  - a SCOOP garbage-collector probe;
  - the README quick start;
  - adversarial cases: hostile safetensors files, extreme values, two trainers on separate SCOOP processors, a 5,000-step stress run.
- Hardening fixed two defects:
  - negative safetensors `data_offsets` were accepted, so header bytes loaded as weights;
  - `set_budget` refused a lower budget mid-run; it now collects first.
- Lean benchmark target `simple_tensor_bench`: matmul at 26–30 GFLOP/s single thread in all transpose cases.
