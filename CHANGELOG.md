# Changelog: simple_tensor

## Unreleased
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
