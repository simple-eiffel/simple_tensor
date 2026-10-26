# Changelog: simple_tensor

## 0.1.0 (2026-10-25): M1, CPU training
- TENSOR and SIMPLE_TENSOR: contiguous `REAL_32` tensors on the C heap; INDEX_TENSOR for token ids and masks.
- 34 primitive operations, each with its backward. Autograd keeps the graph in the tensors themselves (no global tape) and frees intermediates during `backward`.
- Backends: CPU_BACKEND (13 hot kernels in C `blocking` inline) and REFERENCE_BACKEND (the plain-Eiffel `REAL_64` oracle).
- SGD_OPTIMIZER, ADAMW_OPTIMIZER; CONSTANT_SCHEDULE, WARMUP_COSINE_SCHEDULE; GRAD_STORE with norm clipping and non-finite detection.
- TENSOR_RANDOM, TOKEN_BATCHER, SAFETENSORS_FILE (F32/F16/BF16 read, F32 write), STORAGE_BUDGET.
- 18 tests:
  - gradient checks on every primitive;
  - twin parity on every hot kernel;
  - AdamW against a PyTorch hand trace;
  - the schedule against rixgpt.py;
  - bounded memory over 1,000 steps;
  - a SCOOP garbage-collector probe;
  - the README quick start.
- Lean benchmark target `simple_tensor_bench`: matmul at 26–30 GFLOP/s single thread in all transpose cases.
