# M3b design: VULKAN_BACKEND (2026-10-26)

The M1 spec fixed the seam (VULKAN_BACKEND behind TENSOR_BACKEND, VULKAN_STORAGE behind TENSOR_STORAGE, a separate cluster, so CPU users need no Vulkan SDK). This record fixes the internals, as spec 08 deferred ("M2/M3 internals refined at their own phase starts"). Technical forks are settled by the record (S-5, the tests), not handed to Larry.

## Inputs (measured)
- **simple_vulkan 0.2.0 (M3a):**
  - device-local buffers with shader addresses;
  - descriptor-free address pipelines;
  - recorded batches with barriers, one `flush`.
- **S-5 (RTX 5070 Ti):**
  - 1,000 dispatches: 6 ms batched, against 74 ms with a wait each;
  - tiled matmul: 3,616 GFLOP/s at 2048³.
- **Larry, 2026-10-26:** training runs on the GPU. The CPU backends stay as the oracle (REFERENCE) and the fast host path (CPU). CPU threading is dropped.

## Packaging
- `simple_tensor_vulkan.ecf` (same repo) with cluster `src_vulkan/`, depending on simple_tensor, simple_vulkan and simple_shaderc. The core `simple_tensor.ecf` is unchanged: no Vulkan SDK needed for CPU use.
- Test target `simple_tensor_vulkan_tests`:
  - the `testing/` cluster, reused, so the twin cases are written once;
  - a GPU test app that runs twin parity with the candidate facade on VULKAN_BACKEND;
  - gradient checks on the GPU;
  - the training determinism and quick-start tests on the GPU.
- Run GPU executables with `$VULKAN_SDK/Bin` on PATH (shaderc_shared.dll).

## VULKAN_STORAGE (inherits TENSOR_STORAGE, DISPOSABLE)
- **The data:** a VULKAN_DEVICE_BUFFER of `4 * count` bytes.
- **Element access (`item`) through a host mirror:**
  - the first `item` after a change downloads the whole buffer once (one flush), then reads serve from the mirror;
  - every backend kernel that writes a storage calls `invalidate`;
  - `put` writes 4 bytes through to the GPU and updates the mirror.
- **Why the mirror:** contracts such as TENSOR_RANDOM's `in_range` walk every element. Without it, an assertion-on build would flush once per element.
- **Memory:**
  - allocations are noted in STORAGE_BUDGET (the same per-processor budget), so dead GPU tensors trigger collections like C-heap ones;
  - `release` and `dispose` (collector) free through `svk_free_buffer_deferred`: freed immediately when no batch is open, otherwise queued and freed after the next flush's fence. No GPU wait inside the collector.

## VULKAN_BACKEND (inherits TENSOR_BACKEND)
- One VULKAN_CONTEXT per backend object. Kernels are pipelines compiled once (simple_shaderc) and cached by name.
- **Execution model:**
  - kernels record into the context's open batch;
  - the GPU runs at the first host read (`download`, mirror refresh, `sum_of_squares`, `has_non_finite`, loss `item`) or at an index upload.
  - One training step therefore costs a handful of flushes, not one per op.
- **Index tensors** (tokens, targets, masks) live on the host (INDEX_TENSOR). A kernel that takes one uploads it to a device buffer cached by object identity and a modification stamp.
  - **First cut:** upload on every use (a write, which flushes). Measure it before optimizing.
- **Host fallback (completeness from day one):**
  - any kernel without its GLSL yet downloads its inputs (and an in-place output) into CPU_STORAGE, runs CPU_BACKEND's kernel, and writes the result back;
  - `fallback_count` per kernel name is a public query.
  - GLSL kernels then replace fallbacks one by one, each proven by twin parity against REFERENCE_BACKEND within 1e-4.
  - M3b is done when the training-step kernels have no fallbacks: binary, unary, unary_derivative, reduce, reduce_to_shape, matmul (all transpose flags, batched, shared B), softmax / log_softmax, masked_fill, permute, slice/add_slice/concat, gather_rows/scatter_add_rows, cross_entropy (+grad), adamw_update, sgd_update, fill, copy, sum_of_squares, has_non_finite.
- **Randomness:**
  - first cut: TENSOR_RANDOM computes a fill on the host and uploads it once (`backend.upload`), not element by element;
  - before M2 dropout: a GPU Philox4x32-10 kernel. Same (seed, position) means the same words. The keep test is integer-exact on both sides (`word < floor(keep * 2^32)`), so masks match bit for bit.

## Shaders
GLSL 460 with `GL_EXT_buffer_reference`; all buffer addresses and sizes in push constants (≤ 128 bytes). Workgroup 256 for elementwise kernels; 16×16 tiles for matmul (S-5 kernel), extended with the transpose flags and batch strides.

## Gates (M3b)
1. Twin parity: every hot kernel GPU = REFERENCE within 1e-4 relative (forward and backward cases of the M1 suite, unchanged).
2. ~~Gradient checks on VULKAN_BACKEND~~ WITHDRAWN 2026-10-26: GRADIENT_CHECKER requires the reference backend by design (spec 03 A-S04; autograd is backend-independent). Replaced by cold-kernel twin parity: on CPU the cold kernels are HOST_BACKEND code shared with the oracle, but on the GPU they are new GLSL, so they get their own parity cases.
3. `fallback_count` = 0 for every kernel a RixGPT training step uses.
4. A RixGPT-sized step (embedding, attention, MLP, cross-entropy, AdamW) is timed on the GPU against CPU_BACKEND.

## Gate results (2026-10-26)
1. Twin parity, hot kernels: worst 3.6e-7 (log_softmax). Cold kernels: worst 1.4e-7 (reduce_to_shape). Mutation-checked: a wrong cross-entropy gradient fails hot parity; a wrong permute fails cold parity and training.
2. Withdrawn (above).
3. Host fallbacks: 0 across the GPU suite. To re-check on the first RixGPT step (M2).
4. Lean bench (`bench_vulkan/bench_run_2026-10-26.log`), forward + backward + AdamW, loss read back each step:
   - 256x768 . 768x768 layer: CPU 37 ms, GPU 4 ms.
   - 4096x768 -> 3072 -> 768 MLP block: CPU 4,184 ms, GPU 56 ms (about 75x).
   - 2048^3 matmul through TENSOR: 1,644 GFLOP/s, against 3,616 raw (S-5). The gap is per-tensor overhead: a device allocation plus a zero fill for every result. Next optimization: a device-memory pool, with no zero fill for outputs a kernel writes in full.
