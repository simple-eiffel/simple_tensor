# simple_tensor

Tensors, automatic gradients, optimizers and training for Eiffel. Everything is Eiffel: the hot loops are C inline inside Eiffel classes, and every fast kernel has a plain-Eiffel twin that checks it. No Python or outside ML library at run time.

**Status (2026-10-26):** M1 (CPU) and M3b (GPU backend) are complete. M2 passed its gate: RixGPT trained from scratch in Eiffel on the GPU reaches train 1.510 and val 1.242 at step 500, against PyTorch's 1.511 and 1.201 (the gate is 1.201 ± 0.05). The full 5,000-step run, the M3 gate, is in progress.

## Installation

```bash
set SIMPLE_EIFFEL=D:\prod
```

```xml
<library name="simple_tensor" location="$SIMPLE_EIFFEL/simple_tensor/simple_tensor.ecf"/>
```

Needs simple_math 0.2.0+ (`tanh`, `erf`), simple_randomizer 1.2.0+ (the Philox4x32-10 generator) and simple_file 1.2.0+ (bulk binary I/O), plus simple_mml and simple_json.

## Quick start

A 2-8-1 network learns XOR. This is test `test_readme_quick_start`, verbatim; it reaches a loss under 0.01.

```eiffel
local
	f: SIMPLE_TENSOR
	r: TENSOR_RANDOM
	x, y, w1, b1, w2, b2, l_loss: TENSOR
	opt: ADAMW_OPTIMIZER
	step: INTEGER
do
	create f.make
	create r.make (42)
	x := f.from_array (<<0, 0,  0, 1,  1, 0,  1, 1>>, <<4, 2>>)
	y := f.from_array (<<0, 1, 1, 0>>, <<4, 1>>)
	w1 := f.zeros (<<2, 8>>);  r.fill_normal (w1, 0.0, 1.0);  w1 := f.parameter (w1)
	b1 := f.parameter (f.zeros (<<1, 8>>))
	w2 := f.zeros (<<8, 1>>);  r.fill_normal (w2, 0.0, 0.5);  w2 := f.parameter (w2)
	b2 := f.parameter (f.zeros (<<1, 1>>))
	create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w1, b1, w2, b2>>),
		create {CONSTANT_SCHEDULE}.make (0.05), 0.9, 0.999, 1.0e-8, 0.0)
	from step := 1 until step > 500 loop
		l_loss := ((x.matmul (w1) + b1).hyperbolic_tangent.matmul (w2) + b2).mean_squared_error (y)
		opt.step (l_loss.backward)
		step := step + 1
	end
end
```

How it works:
- `f.parameter` returns a trainable copy of a tensor. Every operation on it records where its result came from.
- `backward` walks that record in reverse and returns a GRAD_STORE: one gradient per trainable tensor.
- `opt.step` applies AdamW to every parameter and advances the learning-rate schedule.

## What's in it

**Tensors (TENSOR, SIMPLE_TENSOR).** `REAL_32`, contiguous, any rank. Factories: `zeros`, `ones`, `filled`, `scalar`, `from_array`, `parameter`; integer tensors for token ids and masks (`indices`, `causal_mask`).

**Operations, each with its backward:**
- Elementwise with broadcasting: `+ - * /`, `negated`, `scaled`, `shifted`.
- Unary: `exponential`, `logarithm`, `square_root`, `hyperbolic_tangent`, `sigmoid`, `relu`, `gelu`, `silu`.
- Reductions: `sum`, `mean`, `maximum` (by axis), `sum_all`, `mean_all`, `reduced_to`.
- Shape: `reshaped`, `permuted`, `transposed`, `sliced`, `concatenated`.
- Linear algebra: `matmul` (batched; a shared right-hand side broadcasts), `matmul_transposed`.
- Neural: `softmax`, `log_softmax`, `masked_fill`, `gathered_rows` (embeddings).
- Losses: `cross_entropy` (with an ignore index), `mean_squared_error`.

**Training:**
- SGD_OPTIMIZER and ADAMW_OPTIMIZER (decoupled weight decay, as in PyTorch).
- Learning-rate schedules: CONSTANT_SCHEDULE; WARMUP_COSINE_SCHEDULE (linear warmup, cosine decay to a floor).
- GRAD_STORE: `global_norm`, `clip_global_norm`, `has_non_finite`, `scale_all`, `merge`.
- TENSOR_RANDOM: seeded uniform, normal and dropout-mask fills from Philox4x32-10, a counter-based generator: every value has a fixed address in the stream, so a GPU kernel can reproduce the same mask.
- TOKEN_BATCHER: random blocks of a token stream, with targets shifted by one.

**Files.** SAFETENSORS_FILE loads and saves the Hugging Face `.safetensors` format. It reads F32, F16 and BF16 and writes F32.

**Backends:**

| Backend | Role |
|---|---|
| CPU_BACKEND (default) | The 13 hot kernels in C inline, marked `blocking` so the Eiffel garbage collector never waits on them. |
| REFERENCE_BACKEND | Plain Eiffel in `REAL_64`: the oracle every fast kernel is tested against. |

To use the oracle: `create f.make_with_backend (create {REFERENCE_BACKEND})`.

### The GPU backend (simple_tensor_vulkan.ecf)

VULKAN_BACKEND runs every operation as our own GLSL compute shader on any Vulkan GPU, through simple_vulkan 0.3.0. It lives in a separate ECF, so CPU-only users never need the Vulkan SDK.

```xml
<library name="simple_tensor_vulkan" location="$SIMPLE_EIFFEL/simple_tensor/simple_tensor_vulkan.ecf"/>
```

```eiffel
create f.make_with_backend (create {VULKAN_BACKEND}.make)   -- the rest of the code is unchanged
```

- **Execution:**
  - Kernels are recorded into one batch, and the GPU runs it when the host needs a value: a loss read, a download, a norm, or an index upload.
  - Buffers are addressed straight from the shaders, so no descriptor sets are needed.
  - Memory comes from a device-buffer pool. The collector recycles dead tensors into it without waiting on the GPU.
- **Correctness:** every kernel, hot and cold, passes twin parity against REFERENCE_BACKEND (worst relative error 3.6e-7). Dropout masks drawn on the GPU are bit-identical to the oracle's, and training runs end to end on the GPU.
- **Speed (RTX 5070 Ti, lean build):** a training step of a 4096×768 → 3072 → 768 MLP block takes 56 ms on the GPU, against 4,184 ms single-thread on the CPU.
- **Running GPU programs:** put `%VULKAN_SDK%\Bin` on PATH; it holds shaderc_shared.dll.

## Modules (src/nn)

Building blocks with PyTorch's layouts and parameter names, so trained weights map one to one:

| Class | What it is |
|---|---|
| NN_MODULE / NN_LAYER | Named parameters (`blocks.0.attn.in_proj_weight`); tied weights listed once; training mode. A layer's `forward` requires `accepts (x)`. |
| NN_LINEAR | `y = x Wᵀ + b`, W is out×in; any input rank; can share another module's weight. |
| NN_EMBEDDING | A table of learned rows, looked up by INDEX_TENSOR. |
| NN_LAYER_NORM | Over the last axis (biased variance, eps), built from primitives. |
| NN_DROPOUT | Inverted dropout with Philox masks, drawn on the input's backend. |
| NN_MLP | Linear → GELU → Linear → Dropout. |
| NN_CAUSAL_SELF_ATTENTION | Multi-head causal attention in nn.MultiheadAttention's layout (in_proj 3C×C, out_proj). |

simple_rixgpt's `simple_rixgpt_train.ecf` builds RixGPT from these modules and trains it on the GPU.

## Memory

Tensor data lives on the C heap, and the garbage collector can't see it. Two mechanisms keep memory bounded:
- **Explicit release.** `backward` releases intermediates as soon as it's done with them, and `release` frees a tensor you're finished with.
- **A storage budget.** STORAGE_BUDGET forces a collection after every 256 MB of new tensor data (`set_budget` changes it).

Measured: 1,000 training steps churned about 457 MB, and memory rose 17 MB.

## Measured (single thread, lean build)

| What | Result |
|---|---|
| matmul 256×768 · 768×768 | 29.6 GFLOP/s (transpose A: 30.2, transpose B: 26.0) |
| One training step of a 256×768 → 768 layer (forward, backward, AdamW) | 44 ms |
| Gradient checks (51, every operation) | worst error 1.3e-4 against numerical differentiation |
| C kernels vs. plain-Eiffel twins | worst relative error 2.1e-7 |
| AdamW | matches PyTorch's documented algorithm step for step |
| Worst garbage-collector stall on another SCOOP processor while training runs | 5 ms |

To run the benchmark: `ec.sh release -config simple_tensor.ecf -target simple_tensor_bench`, then run `EIFGENs/simple_tensor_bench/F_code/simple_tensor_lean.exe`.

## Tests

`ec.sh test -config simple_tensor.ecf -target simple_tensor_tests`, then run `EIFGENs/simple_tensor_tests/F_code/simple_tensor.exe`. There are 32 tests, all real. They include adversarial ones: hostile model files, extreme values, two trainers on separate processors, and a 5,000-step stress run. The evidence is in `.eiffel-workflow/evidence/` (phase4-compile.txt, phase5-tests.txt, phase6-tests.txt).

## Roadmap

| Milestone | What | Gate |
|---|---|---|
| M0 ✅ | Spikes: CPU matmul, GC behavior, GPU through simple_vulkan | Numbers measured |
| M1 ✅ | Tensors, operations with backward, autograd, SGD/AdamW, safetensors | Every op passes its gradient check |
| M2 ✅ | RixGPT (a character-level GPT) trained from scratch in Eiffel | Validation loss 1.201 ± 0.05 at step 500: got 1.242 (train 1.510 vs 1.511) |
| M3 | Vulkan GPU backend (M3b ✅: every kernel on the GPU, parity proven) | The full RixGPT run reaches 0.78 (running); then speed: our step takes about 1.5 s against PyTorch's 0.04 s |
| M4 | Cactus Needle 3 forward pass (in simple_needle) | Parity with libneedle.dll |
| M5 | Needle 3 LoRA fine-tune, exported to .cact | libneedle runs it and beats the untuned scores |

Related: `D:\prod\simple_needle` (the Needle binding and the parity oracle), `D:\prod\simple_rixgpt` (Eiffel inference for the M2 model), `D:\prod\simple_vulkan` (GPU).

## License

MIT
