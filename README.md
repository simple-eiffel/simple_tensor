# simple_tensor

Tensors, automatic gradients, optimizers and training for Eiffel. Everything is Eiffel: the hot loops are C inline inside Eiffel classes, and every fast kernel has a plain-Eiffel twin that checks it. No Python or outside ML library at run time.

**Status: M1 complete (2026-10-25). CPU backend; the Vulkan GPU backend is milestone M3.**

## Installation

```bash
set SIMPLE_EIFFEL=D:\prod
```

```xml
<library name="simple_tensor" location="$SIMPLE_EIFFEL/simple_tensor/simple_tensor.ecf"/>
```

Needs simple_math 0.2.0+ (`tanh`, `erf`), simple_randomizer 1.1.0+ (Gaussian draws) and simple_file 1.2.0+ (bulk binary I/O), plus simple_mml and simple_json.

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
- TENSOR_RANDOM: seeded uniform, normal and dropout-mask fills.
- TOKEN_BATCHER: random blocks of a token stream, with targets shifted by one.

**Files.** SAFETENSORS_FILE loads and saves the Hugging Face `.safetensors` format. It reads F32, F16 and BF16 and writes F32.

**Backends:**

| Backend | Role |
|---|---|
| CPU_BACKEND (default) | The 13 hot kernels in C inline, marked `blocking` so the Eiffel garbage collector never waits on them. |
| REFERENCE_BACKEND | Plain Eiffel in `REAL_64`: the oracle every fast kernel is tested against. |

To use the oracle: `create f.make_with_backend (create {REFERENCE_BACKEND})`.

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

`ec.sh test -config simple_tensor.ecf -target simple_tensor_tests`, then run `EIFGENs/simple_tensor_tests/F_code/simple_tensor.exe`. There are 18 tests, all real; the evidence is in `.eiffel-workflow/evidence/phase4-compile.txt`.

## Roadmap

| Milestone | What | Gate |
|---|---|---|
| M0 ✅ | Spikes: CPU matmul, GC behavior, GPU through simple_vulkan | Numbers measured |
| M1 ✅ | Tensors, operations with backward, autograd, SGD/AdamW, safetensors | Every op passes its gradient check |
| M2 | RixGPT (a character-level GPT) trained from scratch in Eiffel | Validation loss 1.201 ± 0.05 at step 500, matching PyTorch |
| M3 | Vulkan GPU backend | GPU = CPU per op; the full RixGPT run reaches 0.78 |
| M4 | Cactus Needle 3 forward pass (in simple_needle) | Parity with libneedle.dll |
| M5 | Needle 3 LoRA fine-tune, exported to .cact | libneedle runs it and beats the untuned scores |

Related: `D:\prod\simple_needle` (the Needle binding and the parity oracle), `D:\prod\simple_rixgpt` (Eiffel inference for the M2 model), `D:\prod\simple_vulkan` (GPU).

## License

MIT
