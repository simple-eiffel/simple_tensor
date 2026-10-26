# SCOPE: simple_tensor (+ simple_needle as first model)

Date: 2026-10-25. Pre-Phase research, /eiffel.research.

## Problem Statement
In one sentence: no Eiffel library can **train** a modern neural model, so every model Larry trains (RixGPT, RixQwen, a Needle 3 fine-tune) has to go through Python (PyTorch or JAX).

- **What's wrong today:**
  - The fleet can *run* models: simple_rixgpt (a pure-Eiffel transformer forward pass), simple_needle (binding to libneedle.dll), and simple_onnx.
  - The only thing that learns is simple_nn, a dense-layer network with REAL_64 numbers and MSE loss.
  - There's no tensor type, no automatic gradients, no AdamW optimizer, no GPU training.
- **Who experiences this:**
  - Larry, who wants "a pure Eiffel AI and trainable model that I can use against anything I like" (2026-10-25).
  - Any Eiffel developer: a web search found no Eiffel tensor, autodiff or GPU ML library (see 02-LANDSCAPE).
- **Impact of not solving:**
  - Training stays a Python island: Store-Python throttles, ComfyUI's embedded Python, and JAX on Windows being CPU-only.
  - Models can't be retrained from inside Eiffel apps.

## Target Users
| User Type | Needs | Pain Level |
|-----------|-------|------------|
| Larry (fleet owner) | Train or fine-tune his own models (char GPT, Needle 3, later others) without Python, on his RTX 5070 Ti | HIGH |
| Eiffel apps in the fleet | Retrain or adapt a model at runtime (an app that learns from its user) | MED |
| Eiffel community (EUG) | A first-of-its-kind contract-checked tensor and autograd library | MED |

## Success Criteria
| Level | Criterion | Measure |
|-------|-----------|---------|
| MVP-1 | Every primitive op's gradient matches finite differences | Gradient-check test per op, relative error ≤ 1e-3 (REAL_32) |
| MVP-2 | RixGPT trained from scratch in Eiffel on `D:\rixgpt\corpus.txt` | Validation loss within 0.05 of the PyTorch run (0.78), and the exported model.bin runs in simple_rixgpt |
| Full-1 | GPU backend through simple_vulkan | Each op on the GPU matches the CPU result; the RixGPT run gets close to PyTorch's ~3 min on the same card |
| Full-2 | Needle 3 forward pass in Eiffel | Matches libneedle.dll (embeddings, and greedy-decoded calls on the 16 + 10 test sets) |
| Full-3 | Needle 3 LoRA fine-tune in Eiffel, exported to .cact, run in libneedle | Beats the untuned score on held-out Bible-study and desktop sets |

## Scope Boundaries
### In Scope (MUST)
- An N-dimensional REAL_32 tensor with shape and strides.
- Reverse-mode automatic gradients at the tensor level, as a tape.
- About 25–35 primitive ops, each with forward, backward and a gradient check.
- Losses: cross-entropy and MSE.
- Optimizers: SGD and AdamW, plus warmup + cosine schedule, gradient clipping and gradient accumulation.
- Parameters and modules, with freezing; LoRA adapters.
- Checkpoint I/O: read and write safetensors, and read bf16/fp16/fp32.
- A CPU backend.
- A GPU backend on simple_vulkan + simple_shaderc.
- Data: a token-stream batcher and a JSONL example loader.

### In Scope (SHOULD)
- Mixed precision (bf16 storage with fp32 master copy).
- Activation checkpointing.
- Multi-threaded CPU kernels.

### Out of Scope
- Distributed or multi-GPU training: one 16 GB card.
- CUDA/cuBLAS: Vulkan is the fleet's GPU path (simple_vulkan) and it works across vendors.
- Python interop at run time: the whole point.

### Deferred to Future
- Pure-Eiffel Needle inference with grammar-constrained decoding: libneedle.dll runs our exported .cact first.
- Big models such as a Qwen 3B LoRA (the RixQwen class): needs GPU memory planning, quantized base weights and a GGUF reader.
- Re-basing simple_nn on simple_tensor: simple_nn has no dependents (survey), so later.

## Constraints
| Type | Constraint |
|------|------------|
| Technical | SCOOP-compatible, void-safe, contracts on every public feature (fleet law) |
| Technical | GC law: any long C call must be `C blocking` and touch only C-heap memory |
| Technical | Only ONE Eiffel compile at a time (ec.sh lock) |
| Ecosystem | simple_* first; a gap gets filled in the owning library, never "fallback to X" |
| Hardware | One RTX 5070 Ti (16 GB VRAM, Blackwell), 29.7 GB RAM, Windows 11 |
| Policy | "Pure Eiffel" must be defined. See 04-DECISIONS D-004 (Larry's gate) |

## Assumptions to Validate
| ID | Assumption | Risk if False |
|----|------------|---------------|
| A-1 | Plain-Eiffel loops are fast enough for CPU kernels | **MEASURED (part false):** 4.0 GFLOP/s lean vs 26.6 GFLOP/s for C on the same arrays. See 06-RISKS RISK-001 |
| A-2 | simple_vulkan can run training-scale compute (many pipelines, device-local memory, batched dispatch) | **Survey says not yet:** host-visible memory only, a submit+wait per dispatch, no `blocking`, about 4 pipelines per context. Gaps to fill |
| A-3 | Needle 3's ops are all differentiable with standard primitives | Read of architecture.py: yes. The exotic ones (engram hashed gather, Hadamard Kronecker MLP, Sinkhorn mixing) reduce to gather, einsum/matmul, elementwise and logsumexp |
| A-4 | A pure non-Python reimplementation can match the reference exactly | needle-rs is token-exact with JAX on Needle 3 (max relative deviation 9.0e-6) |
| A-5 | The .cact format can be written from Eiffel | Format is fully specified in Apache-2.0 `export.py`/`quantize.py` (~900 lines). Unverified until a round trip runs in libneedle |
| A-6 | The tokenizer can be read without protobuf | Yes: the .cact embeds every piece and its score (`_tokenizer_blob`, export.py:406) |

## Research Questions
- Which autograd style suits a contract-checked library: a dynamic tape (candle, micrograd) or hand-written backward per layer (llm.c)? → D-003
- Where should tensor memory live: SPECIAL (Eiffel heap) or C heap (MANAGED_POINTER)? → D-002
- Which language for hot kernels? → D-004
- Which oracle proves each stage, without Python at run time? → D-006
