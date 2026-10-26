# LANDSCAPE: simple_tensor

All URLs were fetched 2026-10-25, by the research sub-agent or directly.

## Existing Solutions

### micrograd (Karpathy)
| Aspect | Assessment |
|--------|------------|
| Type | LIBRARY (teaching) |
| Platform | Python |
| URL | https://github.com/karpathy/micrograd |
| Maturity | EXPERIMENTAL (teaching) |
| License | MIT |

- **Strengths:** about 100-line engine; reverse mode over a dynamically built DAG; the clearest statement of the algorithm (topological sort, then a reverse walk calling each node's local backward).
- **Weaknesses:** scalar nodes only, which is hopeless at 121M parameters.
- **Relevance:** 40%. It's the conceptual spec; nodes must be tensors.

### llm.c (Karpathy)
| Aspect | Assessment |
|--------|------------|
| Type | PROGRAM |
| Platform | C / CUDA |
| URL | https://github.com/karpathy/llm.c |
| Maturity | MATURE (31k stars) |
| License | MIT |

- **Strengths:**
  - Hand-written backward per layer (train_gpt2.c); one contiguous activation arena; `+=` gradients for tied weights; AdamW.
  - **Oracle pattern:** test_gpt2.c loads a PyTorch debug state (inputs, logits, loss, every gradient) and checks logits/loss to 1e-2, gradients to 2e-2, and 10 frozen step losses.
- **Weaknesses:** no autograd, so every new architecture means new hand-written backward code.
- **Relevance:** 70%. Adopt the oracle pattern and the arena idea.
- **Speeds:** CPU about 1.3–1.45 s/step (M3 Max, B=4, T=64); A100 about 178K tok/s (discussion #481).

### tinygrad
| Aspect | Assessment |
|--------|------------|
| Type | FRAMEWORK |
| Platform | Python |
| URL | https://github.com/tinygrad/tinygrad |
| Maturity | GROWING (pre-1.0) |
| License | MIT |

- **Strengths:** lazy graph plus kernel fusion; a new accelerator needs only about 25 low-level ops.
- **Weaknesses:** Python; no Vulkan backend.
- **Relevance:** 50%. Keep the primitive set small; fusion is a later optimization.

### ggml / llama.cpp training (ggml-opt)
| Aspect | Assessment |
|--------|------------|
| Type | LIBRARY |
| Platform | C/C++ |
| URL | https://github.com/ggml-org/ggml, https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-opt.h |
| Maturity | MATURE for inference; training "very much WIP" (llama.cpp examples/training README) |
| License | MIT |

- **Strengths:**
  - Static backward graph built from the forward graph; AdamW and SGD; cross-entropy.
  - The **Vulkan backend has backward ops**: OPT_STEP_ADAMW, CROSS_ENTROPY_LOSS_BACK, SILU_BACK, RMS_NORM_BACK, SOFT_MAX_BACK, ROPE_BACK (docs/ops.md).
- **Weaknesses:** the training README names CPU and CUDA only; end-to-end Vulkan training is unverified.
- **Relevance:** 75%. It's the closest existing **Vulkan training** reference; its GLSL shaders and `*_BACK` op list make a ready inventory.

### Burn (Rust)
| Aspect | Assessment |
|--------|------------|
| Type | FRAMEWORK |
| Platform | Rust (CubeCL: CUDA, ROCm, Metal, Vulkan, WebGPU) |
| URL | https://github.com/tracel-ai/burn |
| Maturity | GROWING (0.22, breaking changes announced) |
| License | MIT / Apache-2.0 |

- **Strengths:** autodiff is a **decorator over a backend** and opt-in per tensor (`require_grad`). That maps directly onto Eiffel deferred classes.
- **Relevance:** 60%. Copy the layering.

### candle (Hugging Face, Rust)
| Aspect | Assessment |
|--------|------------|
| Type | FRAMEWORK |
| Platform | Rust (CPU, CUDA, Metal, WASM; no Vulkan) |
| URL | https://github.com/huggingface/candle (backprop.rs) |
| Maturity | MATURE |
| License | Apache-2.0 / MIT |

- **Strengths:** the cleanest small tensor-level tape. Each tensor records its Op, nodes are topologically sorted, `backward()` matches on each Op, and gradients go into a separate **GradStore** keyed by tensor id.
- **Relevance:** 70%. Model D-003 on it.

### Kompute
| Aspect | Assessment |
|--------|------------|
| Type | LIBRARY |
| Platform | C++ on Vulkan |
| URL | https://github.com/KomputeProject/kompute |
| Maturity | GROWING (LF AI & Data) |
| License | Apache-2.0 |

- **Strengths:** a thin Vulkan compute layer. A Sequence records sync, dispatch, sync and submits once; fences give asynchronous waits; explicit host/device memory. Raw Vulkan costs "500–2000 lines" before any application code.
- **Relevance:** 55%. It's the blueprint for the simple_vulkan gaps (batching, device-local memory, fences).

### needle-rs
| Aspect | Assessment |
|--------|------------|
| Type | LIBRARY (inference only) |
| Platform | Rust → WASM, C FFI |
| URL | https://github.com/geekgineer/needle-rs |
| Maturity | GROWING (121 commits, 369 CI tests) |
| License | MIT |

- **Strengths:** proves a non-Python Needle 3 is possible: token-exact with JAX, max relative deviation 9.0e-6, fixtures committed.
- **Lesson:** use an **f32 oracle**. A bf16 reference makes a correct implementation look wrong (relative error 9.7).
- **Relevance:** 80% for the Needle forward-pass stage.

### Cactus Needle reference trainer (cactus-needle 3.2.8)
| Aspect | Assessment |
|--------|------------|
| Type | PROGRAM |
| Platform | Python: JAX + Flax + optax |
| URL | https://huggingface.co/Cactus-Compute/needle3, https://pypi.org/project/cactus-needle/ |
| License | Apache-2.0 |

- LoRA rank 16, alpha 32, targets q/k/v/gate/out_proj; AdamW, warmup + cosine, clip 1.0.
- Training data is generated through OpenRouter (DeepSeek).
- Export merges LoRA and quantizes to 4-bit .cact. Read locally (finetune.py, 509 lines).
- **JAX has no native NVIDIA GPU on Windows** (https://docs.jax.dev/en/latest/installation.html: "no"; WSL2 "experimental").

## Eiffel Ecosystem Check

### ISE Libraries
- base: SPECIAL, MANAGED_POINTER, C_STRING. No numeric arrays beyond those.

### simple_* Libraries (fleet survey, file:line in the survey report)
| Library | What it gives us |
|---|---|
| simple_rixgpt | Pure-Eiffel char-GPT forward pass: matvec, layernorm, GELU, softmax, KV cache. `ARRAY [REAL_64]`, naive loops, **no backward**. Its RIXG loader is reusable. |
| simple_nn | Per-layer backward, ARRAY2 [REAL_64], MSE + SGD, a gradient-check test (central differences, 1e-6). **No dependents.** |
| simple_linalg / simple_math | MATRIX over `ARRAY [REAL_64]`, naive matmul; several stubs. Not a base for tensors. |
| simple_vulkan | Context, buffers, SPIR-V, 8 storage bindings, push constants, dispatch. Gaps: host-visible memory only; submit + `vkQueueWaitIdle` per dispatch; no `C blocking`; descriptor pool sized for about 4 pipelines; descriptor set never freed. |
| simple_shaderc | `compile_compute (glsl)` → SPIR-V MANAGED_POINTER, feeding `VULKAN_SHADER.make_from_memory`. |
| simple_onnx | ONNX_TOKENIZER = SentencePiece **unigram, Viterbi** (from a text vocab). The algorithm Needle's tokenizer needs. |
| simple_grpc | SIMPLE_PROTOBUF, a hand-written wire-format reader (not needed: the .cact embeds the vocab). |
| simple_json, simple_file | JSONL data, safetensors header (JSON). |
| simple_ai_client | Training-data generation through Claude (replaces Cactus's OpenRouter generator). |
| simple_needle (spike) | libneedle.dll binding; the **oracle** for Needle-stage parity. |

### Gobo Libraries
- None relevant.

### Gap Analysis
Not available in Eiffel anywhere, fleet or web:
- an N-d tensor;
- autograd;
- AdamW;
- safetensors I/O;
- GPU training kernels;
- LoRA.

The web search found one dormant Eiffel neural-net repo (ZeLarpMaster/Neural-Network, GPL-2.0, 2016) and nothing else.

## Comparison Matrix
| Feature | llm.c | candle | ggml-opt | Burn | Our Need |
|---------|-------|--------|----------|------|----------|
| General autograd | ✗ | ✓ | ✓ (static) | ✓ | MUST |
| Hand-checkable oracle tests | ✓ | partial | partial | partial | MUST |
| Vulkan training | ✗ | ✗ | partial | ✓ | MUST |
| LoRA | ✗ | ✓ (ext) | ✗ | ✓ | MUST |
| Contracts / DbC | ✗ | ✗ | ✗ | ✗ | MUST (ours) |

## Patterns Identified
| Pattern | Seen In | Adopt? |
|---------|---------|--------|
| Tensor-level tape + separate GradStore | candle | YES |
| Autodiff as a layer over a backend | Burn | YES |
| Small primitive set, composites in the frontend | tinygrad | YES |
| Reference-state oracle file plus frozen step losses | llm.c | YES |
| f32 fixtures, compare intermediates too | needle-rs | YES |
| Batched command sequences, fences, device-local memory | Kompute | YES (fill into simple_vulkan) |
| Static pre-planned memory arena | ggml, llm.c | LATER (optimization) |
| Lazy graph + kernel fusion | tinygrad, Burn | LATER |

## Build vs Buy vs Adapt
| Option | Effort | Risk | Fit |
|--------|--------|------|-----|
| Build (simple_tensor in Eiffel, ideas from the above) | MED | MED | 90% |
| Adopt (bind ggml through C) | MED | HIGH (training WIP, not Eiffel-owned, "pure" fails) | 30% |
| Adapt simple_nn | HIGH (rewrite of its core) | MED | 25% |

**Initial Recommendation:** BUILD.
