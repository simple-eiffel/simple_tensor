# REQUIREMENTS: simple_tensor (+ simple_needle model stage)

## Functional Requirements
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-001 | N-d REAL_32 tensor: shape, strides, views (reshape, transpose, slice), contiguous copy | MUST | Contract tests; view aliasing tests |
| FR-002 | Primitive ops with forward + backward: add, sub, mul, div, neg, scale, matmul (batched), sum/mean (axis), max, exp, log, sqrt, rsqrt, sigmoid, silu, gelu, tanh, softmax, logsumexp, gather/index_select, scatter-add, concat, pad, slice, permute, where/mask, embedding | MUST | Gradient check per op: central difference, relative error ≤ 1e-3 in REAL_32 (≤ 1e-6 in a REAL_64 check build) |
| FR-003 | Tape autograd: `backward` from a scalar loss; gradients accumulate (`+=`) for reused tensors; `requires_grad` per leaf; no-grad mode | MUST | Tied-weight test (one tensor used twice); frozen leaves get no gradient |
| FR-004 | Losses: cross-entropy (from logits, with ignore index), MSE | MUST | Matches a closed-form value on fixtures |
| FR-005 | Optimizers: SGD, AdamW (decoupled weight decay); warmup + cosine schedule; global-norm clipping; gradient accumulation | MUST | AdamW step matches a hand-computed 3-step trace |
| FR-006 | Modules: LINEAR, EMBEDDING, RMS_NORM, LAYER_NORM, ATTENTION (causal, GQA, sliding window, RoPE), parameter registry, freeze/unfreeze | MUST | Module-level gradient checks |
| FR-007 | LoRA adapter on any LINEAR (rank, alpha, targets); merge into the base | MUST | Merged output = base + LoRA output |
| FR-008 | Checkpoint I/O: safetensors read/write (fp32/fp16/bf16); RIXG (simple_rixgpt) write | MUST | Round trip byte-exact; simple_rixgpt loads our RIXG |
| FR-009 | Data: token-stream batcher (random windows), JSONL example loader, train/val split | MUST | Deterministic under a seed |
| FR-010 | CPU backend | MUST | All gradient checks pass |
| FR-011 | GPU backend (simple_vulkan + simple_shaderc GLSL kernels) with the same op set | MUST | Each op's GPU result = CPU result within 1e-4 relative |
| FR-012 | Training loop helper: steps, eval, logging, checkpointing, resume | SHOULD | Resume continues with an identical loss curve |
| FR-013 | Needle 3 model (in simple_needle): .cact reader (dequantize), safetensors reader, unigram tokenizer from the .cact blob, the full forward pass (engram, Hadamard MLP, mHC lanes + Sinkhorn, conv taps, QK-norm, RoPE, GQA, sliding window, heads) | MUST (Full-2) | Parity with libneedle.dll (embeddings; greedy calls on the test sets) |
| FR-014 | Needle 3 LoRA fine-tune + export: merge, 4-bit quantize, .cact writer | MUST (Full-3) | Exported .cact loads in libneedle and beats the untuned score on held-out sets |
| FR-015 | Training-data generator through simple_ai_client (Claude), Cactus JSONL shape `{query, reasoning, answers}` | SHOULD | Valid JSONL; schema-checked |
| FR-016 | Mixed precision (bf16 activations, fp32 master weights) | SHOULD | Loss curve within tolerance of fp32 |

## Non-Functional Requirements
| ID | Requirement | Category | Measure | Target |
|----|-------------|----------|---------|--------|
| NFR-001 | CPU matmul throughput | PERFORMANCE | GFLOP/s, 256×768×768, lean build | ≥ 20 single thread (the C-inline spike reached 26.6) |
| NFR-002 | GPU matmul throughput on the RTX 5070 Ti | PERFORMANCE | GFLOP/s, same shape and larger | ≥ 2,000 for a first tiled shader (to be measured: spike S-2) |
| NFR-003 | RixGPT training time | PERFORMANCE | Wall clock to val loss ~0.78 | GPU: same order as PyTorch (~3 min); CPU: under 1 h |
| NFR-004 | GC law | RELIABILITY | Allocation latency on another processor during a long kernel | Under 10 ms (simple_process probe style) |
| NFR-005 | Determinism | RELIABILITY | Same seed gives the same loss curve | Bit-identical on CPU; GPU within 1e-5 |
| NFR-006 | Memory | RESOURCE | Peak RAM, Needle LoRA training on CPU | Under 8 GB |
| NFR-007 | No network at run time | SECURITY | Network imports in the built exe (dumpbin) | None, except optional simple_ai_client data generation |

## Constraints
| ID | Constraint | Type | Immutable? |
|----|------------|------|------------|
| C-001 | SCOOP-compatible, void-safe | TECHNICAL | YES |
| C-002 | simple_* over ISE/Gobo; gaps filled in the owning library (e.g. simple_vulkan) | ECOSYSTEM | YES |
| C-003 | Long C calls `C blocking`, C-heap memory only | TECHNICAL | YES |
| C-004 | Invariants O(1): no walks over tensor data in invariants | TECHNICAL | YES |
| C-005 | No Python at run time; Python at dev time only if D-006 needs it (and only if Larry allows) | POLICY | Larry's gate |
| C-006 | Only one Eiffel compile at a time | PROCESS | YES |
