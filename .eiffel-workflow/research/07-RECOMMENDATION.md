# RECOMMENDATION: simple_tensor

## Executive Summary
**Build** simple_tensor, an Eiffel tensor + autograd + optimizer + training library with CPU and Vulkan backends. Prove it end to end by training RixGPT from scratch, then grow simple_needle on it to fine-tune Needle 3 and export a .cact that the vendor's own engine runs. Nothing like it exists in Eiffel; every piece it needs has a working precedent elsewhere (candle's tape, llm.c's oracle discipline, ggml's Vulkan backward ops, needle-rs's token-exact Needle).

## Recommendation
**Action:** BUILD
**Confidence:** HIGH that it can be built. MEDIUM on GPU speed until spike S-2 runs.

## Milestones (each ends at a gate that could fail)

### M0: Spikes (evidence before spec)
- **S-1 (done 2026-10-25):** CPU matmul throughput. Plain Eiffel lean 4.03 GFLOP/s; C inline 26.6 GFLOP/s.
- **S-2:** a GPU matmul on the 5070 Ti through simple_vulkan as it stands today, and with a batched command buffer. **Gate:** GFLOP/s measured; gap list confirmed.
- **S-3:** MANAGED_POINTER vs SPECIAL loop speed, plus the GC-law probe with a 100 ms kernel. **Gate:** settles D-002.

### M1: simple_tensor core (CPU)
- TENSOR, SHAPE, views.
- About 30 primitive ops, each with forward + backward + an Eiffel reference twin.
- Tape and GRAD_STORE; cross-entropy and MSE; SGD and AdamW; schedule, clipping and accumulation.
- Safetensors I/O.
- **Gate:** every op passes its gradient check; AdamW matches a hand-computed trace.

### M2: RixGPT trained in Eiffel (first end-to-end proof)
- Modules: LINEAR, EMBEDDING, LAYER_NORM, causal ATTENTION, GELU MLP.
- Char batcher over `D:\rixgpt\corpus.txt`; RIXG writer.
- **Gate:** val loss within 0.05 of PyTorch's 0.78, and simple_rixgpt runs the Eiffel-trained model.bin.

### M3: GPU
- **M3a:** fill the simple_vulkan gaps, with its own tests.
- **M3b:** GLSL kernels per op, each with a CPU parity test.
- **Gate:** RixGPT trains on the GPU to the same loss; wall clock compared with PyTorch's ~3 min.

### M4: Needle 3 forward pass in simple_needle
- .cact reader (dequantize) and tokenizer from the .cact blob.
- The unusual blocks composed from primitives: engram, Hadamard MLP, mHC + Sinkhorn, conv taps, QK-norm, RoPE, GQA, sliding window, heads.
- **Gate:** parity with libneedle.dll on the 26 test queries (embeddings, greedy calls).

### M5: Needle 3 LoRA fine-tune
- LoRA on q/k/v/gate/out; a JSONL data generator through simple_ai_client; merge, 4-bit quantize, .cact writer.
- **Gate:** our .cact runs in libneedle and beats the untuned score on held-out Bible-study and desktop sets.

### Later
- Pure-Eiffel Needle inference with grammar-constrained decoding.
- Mixed precision; larger models (a RixQwen-class LoRA); re-base simple_nn.

## Key Features
1. Tensor-level tape autograd: any model composed of primitives trains.
2. Contracts as shape checks: the DbC build catches shape bugs at the op that made them.
3. Eiffel twin per fast kernel: the in-library oracle.
4. Vulkan GPU training on the fleet's own stack.
5. LoRA plus vendor-format export.

## Success Criteria
- The M1–M5 gates above, each with pasted run output.

## Dependencies
| Library | Purpose | simple_* Preferred |
|---------|---------|-------------------|
| base | SPECIAL, MANAGED_POINTER | (kernel) |
| simple_vulkan | GPU buffers and dispatch (gaps filled there) | YES |
| simple_shaderc | GLSL → SPIR-V at run time | YES |
| simple_json | safetensors header, JSONL | YES |
| simple_file | I/O | YES |
| simple_ai_client | Training-data generation (optional) | YES |
| simple_testing | Tests | YES |

## Next Steps
1. **Larry's gate on D-004:** what "pure Eiffel" means (strict Eiffel loops, or Eiffel with C-inline hot loops and GLSL GPU kernels, each with an Eiffel twin).
2. Run spikes S-2 and S-3.
3. `/eiffel.debate` on **D-002** (tensor memory placement), with S-3 as the executed evidence.
4. `/eiffel.spec`, then `/eiffel.intent`, then Phase 1 contracts for M1.

## Open Questions
- Do needle-rs's committed fixtures include f32 Needle 3 intermediates usable as an oracle (D-006b)? If yes, no Python is ever needed.
- Does the 5070 Ti's Vulkan driver expose cooperative-matrix (tensor-core) extensions? That would be a later speed-up.
- Factor the unigram Viterbi out of simple_onnx, or keep a copy in simple_needle?
