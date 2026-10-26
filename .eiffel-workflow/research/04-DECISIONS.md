# DECISIONS: simple_tensor

Rule (Larry, 2026-10-06): technical forks are settled by evidence. Larry gates values, policy and adoption. Each decision is marked:
- **DECIDED**: evidence is in.
- **PREFERRED**: chosen, with an overturn trigger.
- **CONTESTED**: goes to /eiffel.debate before spec.
- **LARRY'S GATE**: a values question.

### D-001: Where the code lives — DECIDED
**Question:** Extend simple_nn, or create new libraries?
**Options:**
1. **Grow simple_nn into a tensor/autograd library.** Pro: the "fill the owning library" rule. Con: its core is per-layer ARRAY2 [REAL_64] with MSE + SGD (survey: dense_layer.e:260, neural_network.e:299), so everything would be replaced, and simple_nn is not the owner of "tensors".
2. **New simple_tensor (framework) + simple_needle (the Needle model).** Pro: clean ownership; simple_nn has **no dependents** (no .ecf references it), so nothing breaks. Con: one more library.

**Decision:** Option 2.
- simple_tensor owns tensor, autograd, ops, modules, LoRA, optimizers, data, checkpoint I/O and the CPU/GPU backends.
- simple_needle owns the Needle 3 architecture, the .cact reader/writer, the tokenizer, fine-tune recipes and the libneedle binding (kept as the oracle).
- simple_vulkan receives its gap fills.
- simple_rixgpt gets a training app built on simple_tensor.
- simple_nn can be re-based later.

**Reversible:** YES.

### D-002: Where tensor memory lives — DECIDED 2026-10-25 by executed spike S-3 (`spikes/memory_gc`)

**S-3 results** (lean build, SCOOP). A second processor runs ~2.5–3 s of 256×768×768 matmuls while the root measures its worst allocation burst (simple_process probe routine: forced full_collect per burst, growing live set):

| Kernel shape on the other processor | Worst root allocation |
|---|---|
| ONE unmarked C call on SPECIAL | 2,744 ms |
| ONE `C blocking` call on C-heap MANAGED_POINTER | **7 ms** |
| 250 SHORT unmarked C calls (12 ms each) on SPECIAL | 2,746 ms |
| Plain-Eiffel loop, no allocation inside | 3,129 ms |
| Same plain-Eiffel loop + an empty `C blocking` call every 4 rows | **6 ms** (kernel time unchanged: 3,180 vs 3,233 ms) |

Plain-Eiffel loop speed: SPECIAL 4.06 GFLOP/s; MANAGED_POINTER `read_real_32`/`put_real_32` 2.37 GFLOP/s; same results.

**Decision:** tensor storage is C-heap memory (MANAGED_POINTER); every fast kernel is `C blocking`.

**Rationale:**
- It's the only shape that keeps other processors alive, with no option of a stall.
- Chunking unmarked calls does NOT help. A thread that never allocates never reaches a GC point between calls.

**New law for the library (and the fleet):** any long plain-Eiffel loop (the oracle twins, data preparation) passes a GC point, an empty `C blocking` external, at least every few milliseconds.

**Debate:** /eiffel.debate not run. The fork was settled by executed evidence that could have failed: the blocking/C-heap shape could have stalled, and chunking could have worked. That counts as external confrontation (oracle rule, Larry 2026-10-11).

*Original framing, kept for the record:*
**Question:** SPECIAL [REAL_32] on the Eiffel heap, or C-heap blocks (MANAGED_POINTER)?
**Options:**
1. **SPECIAL.** Pro: plain Eiffel loops; `base_address` gives C kernels a pointer. Con: the GC law. A long C kernel on Eiffel memory **cannot** be marked `blocking` (the collector may move it), so a 100 ms matmul would stall every other processor. GPU uploads and downloads have the same problem.
2. **C heap (MANAGED_POINTER).** Pro: every long kernel can be marked `blocking`; zero-copy staging to Vulkan; no GC pressure from multi-MB blocks. Con: plain-Eiffel access goes through `read_real_32`/`put_real_32` calls, and the speed of that is unmeasured.

**Leaning:** Option 2.
**Deciding test (S-3):** lean-build throughput of an Eiffel loop over MANAGED_POINTER against SPECIAL, plus the GC-law probe with a 100 ms kernel on each.

### D-003: Autograd style — PREFERRED
**Options:**
1. **Hand-written backward per layer (llm.c).** Pro: fastest to a first GPT; easy to read. Con: every architecture means new backward code; Needle 3 alone has about 10 unusual blocks; fails "use against anything I like".
2. **Tensor-level dynamic tape (candle, micrograd).** Pro: write the backward once per primitive (~30), and any model composed of primitives trains. Con: tape overhead per op (small at tensor granularity); memory for saved activations.

**Decision:** Option 2.
- The tape records an op, its inputs and saved values.
- Gradients live in a separate GRAD_STORE keyed by tensor id.
- Autograd is layered over a deferred BACKEND, the Burn pattern.
- Hand-fused composites (attention, cross-entropy) are allowed as single ops with their own checked backward.

**Overturn trigger:** the tape overhead exceeds 10% of step time on RixGPT.

### D-004: Kernel language — LARRY'S GATE on the definition; technical part DECIDED by measurement
**Measured 2026-10-25** (`spikes/matmul`, 256×768 × 768×768, REAL_32, one thread):

| Build | Plain Eiffel (best loop) | C inline in the same Eiffel class, same SPECIAL memory |
|---|---|---|
| `ec.sh test` (contracts + hooks kept) | 0.93 GFLOP/s | 25.0 GFLOP/s |
| `ec.sh release`, lean | **4.03 GFLOP/s** | **26.6 GFLOP/s** |

- The test build wraps every `SPECIAL.item` in a call plus RTHOOK (generated C, ma14.c F26_486). The lean build removes most of that, but plain Eiffel stays **6.6× slower** than C inline.
- Rough sizing for a Needle LoRA run (1,000 examples × ~400 tokens × 3 epochs ≈ 1.2M tokens; ~4 × 50M effective FLOP per token, per the NYU review's "arithmetic of a 50M model"; ≈ 2.4e14 FLOP):
  - plain Eiffel ≈ 17 h single thread;
  - C inline ≈ 2.5 h single thread, well under 1 h with threads;
  - GPU ≈ minutes.
- These are estimates, to be replaced by S-2 measurements.

**Options for "pure Eiffel":**
1. **Everything in Eiffel source (strict).** Every loop is Eiffel; the GPU is impossible (shaders are GLSL by nature).
2. **Eiffel owns everything; hot loops are C inline inside Eiffel classes, and GPU kernels are GLSL compiled at run time by simple_shaderc.** This is exactly how the fleet already works (simple_process, simple_cairo, simple_vulkan). No Python, no third-party ML library, no foreign runtime. Every kernel has an Eiffel reference twin that serves as its test oracle.

**Recommendation:** Option 2, with a plain-Eiffel reference implementation of every op kept as the gradient-check and parity oracle (the "slow path" that proves the fast path).
**Gate: APPROVED by Larry, 2026-10-25 ("#2 recommendation").** "Pure Eiffel" means:
- Eiffel owns design, contracts, autograd, optimizers and data.
- Hot loops are C inline inside Eiffel classes.
- GPU kernels are our own GLSL, compiled at run time by simple_shaderc.
- No Python and no third-party ML library at run time.
- Every fast kernel has a plain-Eiffel twin as its oracle.

### D-005: Numeric precision — DECIDED
- REAL_32 storage and compute.
- REAL_64 accumulation in reductions (sum, softmax normalizer, loss) on the CPU.
- A REAL_64 gradient-check build mode.
- bf16/fp16 only as storage formats on load and save (needle3.safetensors is 242,047,978 bytes ≈ 121M params × 2 bytes).
- Mixed-precision training: SHOULD, later.

### D-006: Oracles — PREFERRED (needs Larry's policy word on C-005)
Each stage has an outside opponent that could fail it:

| Stage | Oracle | Python needed? |
|---|---|---|
| Ops | Finite-difference gradient checks; a REAL_64 twin | No |
| RixGPT training | Same corpus, same hyperparameters, PyTorch's recorded val loss 0.78 (D:\rixgpt); our RIXG must run in simple_rixgpt | No (a recorded number) |
| Needle forward pass | (a) Eiffel forward on **dequantized .cact** weights against libneedle.dll on the same weights (embeddings, greedy calls); (b) f32 fixtures, via needle-rs's committed f32 fixtures (MIT) if they cover what we need, else one JAX-on-CPU dump | (a) No; (b) maybe, once, at dev time |
| Needle fine-tune export | libneedle loads our .cact and scores it on held-out sets | No |

### D-007: GPU path — DECIDED (fill simple_vulkan gaps)
- simple_vulkan + simple_shaderc, no CUDA. Larry's working GPU stack is Vulkan already (llama.cpp Vulkan builds; simple_vulkan's 63 FPS SDF).
- Gaps to fill **in simple_vulkan**:
  - device-local buffers plus staging;
  - recording many dispatches into one command buffer and submitting once;
  - fences;
  - `C blocking` on waits;
  - a descriptor pool sized for 50+ pipelines, and freeing descriptor sets.
- Reference shaders: ggml's Vulkan backend (MIT) for the `*_BACK` ops.

### D-008: First end-to-end target — DECIDED
- RixGPT (char-level, 10.9M params) **before** Needle. It has a known recipe and a known result, its inference already runs in Eiffel (simple_rixgpt), and it has only standard ops. It proves tensor + autograd + AdamW + data + checkpoint end to end.
- Needle's unusual blocks come after the foundation is proven.

### D-009: Needle tokenizer — DECIDED
Read pieces and scores from the .cact tokenizer blob (export.py:406 `_tokenizer_blob`, header `<IIIIIBBH`, records `<fBH`+bytes). Encode with unigram Viterbi; the algorithm is already in simple_onnx ONNX_TOKENIZER (onnx_tokenizer.e:278), so factor it out or reuse it. No protobuf needed.

## Contested forks for /eiffel.debate
None remaining (2026-10-25):
- D-002 was settled by executed spike S-3.
- D-004 was gated by Larry (option 2).
- S-2 measured the GPU path: 332–431 GFLOP/s through simple_vulkan as-is.
