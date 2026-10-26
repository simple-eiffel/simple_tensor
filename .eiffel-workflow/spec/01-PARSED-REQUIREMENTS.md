# PARSED REQUIREMENTS: simple_tensor

Source: `research/01–07` (2026-10-25), with the D-002/D-004 updates and the M0 spike evidence (`evidence/pre-phase-research.txt`).

## Problem Summary
Eiffel can run neural models but cannot train them. simple_tensor gives Eiffel:
- an N-dimensional REAL_32 tensor;
- tensor-level reverse-mode autograd;
- losses, optimizers and modules (including LoRA);
- checkpoint I/O;
- a token batcher;
- three interchangeable backends: a plain-Eiffel reference (the oracle twin), a fast CPU backend (C inline, `blocking`), and later a Vulkan GPU backend.

The first end-to-end proof is training RixGPT (M2). Needle 3 lives in simple_needle on top of this library (M4–M5).

**This spec covers M1–M3** (core, RixGPT modules, backend seam for GPU). The Needle architecture gets its own spec in simple_needle.

## Scope
### In Scope
- TENSOR (REAL_32, C-heap storage, contiguous row-major), INDEX_TENSOR (INTEGER_32 indices/targets).
- Numpy-style broadcasting for elementwise ops; batched matmul.
- ~30 primitive ops with forward + backward. Fused cross-entropy and fused causal attention.
- Graph-in-tensor autograd (each result tensor remembers its op), `backward` → GRAD_STORE.
- Modules: LINEAR, EMBEDDING, LAYER_NORM, RMS_NORM, CAUSAL_SELF_ATTENTION, MLP, LORA_LINEAR; parameter registry; freeze.
- Optimizers: SGD, ADAMW; schedules (constant, warmup + cosine); global-norm clipping; gradient accumulation.
- Checkpoints: safetensors read/write (F32/F16/BF16 → REAL_32); RIXG writer for simple_rixgpt.
- Data: TOKEN_BATCHER (random windows), deterministic seeds.
- Backends: REFERENCE (plain Eiffel + GC points), CPU (C inline `blocking`), the VULKAN seam (M3).

### Out of Scope
- Multi-GPU, CUDA, Python interop (research/01).
- The Needle architecture, .cact, tokenizer (simple_needle spec).
- Lazy graphs / kernel fusion beyond the two hand-fused ops (research/02 "LATER").

## Functional Requirements
| ID | Requirement | Priority | Source | Acceptance |
|----|-------------|----------|--------|------------|
| FR-001 | Tensor + shape + reshape view; copy ops for transpose/permute | MUST | research/03 | Contract tests |
| FR-002 | ~30 primitive ops with backward | MUST | research/03 | Gradient check per op, relative error ≤ 1e-3 (REAL_32), against central differences |
| FR-003 | Autograd: backward from a scalar, `+=` accumulation, requires_grad per leaf, no-grad by construction | MUST | research/03 | Tied-weight test; frozen leaf has no gradient |
| FR-004 | Cross-entropy (logits, INDEX_TENSOR targets, ignore index), MSE | MUST | research/03 | Closed-form fixtures |
| FR-005 | SGD, AdamW, warmup+cosine, clipping, accumulation | MUST | research/03 | AdamW 3-step hand trace |
| FR-006 | Modules incl. causal attention | MUST | research/03 | Module gradient checks |
| FR-007 | LoRA on LINEAR + merge | MUST | research/03 | merged(x) = base(x) + lora(x) within 1e-5 |
| FR-008 | safetensors read/write; RIXG write | MUST | research/03 | Byte-exact round trip; simple_rixgpt loads RIXG |
| FR-009 | Token batcher, seeded | MUST | research/03 | Same seed → same batches |
| FR-010 | CPU backend | MUST | research/03 | Twin parity per op (CPU = REFERENCE within 1e-5 relative) |
| FR-011 | GPU backend through simple_vulkan | MUST (M3) | research/03 | Twin parity per op (GPU = CPU within 1e-4) |
| FR-012 | Training-loop helper (eval, log, checkpoint, resume) | SHOULD | research/03 | Resume reproduces the loss curve |
| FR-016 | Mixed precision | SHOULD (later) | research/03 | n/a for M1 |

## Non-Functional Requirements
| ID | Requirement | Category | Measure | Target |
|----|-------------|----------|---------|--------|
| NFR-001 | CPU matmul | PERFORMANCE | GFLOP/s 256×768×768, lean | ≥ 20 (S-1: 26.6) |
| NFR-002 | GPU matmul | PERFORMANCE | GFLOP/s, same shape | ≥ 400 as-is (S-2: 431); ≥ 2,000 after simple_vulkan gap fills |
| NFR-003 | RixGPT training | PERFORMANCE | Wall clock to val ≈ 0.78 | GPU same order as PyTorch (~3 min); CPU < 1 h |
| NFR-004 | GC law | RELIABILITY | Worst root allocation during a training step on another processor | < 10 ms (S-3 shape: 6–7 ms) |
| NFR-005 | Determinism | RELIABILITY | Same seed → same losses | Bit-identical on CPU/REFERENCE |
| NFR-006 | Memory | RESOURCE | RixGPT training peak RAM | < 4 GB |
| NFR-007 | No network | SECURITY | dumpbin imports | None |

## Constraints (simple_* First)
| ID | Constraint | Type |
|----|------------|------|
| C-001 | simple_* over ISE/Gobo; gaps filled in the owning library | ECOSYSTEM |
| C-002 | SCOOP-compatible | TECHNICAL |
| C-003 | Void-safe | TECHNICAL |
| C-004 | Long C kernels `C blocking` on C-heap memory only; long Eiffel loops pass a GC point | TECHNICAL (S-3 law) |
| C-005 | Invariants O(1) (Larry 2026-09-11): no walks over tensor data | TECHNICAL |
| C-006 | One compile at a time; benchmark lean builds only | PROCESS |
| C-007 | "Pure Eiffel" = D-004 option 2 (Larry, 2026-10-25) | POLICY |

## Decisions Already Made
| ID | Decision | Rationale | From |
|----|----------|-----------|------|
| D-001 | New simple_tensor; Needle in simple_needle | simple_nn has no dependents; clean ownership | research/04 DECIDED |
| D-002 | Storage = C heap (MANAGED_POINTER); kernels `C blocking` | S-3: 7 ms vs 2,744 ms | research/04 DECIDED (spike) |
| D-003 | Tensor-level dynamic graph + separate GRAD_STORE; autograd over a deferred backend | Any composition trains | research/04 PREFERRED |
| D-004 | Eiffel owns all; C inline hot loops; own GLSL; Eiffel twin per kernel | Larry's gate | research/04 GATED |
| D-005 | REAL_32 compute; REAL_64 reductions; bf16/fp16 I/O only | Needle/RixGPT sizes | research/04 DECIDED |
| D-006 | Oracles per stage (gradient check, twin, PyTorch 0.78, libneedle) | External confrontation | research/04 PREFERRED |
| D-007 | GPU via simple_vulkan + gap fills there | Fleet stack; S-2 works as-is | research/04 DECIDED |
| D-008 | RixGPT first | Known recipe and result | research/04 DECIDED |

## Innovations to Implement
| ID | Innovation | Design Impact |
|----|------------|---------------|
| I-001 | Contracts as shape system | Every op's precondition states its shape rule; postcondition states the result shape |
| I-002 | Eiffel twin per kernel | TENSOR_BACKEND deferred; REFERENCE_BACKEND is the oracle; a parity test harness |
| I-003 | Readable autograd contracts | `backward` postcondition: every grad-requiring leaf reached has a gradient of its own shape |
| I-004 | One library from char-GPT to Needle | Ops are primitives; models are compositions |
| I-006 | GC-safe training | C-heap storage; blocking kernels; GC_POINT in Eiffel loops |

## Risks to Address in Design
| ID | Risk | Mitigation Strategy |
|----|------|---------------------|
| RISK-001 | Plain-Eiffel speed | CPU_BACKEND C inline; REFERENCE only for oracles |
| RISK-002 | simple_vulkan gaps | VULKAN_BACKEND behind the same deferred interface; gap fills in simple_vulkan (M3a) |
| RISK-003 | GC stalls | C-heap storage; blocking; GC_POINT |
| RISK-004 | Wrong gradients | Gradient-check harness over every op; twin parity |
| RISK-009 | Activation memory | Saved values only where backward needs them; `no_grad` = untracked inputs |

## Use Cases
### UC-001: Train a model from scratch (RixGPT, M2)
**Actor:** an Eiffel app.
**Precondition:** a corpus encoded to an INTEGER_32 token sequence.
**Main Flow:**
1. Build the model from modules (EMBEDDING, blocks of LAYER_NORM + CAUSAL_SELF_ATTENTION + MLP, final LAYER_NORM, tied head).
2. Create ADAMW over `model.parameters` with a WARMUP_COSINE schedule.
3. Loop:
   - `batch := batcher.next_batch`;
   - `logits := model.forward (batch.inputs)`;
   - `loss := logits.cross_entropy (batch.targets)`;
   - `grads := loss.backward`;
   - `grads.clip_global_norm (1.0)`;
   - `optimizer.step (grads)`.
4. Every N steps, evaluate on the validation set (no gradients: inputs untracked).
5. Save as safetensors and RIXG.

**Postcondition:** val loss logged; the checkpoint loads back to identical tensors.

### UC-002: Fine-tune with LoRA (M5 shape)
**Actor:** simple_needle.
**Main Flow:**
1. Load base weights (safetensors) into the model; `model.freeze`.
2. Wrap the target LINEARs in LORA_LINEAR (rank, alpha); only the A and B leaves require gradients.
3. Train as in UC-001.
4. `lora.merge` into the base; export.

**Postcondition:** the base tensors are unchanged until merge; after merge the LoRA leaves are discarded.

### UC-003: Check a kernel against its twin
**Actor:** the test suite.
**Main Flow:** same random inputs (seeded) → op on REFERENCE_BACKEND and on CPU_BACKEND (or VULKAN) → compare elementwise; then a gradient check on REFERENCE.
**Postcondition:** a PASS/FAIL with the worst relative error.

### UC-004: Run training beside a GUI
**Actor:** an app with a GUI processor and a separate trainer processor.
**Postcondition:** the GUI's worst allocation stays < 10 ms (NFR-004) through a training step.
