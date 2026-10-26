# DESIGN VALIDATION: simple_tensor

## OOSC2 Compliance
| Principle | Status | Evidence |
|-----------|--------|----------|
| Single Responsibility | ✓ | Shape, storage, tensor, op, store, backend, module, optimizer, schedule, batcher and file are each one job (04 inventory) |
| Open/Closed | ✓ | A new op is a new TENSOR_OP_* plus kernels; a new backend is a new TENSOR_BACKEND heir; a new model is a composition (I-004). Nothing existing changes |
| Liskov Substitution | ✓ | Backends share kernel contracts and twin parity; NN_EMBEDDING kept out of NN_LAYER so no `forward` precondition is strengthened (04) |
| Interface Segregation | ✓ | NN_MODULE (parameters) split from NN_LAYER (forward) |
| Dependency Inversion | ✓ | Ops and modules depend on deferred TENSOR_BACKEND / TENSOR_STORAGE, never on CPU/VULKAN |

## Eiffel Excellence
| Criterion | Status | Evidence |
|-----------|--------|----------|
| Command-Query Separation | ✓ | Ops and `backward` are queries returning new objects; TOKEN_BATCHER split into `forth`/`batch` (05); kernels are commands writing `out` |
| Uniform Access | ✓ | `count`, `is_tracked`, `is_scalar` computed or attribute without client impact |
| Design by Contract | ✓ | Shape rules as preconditions (I-001); tracking and shape postconditions; O(1) invariants; MML frames on GRAD_STORE, modules, optimizer state |
| Genericity | ✓ (deliberately unused) | REAL_32 fixed by D-005; justified in 04 |
| Inheritance | ✓ | IS-A only; LoRA by composition |
| Information Hiding | ✓ | Storage mutation selectively exported to {TENSOR_BACKEND, TENSOR, TENSOR_OPTIMIZER, SAFETENSORS_FILE, NN_MODULE} (06) |

## Practical Quality
| Criterion | Status | Evidence |
|-----------|--------|----------|
| Void-safe | ✓ | `origin: detachable TENSOR_OP`; bias detachable |
| SCOOP-compatible | ✓ | No global state (per-thread id counter); C-heap storage; `C blocking` kernels; GC_POINT in Eiffel loops; a GC_PROBE test target (UC-004) |
| simple_* first | ✓ | simple_json, simple_file, simple_randomizer (+ G-2 gap fill), simple_mml, simple_testing; simple_vulkan + simple_shaderc in M3 with gap fills there |
| MML postconditions | ✓ | GRAD_STORE `ids_model`, modules `parameters_model`, shape `dims_model` |
| Testable | ✓ | GRADIENT_CHECKER (every op), TWIN_CHECKER (every kernel, every backend), AdamW hand trace, schedule against rixgpt.py's formula, safetensors round trip, PyTorch curve (iter 500 val 1.201) |

## Requirements Traceability
| Requirement | Addressed By | Status |
|-------------|--------------|--------|
| FR-001 | TENSOR_SHAPE, TENSOR.reshape (view), TENSOR_OP_PERMUTE (copy) | ✓ |
| FR-002 | 33 TENSOR_OP_* + 05 op-semantics table + GRADIENT_CHECKER | ✓ |
| FR-003 | TENSOR.backward (07 algorithm), GRAD_STORE.accumulate (`+=`), NN_MODULE.freeze | ✓ |
| FR-004 | TENSOR_OP_CROSS_ENTROPY (ignore index), TENSOR_OP_MSE | ✓ |
| FR-005 | SGD_OPTIMIZER, ADAMW_OPTIMIZER, WARMUP_COSINE_SCHEDULE, GRAD_STORE.clip_global_norm / merge | ✓ |
| FR-006 | NN_* (M2) | ✓ (designed) |
| FR-007 | NN_LORA_LINEAR.make / merge | ✓ |
| FR-008 | SAFETENSORS_FILE; RIXG writer moved to simple_rixgpt (03) | ✓ (modified) |
| FR-009 | TOKEN_BATCHER (seeded) | ✓ |
| FR-010 | CPU_BACKEND | ✓ |
| FR-011 | VULKAN_BACKEND seam (M3) | ✓ (seam) |
| FR-012 | TENSOR_TRAINER (SHOULD) | ✓ (deferred to M2) |
| FR-NEW-001…006 | to_backend, GC_POINT + GC_PROBE, detached, G-2, named_parameters, matmul flags | ✓ |
| NFR-001 | CPU_BACKEND C inline (S-1 26.6) | ✓ |
| NFR-002 | VULKAN_BACKEND (S-2 431 as-is) | ✓ (M3) |
| NFR-003 | Modified (03): CPU gate = iter-500 curve point with threading; full 0.782 = M3 GPU gate | ✓ (modified) |
| NFR-004 | D-002 + GC_POINT + GC_PROBE target | ✓ |
| NFR-005 | Seeded TENSOR_RANDOM / TOKEN_BATCHER; deterministic kernels (no atomics on CPU) | ✓ |
| NFR-006 | Gradients of intermediates discarded during the sweep | ✓ |
| NFR-007 | No network library in dependencies | ✓ |

## Risk Mitigations Implemented
| Risk | Mitigation in Design |
|------|---------------------|
| RISK-001 | CPU_BACKEND in C inline; REFERENCE only for oracles; threaded matmul MUST for M2 |
| RISK-002 | VULKAN_BACKEND behind TENSOR_BACKEND; simple_vulkan gaps filled in its own repo first (M3a) |
| RISK-003 | C-heap storage, `C blocking` kernels, GC_POINT, GC_PROBE test |
| RISK-004 | GRADIENT_CHECKER on every op; TWIN_CHECKER on every kernel; composed-vs-fused oracle for later fused ops |
| RISK-009 | Intermediate gradients discarded; saved values only where the formula needs them |

## Open Issues
1. **`reduced_to (shape)`** (used by every broadcast backward) must itself be differentiable when called on tracked tensors. Resolution: make it the **34th primitive**, TENSOR_OP_REDUCE_TO (forward = sum over broadcast axes; backward = broadcast g back via add with zeros). Fold into Phase 1.
2. **CPU threading model** for M2: a C-side thread pool inside one `C blocking` call (simplest; no Eiffel objects touched) vs SCOOP workers. Settle by a spike at M2 start (GC-safe by construction for the C pool).
3. **G-2** (simple_randomizer Gaussian) is a separate PR in simple_randomizer, with its downstream dependents checked.
4. PyTorch's RixGPT used bf16 autocast; ours is fp32. Expect a similar, not identical, curve. That's why the M2 gate is ± 0.05.

## Ready for Implementation
- [x] All requirements traced
- [x] All risks mitigated
- [x] All principles satisfied
- [x] Design is complete for M1 (M2/M3 seams fixed; their internals refined at their own phase starts)

**VERDICT:** READY (M1 scope).
