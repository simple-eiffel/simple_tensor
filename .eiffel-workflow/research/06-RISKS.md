# RISKS: simple_tensor

## Risk Register
| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| RISK-001 | Plain-Eiffel kernels too slow for CPU training | **MEASURED: HIGH** (4.0 vs 26.6 GFLOP/s) | HIGH | D-004 option 2: C-inline hot loops with Eiffel twins; GPU path |
| RISK-002 | simple_vulkan not ready for training-scale compute | HIGH (survey: host-visible only, wait per dispatch, ~4 pipelines, no `blocking`, descriptor leak) | HIGH | Fill the gaps in simple_vulkan first (milestone M3a), with its own tests; Kompute's design as the blueprint |
| RISK-003 | GC stalls during long kernels or GPU waits | MED | HIGH (a frozen GUI, as in simple_chat) | D-002 C-heap tensors + `blocking`; the simple_process-style allocation-latency probe as a test |
| RISK-004 | Wrong gradients in an unusual op (Sinkhorn iterations, hashed gather, Kronecker MLP) | MED | HIGH | Gradient check per primitive; composites checked as wholes; f32 oracle comparisons of intermediates (needle-rs lesson) |
| RISK-005 | Cactus changes the architecture (three generations May–Sept 2026) | HIGH | MED | Needle is a *composition* in simple_needle, not engine code; a new generation means a new composition, not new backward code |
| RISK-006 | .cact writer subtly wrong (2/4-bit packing, codebooks) | MED | MED | Round trip: read the shipped needle3.cact → write → byte-compare; then libneedle loads our output |
| RISK-007 | Training-data quality limits the fine-tune more than the trainer does | HIGH | MED | Generator through Claude with schema checks; held-out sets written by hand (as in the 16 + 10 sets) |
| RISK-008 | The 2-bit shipped artifact can't be reproduced (NYU review: Cactus Platform only) | CERTAIN | LOW | Export 4-bit like the official LoRA path; accept a size of ~2× |
| RISK-009 | Memory: Needle LoRA activations at seq 1,024 × 20 layers × 4 lanes | MED | MED | Shorter sequences; activation checkpointing (SHOULD); measure in M4 |
| RISK-010 | Concurrency of compiles while other sessions build simple_widgets | MED | LOW | ec.sh lock; stagger |

## Technical Risks

### RISK-001: CPU kernel speed
- **Description:** the lean plain-Eiffel matmul reached 4.03 GFLOP/s; the same loop in C inline on the same memory reached 26.6 GFLOP/s.
- **Indicators:** step time on RixGPT.
- **Mitigation:** C inline for matmul, attention and softmax; SCOOP-parallel row blocks; GPU.
- **Contingency:** if Larry rules strict-Eiffel (D-004 option 1), training is CPU-only at ~4 GFLOP/s. That's fine for RixGPT-scale (rough estimate: hours), but slow for Needle (a day per run).

### RISK-002: simple_vulkan gaps
- **Indicators:** a dispatch-only benchmark (S-2) far below expectation.
- **Mitigation:** device-local memory + staging; command-buffer batching; fences; descriptor pool. Add a dispatch test, since it has none today (7 tests, none dispatches).

## Ecosystem Risks
- simple_vulkan changes ripple to its dependents. Check them (rule: check downstream dependents after a library change; clean rebuild of clients).
- simple_onnx's tokenizer may need factoring out to share Viterbi. Alternatively, copy the algorithm into simple_needle with a citation (decide in spec).

## Resource Risks
- Larry's token budget: debate cycles at budget settings (Sonnet seats, Opus adjudicator); spikes before seats.
- One-compile-at-a-time: long finalize builds of a growing library. Keep test targets lean.
