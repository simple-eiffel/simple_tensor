# Review Register: simple_tensor Phase 2

Date: 2026-10-25.
**Mode:** code-lite plus lens-split RED. Three RED seats by lens (RED-1 completeness, RED-2 design/seams, RED-3 provenance/dependencies), then BLUE, then a fresh adjudicator. Budget previewed to Larry and approved ("run it").

**Rules:**
- Seats read the General Research ROE **v1.1** directly. `eiffel-debate/ROE-CARD.md` digests v1.0 and declares itself void when the rulebook's version line differs.
- Nobody reads the theological ROE.

## Scope
- ECF: `D:\prod\simple_tensor\simple_tensor.ecf`, targets `simple_tensor` (library, all_classes) and `simple_tensor_tests`.
- Clusters: `src/` (46 classes: root, storage/, backend/, ops/, optim/, data/, io/) and `testing/` (5 classes).
- Inputs: `.eiffel-workflow/intent-v2.md` (APPROVED), `spec/01–08`, `approach.md`, `research/04-DECISIONS.md`, `evidence/*.txt`.

## Criteria (fixed before any finding)
- **AC-1..AC-11:** the acceptance criteria of `intent-v2.md` (in order: gradient checks, twin parity, AdamW trace, schedule, safetensors, determinism, GC probe, memory, CPU throughput, non-finite detection, M2 gate).
- **Spec invariants:** DR-001..DR-010 (spec/02).
- **OOSC2 / CQS / void safety / SCOOP** compatibility.
- **Fleet rules:**
  - (F1) invariants O(1);
  - (F2) long C externals `C blocking` on C-heap memory only;
  - (F3) long Eiffel loops pass a GC point (S-3);
  - (F4) simple_* first, gaps filled in the owning library;
  - (F5) zero warnings;
  - (F6) C helper prefix `stensor_`;
  - (F7) MANAGED_POINTER churn needs a budget or release (S-4).

## Claims under test
- **CL-1** Every TENSOR op's postcondition fixes its result shape and its tracking status exactly.
- **CL-2** No global mutable state besides per-thread `once` objects (id counter, storage budget). `once` is per thread, hence per SCOOP processor.
- **CL-3** `backward` returns a store holding only grad-requiring leaves, each shaped like its tensor.
- **CL-4** Backward builds no graph: every op's `backward` is called only with an untracked gradient.
- **CL-5** Every kernel that can run long is `C blocking` and touches only C-heap memory (the CPU_BACKEND contract, to be honored in Phase 4).
- **CL-6** Memory churn is bounded by STORAGE_BUDGET + release.
- **CL-7** CPU_BACKEND may inherit REFERENCE_BACKEND (IS-A holds; the twin test compares exactly the redefined kernels).
- **CL-8** One TENSOR_OP_UNARY for 11 ops is correct: backward = g · f'(a) via `unary_derivative`, with the saved output y.
- **CL-9** The schedule contract equals rixgpt.py's `lr_at`; the AdamW contract equals torch.optim.AdamW semantics.
- **CL-10** Each intent acceptance criterion maps to a contract or a test.
- **CL-11** Dependencies provide what the contracts assume (simple_mml operators, simple_randomizer seeding, simple_math functions, simple_json, MANAGED_POINTER.resize).

## Re-specification after the verdict (declared; Larry adopted the verdict 2026-10-25)
The criteria are unchanged. The claims are reworded to what the contracts now say:
- **CL-1** (was DISPROVEN): every TENSOR op fixes its result rank, its tracking status and the dims its contract names. Exact batch dims for matmul/transposed land with R1-10 in Phase 4.
- **CL-4** (was DISPROVEN): backward builds no graph *provided* every formula computes on `i.detached` (tensor_op.e doctrine). A violation fails loudly at `GRAD_STORE.accumulate`'s `untracked_grad`.
- **CL-5** (UNTESTED): every kernel that can run long is `C blocking` on C-heap memory (fast kernels); every plain-Eiffel kernel passes a GC point (cold and reference kernels). It's a Phase 4 obligation, checked by `test_gc_probe_under_10ms`.
- **CL-7** (UNSUPPORTED): IS-A withdrawn. CPU_BACKEND reuses REFERENCE_BACKEND for implementation; twin scope comes from a kernel roster (R2-12). HOST_BACKEND is preferred.
- **CL-10** (was DISPROVEN): every acceptance criterion has a home. AC-7 → `test_gc_probe_under_10ms`; AC-9 → `test_cpu_matmul_throughput` plus a lean benchmark target (Phase 3 task); AC-8's measuring method fixed (intent-v2 Q7 annotation).

## S6 declaration
The contracts' author (this session) designed the spec, wrote the contracts and ran the spikes. It leans toward its own design.
