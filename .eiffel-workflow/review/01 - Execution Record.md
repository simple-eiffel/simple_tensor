# Execution Record: simple_tensor Phase 2 (the external opponent)

**Harness:** Windows 11 Pro, EiffelStudio 25.02 Standard (MSVC 14.44), RTX 5070 Ti machine.
**Tool:** `/d/prod/ec.sh` modes only; counts read from the per-run log (`/tmp/ec_output.*.txt`), not the summary line.
**Run:** 2026-10-25, main thread. All items P1-raw.

## E-1 Compile gates (fresh)
```
### ec.sh check -target simple_tensor            (library target: root all_classes)
✓ Syntax and type check passed
log error/warning lines: 0
### ec.sh check -target simple_tensor_tests
✓ Syntax and type check passed
log error/warning lines: 0
### ec.sh test -target simple_tensor_tests       (finalize, DbC kept, C compile)
✓ Built: EIFGENs/simple_tensor_tests/F_code/simple_tensor.exe
log error/warning lines: 0     (pattern includes ^Error code, Syntax error, ^Warning, error C####, LNK####)
```
Phase 1 history (evidence/phase1-compile.txt): four defects were fixed before green.
- Kind constants clashed case-insensitively with the queries `sum`/`sigmoid`.
- `reference:` was used as an assertion tag (it's a keyword).
- A loop variable shadowed a local (VOIT(2)).
- **VAPE:** matmul geometry was used in a public precondition but selectively exported. This was visible only on the library target; the tests target compiles just what TEST_APP reaches.

## E-2 Test run
```
./EIFGENs/simple_tensor_tests/F_code/simple_tensor.exe
Results: 15 passed, 0 failed     exit=0
```
**Caveat (P1):** 13 of the 15 tests have empty Phase-1 bodies. Only `test_shape_basics` (TENSOR_SHAPE.make/rank/count/last, postconditions and invariant evaluated) and `test_scoop_compatibility` (creates SIMPLE_TENSOR on CPU and REFERENCE, TENSOR_SHAPE, GRAD_STORE, WARMUP_COSINE_SCHEDULE, TENSOR_RANDOM, SAFETENSORS_FILE under `concurrency use="scoop"`) exercise code. No contract violation was tripped.

## E-3 Inventory (raw counts)
```
src/*.e classes: 43     testing/*.e: 5
attached-result stubs "check implemented_in_phase_4: False then end": 50
files containing "C blocking": gc_point.e (the only actual blocking external so far);
   cpu_backend.e, cpu_storage.e, tensor_dtype_codec.e mention it in notes only (kernels are Phase-4 stubs)
```
**Consequence for CL-5:** the `C blocking` claim is a design obligation, not yet code. No CPU kernel exists.

## E-4 Spike R-1: is `once` per SCOOP processor? (claim CL-2)
`review/spikes/once_per_processor/` (R1_APP, R1_WORKER; concurrency use="scoop"). Two separate R1_WORKER objects each bump the **same** `once` routine `R1_WORKER.counter` 1,000 times.
```
root counter: 10
worker A counter: 1000
worker B counter: 1000
```
If `once` were process-wide, one worker would have seen 2,000. **Result: `once` is per processor.** TENSOR's id counter and CPU_STORAGE's STORAGE_BUDGET are per processor, as the contracts assume.

## E-5 Prior executed evidence (M0 spikes, `evidence/pre-phase-research.txt`), all P1
- **S-1:** plain-Eiffel matmul 4.03 GFLOP/s (lean) vs C inline 26.6 GFLOP/s on the same memory.
- **S-3:**
  - worst root allocation, ONE unmarked C call on SPECIAL: 2,744 ms;
  - ONE `C blocking` call on C heap: 7 ms;
  - 250 short unmarked calls: 2,746 ms;
  - plain-Eiffel loop: 3,129 ms;
  - plain-Eiffel loop + blocking no-op every 4 rows: 6 ms.
- **S-4:** 2,000 dead 4 MB MANAGED_POINTERs: 8,040 MB peak (GC alone); 21 MB (full_collect every 50); 40 MB (`resize (1)` release).
- **S-2:** simple_vulkan as-is on the 5070 Ti: tiled GLSL matmul 431 GFLOP/s (256×768×768), 332 GFLOP/s (2048³).

## E-5a CORRECTION after the RED seats (declared, not a new criterion): S-4 provenance
RED-3 (R3-04) found that E-5 cited the S-4 figures to `pre-phase-research.txt`, where they are absent. Before this they were recorded only in intent-v2.md and phase0-intent.txt, with no raw log kept (a pipeline error by the record's author). The spike was re-run on 2026-10-25 and the raw log saved at `spikes/memory_pressure/s4_run_2026-10-25.log`:
```
start private MB: 16
A. GC decides            : peak private MB 8040, after collect 21, 23 ms
B. full_collect every 50 : peak private MB 21, after collect 21, 22 ms
C. explicit release      : peak private MB 40, after collect 17, 51 ms
```
The figures reproduce exactly. They are now P1 with a log on disk.

## E-5b POST-VERDICT measurement (declared; answers the adjudicator's A-01 with a run instead of a derivation)
`s4_app.e` was changed to sample `PagefileUsage` every iteration, BEFORE the collect. Rebuilt lean and re-run; log at `spikes/memory_pressure/s4_run_2026-10-25_per_iteration.log`:
```
A. GC decides            : peak private MB 8040, after collect 20, 29 ms
B. full_collect every 50 : peak private MB 221, after collect 20, 29 ms
C. explicit release      : peak private MB 40, after collect 16, 69 ms
```
A-01 is confirmed by measurement: arm B's true peak is **221 MB** (adjudicator's derivation: ~212). The old "21 MB" was a post-collect floor.

## E-6 Dependency facts read raw (P1)
- **simple_mml** MML_SET (src/mml_set.e): `has alias "[]"`, `count alias "#"`, `is_model_equal alias "|=|"`, `extended alias "&"`, `removed alias "/"`, `union alias "+"`. MML_SEQUENCE: `extended alias "&"`, `count alias "#"`, `|=|`.
- **simple_math** SIMPLE_MATH: `pi` (115), `cos` (135), `exp` (205), `log` (211), `sqrt` (241). **No `tanh`, no `erf`.**
- **simple_randomizer** SIMPLE_RANDOMIZER: `make_with_seed (INTEGER)` (45), `random_real: REAL_64` (74), `random_integer_in_range (INTEGER_INTERVAL)` (64). **No Gaussian.** Wraps ISE RANDOM (`random_sequence: RANDOM`, 456).
- **simple_testing** TEST_SET_BASE: `assert`, `refute`, `assert_true`, `assert_equal` and others.
- **ISE C compile:** EiffelStudio's generated C is compiled with `-D_WIN32_WINNT=0x0500` (seen in the S-4 build log). Inline C needing newer Windows declarations fails, as PROCESS_MEMORY_COUNTERS_EX did.
