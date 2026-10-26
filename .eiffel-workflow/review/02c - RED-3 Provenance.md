# RED-3 Provenance and Dependencies (ROE General v1.1; lens: provenance audit)

Condition: C4/C3 (same model family; independent checks = raw source files and the PyTorch docs page). No compiler run. Levels: P1 = read raw this session.

## Findings (most severe first)

| ID | Sev | Location | Finding | Evidence (P1) | Fix | Log |
|---|---|---|---|---|---|---|
| R3-01 | MEDIUM | `src/tensor_random.e:20-24` `make` | No `require a_seed > 0`, but `SIMPLE_RANDOMIZER.make_with_seed` requires `positive_seed`. A caller passing 0 trips a *supplier* precondition (blame lands on TENSOR_RANDOM). Worse, ISE RANDOM here is Lehmer (multiplier 16807, increment 0, modulus 2^31-1): seed 0 or 2147483647 yields the constant 0 sequence forever. Callers passing a negative seed trip `RANDOM.set_seed require s >= 0`. | `simple_randomizer.e:45-52`; ISE `support/random.e:72-90` (modulus 2_147_483_647, multiplier 16_807, increment 0), `:51-53` (`non_negative`) | Add `seed_valid: a_seed > 0 and a_seed < 2_147_483_647` to TENSOR_RANDOM.make (and SIMPLE_TENSOR facade if it takes a seed) | pipeline error (contract gap) |
| R3-02 | MEDIUM | approach.md:63 "simple_json, simple_file: safetensors"; `safetensors_file.e:1-8` | simple_file offers no usable bulk binary read. `read_bytes` loops `RAW_FILE.read_natural_8` one byte at a time into `ARRAY [NATURAL_8].force`; `read_chunk` re-opens the file on every call, reads byte by byte, and takes INTEGER offsets (2 GB cap). Result is an Eiffel-heap array, not a C-heap block, so it contradicts the S-3/S-4 rationale (F2/F7) and is slow. A gap should be declared (G-2/G-3/G-4 exist; none for file). ISE already has `RAW_FILE.read_to_managed_pointer`. | `simple_file.e:440-470` (read_bytes), `:993-1020` (read_chunk); ISE `kernel/file.e:1528`, `:1117` `put_managed_pointer`; gaps listed in 04-DECISIONS.md:109-125: only G-2, G-3, G-4 | Declare gap G-5: simple_file `read_into (MANAGED_POINTER, offset, n)` / `write_from`, or have SAFETENSORS_FILE use RAW_FILE directly; note the choice in approach.md | pipeline error (unflagged gap) |
| R3-03 | MEDIUM | `STORAGE_BUDGET`, `TENSOR_RANDOM.fill_keep_mask` (design vs recipe) | Lehmer period is 2^31-2 (about 2.1e9 draws), computed from the constants in R3-01. The RixGPT recipe uses dropout 0.2 (rixgpt.py:144). One block at batch 64, block 256, embd 384, 6 heads: attention mask 64*6*256*256 = 25.2M plus two residual masks of 6.3M each = about 38M draws per layer, about 230M per iteration for 6 layers, so the whole period is spent in about 9 iterations; 5000 iterations wrap it about 500 times. Masks repeat exactly; per-parameter-init is fine. Derived arithmetic (flagged as such), constants raw. | ISE `random.e:72-90`; rixgpt.py:143-144 defaults | State this limit in TENSOR_RANDOM notes; options are a better generator (gap in simple_randomizer) or seeding per iteration; do not claim AC-6 (determinism) implies statistical quality | source error (library limit) |
| R3-04 | LOW | `review/01 Execution Record.md` E-5 heading; `approach.md:52` | S-4 figures (8,040 / 21 / 40 MB) are attributed to `evidence/pre-phase-research.txt`; they are NOT in that file (grep: 0 hits). They appear only in `intent-v2.md:59` and `phase0-intent.txt:9`. `spikes/memory_pressure` keeps `s4_app.e` but no raw output log. Spike source does support the arms: full_collect every 50 (`s4_app.e:46`), release via `resize (1)` (`:58-61`). Numbers are therefore the author's own summary (P2 own-notes), not P1. | grep results above | Save the S-4 run log under evidence/; fix the E-5 citation | title error (wrong cite) |
| R3-05 | LOW | `cpu_storage.e:46-52,65-69` | D-002 "never moved" is true only while `count` is unchanged: `MANAGED_POINTER.resize (1)` calls `memory_realloc`, which may relocate the block, and `dispose` later frees whatever `item` is then. Any `address` captured before `release` is invalid after it. `address` has `require live`, so the contract is sound; document that kernels must not hold addresses across a `release`. | ISE `support/managed_pointer.e:1173-1190` (resize), `:1196-1206` (dispose) | Add a note in `address` | source error (assumption wording) |
| R3-06 | INFO | `warmup_cosine_schedule.e:3-5` | Contract is stricter than the oracle: `total > warmup` required; rixgpt guards with `max(iters-warmup,1)`. Python also accepts negative `it` (rate(-1) = negative). Not a defect; a Python-parity test must keep `total > warmup`. | rixgpt.py:96-99 | none | n/a |

## Verified (no further argument needed)

| Claim | Result | Evidence |
|---|---|---|
| `MANAGED_POINTER.make` zero-fills C heap (calloc) | TRUE | `managed_pointer.e:35-50` (`memory_calloc`) |
| `resize (1)` shrinks the C block (realloc to 1 byte); `dispose` frees when not shared | TRUE | `managed_pointer.e:1173-1206`; spike arm 40 MB |
| `MEMORY.full_collect` exists, public, callable as `(create {MEMORY}).full_collect` | TRUE | `kernel/memory.e:509-518` |
| `{PLATFORM}.real_32_bytes`, `integer_32_bytes` = 4 | TRUE | `kernel/platform.e:134,140` |
| `CELL [G]` exists; `HASH_TABLE [TENSOR, INTEGER_64]` usage is ordinary | TRUE | `structures/list/cell.e:11`; grad_store.e:144 |
| MML_SET `&`, `/`, `+`, `#`, `|=|` exist; `&` returns a NEW set (or Current if present), never mutates | TRUE | `mml_set.e:95,103,137-155,157,177` |
| `|=|` is value based (same count and subset) | TRUE | `mml_set.e:103-109` |
| Element equality is `~` (value) for INTEGER_64 and STRING_8; for TENSOR it is default shallow `is_equal` (TENSOR does not redefine it), so unique ids keep elements distinct | TRUE | `mml_model.e:73-80` frozen `model_equals`; grep is_equal: only tensor_shape.e |
| simple_math has no tanh/erf (G-3 confirmed); exp/log/sqrt/cos wrap DOUBLE_MATH REAL_64; `log` requires x>0, `sqrt` x>=0 | TRUE | grep 0 hits in simple_math/src and ISE base; `simple_math.e:135-247` |
| simple_randomizer has no Gaussian (G-2 confirmed); wraps ISE RANDOM; `random_real` = item/dmod in [0,1) | TRUE | `simple_randomizer.e:45-86`; `random.e:147-150` |
| `make_with_seed` is deterministic for a seed (pure LCG state, no clock) | TRUE | `random.e` randomize; `simple_randomizer.e:45-52` |
| RANDOM `once` values (dmod, dmul, dinc) are constants; per-processor once is harmless for SCOOP; one RANDOM per TENSOR_RANDOM | TRUE | `random.e:72-90,~190-210` |
| `nn.GELU()` default in rixgpt (so exact erf GELU, hence gap G-3 needed) | TRUE for the recipe | rixgpt.py:15 `nn.GELU()` (default-arg semantics per PyTorch, not re-fetched) |
| `lr_at` formula equals WARMUP_COSINE_SCHEDULE contract (warmup linear peak*it/max(warmup,1); cosine to 0.1*peak; r clamped at 1) | TRUE | rixgpt.py:96-99; contract `warmup_cosine_schedule.e:44-60`; recipe peak 1e-3, warmup 150, iters 5000 at rixgpt.py:147-148 |
| torch.optim.AdamW decays first: `theta <- theta - gamma*lambda*theta`, then bias-corrected m,v, `theta - gamma*m_hat/(sqrt(v_hat)+eps)` | TRUE (P1-raw TeX of PyTorch 2.14 docs) | https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html (raw html lines 3619-3629); matches `adamw_optimizer.e:3-6` |
| Recipe betas (0.9,0.99), wd 0.1 on all params; eps 1e-8 is torch default (not set in recipe) | TRUE | rixgpt.py:94; docs defaults lr=0.001, betas=(0.9,0.999), eps=1e-08, wd=0.01 (do not default these in the Eiffel class) |
| `LIB_TESTS inherit TEST_SET_BASE` compiles with the ECF as written | TRUE | ECF lines for simple_testing in target simple_tensor_tests; execution record E-1 green (0 errors/warnings) |
| Numbers 26.6 / 4.03 GFLOP/s, 2,744 ms, 7 ms, 3,129 ms, 6 ms, 431/332 GFLOP/s | present, not invented | `evidence/pre-phase-research.txt:17,21-29` |
| simple_json can parse the safetensors header (STRING_32 -> value, objects at top level) once text is extracted | TRUE | `simple_json.e:30-76` |
| Safetensors layout (8-byte LE length + JSON + raw data) | not re-fetched; spec text only (P3 for the format itself) | `safetensors_file.e:3-5`; fetch the HF spec at Phase 4 |

Open note: SAFETENSORS_FILE must tolerate the header key `__metadata__` (a string map, not a tensor); not mentioned in the class.
