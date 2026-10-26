# Task Completeness Review: simple_tensor

Review `.eiffel-workflow/tasks.md` against the contracts in `src/**/*.e` and the ruled findings in `evidence/phase2-claude-response.md`.

## Check for
- Features with Phase-4 stubs (`check implemented_in_phase_4: False then end` or `-- Phase 4`) not covered by any task.
- Ruled findings ("land with their class") not attached to a task.
- Missing dependencies between tasks (gap fills G-2/G-3/G-5/G-6).
- Tasks too large (should be split) or too small (should be combined).

## Output
List any gaps or issues found.
