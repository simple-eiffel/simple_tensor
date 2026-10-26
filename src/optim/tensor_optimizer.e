note
	description: "[
		Updates parameters in place from a GRAD_STORE. The one place tensor
		data changes in place (spec 06): through the backend's update kernels.
	]"
	author: "Larry Rix"

deferred class
	TENSOR_OPTIMIZER

feature -- Access

	parameters: ARRAYED_LIST [TENSOR]
			-- The leaves this optimizer updates.

	schedule: LR_SCHEDULE

	step_count: INTEGER
			-- Steps taken.

	parameters_model: MML_SEQUENCE [TENSOR]
			-- Model of `parameters'.
		do
			create Result.from_iterable (parameters)
		ensure
			counted: Result.count = parameters.count
		end

	current_rate: REAL_64
			-- The rate the next `step' will use.
		do
			Result := schedule.rate (step_count)
		end

feature -- Basic operations

	step (a_grads: GRAD_STORE)
			-- Update every parameter that has a gradient in `a_grads'; parameters
			-- without one are left unchanged (as torch does; checked by test).
		require
			finite: not a_grads.has_non_finite
			shaped: a_grads.all_shapes_match
			live: across parameters as p all not p.is_released end
		deferred
		ensure
			counted: step_count = old step_count + 1
			same_parameters: parameters_model |=| old parameters_model
			same_schedule: schedule = old schedule
		end

feature -- Status report

	distinct_ids (a_parameters: ARRAYED_LIST [TENSOR]): BOOLEAN
			-- Does each tensor appear at most once (by id)? A tied weight is
			-- passed once, not twice (verdict R1-14).
		local
			l_seen: HASH_TABLE [BOOLEAN, INTEGER_64]
		do
			create l_seen.make (a_parameters.count)
			Result := True
			across a_parameters as p until not Result loop
				Result := not l_seen.has (p.id)
				l_seen.put (True, p.id)
			end
		end

feature {NONE} -- Implementation

	update_all (a_grads: GRAD_STORE)
			-- Advance `step_count' and update every parameter that has a gradient,
			-- at the rate the schedule gives the step BEFORE this one (rixgpt.py
			-- sets lr_at (it) before update `it', so update 0 uses rate (0)).
		local
			l_rate: REAL_64
		do
			l_rate := current_rate
			step_count := step_count + 1
			across parameters as p loop
				if a_grads.has (p) then
					update_one (p, a_grads.gradient (p), l_rate)
				end
			end
		end

	update_one (a_parameter, a_gradient: TENSOR; a_rate: REAL_64)
			-- Update `a_parameter' in place from `a_gradient' at `a_rate'.
		deferred
		end

invariant
	non_negative_steps: step_count >= 0

end
