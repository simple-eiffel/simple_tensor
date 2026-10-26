note
	description: "[
		AdamW with torch.optim.AdamW semantics (the M2 oracle): decay first,
		param := param · (1 − rate · decay); then bias-corrected Adam,
		param := param − rate · m̂ / (√v̂ + eps). RixGPT recipe: betas
		(0.9, 0.99), eps 1e-8, weight decay 0.1 on all parameters.
	]"
	author: "Larry Rix"

class
	ADAMW_OPTIMIZER

inherit
	TENSOR_OPTIMIZER

create
	make

feature {NONE} -- Initialization

	make (a_parameters: ARRAYED_LIST [TENSOR]; a_schedule: LR_SCHEDULE; a_beta1, a_beta2, a_eps, a_decay: REAL_64)
		require
			leaves_requiring_grad: across a_parameters as p all p.requires_grad end
			distinct: distinct_ids (a_parameters)
			beta1_range: 0.0 <= a_beta1 and a_beta1 < 1.0
			beta2_range: 0.0 <= a_beta2 and a_beta2 < 1.0
			eps_positive: a_eps > 0.0
			decay_non_negative: a_decay >= 0.0
		do
			parameters := a_parameters
			schedule := a_schedule
			beta1 := a_beta1
			beta2 := a_beta2
			eps := a_eps
			decay := a_decay
			create first_moments.make (a_parameters.count)
			create second_moments.make (a_parameters.count)
		ensure
			parameters_set: parameters = a_parameters
			fresh: step_count = 0
			no_state_yet: first_moments.is_empty and second_moments.is_empty
		end

feature -- Access

	beta1, beta2, eps, decay: REAL_64

	state_ids_model: MML_SET [INTEGER_64]
			-- Ids of the parameters that have moment state.
		do
			create Result
			across first_moments.current_keys as k loop
				Result := Result & k
			end
		end

feature -- Basic operations

	step (a_grads: GRAD_STORE)
		do
			update_all (a_grads)
		ensure then
			state_for_updated: across parameters as p all a_grads.has (p) implies state_ids_model.has (p.id) end
		end

feature {NONE} -- Implementation

	update_one (a_parameter, a_gradient: TENSOR; a_rate: REAL_64)
			-- torch.optim.AdamW on one parameter; moment state created on first use.
		local
			m, v: TENSOR_STORAGE
		do
			if attached first_moments.item (a_parameter.id) as l_m and attached second_moments.item (a_parameter.id) as l_v then
				m := l_m
				v := l_v
			else
				m := a_parameter.backend.new_storage (a_parameter.shape.count)
				v := a_parameter.backend.new_storage (a_parameter.shape.count)
				first_moments.force (m, a_parameter.id)
				second_moments.force (v, a_parameter.id)
			end
			a_parameter.backend.adamw_update (a_parameter.storage, a_gradient.storage, m, v, a_rate, beta1, beta2, eps, decay, step_count)
		end

	first_moments: HASH_TABLE [TENSOR_STORAGE, INTEGER_64]
			-- m by parameter id.

	second_moments: HASH_TABLE [TENSOR_STORAGE, INTEGER_64]
			-- v by parameter id.

invariant
	beta1_range: 0.0 <= beta1 and beta1 < 1.0
	beta2_range: 0.0 <= beta2 and beta2 < 1.0
	eps_positive: eps > 0.0
	same_state_count: first_moments.count = second_moments.count

end
