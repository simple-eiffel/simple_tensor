note
	description: "Plain SGD: param := param − rate · grad."
	author: "Larry Rix"

class
	SGD_OPTIMIZER

inherit
	TENSOR_OPTIMIZER

create
	make

feature {NONE} -- Initialization

	make (a_parameters: ARRAYED_LIST [TENSOR]; a_schedule: LR_SCHEDULE)
		require
			leaves_requiring_grad: across a_parameters as p all p.requires_grad end
			distinct: distinct_ids (a_parameters)
		do
			parameters := a_parameters
			schedule := a_schedule
		ensure
			parameters_set: parameters = a_parameters
			schedule_set: schedule = a_schedule
			fresh: step_count = 0
		end

feature -- Basic operations

	step (a_grads: GRAD_STORE)
		do
			update_all (a_grads)
		end

feature {NONE} -- Implementation

	update_one (a_parameter, a_gradient: TENSOR; a_rate: REAL_64)
		do
			a_parameter.backend.sgd_update (a_parameter.storage, a_gradient.storage, a_rate)
		end

end
