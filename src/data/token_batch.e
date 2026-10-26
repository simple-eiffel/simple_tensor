note
	description: "One training batch: inputs and next-token targets, each batch × block."
	author: "Larry Rix"

class
	TOKEN_BATCH

create
	make

feature {NONE} -- Initialization

	make (a_inputs, a_targets: INDEX_TENSOR)
		require
			same_shape: a_inputs.shape ~ a_targets.shape
			rank_two: a_inputs.shape.rank = 2
		do
			inputs := a_inputs
			targets := a_targets
		ensure
			inputs_set: inputs = a_inputs
			targets_set: targets = a_targets
		end

feature -- Access

	inputs: INDEX_TENSOR
	targets: INDEX_TENSOR

invariant
	same_shape: inputs.shape ~ targets.shape

end
