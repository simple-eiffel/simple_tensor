note
	description: "[
		Sum over the axes along which the output shape broadcasts to the input
		shape (spec 08 open issue 1: the broadcast backward, itself
		differentiable). Backward: g broadcast back to the input shape.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_REDUCE_TO

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_output_shape: TENSOR_SHAPE)
		require
			broadcastable: a_output_shape.broadcasts_with (a.shape) and then a_output_shape.broadcast_with (a.shape) ~ a.shape
		do
			inputs := <<a>>
			output_shape := a_output_shape
		ensure
			input_set: inputs [1] = a
		end

feature -- Access

	name: STRING_8 = "reduce_to"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		do
			if inputs [1].is_tracked then
				a_store.accumulate (inputs [1], spread_to (a_output_grad, inputs [1]))
			end
		end

end
