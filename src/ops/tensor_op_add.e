note
	description: "a + b (broadcast). Backward: g, g, each reduced to its input's shape."
	author: "Larry Rix"

class
	TENSOR_OP_ADD

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a, b: TENSOR; a_output_shape: TENSOR_SHAPE)
		require
			broadcast: a.shape.broadcasts_with (b.shape) and then a_output_shape ~ a.shape.broadcast_with (b.shape)
		do
			inputs := <<a, b>>
			output_shape := a_output_shape
		ensure
			inputs_set: inputs [1] = a and inputs [2] = b
			shape_set: output_shape = a_output_shape
		end

feature -- Access

	name: STRING_8 = "add"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		do
			if inputs [1].is_tracked then
				a_store.accumulate (inputs [1], shaped_for (a_output_grad, inputs [1]))
			end
			if inputs [2].is_tracked then
				a_store.accumulate (inputs [2], shaped_for (a_output_grad, inputs [2]))
			end
		end

end
