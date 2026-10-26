note
	description: "Reshape (a free view). Backward: g reshaped back to the input dims."
	author: "Larry Rix"

class
	TENSOR_OP_RESHAPE

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_output_shape: TENSOR_SHAPE)
		require
			same_count: a_output_shape.count = a.shape.count
		do
			inputs := <<a>>
			output_shape := a_output_shape
		ensure
			input_set: inputs [1] = a
		end

feature -- Access

	name: STRING_8 = "reshape"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		do
			if inputs [1].is_tracked then
				a_store.accumulate (inputs [1], a_output_grad.reshaped (inputs [1].shape.to_array))
			end
		end

end
