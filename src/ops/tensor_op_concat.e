note
	description: "Concatenate along an axis. Backward: g split back into the two parts."
	author: "Larry Rix"

class
	TENSOR_OP_CONCAT

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a, b: TENSOR; a_axis: INTEGER; a_output_shape: TENSOR_SHAPE)
		require
			same_rank: a.shape.rank = b.shape.rank
			axis_valid: 1 <= a_axis and a_axis <= a.shape.rank
		do
			inputs := <<a, b>>
			axis := a_axis
			output_shape := a_output_shape
		ensure
			inputs_set: inputs [1] = a and inputs [2] = b
			axis_set: axis = a_axis
		end

feature -- Access

	axis: INTEGER

	name: STRING_8 = "concat"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			na, nb: INTEGER
		do
			na := inputs [1].shape.dim (axis)
			nb := inputs [2].shape.dim (axis)
			if inputs [1].is_tracked then
				a_store.accumulate (inputs [1], a_output_grad.sliced (axis, 1, na))
			end
			if inputs [2].is_tracked then
				a_store.accumulate (inputs [2], a_output_grad.sliced (axis, na + 1, nb))
			end
		end

end
