note
	description: "Slice along an axis (a copy). Backward: g added into zeros at the slice."
	author: "Larry Rix"

class
	TENSOR_OP_SLICE

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_axis, a_start, a_length: INTEGER; a_output_shape: TENSOR_SHAPE)
		require
			axis_valid: 1 <= a_axis and a_axis <= a.shape.rank
			range: a_start >= 1 and a_length >= 1 and a_start + a_length - 1 <= a.shape.dim (a_axis)
		do
			inputs := <<a>>
			axis := a_axis
			start := a_start
			length := a_length
			output_shape := a_output_shape
		ensure
			set: axis = a_axis and start = a_start and length = a_length
		end

feature -- Access

	axis, start, length: INTEGER

	name: STRING_8 = "slice"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			a: TENSOR
			l_full: TENSOR_STORAGE
		do
			a := inputs [1]
			if a.is_tracked then
				l_full := a.backend.new_storage (a.shape.count)
				a.backend.add_slice (a_output_grad.storage, l_full, a.shape, axis, start, length)
				a_store.accumulate (a, leaf_over (l_full, a.shape, a.backend))
			end
		end

end
