note
	description: "[
		Fused causal attention softmax: y = softmax over visible keys of scale * x, with
		future positions 0 (TENSOR.causal_softmax). One saved output instead of the scaled,
		masked and softmax temporaries. Backward: scale * y * (g - sum (g * y)) over the
		last axis; hidden positions have y = 0, so their gradient is 0.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_CAUSAL_SOFTMAX

inherit
	TENSOR_OP
		redefine
			saved_live
		end

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_output: TENSOR_STORAGE; a_scale: REAL_32)
		require
			output_sized: a_output.count = a.shape.count
		do
			inputs := <<a>>
			output_values := a_output
			scale := a_scale
			output_shape := a.shape
		ensure
			input_set: inputs [1] = a
			output_saved: output_values = a_output
			scale_set: scale = a_scale
		end

feature -- Access

	output_values: TENSOR_STORAGE
	scale: REAL_32

	name: STRING_8
		do
			Result := "causal_softmax"
		end

feature -- Status report

	saved_live: BOOLEAN
		do
			Result := Precursor and not output_values.is_released
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			a, y: TENSOR
		do
			a := inputs [1]
			if a.is_tracked then
				y := leaf_over (output_values, a.shape, a.backend)
				a_store.accumulate (a, (y * (a_output_grad - (a_output_grad * y).sum (a.shape.rank, True))).scaled (scale))
			end
		end

end
