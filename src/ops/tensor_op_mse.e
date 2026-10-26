note
	description: "Mean squared error. Backward: g·2(a−b)/n and its negative."
	author: "Larry Rix"

class
	TENSOR_OP_MSE

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a, b: TENSOR)
		require
			same_shape: a.shape ~ b.shape
		do
			inputs := <<a, b>>
			create output_shape.make_scalar
		ensure
			inputs_set: inputs [1] = a and inputs [2] = b
		end

feature -- Access

	name: STRING_8 = "mse"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			l_scaled: TENSOR
		do
				-- d/da = g · 2(a − b)/n; d/db is its negative.
			l_scaled := (inputs [1].detached - inputs [2].detached).scaled ((2.0 * a_output_grad.item (0) / inputs [1].shape.count).truncated_to_real)
			if inputs [1].is_tracked then
				a_store.accumulate (inputs [1], l_scaled)
			end
			if inputs [2].is_tracked then
				a_store.accumulate (inputs [2], l_scaled.negated)
			end
		end

invariant
	scalar_output: output_shape.count = 1

end
