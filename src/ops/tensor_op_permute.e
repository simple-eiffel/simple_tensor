note
	description: "Permute (a copy). Backward: g permuted by the inverse order."
	author: "Larry Rix"

class
	TENSOR_OP_PERMUTE

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_order: ARRAY [INTEGER]; a_output_shape: TENSOR_SHAPE)
		require
			permutation: a.shape.is_permutation (a_order)
		do
			inputs := <<a>>
			order := a_order.twin
			output_shape := a_output_shape
		ensure
			input_set: inputs [1] = a
		end

feature -- Access

	order: ARRAY [INTEGER]

	name: STRING_8 = "permute"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			l_inverse: ARRAY [INTEGER]
			k: INTEGER
		do
			if inputs [1].is_tracked then
				create l_inverse.make_filled (1, 1, order.count)
				from k := 1 until k > order.count loop
					l_inverse [order [order.lower + k - 1]] := k
					k := k + 1
				end
				a_store.accumulate (inputs [1], a_output_grad.permuted (l_inverse))
			end
		end

end
