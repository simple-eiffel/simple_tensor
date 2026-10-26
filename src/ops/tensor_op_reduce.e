note
	description: "[
		Sum or mean along an axis. Backward: g (or g/n) broadcast back to the
		input shape. Max has its own class (gradient to the argmax).
	]"
	author: "Larry Rix"

class
	TENSOR_OP_REDUCE

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a_kind: INTEGER; a: TENSOR; a_axis: INTEGER; a_keep: BOOLEAN; a_output_shape: TENSOR_SHAPE)
		require
			sum_or_mean: a_kind = Sum_kind or a_kind = Mean_kind
			axis_valid: 1 <= a_axis and a_axis <= a.shape.rank
		do
			kind := a_kind
			inputs := <<a>>
			axis := a_axis
			keep := a_keep
			output_shape := a_output_shape
		ensure
			kind_set: kind = a_kind
			axis_set: axis = a_axis
			keep_set: keep = a_keep
		end

feature -- Access

	kind: INTEGER
	axis: INTEGER
	keep: BOOLEAN

	name: STRING_8
		do
			if kind = Sum_kind then Result := "sum" else Result := "mean" end
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			a, g_kept: TENSOR
		do
			a := inputs [1]
			if a.is_tracked then
					-- Put the reduced axis back as size 1, then repeat over it.
				g_kept := a_output_grad.reshaped (a.shape.with_dim (axis, 1).to_array)
				if kind = Mean_kind then
					g_kept := g_kept.scaled ((1.0 / a.shape.dim (axis)).truncated_to_real)
				end
				a_store.accumulate (a, spread_to (g_kept, a))
			end
		end

invariant
	sum_or_mean: kind = Sum_kind or kind = Mean_kind

end
