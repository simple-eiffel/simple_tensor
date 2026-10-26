note
	description: "Max along an axis. Backward: g to the first maximum (saved 0/1 mask)."
	author: "Larry Rix"

class
	TENSOR_OP_MAX

inherit
	TENSOR_OP
		redefine
			saved_live
		end

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_axis: INTEGER; a_keep: BOOLEAN; a_mask: TENSOR_STORAGE; a_output_shape: TENSOR_SHAPE)
		require
			axis_valid: 1 <= a_axis and a_axis <= a.shape.rank
			mask_sized: a_mask.count = a.shape.count
		do
			inputs := <<a>>
			axis := a_axis
			keep := a_keep
			argmax := a_mask
			output_shape := a_output_shape
		ensure
			axis_set: axis = a_axis
			mask_saved: argmax = a_mask
		end

feature -- Access

	axis: INTEGER
	keep: BOOLEAN
	argmax: TENSOR_STORAGE
			-- 1 at the first maximum along `axis', 0 elsewhere.

	name: STRING_8 = "max"

feature -- Status report

	saved_live: BOOLEAN
			-- Inputs and the saved argmax mask are still in memory.
		do
			Result := Precursor and not argmax.is_released
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
			-- g to the first maximum: (g with the axis kept) · mask.
		local
			a, g_kept: TENSOR
		do
			a := inputs [1]
			if a.is_tracked then
				g_kept := a_output_grad.reshaped (a.shape.with_dim (axis, 1).to_array)
				a_store.accumulate (a, leaf_over (argmax, a.shape, a.backend) * g_kept)
			end
		end

end
