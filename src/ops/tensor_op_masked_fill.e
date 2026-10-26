note
	description: "masked_fill. Backward: g with the masked positions zeroed."
	author: "Larry Rix"

class
	TENSOR_OP_MASKED_FILL

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_mask: INDEX_TENSOR)
		require
			mask_broadcasts: a.shape.broadcasts_with (a_mask.shape)
		do
			inputs := <<a>>
			mask := a_mask
			output_shape := a.shape
		ensure
			input_set: inputs [1] = a
			mask_set: mask = a_mask
		end

feature -- Access

	mask: INDEX_TENSOR

	name: STRING_8 = "masked_fill"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		do
			if inputs [1].is_tracked then
				a_store.accumulate (inputs [1], a_output_grad.masked_fill (mask, 0.0))
			end
		end

end
