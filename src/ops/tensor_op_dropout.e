note
	description: "[
		Dropout that stores no mask (TENSOR.dropped): the keep mask is Philox words at a fixed
		stream address, so backward regenerates the identical mask from (seed, position,
		keep) instead of keeping a full-size mask alive between forward and backward.
		Backward: dx = g * mask.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_DROPOUT

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (x: TENSOR; a_keep: REAL_64; a_seed, a_position: NATURAL_64)
		do
			inputs := <<x>>
			keep := a_keep
			seed := a_seed
			position := a_position
			output_shape := x.shape
		ensure
			input_set: inputs [1] = x
			keep_set: keep = a_keep
		end

feature -- Access

	keep: REAL_64
	seed, position: NATURAL_64

	name: STRING_8
		do
			Result := "dropout"
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			x, m: TENSOR
		do
			x := inputs [1]
			if x.is_tracked then
				m := x.detached.zeros_like
				x.backend.fill_random (m.storage, x.backend.Keep_mask_fill_kind, keep, 0.0, seed, position)
				a_store.accumulate (x, a_output_grad * m)
			end
		end

end
