note
	description: "[
		Inverted dropout: in training, each element is kept with probability 1 - p and
		scaled by 1/(1 - p); otherwise the identity. The mask is Philox words reserved from
		the random stream and drawn on the input's backend (TENSOR.dropped): never stored,
		regenerated in backward, bit-identical on every backend.
	]"
	author: "Larry Rix"

class
	NN_DROPOUT

inherit
	NN_LAYER

create
	make

feature {NONE} -- Initialization

	make (a_p: REAL_64; r: TENSOR_RANDOM)
			-- Drop probability `a_p', masks drawn from `r'.
		require
			probability: 0.0 <= a_p and a_p < 1.0
		do
			make_module
			p := a_p
			random := r
		ensure
			p_set: p = a_p
			random_set: random = r
		end

feature -- Access

	p: REAL_64
			-- Drop probability.

	random: TENSOR_RANDOM

feature -- Basic operations

	forward (x: TENSOR): TENSOR
		local
			l_position: NATURAL_64
		do
			if is_training and p > 0.0 then
					-- The same words fill_keep_mask would draw, reserved here and drawn by
					-- TENSOR.dropped, which regenerates them in backward instead of storing a mask.
				l_position := random.position
				random.skip (x.shape.count.to_natural_64)
				Result := x.dropped (1.0 - p, random.seed.to_natural_64, l_position)
			else
				Result := x
			end
		ensure then
			identity_when_off: (not is_training or p = 0.0) implies Result = x
			same_shape: Result.shape ~ x.shape
		end

end
