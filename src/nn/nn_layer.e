note
	description: "[
		A module that maps a tensor to a tensor (spec 04: NN_LAYER). What a layer
		accepts is its own business, so the precondition names `accepts', which each
		layer redefines (no precondition strengthening in descendants).
	]"
	author: "Larry Rix"

deferred class
	NN_LAYER

inherit
	NN_MODULE

feature -- Status report

	accepts (x: TENSOR): BOOLEAN
			-- Can `forward' take `x'? (Any tensor here; layers narrow it.)
		do
			Result := True
		end

feature -- Basic operations

	forward (x: TENSOR): TENSOR
			-- This layer applied to `x'.
		require
			live: not x.is_released
			acceptable: accepts (x)
		deferred
		end

end
