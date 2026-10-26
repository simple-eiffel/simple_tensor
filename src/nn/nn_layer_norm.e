note
	description: "[
		Layer normalization over the last axis (PyTorch nn.LayerNorm, biased variance):
		y = (x - mean) / sqrt (var + eps) * weight + bias. A composition of primitives
		(spec 03 A-S08), so its gradient is the primitives' own; a fused kernel later
		keeps this composition as its oracle.
	]"
	author: "Larry Rix"

class
	NN_LAYER_NORM

inherit
	NN_LAYER
		redefine
			accepts
		end

create
	make

feature {NONE} -- Initialization

	make (f: SIMPLE_TENSOR; a_dim: INTEGER; a_eps: REAL_64)
			-- weight ones, bias zeros (PyTorch's defaults); eps 1e-5 in PyTorch.
		require
			positive: a_dim >= 1
			eps_positive: a_eps > 0.0
		do
			make_module
			eps := a_eps
			weight := f.parameter (f.ones (<<a_dim>>))
			bias := f.parameter (f.zeros (<<a_dim>>))
			register_parameter ("weight", weight)
			register_parameter ("bias", bias)
		ensure
			eps_set: eps = a_eps
			sized: dim = a_dim
		end

feature -- Access

	weight, bias: TENSOR
	eps: REAL_64

	dim: INTEGER
		do
			Result := weight.shape.dim (1)
		end

feature -- Status report

	accepts (x: TENSOR): BOOLEAN
		do
			Result := x.shape.last = dim and x.has_same_backend (weight)
		end

feature -- Basic operations

	forward (x: TENSOR): TENSOR
		local
			l_axis: INTEGER
			xc: TENSOR
		do
			l_axis := x.shape.rank
			xc := x - x.mean (l_axis, True)
			Result := xc / (xc * xc).mean (l_axis, True).shifted (eps.truncated_to_real).square_root * weight + bias
		ensure then
			same_shape: Result.shape ~ x.shape
		end

end
