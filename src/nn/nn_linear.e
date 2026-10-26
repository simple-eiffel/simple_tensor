note
	description: "[
		y = x W^T + b over the last axis (PyTorch nn.Linear layout: `weight' is out x in),
		for x of any rank >= 1: leading axes are flattened into rows for one matmul and
		restored after (reshape is a free view). `make_tied' shares another module's
		weight tensor (RixGPT's head = token embedding).
	]"
	author: "Larry Rix"

class
	NN_LINEAR

inherit
	NN_LAYER
		redefine
			accepts
		end

create
	make_normal,
	make_uniform,
	make_tied

feature {NONE} -- Initialization

	make_normal (f: SIMPLE_TENSOR; a_in, a_out: INTEGER; a_bias: BOOLEAN; a_std: REAL_64; r: TENSOR_RANDOM)
			-- Weight ~ N(0, a_std), bias zeros (RixGPT's _init: std 0.02).
		require
			positive: a_in >= 1 and a_out >= 1
			std_non_negative: a_std >= 0.0
		local
			w: TENSOR
		do
			w := f.zeros (<<a_out, a_in>>)
			r.fill_normal (w, 0.0, a_std)
			setup (f, f.parameter (w), a_bias)
		ensure
			sized: in_features = a_in and out_features = a_out
			biased: has_bias = a_bias
		end

	make_uniform (f: SIMPLE_TENSOR; a_in, a_out: INTEGER; a_bias: BOOLEAN; a_bound: REAL_64; r: TENSOR_RANDOM)
			-- Weight ~ U(-a_bound, a_bound), bias zeros.
		require
			positive: a_in >= 1 and a_out >= 1
			bound_non_negative: a_bound >= 0.0
		local
			w: TENSOR
		do
			w := f.zeros (<<a_out, a_in>>)
			r.fill_uniform (w, -a_bound, a_bound)
			setup (f, f.parameter (w), a_bias)
		ensure
			sized: in_features = a_in and out_features = a_out
			biased: has_bias = a_bias
		end

	make_tied (f: SIMPLE_TENSOR; a_weight: TENSOR; a_bias: BOOLEAN)
			-- Use `a_weight' (out x in), shared with its owner.
		require
			matrix: a_weight.shape.rank = 2
			trainable: a_weight.requires_grad
		do
			setup (f, a_weight, a_bias)
		ensure
			shared: weight = a_weight
		end

	setup (f: SIMPLE_TENSOR; w: TENSOR; a_bias: BOOLEAN)
		do
			make_module
			weight := w
			register_parameter ("weight", w)
			if a_bias then
				bias := f.parameter (f.zeros (<<w.shape.dim (1)>>))
				if attached bias as b then
					register_parameter ("bias", b)
				end
			end
		end

feature -- Access

	weight: TENSOR
			-- out_features x in_features.

	bias: detachable TENSOR

	in_features: INTEGER
		do
			Result := weight.shape.dim (2)
		end

	out_features: INTEGER
		do
			Result := weight.shape.dim (1)
		end

	has_bias: BOOLEAN
		do
			Result := attached bias
		end

feature -- Status report

	accepts (x: TENSOR): BOOLEAN
			-- Last axis = `in_features', on the weight's backend.
		do
			Result := x.shape.last = in_features and x.has_same_backend (weight)
		end

feature -- Basic operations

	forward (x: TENSOR): TENSOR
		local
			l_rows: INTEGER
			l_dims: ARRAY [INTEGER]
		do
			l_rows := x.shape.count // in_features
			if attached bias as b then
				Result := x.reshaped (<<l_rows, in_features>>).linear (weight, b)
			else
				Result := x.reshaped (<<l_rows, in_features>>).matmul_transposed (weight, False, True)
			end
			l_dims := x.shape.to_array
			l_dims [l_dims.upper] := out_features
			Result := Result.reshaped (l_dims)
		ensure then
			shaped: Result.shape.rank = x.shape.rank and Result.shape.last = out_features
		end

invariant
	matrix: weight.shape.rank = 2
	bias_matches: attached bias as b implies (b.shape.rank = 1 and b.shape.dim (1) = out_features)

end
