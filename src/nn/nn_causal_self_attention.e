note
	description: "[
		Causal multi-head self-attention in PyTorch nn.MultiheadAttention's layout (spec 04),
		so RixGPT's trained weights map one to one:
			in_proj_weight (3C x C), in_proj_bias (3C): q, k, v = rows 0..C-1, C..2C-1, 2C..3C-1
			out_proj (C -> C, with bias)
		For x (B, T, C) with H heads of D = C / H:
			qkv = x W_in^T + b_in; q, k, v -> (B*H, T, D)
			a = softmax (mask (q k^T / sqrt D)), dropout on a (training only)
			y = (a v) -> (B, T, C); out = out_proj (y)
		The mask hides future positions (strictly upper triangle): masked logits are set to
		-1e9, which softmax turns into exactly 0 in float.
	]"
	author: "Larry Rix"

class
	NN_CAUSAL_SELF_ATTENTION

inherit
	NN_LAYER
		redefine
			accepts
		end

create
	make

feature {NONE} -- Initialization

	make (f: SIMPLE_TENSOR; a_dim, a_heads: INTEGER; a_dropout: REAL_64; a_std: REAL_64; r: TENSOR_RANDOM)
			-- in_proj ~ U(-b, b), b = sqrt (6 / (C + 3C)) (PyTorch's xavier_uniform_ for
			-- MultiheadAttention); in_proj_bias zeros; out_proj ~ N(0, a_std), bias zeros.
		require
			positive: a_dim >= 1 and a_heads >= 1
			divisible: a_dim \\ a_heads = 0
			probability: 0.0 <= a_dropout and a_dropout < 1.0
		local
			w: TENSOR
		do
			make_module
			facade := f
			heads := a_heads
			w := f.zeros (<<3 * a_dim, a_dim>>)
			r.fill_uniform (w, -{DOUBLE_MATH}.sqrt (6.0 / (4 * a_dim)), {DOUBLE_MATH}.sqrt (6.0 / (4 * a_dim)))
			in_proj_weight := f.parameter (w)
			in_proj_bias := f.parameter (f.zeros (<<3 * a_dim>>))
			create out_proj.make_normal (f, a_dim, a_dim, True, a_std, r)
			create attention_dropout.make (a_dropout, r)
			create masks.make (2)
			register_parameter ("in_proj_weight", in_proj_weight)
			register_parameter ("in_proj_bias", in_proj_bias)
			register_child ("out_proj", out_proj)
			register_child ("attention_dropout", attention_dropout)
		ensure
			sized: dim = a_dim and heads = a_heads
		end

feature -- Access

	in_proj_weight, in_proj_bias: TENSOR
	out_proj: NN_LINEAR
	attention_dropout: NN_DROPOUT
	heads: INTEGER

	dim: INTEGER
		do
			Result := in_proj_weight.shape.dim (2)
		end

	head_dim: INTEGER
		do
			Result := dim // heads
		end

feature -- Status report

	accepts (x: TENSOR): BOOLEAN
			-- (B, T, C) with C = `dim', on this module's backend.
		do
			Result := x.shape.rank = 3 and then x.shape.last = dim and x.has_same_backend (in_proj_weight)
		end

feature -- Basic operations

	forward (x: TENSOR): TENSOR
		local
			b, t, c, h, d: INTEGER
			qkv, q, k, v, a, y: TENSOR
		do
			b := x.shape.dim (1); t := x.shape.dim (2); c := dim; h := heads; d := head_dim
			qkv := x.reshaped (<<b * t, c>>).matmul_transposed (in_proj_weight, False, True) + in_proj_bias
			q := split_heads (qkv.sliced (2, 1, c), b, t, h, d)
			k := split_heads (qkv.sliced (2, c + 1, c), b, t, h, d)
			v := split_heads (qkv.sliced (2, 2 * c + 1, c), b, t, h, d)
			a := q.matmul_transposed (k, False, True).scaled ((1.0 / {DOUBLE_MATH}.sqrt (d)).truncated_to_real)
			a := attention_dropout.forward (a.masked_fill (mask (t), -1.0e9).softmax)
			y := a.matmul (v).reshaped (<<b, h, t, d>>).permuted (<<1, 3, 2, 4>>).reshaped (<<b, t, c>>)
			Result := out_proj.forward (y)
		ensure then
			same_shape: Result.shape ~ x.shape
		end

feature {NONE} -- Implementation

	facade: SIMPLE_TENSOR

	masks: HASH_TABLE [INDEX_TENSOR, INTEGER]
			-- Causal masks by sequence length.

	mask (t: INTEGER): INDEX_TENSOR
		do
			if attached masks.item (t) as m then
				Result := m
			else
				Result := facade.causal_mask (t)
				masks.put (Result, t)
			end
		end

	split_heads (z: TENSOR; b, t, h, d: INTEGER): TENSOR
			-- (B*T, C) -> (B*H, T, D).
		do
			Result := z.reshaped (<<b, t, h, d>>).permuted (<<1, 3, 2, 4>>).reshaped (<<b * h, t, d>>)
		end

invariant
	divisible: dim \\ heads = 0

end
