note
	description: "[
		Transformer feed-forward block: Linear (d -> h), GELU (exact, erf), Linear (h -> d),
		Dropout. Children are named "0" and "2" like RixGPT's nn.Sequential, so parameter
		names match PyTorch's ("mlp.0.weight", "mlp.2.bias").
	]"
	author: "Larry Rix"

class
	NN_MLP

inherit
	NN_LAYER
		redefine
			accepts
		end

create
	make

feature {NONE} -- Initialization

	make (f: SIMPLE_TENSOR; a_dim, a_hidden: INTEGER; a_dropout: REAL_64; a_std: REAL_64; r: TENSOR_RANDOM)
		require
			positive: a_dim >= 1 and a_hidden >= 1
			probability: 0.0 <= a_dropout and a_dropout < 1.0
		do
			make_module
			create fc.make_normal (f, a_dim, a_hidden, True, a_std, r)
			create proj.make_normal (f, a_hidden, a_dim, True, a_std, r)
			create dropout.make (a_dropout, r)
			register_child ("0", fc)
			register_child ("2", proj)
			register_child ("3", dropout)
		ensure
			sized: fc.in_features = a_dim and fc.out_features = a_hidden and proj.out_features = a_dim
		end

feature -- Access

	fc, proj: NN_LINEAR
	dropout: NN_DROPOUT

feature -- Status report

	accepts (x: TENSOR): BOOLEAN
		do
			Result := fc.accepts (x)
		end

feature -- Basic operations

	forward (x: TENSOR): TENSOR
		do
			Result := dropout.forward (proj.forward (fc.forward (x).gelu))
		ensure then
			same_shape: Result.shape ~ x.shape
		end

end
