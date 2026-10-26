note
	description: "[
		A table of `rows' learned vectors of length `dim' (PyTorch nn.Embedding layout).
		Maps indices, not tensors, so it is an NN_MODULE and not an NN_LAYER (spec 04).
	]"
	author: "Larry Rix"

class
	NN_EMBEDDING

inherit
	NN_MODULE

create
	make_normal

feature {NONE} -- Initialization

	make_normal (f: SIMPLE_TENSOR; a_rows, a_dim: INTEGER; a_std: REAL_64; r: TENSOR_RANDOM)
			-- Table ~ N(0, a_std) (RixGPT's _init: std 0.02).
		require
			positive: a_rows >= 1 and a_dim >= 1
			std_non_negative: a_std >= 0.0
		local
			t: TENSOR
		do
			make_module
			t := f.zeros (<<a_rows, a_dim>>)
			r.fill_normal (t, 0.0, a_std)
			table := f.parameter (t)
			register_parameter ("weight", table)
		ensure
			sized: rows = a_rows and dim = a_dim
		end

feature -- Access

	table: TENSOR
			-- rows x dim.

	rows: INTEGER
		do
			Result := table.shape.dim (1)
		end

	dim: INTEGER
		do
			Result := table.shape.dim (2)
		end

feature -- Basic operations

	forward_indices (ix: INDEX_TENSOR): TENSOR
			-- The rows named by `ix', shape ix.shape + [dim].
		require
			in_range: ix.all_in_range (0, rows - 1)
		do
			Result := table.gathered_rows (ix)
		ensure
			shaped: Result.shape ~ ix.shape.appended (dim)
		end

invariant
	matrix: table.shape.rank = 2

end
