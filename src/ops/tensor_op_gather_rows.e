note
	description: "Embedding lookup. Backward: scatter-add g into a zero table (deterministic order)."
	author: "Larry Rix"

class
	TENSOR_OP_GATHER_ROWS

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a_table: TENSOR; a_indices: INDEX_TENSOR; a_output_shape: TENSOR_SHAPE)
		require
			table: a_table.shape.rank = 2
		do
			inputs := <<a_table>>
			indices := a_indices
			output_shape := a_output_shape
		ensure
			table_set: inputs [1] = a_table
			indices_set: indices = a_indices
		end

feature -- Access

	indices: INDEX_TENSOR

	name: STRING_8 = "gather_rows"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			l_table: TENSOR
			l_grad: TENSOR_STORAGE
		do
			l_table := inputs [1]
			if l_table.is_tracked then
				l_grad := l_table.backend.new_storage (l_table.shape.count)
				l_table.backend.scatter_add_rows (a_output_grad.storage, indices, l_grad, l_table.shape.dim (1), l_table.shape.last)
				a_store.accumulate (l_table, leaf_over (l_grad, l_table.shape, l_table.backend))
			end
		end

end
