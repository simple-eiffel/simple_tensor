note
	description: "[
		Batched op(A)·op(B). Backward: dA = g·op(B)^T, dB = op(A)^T·g, with the
		transposes folded into the flags; a shared (rank-2) B sums its
		gradient over the batch.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_MATMUL

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a, b: TENSOR; a_ta, a_tb: BOOLEAN; a_output_shape: TENSOR_SHAPE)
		require
			ranks: a.shape.rank >= 2 and b.shape.rank >= 2
		do
			inputs := <<a, b>>
			transpose_a := a_ta
			transpose_b := a_tb
			output_shape := a_output_shape
		ensure
			inputs_set: inputs [1] = a and inputs [2] = b
			flags_set: transpose_a = a_ta and transpose_b = a_tb
		end

feature -- Access

	transpose_a, transpose_b: BOOLEAN

	name: STRING_8 = "matmul"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
			-- C = op(A)·op(B), G = dL/dC.
			-- dL/dop(A) = G·op(B)ᵀ; dL/dop(B) = op(A)ᵀ·G; transpose back when flagged.
			-- A shared (rank-2) B under a batched A sums its gradient over the batch.
		local
			a, b, g_op_a, g_op_b: TENSOR
			r, l_batch: INTEGER
		do
			a := inputs [1]
			b := inputs [2]
			r := a.shape.rank
			if a.is_tracked then
				g_op_a := a_output_grad.matmul_transposed (b.detached, False, not transpose_b)
				if transpose_a then
					a_store.accumulate (a, g_op_a.transposed)
				else
					a_store.accumulate (a, g_op_a)
				end
			end
			if b.is_tracked then
				g_op_b := a.detached.matmul_transposed (a_output_grad, not transpose_a, False)
				if transpose_b then
					g_op_b := g_op_b.transposed
				end
				if b.shape.rank = 2 and r > 2 then
					l_batch := g_op_b.shape.count // (b.shape.count)
					g_op_b := g_op_b.reshaped (<<l_batch, b.shape.dim (1), b.shape.dim (2)>>).sum (1, False)
				end
				a_store.accumulate (b, g_op_b)
			end
		end

end
