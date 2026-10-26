note
	description: "[
		Fused log-softmax + negative log-likelihood, mean over non-ignored
		positions. Backward: g · (softmax − onehot(target)) / n_valid, 0 on
		ignored rows (backend kernel `cross_entropy_grad').
	]"
	author: "Larry Rix"

class
	TENSOR_OP_CROSS_ENTROPY

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a_logits: TENSOR; a_targets: INDEX_TENSOR; a_ignore: INTEGER)
		require
			logits_rank: a_logits.shape.rank >= 2
		do
			inputs := <<a_logits>>
			targets := a_targets
			ignore_index := a_ignore
			create output_shape.make_scalar
		ensure
			logits_set: inputs [1] = a_logits
			targets_set: targets = a_targets
			ignore_set: ignore_index = a_ignore
		end

feature -- Access

	targets: INDEX_TENSOR
	ignore_index: INTEGER

	name: STRING_8 = "cross_entropy"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			l_logits: TENSOR
			l_grad: TENSOR_STORAGE
		do
			l_logits := inputs [1]
			if l_logits.is_tracked then
				l_grad := l_logits.backend.new_storage (l_logits.shape.count)
				l_logits.backend.cross_entropy_grad (l_logits.storage, targets, ignore_index, l_grad,
					l_logits.shape.count // l_logits.shape.last, l_logits.shape.last, a_output_grad.item (0))
				a_store.accumulate (l_logits, leaf_over (l_grad, l_logits.shape, l_logits.backend))
			end
		end

invariant
	scalar_output: output_shape.count = 1

end
