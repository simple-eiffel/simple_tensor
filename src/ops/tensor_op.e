note
	description: "[
		A node of the autograd graph: a primitive computation that knows its
		inputs, whatever values its backward needs, and how to push an output
		gradient back to its inputs.

		Backward is written with ordinary TENSOR ops on UNTRACKED tensors
		(spec 07). `inputs' stay tracked (the graph walk follows them), so
		every backward formula computes on `i.detached', never on `i'
		itself; with an untracked output gradient no graph is built during
		backward (Phase 2 verdict G-C). Every backend gets backward for free.
	]"
	author: "Larry Rix"

deferred class
	TENSOR_OP

inherit
	TENSOR_OP_KINDS

feature -- Access

	inputs: ARRAY [TENSOR]
			-- Operands, in order.

	output_shape: TENSOR_SHAPE
			-- Shape of the tensor this op produced.

	name: STRING_8
			-- Op name for diagnostics.
		deferred
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
			-- Add this op's contribution to each tracked input's gradient in `a_store'.
		require
			grad_shaped: a_output_grad.shape ~ output_shape
			grad_untracked: not a_output_grad.is_tracked
			grad_backend: a_output_grad.has_same_backend (inputs [inputs.lower])
			saved_values_live: saved_live
		deferred
		ensure
			tracked_inputs_have_gradients: across inputs as i all i.is_tracked implies a_store.has (i) end
		end

feature {NONE} -- Backward helpers

	shaped_for (g: TENSOR; a_input: TENSOR): TENSOR
			-- `g' summed down to `a_input''s shape (the broadcast backward).
		require
			broadcastable: a_input.shape.broadcasts_with (g.shape) and then a_input.shape.broadcast_with (g.shape) ~ g.shape
		do
			if g.shape ~ a_input.shape then
				Result := g
			else
				Result := g.reduced_to (a_input.shape)
			end
		ensure
			shaped: Result.shape ~ a_input.shape
		end

	spread_to (g: TENSOR; a_input: TENSOR): TENSOR
			-- `g' (of a broadcastable shape) repeated up to `a_input''s shape.
		do
			Result := a_input.detached.zeros_like + g
		end

	leaf_over (a_storage: TENSOR_STORAGE; a_shape: TENSOR_SHAPE; a_backend: TENSOR_BACKEND): TENSOR
			-- An untracked tensor over `a_storage'.
		require
			fits: a_storage.count = a_shape.count
			owned: a_backend.owns (a_storage)
			live: not a_storage.is_released
		do
			create Result.make_leaf (a_shape, a_storage, a_backend, False)
		ensure
			untracked: not Result.is_tracked
		end

feature -- Status report

	saved_live: BOOLEAN
			-- Is every value backward needs still in memory: the inputs, and
			-- (in descendants that save them) stored outputs or masks?
		do
			Result := across inputs as i all not i.is_released end
		end

invariant
	has_inputs: inputs.count >= 1

end
