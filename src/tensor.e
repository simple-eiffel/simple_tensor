note
	description: "[
		A shape over a storage, plus an autograd identity.

		Every op is a QUERY returning a new tensor (CQS). When an input is
		tracked (requires_grad, or computed from a tracked input) the result
		remembers the op that made it (`origin'); the graph lives in the
		tensors and dies with them (no global tape, spec 03 A-S02).

		Shape rules are preconditions (I-001): a wrong axis or a mismatched
		dimension fails at the op that made it, with a named tag.

		SCOOP: tensors live on the processor that made them; only basic values
		(numbers, strings, paths) should cross a processor boundary (R2-13).
	]"
	author: "Larry Rix"

class
	TENSOR

inherit
	TENSOR_OP_KINDS

	GC_POINT

create {SIMPLE_TENSOR, TENSOR, TENSOR_OP, GRAD_STORE, TENSOR_OPTIMIZER, SAFETENSORS_FILE}
	make_leaf, make_result

feature {NONE} -- Initialization

	make_leaf (a_shape: TENSOR_SHAPE; a_storage: TENSOR_STORAGE; a_backend: TENSOR_BACKEND; a_requires_grad: BOOLEAN)
			-- A tensor not produced by an op.
		require
			fits: a_storage.count = a_shape.count
			owned: a_backend.owns (a_storage)
			live: not a_storage.is_released
		do
			shape := a_shape
			storage := a_storage
			backend := a_backend
			requires_grad := a_requires_grad
			id := next_id
			a_storage.note_owner
		ensure
			shape_set: shape = a_shape
			storage_set: storage = a_storage
			backend_set: backend = a_backend
			leaf: origin = Void
			grad_flag: requires_grad = a_requires_grad
		end

	make_result (a_shape: TENSOR_SHAPE; a_storage: TENSOR_STORAGE; a_backend: TENSOR_BACKEND; a_origin: detachable TENSOR_OP)
			-- Output of an op; `a_origin' is Void when no input was tracked.
		require
			fits: a_storage.count = a_shape.count
			owned: a_backend.owns (a_storage)
			live: not a_storage.is_released
		do
			shape := a_shape
			storage := a_storage
			backend := a_backend
			origin := a_origin
			id := next_id
			a_storage.note_owner
		ensure
			shape_set: shape = a_shape
			storage_set: storage = a_storage
			backend_set: backend = a_backend
			not_a_parameter: not requires_grad
			origin_set: origin = a_origin
		end

feature -- Access

	shape: TENSOR_SHAPE
	storage: TENSOR_STORAGE
	backend: TENSOR_BACKEND

	id: INTEGER_64
			-- Unique within this thread (each SCOOP processor is a thread; review spike R-1).

	requires_grad: BOOLEAN
			-- Is this a leaf whose gradient `backward' must produce?

	origin: detachable TENSOR_OP
			-- The op that produced this tensor, when an input was tracked.

	item (i: INTEGER): REAL_32
			-- Element `i' (0-based, row-major). Slow path.
		require
			valid: 0 <= i and i < shape.count
			live: not is_released
		do
			Result := storage.item (i)
		end

	to_array: ARRAY [REAL_32]
			-- All elements, row-major (1-based array). Slow path.
		require
			live: not is_released
		do
			Result := backend.download (storage)
		ensure
			counted: Result.count = shape.count
		end

feature -- Status report

	is_tracked: BOOLEAN
			-- Does this tensor take part in autograd?
		do
			Result := requires_grad or origin /= Void
		end

	is_scalar: BOOLEAN
		do
			Result := shape.count = 1
		end

	is_released: BOOLEAN
		do
			Result := storage.is_released
		end

	has_same_backend (o: TENSOR): BOOLEAN
		do
			Result := backend.is_compatible (o.backend)
		end

	storage_shared: BOOLEAN
			-- Does another tensor see this storage? (Conservative: counts every
			-- tensor ever created over it.)
		do
			Result := storage.is_shared
		end

feature -- Elementwise binary (broadcast)

	plus alias "+" (o: TENSOR): TENSOR
		require
			same_backend: has_same_backend (o)
			shapes_broadcast: shape.broadcasts_with (o.shape)
			live: not is_released and not o.is_released
		do
			Result := binary_result (Add_kind, o)
		ensure
			shape_broadcast: Result.shape ~ shape.broadcast_with (o.shape)
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage and Result.storage /= o.storage
		end

	minus alias "-" (o: TENSOR): TENSOR
		require
			same_backend: has_same_backend (o)
			shapes_broadcast: shape.broadcasts_with (o.shape)
			live: not is_released and not o.is_released
		do
			Result := binary_result (Sub_kind, o)
		ensure
			shape_broadcast: Result.shape ~ shape.broadcast_with (o.shape)
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage and Result.storage /= o.storage
		end

	product alias "*" (o: TENSOR): TENSOR
		require
			same_backend: has_same_backend (o)
			shapes_broadcast: shape.broadcasts_with (o.shape)
			live: not is_released and not o.is_released
		do
			Result := binary_result (Mul_kind, o)
		ensure
			shape_broadcast: Result.shape ~ shape.broadcast_with (o.shape)
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage and Result.storage /= o.storage
		end

	quotient alias "/" (o: TENSOR): TENSOR
		require
			same_backend: has_same_backend (o)
			shapes_broadcast: shape.broadcasts_with (o.shape)
			live: not is_released and not o.is_released
		do
			Result := binary_result (Div_kind, o)
		ensure
			shape_broadcast: Result.shape ~ shape.broadcast_with (o.shape)
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage and Result.storage /= o.storage
		end

feature -- Elementwise unary

	unary (a_kind: INTEGER; a_param: REAL_32): TENSOR
			-- f (Current) for unary kind `a_kind' (TENSOR_OP_KINDS); the named
			-- queries below are the client vocabulary.
		require
			valid_kind: is_unary_kind (a_kind)
			param_unused: not uses_param (a_kind) implies a_param = 0.0
			live: not is_released
		local
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_out := backend.new_storage (shape.count)
			backend.unary (a_kind, storage, l_out, a_param)
			if is_tracked then
				create {TENSOR_OP_UNARY} l_op.make (a_kind, Current, l_out, a_param)
			end
			create Result.make_result (shape, l_out, backend, l_op)
		ensure
			same_shape: Result.shape ~ shape
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	negated: TENSOR
		require live: not is_released
		do Result := unary (Neg_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	scaled (c: REAL_32): TENSOR
		require live: not is_released
		do Result := unary (Scale_kind, c) ensure same_shape: Result.shape ~ shape end

	shifted (c: REAL_32): TENSOR
		require live: not is_released
		do Result := unary (Add_scalar_kind, c) ensure same_shape: Result.shape ~ shape end

	exponential: TENSOR
		require live: not is_released
		do Result := unary (Exp_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	logarithm: TENSOR
		require live: not is_released
		do Result := unary (Log_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	square_root: TENSOR
		require live: not is_released
		do Result := unary (Sqrt_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	hyperbolic_tangent: TENSOR
		require live: not is_released
		do Result := unary (Tanh_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	sigmoid: TENSOR
		require live: not is_released
		do Result := unary (Sigmoid_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	relu: TENSOR
		require live: not is_released
		do Result := unary (Relu_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	gelu: TENSOR
			-- Exact GELU: x·Φ(x) (erf form; PyTorch nn.GELU() default, rixgpt.py).
		require live: not is_released
		do Result := unary (Gelu_kind, 0.0) ensure same_shape: Result.shape ~ shape end

	silu: TENSOR
		require live: not is_released
		do Result := unary (Silu_kind, 0.0) ensure same_shape: Result.shape ~ shape end

feature -- Reductions

	sum (a_axis: INTEGER; a_keep: BOOLEAN): TENSOR
			-- Sum along `a_axis'; keep it as size 1 when `a_keep'.
		require
			axis_valid: 1 <= a_axis and a_axis <= shape.rank
			live: not is_released
		do
			Result := reduce_result (Sum_kind, a_axis, a_keep)
		ensure
			kept: a_keep implies (Result.shape ~ shape.with_dim (a_axis, 1))
			dropped: (not a_keep and shape.rank > 1) implies (Result.shape ~ shape.without_dim (a_axis))
			scalar_from_vector: (not a_keep and shape.rank = 1) implies Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	mean (a_axis: INTEGER; a_keep: BOOLEAN): TENSOR
		require
			axis_valid: 1 <= a_axis and a_axis <= shape.rank
			live: not is_released
		do
			Result := reduce_result (Mean_kind, a_axis, a_keep)
		ensure
			kept: a_keep implies (Result.shape ~ shape.with_dim (a_axis, 1))
			dropped: (not a_keep and shape.rank > 1) implies (Result.shape ~ shape.without_dim (a_axis))
			scalar_from_vector: (not a_keep and shape.rank = 1) implies Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	maximum (a_axis: INTEGER; a_keep: BOOLEAN): TENSOR
			-- Max along `a_axis' (gradient flows to the first maximum).
		require
			axis_valid: 1 <= a_axis and a_axis <= shape.rank
			live: not is_released
		do
			Result := reduce_result (Max_kind, a_axis, a_keep)
		ensure
			kept: a_keep implies (Result.shape ~ shape.with_dim (a_axis, 1))
			dropped: (not a_keep and shape.rank > 1) implies (Result.shape ~ shape.without_dim (a_axis))
			scalar_from_vector: (not a_keep and shape.rank = 1) implies Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	sum_all: TENSOR
		require
			live: not is_released
		do
			Result := reshaped (<<shape.count>>).sum (1, False)
		ensure
			scalar: Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
		end

	mean_all: TENSOR
		require
			live: not is_released
		do
			Result := reshaped (<<shape.count>>).mean (1, False)
		ensure
			scalar: Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
		end

	reduced_to (a_shape: TENSOR_SHAPE): TENSOR
			-- Sum over the axes along which `a_shape' broadcasts to `shape' (broadcast backward).
		require
			broadcastable: a_shape.broadcasts_with (shape) and then a_shape.broadcast_with (shape) ~ shape
			live: not is_released
		local
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_out := backend.new_storage (a_shape.count)
			backend.reduce_to_shape (storage, l_out, shape, a_shape)
			if is_tracked then
				create {TENSOR_OP_REDUCE_TO} l_op.make (Current, a_shape)
			end
			create Result.make_result (a_shape, l_out, backend, l_op)
		ensure
			shaped: Result.shape ~ a_shape
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

feature -- Shape

	reshaped (a_dims: ARRAY [INTEGER]): TENSOR
			-- Same data, dims `a_dims' (a free view: same storage).
		require
			at_least_one_dim: a_dims.count >= 1
			dims_positive: across a_dims as d all d >= 1 end
			not_too_large: shape.fits (a_dims)
			same_count: shape.product_of (a_dims) = shape.count
			live: not is_released
		local
			l_shape: TENSOR_SHAPE
			l_op: detachable TENSOR_OP
		do
			l_shape := shape.reshaped (a_dims)
			if is_tracked then
				create {TENSOR_OP_RESHAPE} l_op.make (Current, l_shape)
			end
			create Result.make_result (l_shape, storage, backend, l_op)
		ensure
			shares_storage: Result.storage = storage
			shaped: Result.shape ~ shape.reshaped (a_dims)
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
		end

	permuted (a_order: ARRAY [INTEGER]): TENSOR
			-- Dims reordered (a copy): result dim i = dim (a_order [i]).
		require
			permutation: shape.is_permutation (a_order)
			live: not is_released
		local
			l_shape: TENSOR_SHAPE
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_shape := shape.permuted (a_order)
			l_out := backend.new_storage (shape.count)
			backend.permute (storage, l_out, shape, a_order)
			if is_tracked then
				create {TENSOR_OP_PERMUTE} l_op.make (Current, a_order, l_shape)
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		ensure
			shaped: Result.shape ~ shape.permuted (a_order)
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	transposed: TENSOR
			-- Last two dims swapped (a copy).
		require
			rank_two_or_more: shape.rank >= 2
			live: not is_released
		local
			l_order: ARRAY [INTEGER]
			k: INTEGER
		do
			create l_order.make_filled (1, 1, shape.rank)
			from k := 1 until k > shape.rank loop
				l_order [k] := k
				k := k + 1
			end
			l_order [shape.rank - 1] := shape.rank
			l_order [shape.rank] := shape.rank - 1
			Result := permuted (l_order)
		ensure
			shaped: Result.shape ~ shape.with_dim (shape.rank - 1, shape.last).with_dim (shape.rank, shape.dim (shape.rank - 1))
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	sliced (a_axis, a_start, a_length: INTEGER): TENSOR
			-- Elements a_start .. a_start + a_length - 1 (1-based) along `a_axis' (a copy).
		require
			axis_valid: 1 <= a_axis and a_axis <= shape.rank
			range: a_start >= 1 and a_length >= 1 and a_start + a_length - 1 <= shape.dim (a_axis)
			live: not is_released
		local
			l_shape: TENSOR_SHAPE
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_shape := shape.with_dim (a_axis, a_length)
			l_out := backend.new_storage (l_shape.count)
			backend.slice (storage, l_out, shape, a_axis, a_start, a_length)
			if is_tracked then
				create {TENSOR_OP_SLICE} l_op.make (Current, a_axis, a_start, a_length, l_shape)
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		ensure
			shaped: Result.shape ~ shape.with_dim (a_axis, a_length)
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	concatenated (o: TENSOR; a_axis: INTEGER): TENSOR
			-- Current then `o' along `a_axis'.
		require
			same_backend: has_same_backend (o)
			same_rank: o.shape.rank = shape.rank
			axis_valid: 1 <= a_axis and a_axis <= shape.rank
			other_dims_agree: across 1 |..| shape.rank as i all i /= a_axis implies o.shape.dim (i) = shape.dim (i) end
			live: not is_released and not o.is_released
		local
			l_shape: TENSOR_SHAPE
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_shape := shape.with_dim (a_axis, shape.dim (a_axis) + o.shape.dim (a_axis))
			l_out := backend.new_storage (l_shape.count)
			backend.concat (storage, o.storage, l_out, shape, o.shape, a_axis)
			if is_tracked or o.is_tracked then
				create {TENSOR_OP_CONCAT} l_op.make (Current, o, a_axis, l_shape)
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		ensure
			shaped: Result.shape ~ shape.with_dim (a_axis, shape.dim (a_axis) + o.shape.dim (a_axis))
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage and Result.storage /= o.storage
		end

feature -- Linear algebra

	matmul (o: TENSOR): TENSOR
			-- Batched Current · `o'.
		require
			same_backend: has_same_backend (o)
			ranks: shape.rank >= 2 and o.shape.rank >= 2
			inner_dims_agree: shape.last = o.shape.dim (o.shape.rank - 1)
			batches_agree: o.shape.rank = 2 or else (o.shape.rank = shape.rank and then
				(shape.rank = 2 or else o.shape.without_last.without_last ~ shape.without_last.without_last))
			live: not is_released and not o.is_released
		do
			Result := matmul_transposed (o, False, False)
		ensure
			shaped: Result.shape ~ shape.with_dim (shape.rank, o.shape.last)
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
		end

	matmul_transposed (o: TENSOR; a_ta, a_tb: BOOLEAN): TENSOR
			-- Batched op(Current) · op(o); op transposes the last two dims when flagged.
			-- `o' of rank 2 is shared by every batch item.
		require
			same_backend: has_same_backend (o)
			ranks: shape.rank >= 2 and o.shape.rank >= 2
			inner_dims_agree: matmul_inner (a_ta) = o.matmul_outer (a_tb)
			batches_agree: o.shape.rank = 2 or else (o.shape.rank = shape.rank and then
				(shape.rank = 2 or else o.shape.without_last.without_last ~ shape.without_last.without_last))
			live: not is_released and not o.is_released
		local
			l_shape: TENSOR_SHAPE
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
			m, k, n, l_batch: INTEGER
		do
			m := matmul_rows (a_ta)
			k := matmul_inner (a_ta)
			n := o.matmul_cols (a_tb)
			l_batch := shape.count // (shape.dim (shape.rank - 1) * shape.last)
			l_shape := shape.with_dim (shape.rank - 1, m).with_dim (shape.rank, n)
			l_out := backend.new_storage (l_shape.count)
			backend.matmul (storage, o.storage, l_out, l_batch, m, k, n, a_ta, a_tb, o.shape.rank = 2)
			if is_tracked or o.is_tracked then
				create {TENSOR_OP_MATMUL} l_op.make (Current, o, a_ta, a_tb, l_shape)
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		ensure
			shaped: Result.shape ~ shape.with_dim (shape.rank - 1, matmul_rows (a_ta)).with_dim (shape.rank, o.matmul_cols (a_tb))
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage and Result.storage /= o.storage
		end

feature -- Neural

	softmax: TENSOR
			-- Softmax over the last axis.
		require
			live: not is_released
		do
			Result := softmax_result (False)
		ensure
			same_shape: Result.shape ~ shape
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	log_softmax: TENSOR
			-- Log-softmax over the last axis.
		require
			live: not is_released
		do
			Result := softmax_result (True)
		ensure
			same_shape: Result.shape ~ shape
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	masked_fill (a_mask: INDEX_TENSOR; a_value: REAL_32): TENSOR
			-- `a_value' where the broadcast mask is non-zero (e.g. -inf above the causal diagonal).
		require
			mask_broadcasts: shape.broadcasts_with (a_mask.shape) and then shape.broadcast_with (a_mask.shape) ~ shape
			live: not is_released
		local
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_out := backend.new_storage (shape.count)
			backend.masked_fill (storage, a_mask, l_out, shape, a_value)
			if is_tracked then
				create {TENSOR_OP_MASKED_FILL} l_op.make (Current, a_mask)
			end
			create Result.make_result (shape, l_out, backend, l_op)
		ensure
			same_shape: Result.shape ~ shape
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

	gathered_rows (a_indices: INDEX_TENSOR): TENSOR
			-- Embedding lookup on this rows × dim table: result [..., :] = row [index].
		require
			table: shape.rank = 2
			indices_in_range: a_indices.all_in_range (0, shape.dim (1) - 1)
			live: not is_released
		local
			l_shape: TENSOR_SHAPE
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_shape := a_indices.shape.appended (shape.last)
			l_out := backend.new_storage (l_shape.count)
			backend.gather_rows (storage, a_indices, l_out, shape.dim (1), shape.last)
			if is_tracked then
				create {TENSOR_OP_GATHER_ROWS} l_op.make (Current, a_indices, l_shape)
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		ensure
			shaped: Result.shape ~ a_indices.shape.appended (shape.last)
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
			fresh_storage: Result.storage /= storage
		end

feature -- Losses

	cross_entropy (a_targets: INDEX_TENSOR; a_ignore: INTEGER): TENSOR
			-- Mean over non-ignored positions of -log softmax (Current) [target].
			-- (Dropout is a composition, not an op: x * mask, with the mask filled
			-- by TENSOR_RANDOM.fill_keep_mask: verdict R2-04.)
		require
			logits_rank: shape.rank >= 2
			targets_shape: a_targets.shape ~ shape.without_last
			targets_in_range: a_targets.all_in_range_or (0, shape.last - 1, a_ignore)
			some_target: a_targets.has_value_other_than (a_ignore)
			live: not is_released
		local
			l_shape: TENSOR_SHAPE
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			create l_shape.make_scalar
			l_out := backend.new_storage (1)
			backend.cross_entropy (storage, a_targets, a_ignore, l_out, shape.count // shape.last, shape.last)
			if is_tracked then
				create {TENSOR_OP_CROSS_ENTROPY} l_op.make (Current, a_targets, a_ignore)
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		ensure
			scalar: Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
			not_a_parameter: not Result.requires_grad
		end

	mean_squared_error (o: TENSOR): TENSOR
		require
			same_backend: has_same_backend (o)
			same_shape: o.shape ~ shape
			live: not is_released and not o.is_released
		local
			l_shape: TENSOR_SHAPE
			l_diff, l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			create l_shape.make_scalar
			l_diff := backend.new_storage (shape.count)
			backend.binary (Sub_kind, storage, o.storage, l_diff, shape, o.shape, shape)
			l_out := backend.new_storage (1)
			backend.upload (<<(backend.sum_of_squares (l_diff) / shape.count).truncated_to_real>>, l_out)
			if is_tracked or o.is_tracked then
				create {TENSOR_OP_MSE} l_op.make (Current, o)
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		ensure
			scalar: Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
		end

feature -- Graph and device

	detached: TENSOR
			-- Same data, out of the graph.
		require
			live: not is_released
		do
			create Result.make_leaf (shape, storage, backend, False)
		ensure
			shares_storage: Result.storage = storage
			same_shape: Result.shape = shape
			untracked: not Result.is_tracked
		end

	zeros_like: TENSOR
			-- Untracked zeros of this shape and backend.
		do
			create Result.make_leaf (shape, backend.new_storage (shape.count), backend, False)
		ensure
			same_shape: Result.shape ~ shape
			same_backend: Result.backend = backend
			untracked: not Result.is_tracked
		end

	ones_like: TENSOR
			-- Untracked ones of this shape and backend.
		local
			l_out: TENSOR_STORAGE
		do
			l_out := backend.new_storage (shape.count)
			backend.fill (l_out, 1.0)
			create Result.make_leaf (shape, l_out, backend, False)
		ensure
			same_shape: Result.shape ~ shape
			same_backend: Result.backend = backend
			untracked: not Result.is_tracked
		end

	to_backend (a_backend: TENSOR_BACKEND): TENSOR
			-- A copy on `a_backend' (untracked: crossing devices ends the graph).
		require
			live: not is_released
		local
			l_out: TENSOR_STORAGE
		do
			l_out := a_backend.new_storage (shape.count)
			a_backend.upload (backend.download (storage), l_out)
			create Result.make_leaf (shape, l_out, a_backend, False)
		ensure
			same_shape: Result.shape ~ shape
			moved: Result.backend = a_backend
			untracked: not Result.is_tracked
		end

feature -- Autograd

	backward: GRAD_STORE
			-- Gradients of this scalar with respect to every leaf that requires them.
			-- Intermediates' gradients are discarded during the sweep (spec 07).
		require
			is_scalar: is_scalar
			tracked: is_tracked
			live: not is_released
		local
			l_order: ARRAYED_LIST [TENSOR]
			t: TENSOR
		do
			create Result.make
			l_order := post_order
			Result.accumulate (Current, ones_like)
			from l_order.finish until l_order.before loop
				t := l_order.item
				if attached t.origin as l_op and then Result.has (t) then
					l_op.backward (Result.gradient (t), Result)
					Result.discard (t)
				end
				l_order.back
				if l_order.index \\ 256 = 0 then
					gc_point
				end
			end
		ensure
			only_grad_leaves: Result.holds_only_grad_leaves
			leaves_shaped: Result.all_shapes_match
			exactly_the_reachable_leaves: Result.ids_model |=| grad_leaf_ids_model
			graph_intact: origin = old origin
		end

	grad_leaf_ids_model: MML_SET [INTEGER_64]
			-- Ids of the leaves that require gradients and are reachable from
			-- Current through `origin.inputs'.
		do
			create Result
			across post_order as t loop
				if t.requires_grad then
					Result := Result & t.id
				end
			end
		end

feature -- Memory

	release
			-- Give this tensor's memory back now (intent-v2 Q7; optional per the
			-- Phase 2 verdict). Refused while another tensor sees the storage
			-- (views, detached copies) or when this is a parameter. An op that
			-- saved this storage checks `saved_live' before its backward.
		require
			not_shared: not storage_shared
			not_a_parameter: not requires_grad
		do
			storage.release
		ensure
			released: is_released
		end

feature -- Matmul geometry

	matmul_rows (a_t: BOOLEAN): INTEGER
			-- Rows of op(Current).
		require
			rank_two_or_more: shape.rank >= 2
		do
			if a_t then Result := shape.last else Result := shape.dim (shape.rank - 1) end
		end

	matmul_inner (a_t: BOOLEAN): INTEGER
			-- Columns of op(Current) (the contracted dim when Current is the left operand).
		require
			rank_two_or_more: shape.rank >= 2
		do
			if a_t then Result := shape.dim (shape.rank - 1) else Result := shape.last end
		end

	matmul_outer (a_t: BOOLEAN): INTEGER
			-- Rows of op(Current) (the contracted dim when Current is the right operand).
		require
			rank_two_or_more: shape.rank >= 2
		do
			Result := matmul_rows (a_t)
		end

	matmul_cols (a_t: BOOLEAN): INTEGER
			-- Columns of op(Current).
		require
			rank_two_or_more: shape.rank >= 2
		do
			Result := matmul_inner (a_t)
		end

feature {NONE} -- Implementation

	binary_result (a_kind: INTEGER; o: TENSOR): TENSOR
			-- Current (op) `o' with broadcasting, tracked when an input is.
		local
			l_shape: TENSOR_SHAPE
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_shape := shape.broadcast_with (o.shape)
			l_out := backend.new_storage (l_shape.count)
			backend.binary (a_kind, storage, o.storage, l_out, shape, o.shape, l_shape)
			if is_tracked or o.is_tracked then
				inspect a_kind
				when Add_kind then create {TENSOR_OP_ADD} l_op.make (Current, o, l_shape)
				when Sub_kind then create {TENSOR_OP_SUB} l_op.make (Current, o, l_shape)
				when Mul_kind then create {TENSOR_OP_MUL} l_op.make (Current, o, l_shape)
				else create {TENSOR_OP_DIV} l_op.make (Current, o, l_shape)
				end
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		end

	reduce_result (a_kind: INTEGER; a_axis: INTEGER; a_keep: BOOLEAN): TENSOR
			-- Sum, mean or max along `a_axis'.
		local
			l_shape: TENSOR_SHAPE
			l_out, l_mask: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			if a_keep then
				l_shape := shape.with_dim (a_axis, 1)
			elseif shape.rank > 1 then
				l_shape := shape.without_dim (a_axis)
			else
				create l_shape.make_scalar
			end
			l_out := backend.new_storage (shape.count // shape.dim (a_axis))
			backend.reduce (a_kind, storage, l_out, shape, a_axis)
			if is_tracked then
				if a_kind = Max_kind then
					l_mask := backend.new_storage (shape.count)
					backend.argmax_mask (storage, l_mask, shape, a_axis)
					create {TENSOR_OP_MAX} l_op.make (Current, a_axis, a_keep, l_mask, l_shape)
				else
					create {TENSOR_OP_REDUCE} l_op.make (a_kind, Current, a_axis, a_keep, l_shape)
				end
			end
			create Result.make_result (l_shape, l_out, backend, l_op)
		end

	softmax_result (a_log: BOOLEAN): TENSOR
			-- (Log-)softmax over the last axis.
		local
			l_out: TENSOR_STORAGE
			l_op: detachable TENSOR_OP
		do
			l_out := backend.new_storage (shape.count)
			backend.softmax (storage, l_out, shape.count // shape.last, shape.last, a_log)
			if is_tracked then
				create {TENSOR_OP_SOFTMAX} l_op.make (Current, l_out, a_log)
			end
			create Result.make_result (shape, l_out, backend, l_op)
		end

	post_order: ARRAYED_LIST [TENSOR]
			-- Tracked tensors reachable from Current through `origin.inputs',
			-- each once, inputs before the tensors computed from them
			-- (iterative depth-first walk: no recursion depth limit).
		local
			l_stack: ARRAYED_STACK [TUPLE [node: TENSOR; visited: BOOLEAN]]
			l_seen: HASH_TABLE [BOOLEAN, INTEGER_64]
			l_top: TUPLE [node: TENSOR; visited: BOOLEAN]
		do
			create Result.make (64)
			create l_stack.make (64)
			create l_seen.make (64)
			l_stack.put ([Current, False])
			from until l_stack.is_empty loop
				l_top := l_stack.item
				l_stack.remove
				if l_top.visited then
					Result.extend (l_top.node)
				elseif not l_seen.has (l_top.node.id) then
					l_seen.put (True, l_top.node.id)
					l_stack.put ([l_top.node, True])
					if attached l_top.node.origin as l_op then
						across l_op.inputs as l_input loop
							if l_input.is_tracked and then not l_seen.has (l_input.id) then
								l_stack.put ([l_input, False])
							end
						end
					end
				end
				if Result.count \\ 256 = 0 then
					gc_point
				end
			end
		end

	next_id: INTEGER_64
			-- Next id from this thread's counter.
		do
			id_counter.put (id_counter.item + 1)
			Result := id_counter.item
		end

	id_counter: CELL [INTEGER_64]
			-- Per-thread counter (ISE `once' is per thread; each SCOOP processor is a thread).
		once
			create Result.put (0)
		end

invariant
	storage_fits: storage.count = shape.count
	leaf_has_no_origin: requires_grad implies origin = Void
	backend_owns_storage: backend.owns (storage)

end
