note
	description: "[
		Where and how kernels run (Burn pattern: autograd above, kernels here).
		Kernels are commands writing into a preallocated `out' storage, so a
		GPU backend can keep results on the device.

		Every effective backend must agree with REFERENCE_BACKEND within 1e-5
		relative on seeded random inputs (twin parity, I-002). That shared
		semantic postcondition is executed by TWIN_CHECKER over `Hot_kernels',
		because an Eiffel postcondition cannot name another backend without
		coupling them (Phase 2 verdict R2-05, R2-12).

		Unary derivatives (verdict R2-07), given input a and saved output y:
		  neg -1 | scale c | add_scalar 1 | exp y | log 1/a | sqrt 1/(2y) |
		  tanh 1-y^2 | sigmoid y(1-y) | relu [a>0] (relu'(0) = 0) |
		  gelu (exact, erf) Phi(a) + a*phi(a) | silu s + a*s*(1-s), s = sigmoid(a)
	]"
	author: "Larry Rix"

deferred class
	TENSOR_BACKEND

inherit
	TENSOR_OP_KINDS

	BACKEND_KINDS

feature -- Identity

	kind: INTEGER
			-- Reference_backend_kind, Cpu_backend_kind or Vulkan_backend_kind.
		deferred
		end

	name: STRING_8
		deferred
		end

	is_compatible (o: TENSOR_BACKEND): BOOLEAN
			-- Can tensors of `o' be operands together with tensors of Current?
		do
			Result := o.kind = kind
		end

	Hot_kernels: ARRAY [STRING_8]
			-- The kernels every fast backend redefines; the twin-parity roster.
		once
			Result := <<"binary", "unary", "unary_derivative", "reduce", "sum_of_squares", "has_non_finite",
				"matmul", "softmax", "gather_rows", "scatter_add_rows", "cross_entropy", "cross_entropy_grad", "adamw_update">>
		ensure
			thirteen: Result.count = 13
		end

feature -- Storage

	new_storage (a_count: INTEGER): TENSOR_STORAGE
			-- Zero-filled storage of `a_count' elements.
		require
			positive: a_count >= 1
			not_too_large: a_count <= {TENSOR_SHAPE}.Max_elements
		deferred
		ensure
			owned: owns (Result)
			counted: Result.count = a_count
			live: not Result.is_released
		end

	owns (s: TENSOR_STORAGE): BOOLEAN
			-- Can this backend's kernels run on `s'?
		deferred
		end

feature -- Transfer

	upload (a_values: ARRAY [REAL_32]; a_out: TENSOR_STORAGE)
			-- Copy `a_values' into `a_out'.
		require
			owned: owns (a_out)
			live: not a_out.is_released
			sized: a_values.count = a_out.count
		deferred
		ensure
			loaded: across 1 |..| a_values.count as ix all a_out.item (ix - 1) = a_values [a_values.lower + ix - 1] or a_values [a_values.lower + ix - 1] /= a_values [a_values.lower + ix - 1] end
		end

	download (a: TENSOR_STORAGE): ARRAY [REAL_32]
			-- The values of `a'.
		require
			owned: owns (a)
			live: not a.is_released
		deferred
		ensure
			sized: Result.count = a.count
			one_based: Result.lower = 1
		end

feature -- Kernels: fill and copy

	fill (a_out: TENSOR_STORAGE; a_value: REAL_32)
		require
			owned: owns (a_out)
			live: not a_out.is_released
		deferred
		end

	copy_values (a, a_out: TENSOR_STORAGE)
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			same_count: a.count = a_out.count
			distinct: a /= a_out
		deferred
		end

feature -- Kernels: elementwise

	binary (a_kind: INTEGER; a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b, a_shape_out: TENSOR_SHAPE)
			-- a_out := a (op) b with broadcasting.
		require
			valid_kind: is_binary_kind (a_kind)
			owned: owns (a) and owns (b) and owns (a_out)
			live: not a.is_released and not b.is_released and not a_out.is_released
			sized: a.count = a_shape_a.count and b.count = a_shape_b.count and a_out.count = a_shape_out.count
			broadcast: a_shape_a.broadcasts_with (a_shape_b) and then a_shape_out ~ a_shape_a.broadcast_with (a_shape_b)
			distinct_output: a_out /= a and a_out /= b
		deferred
		end

	unary (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_param: REAL_32)
			-- a_out := f (a); `a_param' is the constant of the scale and add-scalar kinds.
		require
			valid_kind: is_unary_kind (a_kind)
			param_unused: not uses_param (a_kind) implies a_param = 0.0
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			same_count: a.count = a_out.count
			distinct_output: a_out /= a
		deferred
		end

	unary_derivative (a_kind: INTEGER; a, a_y, a_out: TENSOR_STORAGE; a_param: REAL_32)
			-- a_out := f' (a), given the forward output `a_y' (table in the class note).
		require
			valid_kind: is_unary_kind (a_kind)
			param_unused: not uses_param (a_kind) implies a_param = 0.0
			owned: owns (a) and owns (a_y) and owns (a_out)
			live: not a.is_released and not a_y.is_released and not a_out.is_released
			same_count: a.count = a_out.count and a_y.count = a.count
			distinct_output: a_out /= a and a_out /= a_y
		deferred
		end

feature -- Kernels: reductions

	reduce (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
			-- a_out := Sum / Mean / Max of `a' along `a_axis' (that dim becomes 1).
		require
			valid_kind: is_reduce_kind (a_kind)
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			axis_valid: 1 <= a_axis and a_axis <= a_shape_a.rank
			sized: a.count = a_shape_a.count and a_out.count = a_shape_a.count // a_shape_a.dim (a_axis)
			distinct_output: a_out /= a
		deferred
		end

	argmax_mask (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
			-- a_out := 1 at the first maximum along `a_axis', 0 elsewhere.
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			axis_valid: 1 <= a_axis and a_axis <= a_shape_a.rank
			same_count: a.count = a_shape_a.count and a_out.count = a.count
			distinct_output: a_out /= a
		deferred
		end

	reduce_to_shape (a, a_out: TENSOR_STORAGE; a_from, a_to: TENSOR_SHAPE)
			-- a_out := sum of `a' over the axes along which `a_to' broadcasts to `a_from'.
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			broadcastable: a_to.broadcasts_with (a_from) and then a_to.broadcast_with (a_from) ~ a_from
			sized: a.count = a_from.count and a_out.count = a_to.count
			distinct_output: a_out /= a
		deferred
		end

	sum_of_squares (a: TENSOR_STORAGE): REAL_64
			-- Σ a², accumulated in REAL_64.
		require
			owned: owns (a)
			live: not a.is_released
		deferred
		ensure
			non_negative: Result >= 0.0 or Result /= Result
		end

	sum_of_squares_all (a_list: ARRAYED_LIST [TENSOR_STORAGE]): REAL_64
			-- Σ over the storages of Σ a², accumulated in REAL_64 (a gradient's global norm).
			-- One host read on backends that batch it (VULKAN_BACKEND); here, one per storage.
		require
			owned: across a_list as s all owns (s) end
			live: across a_list as s all not s.is_released end
		do
			across a_list as s loop
				Result := Result + sum_of_squares (s)
			end
		ensure
			non_negative: Result >= 0.0 or Result /= Result
		end

	has_non_finite (a: TENSOR_STORAGE): BOOLEAN
			-- Does `a' hold a NaN or an infinity?
		require
			owned: owns (a)
			live: not a.is_released
		deferred
		end

feature -- Kernels: linear algebra

	matmul (a, b, a_out: TENSOR_STORAGE; a_batch, a_m, a_k, a_n: INTEGER; a_ta, a_tb, a_b_shared: BOOLEAN)
			-- For each batch item: out (m×n) := op(A) (m×k) · op(B) (k×n),
			-- op = transpose when flagged; `a_b_shared': one B for every batch item.
		require
			owned: owns (a) and owns (b) and owns (a_out)
			live: not a.is_released and not b.is_released and not a_out.is_released
			positive: a_batch >= 1 and a_m >= 1 and a_k >= 1 and a_n >= 1
			a_sized: a.count = a_batch * a_m * a_k
			b_sized: (a_b_shared implies b.count = a_k * a_n) and (not a_b_shared implies b.count = a_batch * a_k * a_n)
			out_sized: a_out.count = a_batch * a_m * a_n
			distinct_output: a_out /= a and a_out /= b
		deferred
		end

feature -- Kernels: neural

	softmax (a, a_out: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_log: BOOLEAN)
			-- Row-wise (log-)softmax of an a_rows × a_cols block.
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			sized: a.count = a_rows * a_cols and a_out.count = a.count
			distinct_output: a_out /= a
		deferred
		end

	masked_fill (a: TENSOR_STORAGE; a_mask: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_value: REAL_32)
			-- a_out := `a_value' where the (broadcast) mask is non-zero, else a.
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			mask_broadcasts: a_shape_a.broadcasts_with (a_mask.shape) and then a_shape_a.broadcast_with (a_mask.shape) ~ a_shape_a
			sized: a.count = a_shape_a.count and a_out.count = a.count
			distinct_output: a_out /= a
		deferred
		end

feature -- Kernels: shape

	permute (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_order: ARRAY [INTEGER])
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			permutation: a_shape_a.is_permutation (a_order)
			sized: a.count = a_shape_a.count and a_out.count = a.count
			distinct_output: a_out /= a
		deferred
		end

	slice (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
			-- a_out := elements a_start .. a_start + a_length - 1 (1-based) along `a_axis'.
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			axis_valid: 1 <= a_axis and a_axis <= a_shape_a.rank
			range: a_start >= 1 and a_length >= 1 and a_start + a_length - 1 <= a_shape_a.dim (a_axis)
			sized: a.count = a_shape_a.count and a_out.count = (a_shape_a.count // a_shape_a.dim (a_axis)) * a_length
			distinct_output: a_out /= a
		deferred
		end

	add_slice (a_part, a_out: TENSOR_STORAGE; a_shape_out: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
			-- a_out [slice] += a_part (slice backward).
		require
			owned: owns (a_part) and owns (a_out)
			live: not a_part.is_released and not a_out.is_released
			axis_valid: 1 <= a_axis and a_axis <= a_shape_out.rank
			range: a_start >= 1 and a_length >= 1 and a_start + a_length - 1 <= a_shape_out.dim (a_axis)
			sized: a_out.count = a_shape_out.count and a_part.count = (a_shape_out.count // a_shape_out.dim (a_axis)) * a_length
			distinct_output: a_out /= a_part
		deferred
		end

	concat (a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b: TENSOR_SHAPE; a_axis: INTEGER)
		require
			owned: owns (a) and owns (b) and owns (a_out)
			live: not a.is_released and not b.is_released and not a_out.is_released
			same_rank: a_shape_a.rank = a_shape_b.rank
			axis_valid: 1 <= a_axis and a_axis <= a_shape_a.rank
			sized: a.count = a_shape_a.count and b.count = a_shape_b.count and a_out.count = a.count + b.count
			distinct_output: a_out /= a and a_out /= b
		deferred
		end

feature -- Kernels: indexing

	gather_rows (a_table: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
			-- a_out [i, :] := table [indices [i], :].
		require
			owned: owns (a_table) and owns (a_out)
			live: not a_table.is_released and not a_out.is_released
			table_sized: a_table.count = a_rows * a_dim
			indices_in_range: a_indices.all_in_range (0, a_rows - 1)
			out_sized: a_out.count = a_indices.count * a_dim
			distinct_output: a_out /= a_table
		deferred
		end

	scatter_add_rows (a_grad: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
			-- a_out [indices [i], :] += grad [i, :] (embedding backward; deterministic order).
		require
			owned: owns (a_grad) and owns (a_out)
			live: not a_grad.is_released and not a_out.is_released
			out_sized: a_out.count = a_rows * a_dim
			indices_in_range: a_indices.all_in_range (0, a_rows - 1)
			grad_sized: a_grad.count = a_indices.count * a_dim
			distinct_output: a_out /= a_grad
		deferred
		end

feature -- Kernels: losses

	cross_entropy (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER)
			-- a_out [0] := mean over non-ignored rows of -log softmax (logits [r]) [target r].
		require
			owned: owns (a_logits) and owns (a_out)
			live: not a_logits.is_released and not a_out.is_released
			sized: a_logits.count = a_rows * a_classes and a_targets.count = a_rows and a_out.count = 1
			targets_in_range: a_targets.all_in_range_or (0, a_classes - 1, a_ignore)
			some_target: a_targets.has_value_other_than (a_ignore)
			distinct_output: a_out /= a_logits
		deferred
		end

	cross_entropy_grad (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER; a_scale: REAL_32)
			-- a_out := a_scale · (softmax − onehot (target)) / n_valid, 0 on ignored rows.
		require
			owned: owns (a_logits) and owns (a_out)
			live: not a_logits.is_released and not a_out.is_released
			sized: a_logits.count = a_rows * a_classes and a_targets.count = a_rows and a_out.count = a_logits.count
			targets_in_range: a_targets.all_in_range_or (0, a_classes - 1, a_ignore)
			some_target: a_targets.has_value_other_than (a_ignore)
			distinct_output: a_out /= a_logits
		deferred
		end

feature -- Kernels: fused (memory: one output instead of a chain of temporaries)

	causal_softmax (a, a_out: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_scale: REAL_32)
			-- Rows of `a_cols' scores, in blocks of `a_cols' rows (square T x T matrices);
			-- row r is query i = r \\ a_cols. For c <= i: out [r, c] = softmax over c' <= i of
			-- a_scale * a [r, c']; for c > i (future positions): 0.
		require
			owned: owns (a) and owns (a_out)
			live: not a.is_released and not a_out.is_released
			sized: a.count = a_rows * a_cols and a_out.count = a.count
			square_blocks: a_cols >= 1 and a_rows \\ a_cols = 0
		deferred
		end

	add_bias (a_out, a_bias: TENSOR_STORAGE; a_rows, a_cols: INTEGER)
			-- In place: out [r, c] := out [r, c] + bias [c] (a linear layer's epilogue).
		require
			owned: owns (a_out) and owns (a_bias)
			live: not a_out.is_released and not a_bias.is_released
			sized: a_out.count = a_rows * a_cols and a_bias.count = a_cols
			distinct: a_out /= a_bias
		deferred
		end

	layer_norm (a, a_weight, a_bias, a_out, a_mean, a_rstd: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_eps: REAL_64)
			-- Per row: mean, rstd = 1 / sqrt (biased variance + a_eps);
			-- out = (a - mean) * rstd * weight + bias. Row statistics go to `a_mean', `a_rstd'.
		require
			owned: owns (a) and owns (a_weight) and owns (a_bias) and owns (a_out) and owns (a_mean) and owns (a_rstd)
			live: not a.is_released and not a_out.is_released and not a_mean.is_released and not a_rstd.is_released
			sized: a.count = a_rows * a_cols and a_out.count = a.count
			params_sized: a_weight.count = a_cols and a_bias.count = a_cols
			stats_sized: a_mean.count = a_rows and a_rstd.count = a_rows
			eps_positive: a_eps > 0.0
		deferred
		end

feature -- Kernels: random fill (gap G-6)

	fill_random (a_out: TENSOR_STORAGE; a_kind: INTEGER; p1, p2: REAL_64; a_seed, a_position: NATURAL_64)
			-- Fill from Philox4x32-10 keyed by `a_seed', stream 0, starting at word `a_position'.
			-- Words are addressed, so every backend gives element i the same word(s):
			--   Uniform_fill_kind: p1 + (p2 - p1) * u, u = (word (p + i) >> 8) / 2^24, in [p1, p2]
			--   Normal_fill_kind: mean p1, std p2, Box-Muller from words p + 2i, p + 2i + 1:
			--     u1 = ((w1 >> 8) + 1) / 2^24 in (0, 1], u2 = (w2 >> 8) / 2^24
			--   Keep_mask_fill_kind: 1/p1 when word (p + i) < ceiling (p1 * 2^32), else 0
			--     (integer-exact: masks are bit-identical on every backend)
		require
			owned: owns (a_out)
			live: not a_out.is_released
			kind_known: is_fill_kind (a_kind)
			uniform_ordered: a_kind = Uniform_fill_kind implies p1 <= p2
			normal_spread: a_kind = Normal_fill_kind implies p2 >= 0.0
			keep_probability: a_kind = Keep_mask_fill_kind implies (0.0 < p1 and p1 <= 1.0)
		deferred
		end

feature -- Kernels: optimizers (in place on the parameter)

	sgd_update (a_param, a_grad: TENSOR_STORAGE; a_rate: REAL_64)
			-- param := param − rate · grad.
		require
			owned: owns (a_param) and owns (a_grad)
			live: not a_param.is_released and not a_grad.is_released
			same_count: a_param.count = a_grad.count
			distinct: a_param /= a_grad
		deferred
		end

	adamw_update (a_param, a_grad, a_m, a_v: TENSOR_STORAGE; a_rate, a_beta1, a_beta2, a_eps, a_decay: REAL_64; a_step: INTEGER)
			-- torch.optim.AdamW: param := param·(1 − rate·decay); m, v updated;
			-- param := param − rate · m̂ / (√v̂ + eps), bias-corrected with `a_step'.
		require
			owned: owns (a_param) and owns (a_grad) and owns (a_m) and owns (a_v)
			live: not a_param.is_released and not a_grad.is_released and not a_m.is_released and not a_v.is_released
			same_count: a_grad.count = a_param.count and a_m.count = a_param.count and a_v.count = a_param.count
			distinct: a_param /= a_grad and a_m /= a_v and a_m /= a_param and a_v /= a_param
			step_positive: a_step >= 1
		deferred
		end

end
