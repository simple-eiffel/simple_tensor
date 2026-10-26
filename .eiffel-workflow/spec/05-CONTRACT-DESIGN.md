# CONTRACT DESIGN: simple_tensor

Rules applied:
- **Invariants O(1)** (Larry 2026-09-11). No clause walks tensor data or builds an MML model of it.
- **Data-dependent checks** (index ranges, finiteness) are preconditions that DbC builds evaluate and lean builds skip, written as calls to `{GC_POINT}`-aware validator queries.
- **MML** is used only where the collection is small and the check is a postcondition: shape dims (MML_SEQUENCE [INTEGER]), the GRAD_STORE key set (MML_SET [INTEGER_64]), module parameter lists (MML_SEQUENCE [TENSOR]).

## MML Model Queries
| Attribute | Type | Model Query | MML Type |
|-----------|------|-------------|----------|
| TENSOR_SHAPE.dims | SPECIAL [INTEGER] | dims_model | MML_SEQUENCE [INTEGER] |
| GRAD_STORE.table | HASH_TABLE [TENSOR, INTEGER_64] | ids_model | MML_SET [INTEGER_64] |
| NN_MODULE parameters | ARRAYED_LIST [TENSOR] | parameters_model | MML_SEQUENCE [TENSOR] |
| ADAMW_OPTIMIZER state | HASH_TABLE [TENSOR (m), INTEGER_64] | state_ids_model | MML_SET [INTEGER_64] |

## Class Contracts

### TENSOR_SHAPE (immutable)
```eiffel
make (a_dims: ARRAY [INTEGER])
	require
		at_least_one_dim: a_dims.count >= 1
		dims_positive: across a_dims as d all d >= 1 end
	ensure
		rank_set: rank = a_dims.count
		dims_copied: across 1 |..| rank as i all dim (i) = a_dims [a_dims.lower + i - 1] end
		count_is_product: count = product (a_dims)

broadcast_with (o: TENSOR_SHAPE): TENSOR_SHAPE
	require
		compatible: broadcasts_with (o)
	ensure
		rank_is_max: Result.rank = rank.max (o.rank)
		-- trailing-aligned: each result dim = the non-1 dim (or 1)

reshaped (a_dims: ARRAY [INTEGER]): TENSOR_SHAPE
	require
		same_count: product (a_dims) = count
	ensure
		same_count: Result.count = count

invariant
	rank_positive: rank >= 1
	count_positive: count >= 1      -- cached at creation; O(1)
```

### TENSOR
```eiffel
invariant
	storage_fits: storage.count = shape.count                -- DR-008, O(1)
	leaf_has_no_origin: requires_grad implies origin = Void  -- DR-007
	tracked_definition: is_tracked = (requires_grad or origin /= Void)
	backend_owns_storage: backend.owns (storage)

-- Every binary elementwise op (+, -, *, /)
plus alias "+" (o: TENSOR): TENSOR
	require
		same_backend: o.backend = backend                      -- DR-003
		shapes_broadcast: shape.broadcasts_with (o.shape)      -- DR-001
	ensure
		shape_broadcast: Result.shape ~ shape.broadcast_with (o.shape)
		new_tensor: Result /= Current and Result /= o
		tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
		leaf_status: not Result.requires_grad
		origin_iff_tracked: (Result.origin /= Void) = Result.is_tracked
		inputs_unchanged: -- storage of Current and o not modified (frame; checked by twin tests)

matmul_transposed (o: TENSOR; a_ta, a_tb: BOOLEAN): TENSOR
	require
		same_backend: o.backend = backend
		ranks: shape.rank >= 2 and o.shape.rank >= 2
		inner_dims_agree: inner_dim (a_ta) = o.outer_dim (a_tb)   -- DR-002
		batch_dims_broadcast: batch_shape.broadcasts_with (o.batch_shape)
	ensure
		rows: Result.shape.dim (Result.shape.rank - 1) = rows_after (a_ta)
		cols: Result.shape.last = o.cols_after (a_tb)
		tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)

sum (a_axis: INTEGER; a_keep: BOOLEAN): TENSOR
	require
		axis_valid: 1 <= a_axis and a_axis <= shape.rank
	ensure
		reduced: a_keep implies Result.shape.dim (a_axis) = 1
		dropped: (not a_keep and shape.rank > 1) implies Result.shape.rank = shape.rank - 1

reshape (a_dims: ARRAY [INTEGER]): TENSOR
	require
		same_count: shape.product_of (a_dims) = shape.count
	ensure
		shares_storage: Result.storage = storage    -- free view
		shape_set: Result.shape.count = shape.count

cross_entropy (a_targets: INDEX_TENSOR; a_ignore: INTEGER): TENSOR
	require
		logits_rank: shape.rank >= 2
		targets_shape: a_targets.shape ~ shape.without_last        -- B×T for B×T×V
		targets_in_range: a_targets.all_in_range_or (0, shape.last - 1, a_ignore)  -- DR-009
	ensure
		scalar: Result.is_scalar

embedding (a_table: TENSOR; a_indices: INDEX_TENSOR)    -- on SIMPLE_TENSOR / NN_EMBEDDING
	require
		table_rank: a_table.shape.rank = 2
		indices_in_range: a_indices.all_in_range (0, a_table.shape.dim (1) - 1)
	ensure
		shape: Result.shape ~ a_indices.shape.appended (a_table.shape.dim (2))

detached: TENSOR
	ensure
		shares_storage: Result.storage = storage
		untracked: not Result.is_tracked

backward: GRAD_STORE
	require
		is_scalar: is_scalar                                     -- DR-004
		tracked: is_tracked
	ensure
		-- Intermediate gradients are discarded during the sweep (07 "Backward"), so the store holds leaves only.
		only_grad_leaves: Result.holds_only_grad_leaves
		-- I-003 / DR-006: every grad-requiring leaf reached has a gradient of its own shape
		leaves_shaped: Result.all_shapes_match
		no_tensor_changed: -- tensors and storages unchanged (frame; twin and gradient tests)
```

### TENSOR_OP (deferred)
```eiffel
inputs: ARRAY [TENSOR]
output_shape: TENSOR_SHAPE

backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		-- Add this op's contribution to each tracked input's gradient.
	require
		grad_shaped: a_output_grad.shape ~ output_shape
		grad_untracked: not a_output_grad.is_tracked        -- no higher-order graphs in M1
	deferred
	ensure
		only_tracked_inputs_touched: across inputs as i all (not i.is_tracked) implies
			(a_store.has (i) = old a_store.has (i)) end
		tracked_inputs_have_gradients: across inputs as i all i.is_tracked implies a_store.has (i) end

invariant
	has_inputs: inputs.count >= 1
```

### GRAD_STORE
```eiffel
accumulate (a_tensor: TENSOR; a_grad: TENSOR)
	require
		shaped: a_grad.shape ~ a_tensor.shape                   -- DR-006
		same_backend: a_grad.backend = a_tensor.backend
	ensure
		has_it: has (a_tensor)
		others_unchanged: ids_model |=| (old ids_model & a_tensor.id)
		summed: old has (a_tensor) implies -- gradient = old gradient + a_grad (twin-tested)

gradient (a_tensor: TENSOR): TENSOR
	require
		present: has (a_tensor)
	ensure
		shaped: Result.shape ~ a_tensor.shape
		untracked: not Result.is_tracked

global_norm: REAL_64            -- sqrt of the sum of squares over all gradients
	ensure
		non_negative: Result >= 0.0

scale_all (a_factor: REAL_32)    -- command (clipping, accumulation)
	require
		not_nan: a_factor = a_factor
	ensure
		same_keys: ids_model |=| old ids_model

clip_global_norm (a_max: REAL_64)
	require
		positive: a_max > 0.0
	ensure
		bounded: global_norm <= a_max * (1.0 + 1.0e-5)
		same_keys: ids_model |=| old ids_model
```

### TENSOR_BACKEND kernels (deferred; every effective backend inherits these)
```eiffel
matmul (a, b, out: TENSOR_STORAGE; a_m, a_k, a_n, a_batch: INTEGER; a_ta, a_tb: BOOLEAN)
	require
		owned: owns (a) and owns (b) and owns (out)
		sizes: a.count = a_batch * a_m * a_k and b.count >= a_k * a_n and out.count = a_batch * a_m * a_n
		distinct_output: out /= a and out /= b
	deferred
	ensure
		inputs_unchanged: -- frame (twin-tested)
```
Twin parity is the semantic postcondition shared by every backend: for random inputs, `CPU.k (x) ≈ REFERENCE.k (x)` within 1e-5 relative. It's executed by TWIN_CHECKER, since an Eiffel postcondition can't name another backend without coupling them.

### NN_MODULE
```eiffel
parameters: ARRAYED_LIST [TENSOR]
	ensure
		all_leaves: across Result as p all p.origin = Void end
		-- each parameter appears once (tied weights are shared, not duplicated)

freeze
	ensure
		all_frozen: across parameters as p all not p.requires_grad end
		same_parameters: parameters_model |=| old parameters_model

set_training (a_on: BOOLEAN)
	ensure
		set: is_training = a_on
		children_set: across children as c all c.is_training = a_on end
```

### NN_LINEAR (NN_LAYER)
```eiffel
forward (x: TENSOR): TENSOR
	require
		in_features: x.shape.last = in_features
		same_backend: x.backend = weight.backend
	ensure
		out_features: Result.shape.last = out_features
		leading_dims_kept: Result.shape.without_last ~ x.shape.without_last
invariant
	weight_shape: weight.shape.rank = 2 and weight.shape.dim (1) = out_features and weight.shape.dim (2) = in_features
	bias_shape: attached bias as b implies (b.shape.rank = 1 and b.shape.last = out_features)
```

### NN_LORA_LINEAR
```eiffel
make (a_base: NN_LINEAR; a_rank: INTEGER; a_alpha: REAL_64; a_random: TENSOR_RANDOM)
	require
		rank_positive: a_rank >= 1 and a_rank <= a_base.in_features.min (a_base.out_features)
	ensure
		base_frozen: not a_base.weight.requires_grad
		b_zero_initially: -- forward = base forward exactly at creation (B = 0)
		only_adapters_train: parameters_model |=| << lora_a, lora_b >>

merge
	require
		not_merged: not is_merged
	ensure
		merged: is_merged
		-- base.weight = old base.weight + scale · B·A (twin-tested)
```

### ADAMW_OPTIMIZER
```eiffel
make (a_params: ARRAYED_LIST [TENSOR]; a_schedule: LR_SCHEDULE; a_beta1, a_beta2, a_eps, a_weight_decay: REAL_64)
	require
		betas: 0.0 <= a_beta1 and a_beta1 < 1.0 and 0.0 <= a_beta2 and a_beta2 < 1.0
		eps_positive: a_eps > 0.0
		decay_non_negative: a_weight_decay >= 0.0
		params_are_leaves: across a_params as p all p.requires_grad end
	ensure
		step_zero: step_count = 0

step (a_grads: GRAD_STORE)
		-- One AdamW update of every parameter that has a gradient in `a_grads'.
	ensure
		counted: step_count = old step_count + 1
		-- updated in place: param := param - lr·(m̂/(√v̂+eps) + wd·param)  (hand-trace test)
		params_without_grad_unchanged: -- frame (test)
invariant
	state_matches_params: state_ids_model.count <= parameters.count
```
PyTorch semantics (the M2 oracle): decay applied as `p := p·(1 - lr·wd)` before the Adam update, bias-corrected m̂, v̂. The test reproduces torch.optim.AdamW's documented algorithm on a 3-step hand trace.

### WARMUP_COSINE_SCHEDULE
```eiffel
rate (a_step: INTEGER): REAL_64
	require
		non_negative: a_step >= 0
	ensure
		warmup: a_step < warmup implies Result = peak * a_step / warmup.max (1)
		floor: a_step >= total implies Result = floor_ratio * peak
		-- otherwise floor + 0.5·(peak - floor)·(1 + cos(π·r)), r = (step - warmup)/(total - warmup)
```
This is rixgpt.py's `lr_at` exactly (floor_ratio 0.1).

### TOKEN_BATCHER
```eiffel
make (a_tokens: INDEX_TENSOR; a_block, a_batch: INTEGER; a_seed: INTEGER)
	require
		long_enough: a_tokens.count > a_block + 1
	ensure
		deterministic_start: -- same seed ⇒ same sequence of batches (test)

next_batch: TOKEN_BATCH      -- command-query exception: advances RNG (documented, like ITERATION_CURSOR.forth + item)
	ensure
		shapes: Result.inputs.shape.dims_model ~ << batch, block >> and Result.targets.shape ~ Result.inputs.shape
		shifted_by_one: -- targets[b, t] = tokens[start_b + t + 1] = inputs[b, t+1] (test)
```
CQS: split into `forth` (command: draws the next starts) and `batch: TOKEN_BATCH` (query).

### SAFETENSORS_FILE
```eiffel
load (a_path: READABLE_STRING_GENERAL; a_backend: TENSOR_BACKEND)
	ensure
		loaded_or_error: is_loaded xor has_error
		names_from_header: is_loaded implies names_model ~ header_names_model

tensor (a_name: READABLE_STRING_GENERAL): TENSOR
	require
		loaded: is_loaded
		known: has (a_name)

save (a_path: READABLE_STRING_GENERAL; a_named: ARRAYED_LIST [TUPLE [name: STRING_8; tensor: TENSOR]])
	require
		names_unique: -- no duplicate names
	ensure
		round_trip: -- load (a_path) gives byte-identical F32 data (test)
```

## Contract Completeness Checklist
- [x] **What changed?** Ops produce a new tensor (shape and tracking stated); optimizers advance step_count and parameters; GRAD_STORE gains keys.
- [x] **How did it change?** Shape formulas exact; tracking stated; numeric effects via hand-trace and twin tests where an Eiffel postcondition would be O(n) or cross-backend.
- [x] **What did NOT change?** Inputs (frame, twin-tested); GRAD_STORE `ids_model |=|`; module parameter lists `parameters_model |=|`.

## Op semantics (the gradient-check spec)
y = output, g = output gradient. Broadcast inputs receive `reduce_to_shape (grad, input.shape)` (sum over broadcast axes).

| Op | Forward | Backward to inputs |
|----|---------|--------------------|
| add | a + b | g, g |
| sub | a − b | g, −g |
| mul | a·b | g·b, g·a |
| div | a/b | g/b, −g·a/b² |
| neg, scale c, add_scalar c | −a, c·a, a+c | −g, c·g, g |
| exp | eᵃ | g·y |
| log | ln a | g/a |
| sqrt | √a | g/(2y) |
| tanh | tanh a | g·(1−y²) |
| sigmoid | σ(a) | g·y·(1−y) |
| relu | max(0,a) | g·[a>0] |
| gelu (exact) | a·Φ(a) | g·(Φ(a) + a·φ(a)) |
| silu | a·σ(a) | g·(σ + a·σ(1−σ)) |
| sum(axis) | Σ | broadcast g back |
| mean(axis) | Σ/n | broadcast g/n |
| max(axis) | max | g to the argmax position (first on ties) |
| reshape | view | reshape g back |
| permute(p) | copy | permute g by p⁻¹ |
| slice | copy | g into zeros at the slice |
| concat | copy | split g |
| matmul (ta, tb) | op(A)·op(B) | dA = g·op(B)ᵀ (transposed as flagged), dB = op(A)ᵀ·g; batch-broadcast reduced |
| softmax | eᵃ/Σeᵃ | y·(g − Σ(g·y)) |
| log_softmax | a − logΣeᵃ | g − softmax·Σg |
| masked_fill (mask, v) | mask ? v : a | g·(1−mask) |
| dropout (p, mask) | a·mask/(1−p) | g·mask/(1−p) |
| embedding | rows | scatter-add g into table rows |
| cross_entropy | mean over non-ignored of −log softmax(a)[t] | (softmax − onehot(t))/n_valid, 0 for ignored |
| mse | mean (a−b)² | 2(a−b)/n, −2(a−b)/n |
