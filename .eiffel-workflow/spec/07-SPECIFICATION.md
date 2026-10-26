# SPECIFICATION: simple_tensor

## Overview
Tensor + autograd + training for Eiffel (research/07, BUILD).
- **This spec fixes M1:** core, CPU and REFERENCE backends, 33 primitives, autograd, optimizers, I/O, batcher.
- **It fixes the seams for M2** (modules for RixGPT) **and M3** (VULKAN_BACKEND behind TENSOR_BACKEND).
- Class texts below are normative for Phase 1 (`/eiffel.contracts`). Bodies marked `-- (Phase 4)` are left to implementation.

## Key algorithms

### Backward (graph in tensors, A-S02)
1. **Order:** iterative depth-first walk from the loss over `origin.inputs`, visiting each tensor id once (HASH_TABLE), emitting post-order. No recursion, so deep graphs (20 layers × 4 lanes in Needle) can't overflow the stack. A GC point every 256 nodes.
2. **Seed:** `store.accumulate (loss, ones_like (loss))`.
3. **Sweep:** in reverse post-order, for each tensor `t` with an origin and a gradient in the store, run `t.origin.backward (store.gradient (t), store)`, then discard `t`'s gradient (intermediates are as large as activations).
4. **Result:** the store holds gradients for the **leaves that require gradients** only.

### Backward formulas are written with the TENSOR API itself
Each `TENSOR_OP.backward` computes its input gradients with ordinary TENSOR ops on **untracked** tensors. The output gradient is untracked (precondition), so no graph is built during backward. Consequences:
- Every backend that implements the forward kernels gets backward for free.
- Only four kernels exist purely for backward: `reduce_to_shape` (broadcast sum), `scatter_add_rows` (embedding), `argmax_mask` (max), `cross_entropy_grad` (fused).

### IDs
TENSOR.id comes from a per-thread `once` counter. ISE's default `once` is per thread, and each SCOOP processor is a thread. Graphs never cross processors (tensors are not `separate`), so ids are unique within every graph and GRAD_STORE.

## Class Specifications

### GC_POINT (mixin)
```eiffel
note
	description: "[
		An empty `C blocking' external. A long Eiffel loop calls `gc_point'
		every few milliseconds so a collection pending on another processor
		can proceed (simple_tensor spike S-3: 3,129 ms stall without it,
		6 ms with it, no speed cost).
	]"

class
	GC_POINT

feature {NONE} -- GC

	gc_point
			-- Let a pending collection pass this thread.
		external
			"C blocking inline"
		alias
			"/* GC point */"
		end

end
```

### TENSOR_SHAPE
```eiffel
note
	description: "Immutable tensor dimensions, outermost first (row-major)."

class
	TENSOR_SHAPE

inherit
	ANY
		redefine
			is_equal
		end

create
	make, make_scalar

feature {NONE} -- Initialization

	make (a_dims: ARRAY [INTEGER])
			-- Shape with dimensions `a_dims', outermost first.
		require
			at_least_one_dim: a_dims.count >= 1
			dims_positive: across a_dims as d all d >= 1 end
		do
			-- (Phase 4) copy into `dims', cache `count'
		ensure
			rank_set: rank = a_dims.count
			dims_copied: across 1 |..| rank as i all dim (i) = a_dims [a_dims.lower + i - 1] end
		end

	make_scalar
			-- Shape [1].
		do
			-- (Phase 4)
		ensure
			scalar: rank = 1 and count = 1
		end

feature -- Access

	rank: INTEGER
			-- Number of dimensions.

	count: INTEGER
			-- Number of elements (product of dims), cached.

	dim (i: INTEGER): INTEGER
			-- Size of dimension `i' (1 = outermost).
		require
			valid: 1 <= i and i <= rank
		do
			-- (Phase 4)
		ensure
			positive: Result >= 1
		end

	last: INTEGER
			-- Innermost dimension.
		do
			Result := dim (rank)
		end

	dims_model: MML_SEQUENCE [INTEGER]
			-- Model of the dimensions.
		do
			-- (Phase 4)
		ensure
			counted: Result.count = rank
		end

feature -- Status report

	broadcasts_with (o: TENSOR_SHAPE): BOOLEAN
			-- Numpy rule: trailing-aligned dims are equal or one of them is 1.
		do
			-- (Phase 4)
		end

	is_equal (o: like Current): BOOLEAN
			-- Same rank and dims.
		do
			-- (Phase 4)
		ensure then
			same_dims: Result = (dims_model ~ o.dims_model)
		end

feature -- Derived shapes

	broadcast_with (o: TENSOR_SHAPE): TENSOR_SHAPE
		require
			compatible: broadcasts_with (o)
		do
			-- (Phase 4)
		ensure
			rank_is_max: Result.rank = rank.max (o.rank)
		end

	reshaped (a_dims: ARRAY [INTEGER]): TENSOR_SHAPE
		require
			same_count: product_of (a_dims) = count
		do
			create Result.make (a_dims)
		ensure
			same_count: Result.count = count
		end

	without_last: TENSOR_SHAPE
		require
			rank_above_one: rank >= 2
		do
			-- (Phase 4)
		ensure
			one_less: Result.rank = rank - 1
		end

	appended (a_dim: INTEGER): TENSOR_SHAPE
		require
			positive: a_dim >= 1
		do
			-- (Phase 4)
		ensure
			one_more: Result.rank = rank + 1 and Result.last = a_dim
		end

	product_of (a_dims: ARRAY [INTEGER]): INTEGER
			-- Product of `a_dims'.
		do
			-- (Phase 4)
		end

feature {NONE} -- Implementation

	dims: SPECIAL [INTEGER]

invariant
	rank_positive: rank >= 1
	count_positive: count >= 1
	dims_sized: dims.count = rank

end
```

### TENSOR_STORAGE (deferred) and CPU_STORAGE
```eiffel
deferred class
	TENSOR_STORAGE

feature -- Access

	count: INTEGER
			-- Number of REAL_32 elements.
		deferred
		end

	item (i: INTEGER): REAL_32
			-- Element `i' (0-based). Slow path: tests and small data.
		require
			valid: 0 <= i and i < count
		deferred
		end

feature {TENSOR_BACKEND, TENSOR, TENSOR_OPTIMIZER, SAFETENSORS_FILE, NN_MODULE} -- Element change

	put (v: REAL_32; i: INTEGER)
		require
			valid: 0 <= i and i < count
		deferred
		ensure
			set: item (i) = v
		end

invariant
	count_positive: count >= 1

end

class
	CPU_STORAGE

inherit
	TENSOR_STORAGE

create
	make

feature {NONE} -- Initialization

	make (a_count: INTEGER)
			-- Zero-filled block of `a_count' REAL_32 on the C heap.
		require
			positive: a_count >= 1
		do
			create block.make (a_count * {PLATFORM}.real_32_bytes)
		ensure
			counted: count = a_count
		end

feature -- Access

	count: INTEGER
		do
			Result := block.count // {PLATFORM}.real_32_bytes
		end

	item (i: INTEGER): REAL_32
		do
			Result := block.read_real_32 (i * {PLATFORM}.real_32_bytes)
		end

	address: POINTER
			-- C-heap address for `C blocking' kernels (D-002).
		do
			Result := block.item
		end

feature {TENSOR_BACKEND, TENSOR, TENSOR_OPTIMIZER, SAFETENSORS_FILE, NN_MODULE} -- Element change

	put (v: REAL_32; i: INTEGER)
		do
			block.put_real_32 (v, i * {PLATFORM}.real_32_bytes)
		end

feature {NONE} -- Implementation

	block: MANAGED_POINTER
			-- C heap: never moved by the collector, so kernels may be `blocking'.

invariant
	block_sized: block.count = count * {PLATFORM}.real_32_bytes

end
```

### TENSOR (core; ops follow the same pattern)
```eiffel
class
	TENSOR

create {SIMPLE_TENSOR, TENSOR, TENSOR_OP, NN_MODULE, SAFETENSORS_FILE, TENSOR_OPTIMIZER, GRAD_STORE}
	make_leaf, make_result

feature {NONE} -- Initialization

	make_leaf (a_shape: TENSOR_SHAPE; a_storage: TENSOR_STORAGE; a_backend: TENSOR_BACKEND; a_requires_grad: BOOLEAN)
		require
			fits: a_storage.count = a_shape.count
			owned: a_backend.owns (a_storage)
		do
			-- (Phase 4) set fields; id := next_id
		ensure
			leaf: origin = Void
			grad_flag: requires_grad = a_requires_grad
		end

	make_result (a_shape: TENSOR_SHAPE; a_storage: TENSOR_STORAGE; a_backend: TENSOR_BACKEND; a_origin: detachable TENSOR_OP)
			-- Output of an op; `a_origin' Void when no input was tracked.
		require
			fits: a_storage.count = a_shape.count
			owned: a_backend.owns (a_storage)
		do
			-- (Phase 4)
		ensure
			not_leaf_param: not requires_grad
			origin_set: origin = a_origin
		end

feature -- Access

	shape: TENSOR_SHAPE
	storage: TENSOR_STORAGE
	backend: TENSOR_BACKEND
	id: INTEGER_64
	requires_grad: BOOLEAN
	origin: detachable TENSOR_OP

feature -- Status report

	is_tracked: BOOLEAN
		do
			Result := requires_grad or origin /= Void
		end

	is_scalar: BOOLEAN
		do
			Result := shape.count = 1
		end

feature -- Elementwise (representative; sub, div, exp, ... follow the same contract shape)

	times alias "*" (o: TENSOR): TENSOR
			-- Broadcast product.
		require
			same_backend: o.backend = backend
			shapes_broadcast: shape.broadcasts_with (o.shape)
		do
			-- (Phase 4) out := backend storage; backend.mul (...);
			-- origin := if is_tracked or o.is_tracked then create {TENSOR_OP_MUL}.make (Current, o) else Void
		ensure
			shape_broadcast: Result.shape ~ shape.broadcast_with (o.shape)
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			not_a_parameter: not Result.requires_grad
		end

	matmul_transposed (o: TENSOR; a_ta, a_tb: BOOLEAN): TENSOR
			-- op(Current) · op(o), op = transpose of the last two dims when flagged.
		require
			same_backend: o.backend = backend
			ranks: shape.rank >= 2 and o.shape.rank >= 2
			inner_dims_agree: matmul_inner (a_ta) = o.matmul_outer (a_tb)
		do
			-- (Phase 4)
		ensure
			tracked_iff_input_tracked: Result.is_tracked = (is_tracked or o.is_tracked)
			cols: Result.shape.last = o.matmul_cols (a_tb)
		end

	cross_entropy (a_targets: INDEX_TENSOR; a_ignore: INTEGER): TENSOR
		require
			logits_rank: shape.rank >= 2
			targets_shape: a_targets.shape ~ shape.without_last
			targets_in_range: a_targets.all_in_range_or (0, shape.last - 1, a_ignore)
		do
			-- (Phase 4)
		ensure
			scalar: Result.is_scalar
			tracked_iff_input_tracked: Result.is_tracked = is_tracked
		end

	reshape (a_dims: ARRAY [INTEGER]): TENSOR
		require
			same_count: shape.product_of (a_dims) = shape.count
		do
			-- (Phase 4) view: same storage, TENSOR_OP_RESHAPE origin when tracked
		ensure
			shares_storage: Result.storage = storage
			same_count: Result.shape.count = shape.count
		end

	detached: TENSOR
		do
			-- (Phase 4) make_leaf (shape, storage, backend, False)
		ensure
			shares_storage: Result.storage = storage
			untracked: not Result.is_tracked
		end

feature -- Autograd

	backward: GRAD_STORE
			-- Gradients of this scalar with respect to every leaf that requires them.
		require
			is_scalar: is_scalar
			tracked: is_tracked
		do
			-- (Phase 4) the algorithm above
		ensure
			only_grad_leaves: Result.holds_only_grad_leaves
			leaves_shaped: Result.all_shapes_match
		end

feature -- Conversion (slow path)

	to_array: ARRAY [REAL_32]
		do
			-- (Phase 4)
		ensure
			counted: Result.count = shape.count
		end

feature {TENSOR} -- Matmul geometry

	matmul_inner (a_t: BOOLEAN): INTEGER
		do
			if a_t then Result := shape.dim (shape.rank - 1) else Result := shape.last end
		end

	matmul_outer (a_t: BOOLEAN): INTEGER
		do
			if a_t then Result := shape.last else Result := shape.dim (shape.rank - 1) end
		end

	matmul_cols (a_t: BOOLEAN): INTEGER
		do
			if a_t then Result := shape.dim (shape.rank - 1) else Result := shape.last end
		end

invariant
	storage_fits: storage.count = shape.count
	leaf_has_no_origin: requires_grad implies origin = Void
	backend_owns_storage: backend.owns (storage)

end
```

### TENSOR_OP (deferred) and one effective op
```eiffel
deferred class
	TENSOR_OP

feature -- Access

	inputs: ARRAY [TENSOR]
	output_shape: TENSOR_SHAPE
	name: STRING_8
		deferred
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
			-- Add this op's contribution to each tracked input's gradient.
		require
			grad_shaped: a_output_grad.shape ~ output_shape
			grad_untracked: not a_output_grad.is_tracked
		deferred
		ensure
			tracked_inputs_have_gradients: across inputs as i all i.is_tracked implies a_store.has (i) end
		end

invariant
	has_inputs: inputs.count >= 1

end

class
	TENSOR_OP_MUL

inherit
	TENSOR_OP

create
	make

feature {NONE} -- Initialization

	make (a, b: TENSOR; a_output_shape: TENSOR_SHAPE)
		do
			inputs := <<a, b>>
			output_shape := a_output_shape
		ensure
			inputs_set: inputs [1] = a and inputs [2] = b
		end

feature -- Access

	name: STRING_8 = "mul"

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
			-- d/da = g·b, d/db = g·a, each reduced to its input's shape.
		local
			a, b: TENSOR
		do
			a := inputs [1]
			b := inputs [2]
			if a.is_tracked then
				a_store.accumulate (a, (a_output_grad * b.detached).reduced_to (a.shape))
			end
			if b.is_tracked then
				a_store.accumulate (b, (a_output_grad * a.detached).reduced_to (b.shape))
			end
		end

end
```

### GRAD_STORE
```eiffel
class
	GRAD_STORE

create
	make

feature -- Access

	has (t: TENSOR): BOOLEAN
	gradient (t: TENSOR): TENSOR
		require
			present: has (t)
		ensure
			shaped: Result.shape ~ t.shape
			untracked: not Result.is_tracked
	ids_model: MML_SET [INTEGER_64]

feature -- Measurement

	global_norm: REAL_64
		ensure
			non_negative: Result >= 0.0

feature -- Status

	holds_only_grad_leaves: BOOLEAN
	all_shapes_match: BOOLEAN

feature -- Element change

	accumulate (t: TENSOR; g: TENSOR)
		require
			shaped: g.shape ~ t.shape
			same_backend: g.backend = t.backend
			untracked_grad: not g.is_tracked
		ensure
			has_it: has (t)
			keys: ids_model |=| (old ids_model & t.id)

	discard (t: TENSOR)
		ensure
			gone: not has (t)
			keys: ids_model |=| (old ids_model / t.id)

	scale_all (a_factor: REAL_32)
		require
			not_nan: a_factor = a_factor
		ensure
			keys: ids_model |=| old ids_model

	clip_global_norm (a_max: REAL_64)
		require
			positive: a_max > 0.0
		ensure
			bounded: global_norm <= a_max * (1.0 + 1.0e-5)
			keys: ids_model |=| old ids_model

	merge (o: GRAD_STORE)
		ensure
			union: ids_model |=| (old ids_model + o.ids_model)

end
```

### TENSOR_BACKEND (deferred; excerpt, one feature per kernel)
```eiffel
deferred class
	TENSOR_BACKEND

feature -- Storage

	new_storage (a_count: INTEGER): TENSOR_STORAGE
		require
			positive: a_count >= 1
		deferred
		ensure
			owned: owns (Result)
			counted: Result.count = a_count
		end

	owns (s: TENSOR_STORAGE): BOOLEAN
		deferred
		end

feature -- Kernels

	binary (a_kind: INTEGER; a, b, out: TENSOR_STORAGE; a_shape_a, a_shape_b, a_shape_out: TENSOR_SHAPE)
			-- out := a (op) b with broadcasting. `a_kind' ∈ {Add, Sub, Mul, Div}.
		require
			valid_kind: is_binary_kind (a_kind)
			owned: owns (a) and owns (b) and owns (out)
			sized: a.count = a_shape_a.count and b.count = a_shape_b.count and out.count = a_shape_out.count
			broadcast: a_shape_out ~ a_shape_a.broadcast_with (a_shape_b)
			distinct_output: out /= a and out /= b
		deferred
		end

	matmul (a, b, out: TENSOR_STORAGE; a_batch, a_m, a_k, a_n: INTEGER; a_ta, a_tb, a_b_broadcast: BOOLEAN)
		require
			owned: owns (a) and owns (b) and owns (out)
			sizes: a.count = a_batch * a_m * a_k and out.count = a_batch * a_m * a_n
			b_size: b.count = (if a_b_broadcast then 1 else a_batch end) * a_k * a_n
			distinct_output: out /= a and out /= b
		deferred
		end

	-- unary (kind), reduce (kind, axis), reduce_to_shape, softmax, log_softmax, permute,
	-- slice, concat, masked_fill, dropout_apply, gather_rows, scatter_add_rows,
	-- argmax_mask, cross_entropy_forward, cross_entropy_grad, adamw_update, fill, copy

end
```
CPU_BACKEND implements each kernel as `external "C blocking inline"` on `CPU_STORAGE.address` (the C-heap pointer), safe to mark per D-002. REFERENCE_BACKEND implements each in Eiffel with REAL_64 accumulators and a `gc_point` per outer loop.

### WARMUP_COSINE_SCHEDULE
```eiffel
class
	WARMUP_COSINE_SCHEDULE

inherit
	LR_SCHEDULE

create
	make

feature {NONE} -- Initialization

	make (a_peak: REAL_64; a_warmup, a_total: INTEGER; a_floor_ratio: REAL_64)
		require
			peak_positive: a_peak > 0.0
			warmup_non_negative: a_warmup >= 0
			total_after_warmup: a_total > a_warmup
			floor_ratio_valid: 0.0 <= a_floor_ratio and a_floor_ratio <= 1.0
		do
			peak := a_peak; warmup := a_warmup; total := a_total; floor_ratio := a_floor_ratio
		end

feature -- Access

	peak: REAL_64
	warmup, total: INTEGER
	floor_ratio: REAL_64

	rate (a_step: INTEGER): REAL_64
			-- rixgpt.py `lr_at', exactly.
		local
			r: REAL_64
			l_math: DOUBLE_MATH
		do
			if a_step < warmup then
				Result := peak * a_step / warmup.max (1)
			else
				create l_math
				r := ((a_step - warmup) / (total - warmup).max (1)).min (1.0)
				Result := peak * floor_ratio + 0.5 * (peak - peak * floor_ratio) * (1.0 + l_math.cosine (l_math.pi * r))
			end
		ensure then
			warmup_linear: a_step < warmup implies Result = peak * a_step / warmup.max (1)
			never_below_floor: a_step >= warmup implies Result >= peak * floor_ratio - 1.0e-12
		end

invariant
	ordered: total > warmup

end
```

## Dependencies
| Library | Purpose | Version |
|---------|---------|---------|
| base | MANAGED_POINTER, SPECIAL, DOUBLE_MATH | ISE 25.02 |
| simple_mml | Small model queries | 1.0.2 |
| simple_json | safetensors header | current |
| simple_file | File I/O | current |
| simple_randomizer | Seeded RNG + **G-2 gap fill: random_gaussian** | bump |
| simple_testing | TEST_SET_BASE | current |
| simple_vulkan, simple_shaderc | M3 only (separate cluster / target) | after gap fills |

## File Structure
```
simple_tensor/
├── simple_tensor.ecf      targets: simple_tensor (lib), simple_tensor_tests,
│                          simple_tensor_parity (lean sweep), simple_tensor_scoop_probe
├── src/
│   ├── simple_tensor.e, tensor.e, tensor_shape.e, index_tensor.e, grad_store.e, gc_point.e, tensor_random.e
│   ├── storage/   tensor_storage.e, cpu_storage.e
│   ├── backend/   tensor_backend.e, reference_backend.e, cpu_backend.e
│   ├── ops/       tensor_op.e, tensor_op_*.e (33)
│   ├── nn/        nn_module.e, nn_layer.e, nn_linear.e, nn_embedding.e, nn_layer_norm.e, nn_rms_norm.e,
│   │              nn_dropout.e, nn_mlp.e, nn_causal_self_attention.e, nn_sequential.e, nn_module_list.e, nn_lora_linear.e
│   ├── optim/     tensor_optimizer.e, sgd_optimizer.e, adamw_optimizer.e, lr_schedule.e,
│   │              constant_schedule.e, warmup_cosine_schedule.e
│   ├── data/      token_batcher.e, token_batch.e
│   └── io/        safetensors_file.e, tensor_dtype_codec.e
├── src_vulkan/    (M3) vulkan_backend.e, vulkan_storage.e, shaders as manifest strings
├── testing/       test_app.e, lib_tests.e, gradient_checker.e, twin_checker.e, gc_probe.e, test_*.e
└── spikes/        matmul/, memory_gc/, gpu_matmul/ (M0 evidence)
```

## Milestone mapping
| Milestone | Classes |
|---|---|
| M1 | GC_POINT, TENSOR_SHAPE, storages, TENSOR, INDEX_TENSOR, GRAD_STORE, REFERENCE/CPU_BACKEND, 33 ops, optimizers, schedules, TOKEN_BATCHER, SAFETENSORS_FILE, TENSOR_RANDOM (+G-2), testing support |
| M2 | NN_* modules; RIXGPT_TRAINABLE + RIXG writer **in simple_rixgpt**; CPU threading in CPU_BACKEND.matmul |
| M3 | simple_vulkan gap fills (own PRs), VULKAN_STORAGE, VULKAN_BACKEND |
