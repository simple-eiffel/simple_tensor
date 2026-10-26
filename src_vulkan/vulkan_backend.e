note
	description: "[
		GPU backend on Vulkan (M3b; design .eiffel-workflow/m3/design.md).
		Every kernel is a GLSL compute shader (VULKAN_KERNELS) reading its buffers by
		device address. Kernels record into the context's open batch; the GPU runs at
		the first host read (download, a mirror refresh, sum_of_squares,
		has_non_finite) or index upload, so a training step costs a few flushes.

		Shapes of rank above 6 (beyond the push-constant stride arrays) run as a HOST
		FALLBACK through CPU_BACKEND, counted in `fallback_counts'; M3b gate 3 requires
		none during a RixGPT step.
	]"
	author: "Larry Rix"

class
	VULKAN_BACKEND

inherit
	TENSOR_BACKEND
		redefine
			is_compatible,
			sum_of_squares_all
		end

create
	make,
	make_with_context

feature {NONE} -- Initialization

	make
			-- Backend on the best GPU (discrete first).
		require
			vulkan_available: (create {VULKAN_CONTEXT}.make).is_valid
		do
			make_with_context (create {VULKAN_CONTEXT}.make)
		end

	make_with_context (a_context: VULKAN_CONTEXT)
			-- Backend on `a_context'.
		require
			valid: a_context.is_valid
			addressable: a_context.has_device_address
		do
			context := a_context
			create host
			create fallback_counts.make (16)
			create programs.make (32)
			create kernels
			create shaderc.make
			wide_tiles := True
			split_k := True
		ensure
			context_set: context = a_context
			no_fallbacks_yet: fallback_counts.is_empty
			wide_tiles_on: wide_tiles
			split_k_on: split_k
		end

feature -- Identity

	kind: INTEGER
		do
			Result := Vulkan_backend_kind
		end

	name: STRING_8
		do
			Result := "vulkan"
		end

	is_compatible (o: TENSOR_BACKEND): BOOLEAN
			-- Same GPU context: storages of another context cannot be operands.
		do
			Result := attached {VULKAN_BACKEND} o as v and then v.context = context
		end

feature -- Access

	context: VULKAN_CONTEXT

	device_name: STRING_8
		do
			Result := context.device_name
		end

	fallback_counts: HASH_TABLE [INTEGER, STRING_8]
			-- Host-fallback runs by kernel name (absent = never fell back).

	fallback_total: INTEGER
			-- All host-fallback runs.
		do
			across fallback_counts as c loop
				Result := Result + c
			end
		end

	Max_rank: INTEGER = 6
			-- Largest rank the stride-array kernels take.

feature -- Storage

	new_storage (a_count: INTEGER): TENSOR_STORAGE
		do
			create {VULKAN_STORAGE} Result.make (context, a_count)
		end

	owns (s: TENSOR_STORAGE): BOOLEAN
		do
			Result := attached {VULKAN_STORAGE} s as v and then v.context = context
		end

feature -- Transfer

	upload (a_values: ARRAY [REAL_32]; a_out: TENSOR_STORAGE)
		local
			l_block: MANAGED_POINTER
			i: INTEGER
		do
			create l_block.make (a_values.count * 4)
			from i := 0 until i = a_values.count loop
				l_block.put_real_32 (a_values [a_values.lower + i], i * 4)
				i := i + 1
			end
			gpu (a_out).write_from (l_block.item)
		end

	download (a: TENSOR_STORAGE): ARRAY [REAL_32]
		local
			l_block: MANAGED_POINTER
			i: INTEGER
		do
			create l_block.make (a.count * 4)
			gpu (a).read_into (l_block.item)
			create Result.make_filled (0.0, 1, a.count)
			from i := 0 until i = a.count loop
				Result [i + 1] := l_block.read_real_32 (i * 4)
				i := i + 1
			end
		end

feature -- Kernels: fill and copy (recorded transfer commands)

	fill (a_out: TENSOR_STORAGE; a_value: REAL_32)
		local
			l_word: MANAGED_POINTER
		do
			create l_word.make (4)
			l_word.put_real_32 (a_value, 0)
			last_record_ok := gpu (a_out).buffer.record_fill (0, gpu (a_out).byte_count, l_word.read_natural_32 (0))
			if not last_record_ok then
				fail_loudly ("GPU command failed (recorded) - device lost or out of memory")
			end
			gpu (a_out).invalidate
		end

	copy_values (a, a_out: TENSOR_STORAGE)
		do
			last_record_ok := gpu (a).buffer.record_copy_to (gpu (a_out).buffer, 0, 0, gpu (a).byte_count)
			if not last_record_ok then
				fail_loudly ("GPU command failed (recorded) - device lost or out of memory")
			end
			gpu (a_out).invalidate
		end

feature -- Kernels: elementwise

	binary (a_kind: INTEGER; a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b, a_shape_out: TENSOR_SHAPE)
		local
			p: VULKAN_PUSH_BLOCK
			ha, hb, hout: CPU_STORAGE
		do
			if a_shape_a ~ a_shape_out and a_shape_b ~ a_shape_out then
				create p.make
				p.address (addr (a)); p.address (addr (b)); p.address (addr (a_out))
				p.natural (a_out.count); p.integer (a_kind)
				run ("binary_same", p, a_out.count)
			elseif a_shape_out.rank <= Max_rank then
				create p.make
				p.address (addr (a)); p.address (addr (b)); p.address (addr (a_out))
				p.natural (a_out.count); p.integer (a_kind); p.integer (a_shape_out.rank)
				p.integers_6 (a_shape_out.to_array)
				p.integers_6 (broadcast_strides (a_shape_out, a_shape_a))
				p.integers_6 (broadcast_strides (a_shape_out, a_shape_b))
				run ("binary_broadcast", p, a_out.count)
			else
				note_fallback ("binary")
				ha := to_host (a); hb := to_host (b); hout := host_out (a_out, False)
				host.binary (a_kind, ha, hb, hout, a_shape_a, a_shape_b, a_shape_out)
				back (hout, a_out); free (ha); free (hb)
			end
			gpu (a_out).invalidate
		end

	unary (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_param: REAL_32)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a)); p.address (addr (a_out))
			p.natural (a.count); p.integer (a_kind); p.real (a_param)
			run ("unary", p, a.count)
			gpu (a_out).invalidate
		end

	unary_derivative (a_kind: INTEGER; a, a_y, a_out: TENSOR_STORAGE; a_param: REAL_32)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a)); p.address (addr (a_y)); p.address (addr (a_out))
			p.natural (a.count); p.integer (a_kind); p.real (a_param)
			run ("unary_derivative", p, a.count)
			gpu (a_out).invalidate
		end

feature -- Kernels: reductions

	reduce (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
			l_outer, l_inner: INTEGER
		do
			l_outer := outer_size (a_shape_a, a_axis)
			l_inner := inner_size (a_shape_a, a_axis)
			create p.make
			p.address (addr (a)); p.address (addr (a_out))
			p.natural (l_outer); p.natural (a_shape_a.dim (a_axis)); p.natural (l_inner); p.integer (a_kind)
			run ("reduce", p, l_outer * l_inner)
			gpu (a_out).invalidate
		end

	argmax_mask (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
			l_outer, l_inner: INTEGER
		do
			l_outer := outer_size (a_shape_a, a_axis)
			l_inner := inner_size (a_shape_a, a_axis)
			create p.make
			p.address (addr (a)); p.address (addr (a_out))
			p.natural (l_outer); p.natural (a_shape_a.dim (a_axis)); p.natural (l_inner)
			run ("argmax_mask", p, l_outer * l_inner)
			gpu (a_out).invalidate
		end

	reduce_to_shape (a, a_out: TENSOR_STORAGE; a_from, a_to: TENSOR_SHAPE)
		local
			p: VULKAN_PUSH_BLOCK
			ha, hout: CPU_STORAGE
			k, k_to, l_kept: INTEGER
		do
			if a_from.rank <= Max_rank then
				from k := 1 until k > a_from.rank loop
					k_to := k - (a_from.rank - a_to.rank)
					if k_to >= 1 and then a_to.dim (k_to) = a_from.dim (k) then
						l_kept := l_kept | (1 |<< (k - 1))
					end
					k := k + 1
				end
				create p.make
				p.address (addr (a)); p.address (addr (a_out))
				p.natural (a_out.count); p.integer (a_from.rank); p.integer (l_kept)
				p.integers_6 (a_from.to_array)
				run_per_output ("reduce_to", p, a_out.count)
			else
				note_fallback ("reduce_to_shape")
				ha := to_host (a); hout := host_out (a_out, False)
				host.reduce_to_shape (ha, hout, a_from, a_to)
				back (hout, a_out); free (ha)
			end
			gpu (a_out).invalidate
		end

	sum_of_squares (a: TENSOR_STORAGE): REAL_64
		do
			Result := norm_partials (a).sum
		end

	has_non_finite (a: TENSOR_STORAGE): BOOLEAN
		do
			Result := norm_partials (a).bad
		end

feature -- Kernels: linear algebra

	matmul (a, b, a_out: TENSOR_STORAGE; a_batch, a_m, a_k, a_n: INTEGER; a_ta, a_tb, a_b_shared: BOOLEAN)
			-- Three routes: split-K for one long-K product (a weight gradient, K = tokens,
			-- whose few output tiles cannot fill the GPU), 128 x 128 tiles for large
			-- products, 64 x 64 tiles otherwise.
		local
			p, q: VULKAN_PUSH_BLOCK
			l_groups, l_slices, l_kc: INTEGER
			l_part: VULKAN_DEVICE_BUFFER
		do
			l_groups := ((a_n + 63) // 64) * ((a_m + 63) // 64)
			if split_k and a_batch = 1 and a_k >= 512 and a_k >= 4 * a_m.max (a_n) and l_groups < Split_target_groups then
				l_slices := (Split_target_groups // l_groups).max (2).min (a_k // 256)
				l_kc := ((a_k + l_slices - 1) // l_slices + 15) // 16 * 16
				l_slices := (a_k + l_kc - 1) // l_kc
			end
			create p.make
			p.address (addr (a)); p.address (addr (b))
			if l_slices >= 2 then
				l_part := scratch (l_slices.to_integer_64 * a_m * a_n * 4)
				p.address (l_part.address)
			else
				p.address (addr (a_out))
			end
			p.natural (a_m); p.natural (a_k); p.natural (a_n)
			p.natural (a_ta.to_integer); p.natural (a_tb.to_integer); p.natural (a_b_shared.to_integer)
			if l_slices >= 2 and then attached l_part as l_p then
				p.natural (l_kc)
				run_groups ("matmul", p, (a_n + 63) // 64, (a_m + 63) // 64, l_slices)
				create q.make
				q.address (l_p.address); q.address (addr (a_out)); q.natural (a_m * a_n); q.natural (l_slices)
				run ("sum_slices", q, a_m * a_n)
					-- Back to the pool now (dispatches in a batch are barrier-separated, so a
					-- later product may reuse it); left to the collector, `dispose' would flush.
				l_p.recycle
			else
				p.natural (0)
				if wide_tiles and a_m >= 128 and a_n >= 128 then
					run_groups ("matmul_128", p, (a_n + 127) // 128, (a_m + 127) // 128, a_batch)
				else
					run_groups ("matmul", p, (a_n + 63) // 64, (a_m + 63) // 64, a_batch)
				end
			end
			gpu (a_out).invalidate
		end

	Split_target_groups: INTEGER = 512
			-- Workgroups a split-K product aims for (several per SM on a 70-SM GPU).

	split_k: BOOLEAN
			-- Are long-K products split across K slices?

	set_split_k (a_on: BOOLEAN)
			-- Split (or stop splitting) long-K products across K slices.
		do
			split_k := a_on
		ensure
			set: split_k = a_on
		end

	wide_tiles: BOOLEAN
			-- Do products of at least 128 x 128 use the 128 x 128 tile kernel?

	set_wide_tiles (a_on: BOOLEAN)
			-- Use (or stop using) the 128 x 128 tile kernel for large products.
		do
			wide_tiles := a_on
		ensure
			set: wide_tiles = a_on
		end

feature -- Kernels: neural

	softmax (a, a_out: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_log: BOOLEAN)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a)); p.address (addr (a_out))
			p.natural (a_rows); p.natural (a_cols); p.natural (a_log.to_integer)
			run ("softmax", p, a_rows)
			gpu (a_out).invalidate
		end

	masked_fill (a: TENSOR_STORAGE; a_mask: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_value: REAL_32)
		local
			p: VULKAN_PUSH_BLOCK
			l_mask: VULKAN_DEVICE_BUFFER
			ha, hout: CPU_STORAGE
		do
			if a_shape_a.rank <= Max_rank then
				l_mask := index_buffer (a_mask)
				create p.make
				p.address (addr (a)); p.address (l_mask.address); p.address (addr (a_out))
				p.natural (a.count); p.real (a_value); p.integer (a_shape_a.rank)
				p.integers_6 (a_shape_a.to_array)
				p.integers_6 (broadcast_strides (a_shape_a, a_mask.shape))
				run ("masked_fill", p, a.count)
				l_mask.release_deferred
			else
				note_fallback ("masked_fill")
				ha := to_host (a); hout := host_out (a_out, False)
				host.masked_fill (ha, a_mask, hout, a_shape_a, a_value)
				back (hout, a_out); free (ha)
			end
			gpu (a_out).invalidate
		end

feature -- Kernels: shape

	permute (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_order: ARRAY [INTEGER])
		local
			p: VULKAN_PUSH_BLOCK
			l_in, l_mapped: ARRAY [INTEGER]
			k: INTEGER
			ha, hout: CPU_STORAGE
		do
			if a_shape_a.rank <= Max_rank then
				l_in := row_major_strides (a_shape_a)
				create l_mapped.make_filled (0, 1, a_shape_a.rank)
				from k := 1 until k > a_shape_a.rank loop
					l_mapped [k] := l_in [a_order [a_order.lower + k - 1]]
					k := k + 1
				end
				create p.make
				p.address (addr (a)); p.address (addr (a_out))
				p.natural (a.count); p.integer (a_shape_a.rank)
				p.integers_6 (a_shape_a.permuted (a_order).to_array)
				p.integers_6 (l_mapped)
				run ("permute", p, a.count)
			else
				note_fallback ("permute")
				ha := to_host (a); hout := host_out (a_out, False)
				host.permute (ha, hout, a_shape_a, a_order)
				back (hout, a_out); free (ha)
			end
			gpu (a_out).invalidate
		end

	slice (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a)); p.address (addr (a_out))
			p.natural (a_out.count); p.natural (a_shape_a.dim (a_axis)); p.natural (inner_size (a_shape_a, a_axis))
			p.natural (a_start - 1); p.natural (a_length)
			run ("slice", p, a_out.count)
			gpu (a_out).invalidate
		end

	add_slice (a_part, a_out: TENSOR_STORAGE; a_shape_out: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a_part)); p.address (addr (a_out))
			p.natural (a_part.count); p.natural (a_shape_out.dim (a_axis)); p.natural (inner_size (a_shape_out, a_axis))
			p.natural (a_start - 1); p.natural (a_length)
			run ("add_slice", p, a_part.count)
			gpu (a_out).invalidate
		end

	concat (a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b: TENSOR_SHAPE; a_axis: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a)); p.address (addr (b)); p.address (addr (a_out))
			p.natural (a_out.count); p.natural (a_shape_a.dim (a_axis)); p.natural (a_shape_b.dim (a_axis))
			p.natural (inner_size (a_shape_a, a_axis))
			run ("concat", p, a_out.count)
			gpu (a_out).invalidate
		end

feature -- Kernels: indexing

	gather_rows (a_table: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
			l_ix: VULKAN_DEVICE_BUFFER
		do
			l_ix := index_buffer (a_indices)
			create p.make
			p.address (addr (a_table)); p.address (l_ix.address); p.address (addr (a_out))
			p.natural (a_indices.count); p.natural (a_dim)
			run ("gather_rows", p, a_indices.count * a_dim)
			l_ix.release_deferred
			gpu (a_out).invalidate
		end

	scatter_add_rows (a_grad: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
			-- The indices are host data, so the host groups positions by target row (CSR:
			-- offsets per row, positions in ascending order) in one pass; each GPU thread then
			-- adds only its own row's positions, in the order the plain loop would.
		local
			p: VULKAN_PUSH_BLOCK
			l_offsets_buffer, l_positions_buffer: VULKAN_DEVICE_BUFFER
			l_offsets, l_positions: MANAGED_POINTER
			l_rows, i, r, l_at: INTEGER
			l_next: SPECIAL [INTEGER]
		do
			l_rows := a_out.count // a_dim
			create l_offsets.make ((l_rows + 1) * 4)
			create l_positions.make (a_indices.count.max (1) * 4)
			from i := 0 until i = a_indices.count loop
				r := a_indices.item (i)
				l_offsets.put_integer_32 (l_offsets.read_integer_32 ((r + 1) * 4) + 1, (r + 1) * 4)
				i := i + 1
			end
			from r := 1 until r > l_rows loop
				l_offsets.put_integer_32 (l_offsets.read_integer_32 (r * 4) + l_offsets.read_integer_32 ((r - 1) * 4), r * 4)
				r := r + 1
			end
			create l_next.make_filled (0, l_rows)
			from r := 0 until r = l_rows loop
				l_next [r] := l_offsets.read_integer_32 (r * 4)
				r := r + 1
			end
			from i := 0 until i = a_indices.count loop
				r := a_indices.item (i)
				l_at := l_next [r]
				l_positions.put_integer_32 (i, l_at * 4)
				l_next [r] := l_at + 1
				i := i + 1
			end
			l_offsets_buffer := scratch (l_offsets.count.to_integer_64)
			l_positions_buffer := scratch (l_positions.count.to_integer_64)
			last_record_ok := l_offsets_buffer.record_write (l_offsets.item, l_offsets.count.to_integer_64, 0)
				and l_positions_buffer.record_write (l_positions.item, l_positions.count.to_integer_64, 0)
			if not last_record_ok then
				fail_loudly ("GPU command failed (written) - device lost or out of memory")
			end
			create p.make
			p.address (addr (a_grad)); p.address (l_offsets_buffer.address); p.address (l_positions_buffer.address)
			p.address (addr (a_out))
			p.natural (a_dim); p.natural (l_rows)
			run ("scatter_add_rows_csr", p, a_out.count)
			l_offsets_buffer.release_deferred
			l_positions_buffer.release_deferred
			gpu (a_out).invalidate
		end

feature -- Kernels: losses

	cross_entropy (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
			l_t, l_rows: VULKAN_DEVICE_BUFFER
		do
			l_t := index_buffer (a_targets)
			l_rows := scratch (a_rows.to_integer_64 * 4)
			create p.make
			p.address (addr (a_logits)); p.address (l_t.address); p.address (l_rows.address)
			p.natural (a_rows); p.natural (a_classes); p.integer (a_ignore)
			run ("ce_rows", p, a_rows)
			create p.make
			p.address (l_rows.address); p.address (addr (a_out))
			p.natural (a_rows); p.real (1.0 / valid_targets (a_targets, a_ignore))
			run_groups ("ce_sum", p, 1, 1, 1)
			l_t.release_deferred
			l_rows.release_deferred
			gpu (a_out).invalidate
		end

	cross_entropy_grad (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER; a_scale: REAL_32)
		local
			p: VULKAN_PUSH_BLOCK
			l_t: VULKAN_DEVICE_BUFFER
		do
			l_t := index_buffer (a_targets)
			create p.make
			p.address (addr (a_logits)); p.address (l_t.address); p.address (addr (a_out))
			p.natural (a_rows); p.natural (a_classes); p.integer (a_ignore)
			p.real (a_scale / valid_targets (a_targets, a_ignore))
			run ("ce_grad", p, a_rows)
			l_t.release_deferred
			gpu (a_out).invalidate
		end

feature -- Kernels: fused

	causal_softmax (a, a_out: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_scale: REAL_32)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a)); p.address (addr (a_out))
			p.natural (a_rows); p.natural (a_cols); p.real (a_scale)
			run ("causal_softmax", p, a_rows)
			gpu (a_out).invalidate
		end

	add_bias (a_out, a_bias: TENSOR_STORAGE; a_rows, a_cols: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a_out)); p.address (addr (a_bias))
			p.natural (a_out.count); p.natural (a_cols)
			run ("add_bias", p, a_out.count)
			gpu (a_out).invalidate
		end

	layer_norm (a, a_weight, a_bias, a_out, a_mean, a_rstd: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_eps: REAL_64)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a)); p.address (addr (a_weight)); p.address (addr (a_bias))
			p.address (addr (a_out)); p.address (addr (a_mean)); p.address (addr (a_rstd))
			p.natural (a_rows); p.natural (a_cols); p.real (a_eps)
			run ("layer_norm", p, a_rows)
			gpu (a_out).invalidate; gpu (a_mean).invalidate; gpu (a_rstd).invalidate
		end

feature -- Kernels: random fill

	fill_random (a_out: TENSOR_STORAGE; a_kind: INTEGER; p1, p2: REAL_64; a_seed, a_position: NATURAL_64)
		local
			p: VULKAN_PUSH_BLOCK
			l_threshold: NATURAL_64
			l_keep_value: REAL_64
		do
			if a_kind = Keep_mask_fill_kind then
				l_threshold := (p1 * 4294967296.0).ceiling_real_64.truncated_to_integer_64.to_natural_64
				l_keep_value := 1.0 / p1
			end
			create p.make
			p.address (addr (a_out))
			p.natural (a_out.count); p.integer (a_kind); p.real (p1); p.real (p2); p.real (l_keep_value)
			p.word (a_seed.as_natural_32); p.word ((a_seed |>> 32).as_natural_32)
			p.word (a_position.as_natural_32); p.word ((a_position |>> 32).as_natural_32)
			if l_threshold >= 0x1_0000_0000 then
				p.word (0); p.word (1)
			else
				p.word (l_threshold.as_natural_32); p.word (0)
			end
			run ("fill_random", p, a_out.count)
			gpu (a_out).invalidate
		end

feature -- Kernels: optimizers

	sgd_update (a_param, a_grad: TENSOR_STORAGE; a_rate: REAL_64)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a_param)); p.address (addr (a_grad))
			p.natural (a_param.count); p.real (a_rate)
			run ("sgd", p, a_param.count)
			gpu (a_param).invalidate
		end

	adamw_update (a_param, a_grad, a_m, a_v: TENSOR_STORAGE; a_rate, a_beta1, a_beta2, a_eps, a_decay: REAL_64; a_step: INTEGER)
		local
			p: VULKAN_PUSH_BLOCK
		do
			create p.make
			p.address (addr (a_param)); p.address (addr (a_grad)); p.address (addr (a_m)); p.address (addr (a_v))
			p.natural (a_param.count)
			p.real (a_rate); p.real (a_beta1); p.real (a_beta2); p.real (a_eps); p.real (a_decay)
			p.real (1.0 - a_beta1 ^ a_step); p.real (1.0 - a_beta2 ^ a_step)
			run ("adamw", p, a_param.count)
			gpu (a_param).invalidate; gpu (a_m).invalidate; gpu (a_v).invalidate
		end

feature -- Profiling

	set_profiling (a_on: BOOLEAN)
			-- Time every kernel on the GPU (simple_vulkan 0.4.0 timestamps).
		do
			context.set_profiling (a_on)
		end

	reset_profile
			-- Forget the times gathered so far.
		do
			across programs as pr loop
				pr.reset_profile
			end
		end

	profiled_ms (a_kernel: STRING_8): REAL_64
			-- GPU time gathered so far for the kernel named `a_kernel' (0 when it has not run).
		do
			if attached programs.item (a_kernel) as pr then
				Result := pr.profiled_ms
			end
		ensure
			non_negative: Result >= 0.0
		end

	profile_report: STRING_8
			-- Kernels by GPU time, most first: name, calls, ms, share of the total.
		local
			l_rows: ARRAYED_LIST [TUPLE [name: STRING_8; ms: REAL_64; calls: INTEGER_64]]
			l_total: REAL_64
			i, j: INTEGER
			l_swap: TUPLE [name: STRING_8; ms: REAL_64; calls: INTEGER_64]
			fd: FORMAT_DOUBLE
		do
			create l_rows.make (programs.count)
			across programs as pr loop
				if pr.profiled_calls > 0 then
					l_rows.extend ([@pr.key, pr.profiled_ms, pr.profiled_calls])
					l_total := l_total + pr.profiled_ms
				end
			end
			from i := 1 until i > l_rows.count loop
				from j := i + 1 until j > l_rows.count loop
					if l_rows [j].ms > l_rows [i].ms then
						l_swap := l_rows [i]; l_rows [i] := l_rows [j]; l_rows [j] := l_swap
					end
					j := j + 1
				end
				i := i + 1
			end
			create fd.make (10, 1)
			create Result.make (1024)
			Result.append ("GPU kernel time (" + fd.formatted (l_total).out + " ms total)%N")
			across l_rows as r loop
				Result.append ("  " + r.name)
				from until Result.count - Result.last_index_of ('%N', Result.count) > 24 loop
					Result.append_character (' ')
				end
				Result.append (fd.formatted (r.ms) + " ms " + fd.formatted (100.0 * r.ms / l_total.max (1.0e-9)) + " %%  "
					+ r.calls.out + " calls%N")
			end
		end

feature -- Status

	last_record_ok: BOOLEAN
			-- Did the last recorded command go into the batch?

feature {NONE} -- Dispatch

	kernels: VULKAN_KERNELS

	shaderc: SIMPLE_SHADERC

	programs: HASH_TABLE [VULKAN_ADDRESS_PIPELINE, STRING_8]
			-- Compiled kernels by name.

	pipeline (a_name: STRING_8): VULKAN_ADDRESS_PIPELINE
			-- Kernel `a_name', compiled on first use.
		require
			known: kernels.has (a_name)
		local
			s: VULKAN_SHADER
			l_failure: DEVELOPER_EXCEPTION
		do
			if attached programs.item (a_name) as l_p then
				Result := l_p
			elseif attached shaderc.compile_compute (kernels.source (a_name)) as l_code then
				create s.make_from_memory (context, l_code.item, l_code.count.to_integer_64)
				create Result.make (context, s)
				s.dispose
				programs.put (Result, a_name)
			else
				l_failure := create {DEVELOPER_EXCEPTION}
				l_failure.set_description ("GLSL kernel " + a_name + " failed to compile: " + shaderc.last_error)
				l_failure.raise
				check unreachable: False then end
			end
		ensure
			cached: programs.has (a_name)
		end

	run (a_name: STRING_8; p: VULKAN_PUSH_BLOCK; a_threads: INTEGER)
			-- Record `a_name' over `a_threads' threads (256 per group, 2-D group grid).
		local
			l_groups, gx, gy: INTEGER
		do
			l_groups := ((a_threads + 255) // 256).max (1)
			gx := l_groups.min (65535)
			gy := (l_groups + gx - 1) // gx
			run_groups (a_name, p, gx, gy, 1)
		end

	run_per_output (a_name: STRING_8; p: VULKAN_PUSH_BLOCK; a_outputs: INTEGER)
			-- Record `a_name' with one workgroup per output element (2-D group grid).
		local
			gx, gy: INTEGER
		do
			gx := a_outputs.max (1).min (65535)
			gy := (a_outputs.max (1) + gx - 1) // gx
			run_groups (a_name, p, gx, gy, 1)
		end

	run_groups (a_name: STRING_8; p: VULKAN_PUSH_BLOCK; gx, gy, gz: INTEGER)
		do
			last_record_ok := pipeline (a_name).record (p.block.item, p.size, gx, gy, gz)
			if not last_record_ok then
				fail_loudly ("GPU command failed (recorded) - device lost or out of memory")
			end
		end

	sum_of_squares_all (a_list: ARRAYED_LIST [TENSOR_STORAGE]): REAL_64
			-- Every storage's 256 partials recorded into one buffer, read back once.
		local
			p: VULKAN_PUSH_BLOCK
			l_part: VULKAN_DEVICE_BUFFER
			l_host: MANAGED_POINTER
			i, k: INTEGER
		do
			if not a_list.is_empty then
				l_part := scratch (a_list.count.to_integer_64 * 512 * 4)
				k := 0
				across a_list as s loop
					create p.make
					p.address (addr (s)); p.address (l_part.address + (k * 512 * 4).to_natural_64); p.natural (s.count)
					run_groups ("norm_partials", p, 256, 1, 1)
					k := k + 1
				end
				create l_host.make (a_list.count * 512 * 4)
				last_record_ok := l_part.read (l_host.item, a_list.count.to_integer_64 * 512 * 4, 0)
				if not last_record_ok then
					fail_loudly ("GPU command failed (read) - device lost or out of memory")
				end
				from i := 0 until i = a_list.count * 256 loop
					Result := Result + l_host.read_real_32 (8 * i)
					if l_host.read_real_32 (8 * i + 4) /= 0.0 then
						Result := {REAL_64}.nan
					end
					i := i + 1
				end
				l_part.release_deferred
			end
		end

	norm_partials (a: TENSOR_STORAGE): TUPLE [sum: REAL_64; bad: BOOLEAN]
			-- Sum of squares and any-non-finite of `a': 256 GPU partials, summed here.
		local
			p: VULKAN_PUSH_BLOCK
			l_part: VULKAN_DEVICE_BUFFER
			l_host: MANAGED_POINTER
			i: INTEGER
			l_sum: REAL_64
			l_bad: BOOLEAN
		do
			l_part := scratch (512 * 4)
			create p.make
			p.address (addr (a)); p.address (l_part.address); p.natural (a.count)
			run_groups ("norm_partials", p, 256, 1, 1)
			create l_host.make (512 * 4)
			last_record_ok := l_part.read (l_host.item, 512 * 4, 0)
			if not last_record_ok then
				fail_loudly ("GPU command failed (read) - device lost or out of memory")
			end
			from i := 0 until i = 256 loop
				l_sum := l_sum + l_host.read_real_32 (8 * i)
				l_bad := l_bad or l_host.read_real_32 (8 * i + 4) /= 0.0
				i := i + 1
			end
			l_part.release_deferred
			Result := [l_sum, l_bad]
		end

	index_buffer (ix: INDEX_TENSOR): VULKAN_DEVICE_BUFFER
			-- `ix' uploaded to the GPU as a recorded copy (no flush: simple_vulkan 0.5.0).
			-- The caller releases it (deferred) after recording its use.
		do
			Result := scratch (ix.count.to_integer_64 * 4)
			last_record_ok := Result.record_write (ix.address, ix.count.to_integer_64 * 4, 0)
			if not last_record_ok then
				fail_loudly ("GPU command failed (written) - device lost or out of memory")
			end
		end

	fail_loudly (a_message: STRING_8)
			-- Raise: a failed GPU command must never pass silently (in a lean build a
			-- `check' is off and kernels would go on writing garbage).
		local
			l_failure: DEVELOPER_EXCEPTION
		do
			l_failure := create {DEVELOPER_EXCEPTION}
			l_failure.set_description (a_message)
			l_failure.raise
		end

	scratch (a_bytes: INTEGER_64): VULKAN_DEVICE_BUFFER
			-- A device buffer for a kernel's indices or partial results. Out of memory is
			-- handled as VULKAN_STORAGE does: collect, trim the pool, flush, retry once,
			-- then raise (never hand a kernel an invalid buffer: address 0).
		require
			positive: a_bytes > 0
		local
			l_failure: DEVELOPER_EXCEPTION
		do
			create Result.make (context, a_bytes)
			if not Result.is_valid then
				(create {MEMORY}).full_collect
				context.trim_pool
				context.flush
				create Result.make (context, a_bytes)
				if not Result.is_valid then
					l_failure := create {DEVELOPER_EXCEPTION}
					l_failure.set_description ("GPU out of memory: cannot allocate " + a_bytes.out + " bytes of kernel scratch")
					l_failure.raise
				end
			end
		ensure
			valid: Result.is_valid
		end

	valid_targets (a_targets: INDEX_TENSOR; a_ignore: INTEGER): INTEGER
			-- Targets not equal to `a_ignore' (at least one: cross_entropy's precondition).
		local
			i: INTEGER
		do
			from i := 0 until i = a_targets.count loop
				if a_targets.item (i) /= a_ignore then
					Result := Result + 1
				end
				i := i + 1
			end
			Result := Result.max (1)
		end

feature {NONE} -- Geometry

	addr (s: TENSOR_STORAGE): NATURAL_64
		do
			Result := gpu (s).address
		end

	broadcast_strides (a_out, a_in: TENSOR_SHAPE): ARRAY [INTEGER]
			-- Per dim of `a_out': the stride of `a_in' along it (0 where `a_in' broadcasts
			-- or has no such dim). As CPU_BACKEND.
		local
			k, k_in, s: INTEGER
		do
			create Result.make_filled (0, 1, a_out.rank)
			s := 1
			from k := a_out.rank until k = 0 loop
				k_in := k - (a_out.rank - a_in.rank)
				if k_in >= 1 then
					if a_in.dim (k_in) /= 1 then
						Result [k] := s
					end
					s := s * a_in.dim (k_in)
				end
				k := k - 1
			end
		end

	row_major_strides (a_shape: TENSOR_SHAPE): ARRAY [INTEGER]
		local
			k, s: INTEGER
		do
			create Result.make_filled (0, 1, a_shape.rank)
			s := 1
			from k := a_shape.rank until k = 0 loop
				Result [k] := s
				s := s * a_shape.dim (k)
				k := k - 1
			end
		end

	outer_size (a_shape: TENSOR_SHAPE; a_axis: INTEGER): INTEGER
			-- Product of the dims before `a_axis'.
		local
			k: INTEGER
		do
			Result := 1
			from k := 1 until k = a_axis loop
				Result := Result * a_shape.dim (k)
				k := k + 1
			end
		end

	inner_size (a_shape: TENSOR_SHAPE; a_axis: INTEGER): INTEGER
			-- Product of the dims after `a_axis'.
		local
			k: INTEGER
		do
			Result := 1
			from k := a_axis + 1 until k > a_shape.rank loop
				Result := Result * a_shape.dim (k)
				k := k + 1
			end
		end

feature {NONE} -- Host fallback (rank above Max_rank)

	host: CPU_BACKEND

	note_fallback (a_kernel: STRING_8)
		do
			fallback_counts.force (fallback_counts.item (a_kernel) + 1, a_kernel)
		ensure
			counted: fallback_counts.item (a_kernel) = old fallback_counts.item (a_kernel) + 1
		end

	gpu (s: TENSOR_STORAGE): VULKAN_STORAGE
			-- `s' as this backend's storage (preconditions guarantee ownership).
		do
			check attached {VULKAN_STORAGE} s as v then
				Result := v
			end
		end

	to_host (s: TENSOR_STORAGE): CPU_STORAGE
			-- A host copy of `s' (runs recorded work first).
		do
			create Result.make (s.count)
			gpu (s).read_into (Result.address)
		ensure
			sized: Result.count = s.count
		end

	host_out (s: TENSOR_STORAGE; a_in_place: BOOLEAN): CPU_STORAGE
			-- Host storage for an output: a copy of `s' when the kernel reads it too.
		do
			if a_in_place then
				Result := to_host (s)
			else
				create Result.make (s.count)
			end
		end

	back (h: CPU_STORAGE; s: TENSOR_STORAGE)
			-- Write host results `h' into `s', then free `h'.
		do
			gpu (s).write_from (h.address)
			h.release
		end

	free (h: CPU_STORAGE)
		do
			h.release
		end

invariant
	addressable: context.is_valid and context.has_device_address

end
