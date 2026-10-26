note
	description: "[
		GPU backend on Vulkan (M3b; design .eiffel-workflow/m3/design.md).
		Kernels record into the context's open batch; the GPU runs at the first host
		read (download, a mirror refresh, sum_of_squares, has_non_finite) or index upload.

		Every kernel works from day one: a kernel without its GLSL yet runs as a HOST
		FALLBACK (download the inputs, run CPU_BACKEND's kernel, write the result back),
		counted in `fallback_counts'. GLSL kernels replace fallbacks one by one, each
		proven by twin parity against REFERENCE_BACKEND.
	]"
	author: "Larry Rix"

class
	VULKAN_BACKEND

inherit
	TENSOR_BACKEND
		redefine
			is_compatible
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
		ensure
			context_set: context = a_context
			no_fallbacks_yet: fallback_counts.is_empty
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

feature -- Kernels: fill and copy (native: recorded transfer commands)

	fill (a_out: TENSOR_STORAGE; a_value: REAL_32)
		local
			l_word: MANAGED_POINTER
			ok: BOOLEAN
		do
			create l_word.make (4)
			l_word.put_real_32 (a_value, 0)
			ok := gpu (a_out).buffer.record_fill (0, gpu (a_out).byte_count, l_word.read_natural_32 (0))
			check recorded: ok end
			gpu (a_out).invalidate
		end

	copy_values (a, a_out: TENSOR_STORAGE)
		local
			ok: BOOLEAN
		do
			ok := gpu (a).buffer.record_copy_to (gpu (a_out).buffer, 0, 0, gpu (a).byte_count)
			check recorded: ok end
			gpu (a_out).invalidate
		end

feature -- Kernels: elementwise

	binary (a_kind: INTEGER; a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b, a_shape_out: TENSOR_SHAPE)
		local
			ha, hb, hout: CPU_STORAGE
		do
			note_fallback ("binary")
			ha := to_host (a); hb := to_host (b); hout := host_out (a_out, False)
			host.binary (a_kind, ha, hb, hout, a_shape_a, a_shape_b, a_shape_out)
			back (hout, a_out); free (ha); free (hb)
		end

	unary (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_param: REAL_32)
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("unary")
			ha := to_host (a); hout := host_out (a_out, False)
			host.unary (a_kind, ha, hout, a_param)
			back (hout, a_out); free (ha)
		end

	unary_derivative (a_kind: INTEGER; a, a_y, a_out: TENSOR_STORAGE; a_param: REAL_32)
		local
			ha, hy, hout: CPU_STORAGE
		do
			note_fallback ("unary_derivative")
			ha := to_host (a); hy := to_host (a_y); hout := host_out (a_out, False)
			host.unary_derivative (a_kind, ha, hy, hout, a_param)
			back (hout, a_out); free (ha); free (hy)
		end

feature -- Kernels: reductions

	reduce (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("reduce")
			ha := to_host (a); hout := host_out (a_out, False)
			host.reduce (a_kind, ha, hout, a_shape_a, a_axis)
			back (hout, a_out); free (ha)
		end

	argmax_mask (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("argmax_mask")
			ha := to_host (a); hout := host_out (a_out, False)
			host.argmax_mask (ha, hout, a_shape_a, a_axis)
			back (hout, a_out); free (ha)
		end

	reduce_to_shape (a, a_out: TENSOR_STORAGE; a_from, a_to: TENSOR_SHAPE)
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("reduce_to_shape")
			ha := to_host (a); hout := host_out (a_out, False)
			host.reduce_to_shape (ha, hout, a_from, a_to)
			back (hout, a_out); free (ha)
		end

	sum_of_squares (a: TENSOR_STORAGE): REAL_64
		local
			ha: CPU_STORAGE
		do
			note_fallback ("sum_of_squares")
			ha := to_host (a)
			Result := host.sum_of_squares (ha)
			free (ha)
		end

	has_non_finite (a: TENSOR_STORAGE): BOOLEAN
		local
			ha: CPU_STORAGE
		do
			note_fallback ("has_non_finite")
			ha := to_host (a)
			Result := host.has_non_finite (ha)
			free (ha)
		end

feature -- Kernels: linear algebra

	matmul (a, b, a_out: TENSOR_STORAGE; a_batch, a_m, a_k, a_n: INTEGER; a_ta, a_tb, a_b_shared: BOOLEAN)
		local
			ha, hb, hout: CPU_STORAGE
		do
			note_fallback ("matmul")
			ha := to_host (a); hb := to_host (b); hout := host_out (a_out, False)
			host.matmul (ha, hb, hout, a_batch, a_m, a_k, a_n, a_ta, a_tb, a_b_shared)
			back (hout, a_out); free (ha); free (hb)
		end

feature -- Kernels: neural

	softmax (a, a_out: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_log: BOOLEAN)
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("softmax")
			ha := to_host (a); hout := host_out (a_out, False)
			host.softmax (ha, hout, a_rows, a_cols, a_log)
			back (hout, a_out); free (ha)
		end

	masked_fill (a: TENSOR_STORAGE; a_mask: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_value: REAL_32)
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("masked_fill")
			ha := to_host (a); hout := host_out (a_out, False)
			host.masked_fill (ha, a_mask, hout, a_shape_a, a_value)
			back (hout, a_out); free (ha)
		end

feature -- Kernels: shape

	permute (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_order: ARRAY [INTEGER])
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("permute")
			ha := to_host (a); hout := host_out (a_out, False)
			host.permute (ha, hout, a_shape_a, a_order)
			back (hout, a_out); free (ha)
		end

	slice (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
		local
			ha, hout: CPU_STORAGE
		do
			note_fallback ("slice")
			ha := to_host (a); hout := host_out (a_out, False)
			host.slice (ha, hout, a_shape_a, a_axis, a_start, a_length)
			back (hout, a_out); free (ha)
		end

	add_slice (a_part, a_out: TENSOR_STORAGE; a_shape_out: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
		local
			hp, hout: CPU_STORAGE
		do
			note_fallback ("add_slice")
			hp := to_host (a_part); hout := host_out (a_out, True)
			host.add_slice (hp, hout, a_shape_out, a_axis, a_start, a_length)
			back (hout, a_out); free (hp)
		end

	concat (a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b: TENSOR_SHAPE; a_axis: INTEGER)
		local
			ha, hb, hout: CPU_STORAGE
		do
			note_fallback ("concat")
			ha := to_host (a); hb := to_host (b); hout := host_out (a_out, False)
			host.concat (ha, hb, hout, a_shape_a, a_shape_b, a_axis)
			back (hout, a_out); free (ha); free (hb)
		end

feature -- Kernels: indexing

	gather_rows (a_table: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
		local
			ht, hout: CPU_STORAGE
		do
			note_fallback ("gather_rows")
			ht := to_host (a_table); hout := host_out (a_out, False)
			host.gather_rows (ht, a_indices, hout, a_rows, a_dim)
			back (hout, a_out); free (ht)
		end

	scatter_add_rows (a_grad: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
		local
			hg, hout: CPU_STORAGE
		do
			note_fallback ("scatter_add_rows")
			hg := to_host (a_grad); hout := host_out (a_out, True)
			host.scatter_add_rows (hg, a_indices, hout, a_rows, a_dim)
			back (hout, a_out); free (hg)
		end

feature -- Kernels: losses

	cross_entropy (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER)
		local
			hl, hout: CPU_STORAGE
		do
			note_fallback ("cross_entropy")
			hl := to_host (a_logits); hout := host_out (a_out, False)
			host.cross_entropy (hl, a_targets, a_ignore, hout, a_rows, a_classes)
			back (hout, a_out); free (hl)
		end

	cross_entropy_grad (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER; a_scale: REAL_32)
		local
			hl, hout: CPU_STORAGE
		do
			note_fallback ("cross_entropy_grad")
			hl := to_host (a_logits); hout := host_out (a_out, False)
			host.cross_entropy_grad (hl, a_targets, a_ignore, hout, a_rows, a_classes, a_scale)
			back (hout, a_out); free (hl)
		end

feature -- Kernels: optimizers

	sgd_update (a_param, a_grad: TENSOR_STORAGE; a_rate: REAL_64)
		local
			hp, hg: CPU_STORAGE
		do
			note_fallback ("sgd_update")
			hp := host_out (a_param, True); hg := to_host (a_grad)
			host.sgd_update (hp, hg, a_rate)
			back (hp, a_param); free (hg)
		end

	adamw_update (a_param, a_grad, a_m, a_v: TENSOR_STORAGE; a_rate, a_beta1, a_beta2, a_eps, a_decay: REAL_64; a_step: INTEGER)
		local
			hp, hg, hm, hv: CPU_STORAGE
		do
			note_fallback ("adamw_update")
			hp := host_out (a_param, True); hg := to_host (a_grad); hm := host_out (a_m, True); hv := host_out (a_v, True)
			host.adamw_update (hp, hg, hm, hv, a_rate, a_beta1, a_beta2, a_eps, a_decay, a_step)
			back (hp, a_param); back (hm, a_m); back (hv, a_v); free (hg)
		end

feature {NONE} -- Host fallback

	host: CPU_BACKEND
			-- Runs the kernels that have no GLSL yet.

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

feature {NONE} -- Programs

	programs: HASH_TABLE [VULKAN_ADDRESS_PIPELINE, STRING_8]
			-- Compiled kernels by name.

invariant
	addressable: context.is_valid and context.has_device_address

end
