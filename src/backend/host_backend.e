note
	description: "[
		Common parent of the two host-memory backends (Phase 2 verdict R2-05):
		C-heap CPU_STORAGE, transfer, and the COLD kernels in plain Eiffel
		(fill, copy, argmax mask, broadcast reduce, masked fill, permute,
		slice, concat, SGD). The HOT kernels (`Hot_kernels') stay deferred so
		REFERENCE_BACKEND (plain Eiffel, the oracle) and CPU_BACKEND (C inline,
		`blocking') must each provide all of them: completeness is a compile-time
		fact, and twin parity covers exactly the roster.

		Every long loop passes `gc_point' (spike S-3: a non-allocating Eiffel
		loop otherwise stalls every other processor's collector).
	]"
	author: "Larry Rix"

deferred class
	HOST_BACKEND

inherit
	TENSOR_BACKEND

	GC_POINT

feature -- Storage

	new_storage (a_count: INTEGER): TENSOR_STORAGE
		do
			create {CPU_STORAGE} Result.make (a_count)
		end

	owns (s: TENSOR_STORAGE): BOOLEAN
		do
			Result := attached {CPU_STORAGE} s
		end

feature -- Transfer

	upload (a_values: ARRAY [REAL_32]; a_out: TENSOR_STORAGE)
		local
			i: INTEGER
		do
			from i := 0 until i = a_out.count loop
				a_out.put (a_values [a_values.lower + i], i)
				i := i + 1
				pace (i)
			end
		end

	download (a: TENSOR_STORAGE): ARRAY [REAL_32]
		local
			i: INTEGER
		do
			create Result.make_filled (0.0, 1, a.count)
			from i := 0 until i = a.count loop
				Result [i + 1] := a.item (i)
				i := i + 1
				pace (i)
			end
		end

feature -- Kernels: fill and copy

	fill (a_out: TENSOR_STORAGE; a_value: REAL_32)
		local
			i: INTEGER
		do
			from i := 0 until i = a_out.count loop
				a_out.put (a_value, i)
				i := i + 1
				pace (i)
			end
		end

	copy_values (a, a_out: TENSOR_STORAGE)
		local
			i: INTEGER
		do
			from i := 0 until i = a.count loop
				a_out.put (a.item (i), i)
				i := i + 1
				pace (i)
			end
		end

feature -- Kernels: cold

	argmax_mask (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
		local
			l_outer, n, l_inner, o, i, j, l_best, l_index: INTEGER
		do
			l_outer := outer_size (a_shape_a, a_axis)
			n := a_shape_a.dim (a_axis)
			l_inner := inner_size (a_shape_a, a_axis)
			from o := 0 until o = l_outer loop
				from i := 0 until i = l_inner loop
					l_best := 0
					from j := 1 until j = n loop
						if a.item ((o * n + j) * l_inner + i) > a.item ((o * n + l_best) * l_inner + i) then
							l_best := j
						end
						j := j + 1
					end
					from j := 0 until j = n loop
						l_index := (o * n + j) * l_inner + i
						if j = l_best then a_out.put (1.0, l_index) else a_out.put (0.0, l_index) end
						j := j + 1
					end
					i := i + 1
				end
				o := o + 1
				gc_point
			end
		end

	reduce_to_shape (a, a_out: TENSOR_STORAGE; a_from, a_to: TENSOR_SHAPE)
		local
			l_acc: SPECIAL [REAL_64]
			i, t: INTEGER
		do
			create l_acc.make_filled (0.0, a_to.count)
			from i := 0 until i = a_from.count loop
				t := source_index (i, a_from, a_to)
				l_acc [t] := l_acc [t] + a.item (i)
				i := i + 1
				pace (i)
			end
			from i := 0 until i = a_to.count loop
				a_out.put (l_acc [i].truncated_to_real, i)
				i := i + 1
			end
		end

	masked_fill (a: TENSOR_STORAGE; a_mask: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_value: REAL_32)
		local
			i: INTEGER
		do
			from i := 0 until i = a.count loop
				if a_mask.item (source_index (i, a_shape_a, a_mask.shape)) /= 0 then
					a_out.put (a_value, i)
				else
					a_out.put (a.item (i), i)
				end
				i := i + 1
				pace (i)
			end
		end

	permute (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_order: ARRAY [INTEGER])
		local
			l_out_shape: TENSOR_SHAPE
			l_in_strides: SPECIAL [INTEGER]
			f, rem, k, c, l_source, r: INTEGER
		do
			r := a_shape_a.rank
			l_out_shape := a_shape_a.permuted (a_order)
			l_in_strides := strides_of (a_shape_a)
			from f := 0 until f = a.count loop
				rem := f
				l_source := 0
				from k := r until k = 0 loop
					c := rem \\ l_out_shape.dim (k)
					rem := rem // l_out_shape.dim (k)
					l_source := l_source + c * l_in_strides [a_order [a_order.lower + k - 1] - 1]
					k := k - 1
				end
				a_out.put (a.item (l_source), f)
				f := f + 1
				pace (f)
			end
		end

	slice (a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
		local
			l_outer, n, l_inner, o, j, i: INTEGER
		do
			l_outer := outer_size (a_shape_a, a_axis)
			n := a_shape_a.dim (a_axis)
			l_inner := inner_size (a_shape_a, a_axis)
			from o := 0 until o = l_outer loop
				from j := 0 until j = a_length loop
					from i := 0 until i = l_inner loop
						a_out.put (a.item ((o * n + a_start - 1 + j) * l_inner + i), (o * a_length + j) * l_inner + i)
						i := i + 1
					end
					j := j + 1
				end
				o := o + 1
				gc_point
			end
		end

	add_slice (a_part, a_out: TENSOR_STORAGE; a_shape_out: TENSOR_SHAPE; a_axis, a_start, a_length: INTEGER)
		local
			l_outer, n, l_inner, o, j, i, l_target: INTEGER
		do
			l_outer := outer_size (a_shape_out, a_axis)
			n := a_shape_out.dim (a_axis)
			l_inner := inner_size (a_shape_out, a_axis)
			from o := 0 until o = l_outer loop
				from j := 0 until j = a_length loop
					from i := 0 until i = l_inner loop
						l_target := (o * n + a_start - 1 + j) * l_inner + i
						a_out.put (a_out.item (l_target) + a_part.item ((o * a_length + j) * l_inner + i), l_target)
						i := i + 1
					end
					j := j + 1
				end
				o := o + 1
				gc_point
			end
		end

	concat (a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b: TENSOR_SHAPE; a_axis: INTEGER)
		local
			l_outer, na, nb, l_inner, o, j, i: INTEGER
		do
			l_outer := outer_size (a_shape_a, a_axis)
			na := a_shape_a.dim (a_axis)
			nb := a_shape_b.dim (a_axis)
			l_inner := inner_size (a_shape_a, a_axis)
			from o := 0 until o = l_outer loop
				from j := 0 until j = na + nb loop
					from i := 0 until i = l_inner loop
						if j < na then
							a_out.put (a.item ((o * na + j) * l_inner + i), (o * (na + nb) + j) * l_inner + i)
						else
							a_out.put (b.item ((o * nb + j - na) * l_inner + i), (o * (na + nb) + j) * l_inner + i)
						end
						i := i + 1
					end
					j := j + 1
				end
				o := o + 1
				gc_point
			end
		end

	sgd_update (a_param, a_grad: TENSOR_STORAGE; a_rate: REAL_64)
		local
			i: INTEGER
		do
			from i := 0 until i = a_param.count loop
				a_param.put ((a_param.item (i) - a_rate * a_grad.item (i)).truncated_to_real, i)
				i := i + 1
				pace (i)
			end
		end

feature -- Kernels: random fill

	fill_random (a_out: TENSOR_STORAGE; a_kind: INTEGER; p1, p2: REAL_64; a_seed, a_position: NATURAL_64)
		local
			l_philox: SIMPLE_PHILOX
			i: INTEGER
			p, l_threshold: NATURAL_64
			v, u1, u2, l_keep_value: REAL_64
		do
			create l_philox.make (a_seed, 0)
			p := a_position
			l_threshold := (p1 * 4294967296.0).ceiling_real_64.truncated_to_integer_64.to_natural_64
			l_keep_value := (1.0 / p1).truncated_to_real
			from i := 0 until i = a_out.count loop
				if a_kind = Uniform_fill_kind then
					v := p1 + (p2 - p1) * ((l_philox.word (p + i.to_natural_64) |>> 8).to_real_64 / 16777216.0)
				elseif a_kind = Normal_fill_kind then
					u1 := ((l_philox.word (p + 2 * i.to_natural_64) |>> 8).to_real_64 + 1.0) / 16777216.0
					u2 := (l_philox.word (p + 2 * i.to_natural_64 + 1) |>> 8).to_real_64 / 16777216.0
					v := p1 + p2 * {DOUBLE_MATH}.sqrt (-2.0 * {DOUBLE_MATH}.log (u1)) * {DOUBLE_MATH}.cosine (2.0 * {DOUBLE_MATH}.pi * u2)
				elseif l_philox.word (p + i.to_natural_64).to_natural_64 < l_threshold then
					v := l_keep_value
				else
					v := 0.0
				end
				a_out.put (v.truncated_to_real, i)
				i := i + 1
				pace (i)
			end
		end

feature {NONE} -- Geometry helpers

	source_index (a_flat: INTEGER; a_out, a_in: TENSOR_SHAPE): INTEGER
			-- Flat index in `a_in' read by flat index `a_flat' of `a_out' under
			-- numpy broadcasting (trailing-aligned; size-1 dims repeat).
		local
			rem, k, k_in, c, d, l_stride: INTEGER
		do
			rem := a_flat
			l_stride := 1
			from k := a_out.rank until k = 0 loop
				c := rem \\ a_out.dim (k)
				rem := rem // a_out.dim (k)
				k_in := k - (a_out.rank - a_in.rank)
				if k_in >= 1 then
					d := a_in.dim (k_in)
					if d /= 1 then
						Result := Result + c * l_stride
					end
					l_stride := l_stride * d
				end
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

	strides_of (a_shape: TENSOR_SHAPE): SPECIAL [INTEGER]
			-- Row-major strides, 0-based by dim.
		local
			k, s: INTEGER
		do
			create Result.make_filled (1, a_shape.rank)
			s := 1
			from k := a_shape.rank until k = 0 loop
				Result [k - 1] := s
				s := s * a_shape.dim (k)
				k := k - 1
			end
		end

	pace (i: INTEGER)
			-- GC point every `Gc_stride' elements of a long loop.
		do
			if i \\ Gc_stride = 0 then
				gc_point
			end
		end

	Gc_stride: INTEGER = 4096

end
