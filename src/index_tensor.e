note
	description: "[
		Integer data that indexes: token ids, targets, gather indices, masks.
		INTEGER_32 on the C heap (D-002), with a shape. Never differentiable.
	]"
	author: "Larry Rix"

class
	INDEX_TENSOR

inherit
	STORAGE_BUDGET_ACCESS

	GC_POINT

create
	make, make_from_array

feature {NONE} -- Initialization

	make (a_shape: TENSOR_SHAPE)
			-- Zero-filled indices of shape `a_shape'.
		do
			shape := a_shape
			storage_budget.note_allocation (a_shape.count.to_integer_64 * {PLATFORM}.integer_32_bytes)
			create block.make (a_shape.count * {PLATFORM}.integer_32_bytes)
		ensure
			shape_set: shape = a_shape
		end

	make_from_array (a_values: ARRAY [INTEGER]; a_shape: TENSOR_SHAPE)
			-- Indices `a_values' laid out in shape `a_shape'.
		require
			sized: a_values.count = a_shape.count
		do
			shape := a_shape
			storage_budget.note_allocation (a_shape.count.to_integer_64 * {PLATFORM}.integer_32_bytes)
			create block.make (a_shape.count * {PLATFORM}.integer_32_bytes)
			across a_values as v loop
				block.put_integer_32 (v, (@v.cursor_index - 1) * {PLATFORM}.integer_32_bytes)
				if @v.cursor_index \\ Gc_stride = 0 then
					gc_point
				end
			end
		ensure
			shape_set: shape = a_shape
		end

feature -- Access

	shape: TENSOR_SHAPE

	count: INTEGER
		do
			Result := shape.count
		end

	item (i: INTEGER): INTEGER
			-- Index `i' (0-based position).
		require
			valid: 0 <= i and i < count
		do
			Result := block.read_integer_32 (i * {PLATFORM}.integer_32_bytes)
		end

feature {TENSOR_BACKEND} -- Kernel access

	address: POINTER
			-- C-heap address for kernels (exported to backends only: Phase 2
			-- verdict R2-03; a device INDEX_STORAGE is an M3 obligation).
		do
			Result := block.item
		end

feature -- Status report

	has_value_other_than (a_value: INTEGER): BOOLEAN
			-- Is some index different from `a_value'? (cross-entropy needs at least
			-- one non-ignored target: verdict R1-15.)
		local
			i: INTEGER
		do
			from i := 0 until Result or i = count loop
				Result := item (i) /= a_value
				i := i + 1
				if i \\ Gc_stride = 0 then
					gc_point
				end
			end
		end

	all_in_range (a_low, a_high: INTEGER): BOOLEAN
			-- Is every index in a_low .. a_high? (Walks the data: precondition use in DbC builds.)
		local
			i, v: INTEGER
		do
			Result := True
			from i := 0 until not Result or i = count loop
				v := item (i)
				Result := a_low <= v and v <= a_high
				i := i + 1
				if i \\ Gc_stride = 0 then
					gc_point
				end
			end
		end

	all_in_range_or (a_low, a_high, a_exception: INTEGER): BOOLEAN
			-- Is every index in a_low .. a_high or equal to `a_exception'?
		local
			i, v: INTEGER
		do
			Result := True
			from i := 0 until not Result or i = count loop
				v := item (i)
				Result := (a_low <= v and v <= a_high) or v = a_exception
				i := i + 1
				if i \\ Gc_stride = 0 then
					gc_point
				end
			end
		end

feature {SIMPLE_TENSOR, INDEX_TENSOR, TOKEN_BATCHER, TENSOR_BACKEND, SAFETENSORS_FILE} -- Element change

	put (v: INTEGER; i: INTEGER)
		require
			valid: 0 <= i and i < count
		do
			block.put_integer_32 (v, i * {PLATFORM}.integer_32_bytes)
		ensure
			set: item (i) = v
		end

feature -- Derived

	reshaped (a_dims: ARRAY [INTEGER]): INDEX_TENSOR
			-- Same data, new shape (copy).
		require
			at_least_one_dim: a_dims.count >= 1
			dims_positive: across a_dims as d all d >= 1 end
			not_too_large: shape.fits (a_dims)
			same_count: shape.product_of (a_dims) = count
		do
			create Result.make (shape.reshaped (a_dims))
			Result.block.item.memory_copy (block.item, block.count)
		ensure
			same_count: Result.count = count
		end

feature {NONE} -- Constants

	Gc_stride: INTEGER = 65536
			-- Elements between GC points in long loops (S-3 law).

feature {INDEX_TENSOR} -- Implementation

	block: MANAGED_POINTER

invariant
	sized: block.count = shape.count * {PLATFORM}.integer_32_bytes

end
