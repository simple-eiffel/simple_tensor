note
	description: "[
		REAL_32 block on the C heap (D-002). The collector never moves it, so
		kernels on `address' may be `C blocking' (spike S-3: 7 ms worst stall
		on another processor, against 2,744 ms for an unmarked kernel on
		Eiffel memory). Allocation is reported to the thread's STORAGE_BUDGET.
	]"
	author: "Larry Rix"

class
	CPU_STORAGE

inherit
	TENSOR_STORAGE

	STORAGE_BUDGET_ACCESS

create
	make

feature {NONE} -- Initialization

	make (a_count: INTEGER)
			-- Zero-filled block of `a_count' REAL_32.
		require
			positive: a_count >= 1
			not_too_large: a_count <= {TENSOR_SHAPE}.Max_elements
		do
			element_count := a_count
			storage_budget.note_allocation (a_count.to_integer_64 * {PLATFORM}.real_32_bytes)
			create block.make (a_count * {PLATFORM}.real_32_bytes)
		ensure
			counted: count = a_count
			live: not is_released
		end

feature -- Access

	count: INTEGER
		do
			Result := element_count
		end

	item (i: INTEGER): REAL_32
		do
			Result := block.read_real_32 (i * {PLATFORM}.real_32_bytes)
		end

	address: POINTER
			-- C-heap address for `C blocking' kernels.
		require
			live: not is_released
		do
			Result := block.item
		end

feature -- Status report

	is_released: BOOLEAN

feature {TENSOR_BACKEND, TENSOR, TENSOR_OP, TENSOR_OPTIMIZER, SAFETENSORS_FILE, TENSOR_RANDOM, SIMPLE_TENSOR, GRAD_STORE} -- Element change

	put (v: REAL_32; i: INTEGER)
		do
			block.put_real_32 (v, i * {PLATFORM}.real_32_bytes)
		end

	release
		do
			if not is_released then
					-- Give the C-heap block back now (spike S-4: 40 MB peak instead of 8,040).
				block.resize (1)
				is_released := True
			end
		end

feature {NONE} -- Implementation

	element_count: INTEGER

	block: MANAGED_POINTER
			-- The C-heap memory. `resize (1)' on release may move it (realloc):
			-- no kernel holds `address' across a `release' (verdict R3-05).

invariant
	sized_while_live: not is_released implies block.count = element_count * {PLATFORM}.real_32_bytes

end
