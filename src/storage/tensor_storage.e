note
	description: "[
		A contiguous block of REAL_32 on one backend. Element access here is the
		slow path (tests, small data); kernels work on whole blocks.
	]"
	author: "Larry Rix"

deferred class
	TENSOR_STORAGE

feature -- Access

	count: INTEGER
			-- Number of REAL_32 elements.
		deferred
		end

	item (i: INTEGER): REAL_32
			-- Element `i' (0-based).
		require
			valid_index: 0 <= i and i < count
			live: not is_released
		deferred
		end

feature -- Status report

	is_released: BOOLEAN
			-- Has the memory been given back?
		deferred
		end

	owner_count: INTEGER
			-- Tensors created over this storage. Monotone: never decremented
			-- (conservative ownership, verdict G-A; a precise view model later
			-- only weakens `release''s precondition).

	is_shared: BOOLEAN
			-- Is this storage seen by more than one tensor (view, detached copy)?
		do
			Result := owner_count > 1
		end

feature {TENSOR} -- Ownership

	note_owner
			-- One more tensor sees this storage.
		do
			owner_count := owner_count + 1
		ensure
			counted: owner_count = old owner_count + 1
		end

feature {TENSOR_BACKEND, TENSOR, TENSOR_OP, TENSOR_OPTIMIZER, SAFETENSORS_FILE, TENSOR_RANDOM, SIMPLE_TENSOR, GRAD_STORE} -- Element change

	put (v: REAL_32; i: INTEGER)
			-- Set element `i' to `v'.
		require
			valid_index: 0 <= i and i < count
			live: not is_released
		deferred
		ensure
			set: item (i) = v or else (v /= v)
		end

	release
			-- Give the memory back now (intent-v2 Q7); the element count is kept.
		deferred
		ensure
			released: is_released
			count_kept: count = old count
		end

invariant
	count_positive: count >= 1
	owners_non_negative: owner_count >= 0

end
