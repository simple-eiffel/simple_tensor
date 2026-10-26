note
	description: "[
		Packs a kernel's push constants in std430 order: buffer addresses first (8-byte
		aligned), then 32-bit integers, reals and fixed-length integer arrays. At most
		128 bytes (VULKAN_ADDRESS_PIPELINE.Max_push_bytes).
	]"
	author: "Larry Rix"

class
	VULKAN_PUSH_BLOCK

create
	make

feature {NONE} -- Initialization

	make
		do
			create block.make (Capacity)
		ensure
			empty: size = 0
		end

feature -- Access

	block: MANAGED_POINTER

	size: INTEGER
			-- Bytes packed so far.

	Capacity: INTEGER = 128

feature -- Packing

	address (a: NATURAL_64)
			-- A buffer address (must come before any 32-bit field).
		require
			aligned: size \\ 8 = 0
			room: size + 8 <= Capacity
		do
			block.put_natural_64 (a, size)
			size := size + 8
		ensure
			grown: size = old size + 8
		end

	natural (v: INTEGER)
			-- A uint field.
		require
			non_negative: v >= 0
			room: size + 4 <= Capacity
		do
			block.put_natural_32 (v.to_natural_32, size)
			size := size + 4
		ensure
			grown: size = old size + 4
		end

	word (v: NATURAL_32)
			-- A uint field given as its 32 bits.
		require
			room: size + 4 <= Capacity
		do
			block.put_natural_32 (v, size)
			size := size + 4
		ensure
			grown: size = old size + 4
		end

	integer (v: INTEGER)
			-- An int field.
		require
			room: size + 4 <= Capacity
		do
			block.put_integer_32 (v, size)
			size := size + 4
		ensure
			grown: size = old size + 4
		end

	real (v: REAL_64)
			-- A float field.
		require
			room: size + 4 <= Capacity
		do
			block.put_real_32 (v.truncated_to_real, size)
			size := size + 4
		ensure
			grown: size = old size + 4
		end

	integers_6 (a: ARRAY [INTEGER])
			-- An `int x[6]' field: `a' then zeros.
		require
			fits: a.count <= 6
			room: size + 24 <= Capacity
		local
			i: INTEGER
		do
			from i := 0 until i = 6 loop
				if i < a.count then
					block.put_integer_32 (a [a.lower + i], size + 4 * i)
				else
					block.put_integer_32 (0, size + 4 * i)
				end
				i := i + 1
			end
			size := size + 24
		ensure
			grown: size = old size + 24
		end

invariant
	bounded: 0 <= size and size <= Capacity

end
