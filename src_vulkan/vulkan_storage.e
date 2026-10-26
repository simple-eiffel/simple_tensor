note
	description: "[
		REAL_32 storage in GPU device-local memory (M3b, design .eiffel-workflow/m3/design.md).
		Kernels address it by `address'. Element access (`item') is served from a host
		mirror, downloaded whole on the first read after a change, so a contract that walks
		every element costs one transfer, not one per element. Writes by kernels call
		`invalidate'; `put' writes through. Memory is noted in the per-processor
		STORAGE_BUDGET; `release' and the collector (`dispose') recycle the buffer into the
		context's pool (simple_vulkan 0.3.0), so the collector never waits on the GPU.
	]"
	author: "Larry Rix"

class
	VULKAN_STORAGE

inherit
	TENSOR_STORAGE

	STORAGE_BUDGET_ACCESS

	DISPOSABLE

create {VULKAN_BACKEND}
	make

feature {NONE} -- Initialization

	make (a_context: VULKAN_CONTEXT; a_count: INTEGER)
			-- Zero-filled storage of `a_count' elements on `a_context'.
		require
			positive: a_count >= 1
			not_too_large: a_count <= {TENSOR_SHAPE}.Max_elements
			addressable: a_context.is_valid and then a_context.has_device_address
		do
			element_count := a_count
			context := a_context
			storage_budget.note_allocation (byte_count)
			create buffer.make (a_context, byte_count)
			last_transfer_ok := buffer.is_valid and then buffer.record_fill (0, byte_count, 0)
			check
				zeroed: last_transfer_ok
			end
		ensure
			counted: count = a_count
			live: not is_released
			context_set: context = a_context
		end

feature -- Access

	count: INTEGER
		do
			Result := element_count
		end

	item (i: INTEGER): REAL_32
		do
			refresh_mirror
			check attached mirror as m then
				Result := m.read_real_32 (i * 4)
			end
		end

	context: VULKAN_CONTEXT
			-- The GPU this storage lives on.

	address: NATURAL_64
			-- Device address of element 0, for kernels.
		require
			live: not is_released
		do
			Result := buffer.address
		end

	byte_count: INTEGER_64
		do
			Result := element_count.to_integer_64 * 4
		end

feature -- Status report

	is_released: BOOLEAN

	is_mirror_valid: BOOLEAN
			-- Does the host mirror hold the current values?

	last_transfer_ok: BOOLEAN
			-- Did the last GPU transfer (fill, read, write) succeed?

feature {TENSOR_BACKEND, TENSOR, TENSOR_OP, TENSOR_OPTIMIZER, SAFETENSORS_FILE, TENSOR_RANDOM, SIMPLE_TENSOR, GRAD_STORE} -- Element change

	put (v: REAL_32; i: INTEGER)
		local
			l_one: MANAGED_POINTER
		do
			create l_one.make (4)
			l_one.put_real_32 (v, 0)
			last_transfer_ok := buffer.write (l_one.item, 4, i.to_integer_64 * 4)
			check
				written: last_transfer_ok
			end
			if is_mirror_valid and then attached mirror as m then
				m.put_real_32 (v, i * 4)
			end
		end

	release
		do
			if not is_released then
				buffer.recycle
				is_released := True
				is_mirror_valid := False
				mirror := Void
			end
		end

feature {VULKAN_BACKEND} -- Kernel access

	buffer: VULKAN_DEVICE_BUFFER

	invalidate
			-- A kernel has written (or will write) this storage.
		do
			is_mirror_valid := False
		ensure
			stale: not is_mirror_valid
		end

	read_into (a_target: POINTER)
			-- Copy all `count' values to host memory `a_target' (runs recorded work first).
		require
			live: not is_released
			target: a_target /= default_pointer
		do
			last_transfer_ok := buffer.read (a_target, byte_count, 0)
			check
				read: last_transfer_ok
			end
		end

	write_from (a_source: POINTER)
			-- Replace all values with `count' REAL_32 at host memory `a_source'.
		require
			live: not is_released
			source: a_source /= default_pointer
		do
			last_transfer_ok := buffer.write (a_source, byte_count, 0)
			check
				written: last_transfer_ok
			end
			is_mirror_valid := False
		ensure
			stale: not is_mirror_valid
		end

feature {NONE} -- Implementation

	element_count: INTEGER

	mirror: detachable MANAGED_POINTER
			-- Host copy for `item'.

	refresh_mirror
			-- Download the values when the mirror is stale.
		local
			m: MANAGED_POINTER
		do
			if not is_mirror_valid then
				if attached mirror as l_m then
					m := l_m
				else
					create m.make (byte_count.to_integer_32)
					mirror := m
				end
				last_transfer_ok := buffer.read (m.item, byte_count, 0)
				check
					read: last_transfer_ok
				end
				is_mirror_valid := True
			end
		ensure
			fresh: is_mirror_valid and attached mirror
		end

	dispose
			-- Collector: hand the GPU memory back without waiting on the GPU.
		do
			if not is_released then
				buffer.recycle
				is_released := True
			end
		end

invariant
	mirror_only_when_live: is_mirror_valid implies not is_released

end
