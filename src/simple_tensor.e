note
	description: "[
		Facade: creates tensors on one explicit backend (never a global one,
		spec 03 A-S06). `make' uses the fast CPU backend; tests use
		`make_with_backend (create {REFERENCE_BACKEND})' for the oracle twin.

		Random initialization is a command on the generator, not a factory
		(CQS, verdict R2-04): `t := zeros (dims); rng.fill_normal (t, 0, 0.02);
		w := parameter (t)'.

		SCOOP: tensors live on the processor that made them; only basic values
		should cross a processor boundary (R2-13).
	]"
	author: "Larry Rix"

class
	SIMPLE_TENSOR

create
	make, make_with_backend

feature {NONE} -- Initialization

	make
			-- Facade on the CPU backend.
		do
			create {CPU_BACKEND} backend
		ensure
			cpu: backend.kind = backend.Cpu_backend_kind
		end

	make_with_backend (a_backend: TENSOR_BACKEND)
			-- Facade on `a_backend'.
		do
			backend := a_backend
		ensure
			backend_set: backend = a_backend
		end

feature -- Access

	backend: TENSOR_BACKEND

feature -- Factories

	zeros (a_dims: ARRAY [INTEGER]): TENSOR
		require
			valid_dims: valid_dims (a_dims)
		do
			Result := filled (a_dims, 0.0)
		ensure
			shaped: Result.shape.to_array ~ a_dims
			untracked: not Result.is_tracked
			on_backend: Result.backend = backend
		end

	ones (a_dims: ARRAY [INTEGER]): TENSOR
		require
			valid_dims: valid_dims (a_dims)
		do
			Result := filled (a_dims, 1.0)
		ensure
			untracked: not Result.is_tracked
			on_backend: Result.backend = backend
		end

	filled (a_dims: ARRAY [INTEGER]; a_value: REAL_32): TENSOR
		require
			valid_dims: valid_dims (a_dims)
		do
			create Result.make_leaf (create {TENSOR_SHAPE}.make (a_dims), filled_storage (product (a_dims), a_value), backend, False)
		ensure
			untracked: not Result.is_tracked
			on_backend: Result.backend = backend
		end

	scalar (a_value: REAL_32): TENSOR
		do
			Result := filled (<<1>>, a_value)
		ensure
			scalar: Result.is_scalar
			untracked: not Result.is_tracked
		end

	from_array (a_values: ARRAY [REAL_32]; a_dims: ARRAY [INTEGER]): TENSOR
			-- Tensor of `a_values', row-major, in dims `a_dims'.
		require
			valid_dims: valid_dims (a_dims)
			sized: a_values.count = product (a_dims)
		do
			create Result.make_leaf (create {TENSOR_SHAPE}.make (a_dims), uploaded (a_values), backend, False)
		ensure
			untracked: not Result.is_tracked
			on_backend: Result.backend = backend
		end

	parameter (a_tensor: TENSOR): TENSOR
			-- A leaf copy of `a_tensor' that requires gradients.
		require
			same_backend: a_tensor.backend.is_compatible (backend)
		do
			create Result.make_leaf (a_tensor.shape, copy_of (a_tensor), backend, True)
		ensure
			leaf: Result.origin = Void
			requires_grad: Result.requires_grad
			same_shape: Result.shape ~ a_tensor.shape
			copied: Result.storage /= a_tensor.storage
		end

	indices (a_values: ARRAY [INTEGER]; a_dims: ARRAY [INTEGER]): INDEX_TENSOR
		require
			valid_dims: valid_dims (a_dims)
			sized: a_values.count = product (a_dims)
		do
			create Result.make_from_array (a_values, create {TENSOR_SHAPE}.make (a_dims))
		ensure
			counted: Result.count = a_values.count
		end

	causal_mask (a_size: INTEGER): INDEX_TENSOR
			-- a_size × a_size, 1 strictly above the diagonal (positions a query may not see).
		require
			positive: a_size >= 1
		do
			create Result.make (create {TENSOR_SHAPE}.make (<<a_size, a_size>>))
			mark_upper_triangle (Result, a_size)
		ensure
			square: Result.shape.rank = 2 and Result.shape.dim (1) = a_size and Result.shape.dim (2) = a_size
		end

feature -- Status report

	valid_dims (a_dims: ARRAY [INTEGER]): BOOLEAN
			-- At least one dimension, all positive, element count within TENSOR_SHAPE.Max_elements.
		do
			Result := a_dims.count >= 1 and then across a_dims as d all d >= 1 end
				and then (create {TENSOR_SHAPE}.make_scalar).fits (a_dims)
		end

	product (a_dims: ARRAY [INTEGER]): INTEGER
		do
			Result := 1
			across a_dims as d loop
				Result := Result * d
			end
		end

feature {NONE} -- Implementation

	filled_storage (a_count: INTEGER; a_value: REAL_32): TENSOR_STORAGE
			-- New storage of `a_count' elements, all `a_value'.
		do
			Result := backend.new_storage (a_count)
			if a_value /= 0.0 then
				backend.fill (Result, a_value)
			end
		end

	uploaded (a_values: ARRAY [REAL_32]): TENSOR_STORAGE
			-- New storage holding `a_values'.
		do
			Result := backend.new_storage (a_values.count)
			backend.upload (a_values, Result)
		end

	copy_of (a_tensor: TENSOR): TENSOR_STORAGE
			-- New storage holding a copy of `a_tensor''s values.
		do
			Result := backend.new_storage (a_tensor.shape.count)
			backend.copy_values (a_tensor.storage, Result)
		end

	mark_upper_triangle (a_mask: INDEX_TENSOR; a_size: INTEGER)
			-- 1 where column > row.
		local
			r, c: INTEGER
		do
			from r := 0 until r = a_size loop
				from c := r + 1 until c >= a_size loop
					a_mask.put (1, r * a_size + c)
					c := c + 1
				end
				r := r + 1
			end
		end

end
