note
	description: "[
		The gradients of one backward pass, keyed by tensor id. Accumulation is
		`+=', so a tensor used twice (tied weights) collects both contributions.
		After `backward' it holds only leaves that require gradients.
	]"
	author: "Larry Rix"

class
	GRAD_STORE

create
	make

feature {NONE} -- Initialization

	make
			-- Empty store.
		do
			create gradients.make (64)
			create owners.make (64)
		ensure
			empty: count = 0
		end

feature -- Access

	count: INTEGER
			-- Number of tensors with a gradient.
		do
			Result := gradients.count
		end

	has (t: TENSOR): BOOLEAN
			-- Does `t' have a gradient here?
		do
			Result := gradients.has (t.id)
		end

	gradient (t: TENSOR): TENSOR
			-- The gradient of `t'.
		require
			present: has (t)
		do
			check attached gradients.item (t.id) as g then
				Result := g
			end
		ensure
			shaped: Result.shape ~ t.shape
			untracked: not Result.is_tracked
		end

	ids_model: MML_SET [INTEGER_64]
			-- Model: ids of the tensors with a gradient.
		do
			create Result
			across gradients.current_keys as k loop
				Result := Result & k
			end
		ensure
			counted: Result.count = count
			faithful: Result.for_all (agent has_id)
		end

	has_id (a_id: INTEGER_64): BOOLEAN
			-- Is there a gradient for the tensor with id `a_id'?
		do
			Result := gradients.has (a_id)
		end

	common_ids_shaped (o: GRAD_STORE): BOOLEAN
			-- Do the gradients the two stores share have equal shapes and compatible backends?
		do
			Result := True
			across o.gradients as g until not Result loop
				if attached gradients.item (@g.key) as mine then
					Result := mine.shape ~ g.shape and mine.has_same_backend (g)
				end
			end
		end

feature -- Measurement

	global_norm: REAL_64
			-- √(Σ over all gradients of Σ g²).
		local
			l_sum: REAL_64
			l_storages: ARRAYED_LIST [TENSOR_STORAGE]
			l_backend: detachable TENSOR_BACKEND
			l_one_backend: BOOLEAN
		do
				-- One backend call when every gradient lives on the same backend (the GPU then
				-- reads the norm back once instead of once per parameter).
			create l_storages.make (gradients.count)
			l_one_backend := True
			across gradients as g loop
				if attached l_backend as b then
					l_one_backend := l_one_backend and b.is_compatible (g.backend) and b.owns (g.storage)
				else
					l_backend := g.backend
				end
				l_storages.extend (g.storage)
			end
			if l_one_backend and attached l_backend as b then
				l_sum := b.sum_of_squares_all (l_storages)
			else
				across gradients as g loop
					l_sum := l_sum + g.backend.sum_of_squares (g.storage)
				end
			end
			Result := {DOUBLE_MATH}.sqrt (l_sum)
		ensure
			non_negative: Result >= 0.0 or Result /= Result
		end

feature -- Status report

	holds_only_grad_leaves: BOOLEAN
			-- Is every entry a leaf that requires gradients?
		do
			Result := across owners as t all t.requires_grad end
		end

	all_shapes_match: BOOLEAN
			-- Does every gradient have its tensor's shape?
		do
			Result := across gradients as g all attached owners.item (@g.key) as t and then t.shape ~ g.shape end
		end

	has_non_finite: BOOLEAN
			-- Does any gradient hold a NaN or an infinity? (intent-v2 Q5)
		do
			Result := across gradients as g some g.backend.has_non_finite (g.storage) end
		end

feature -- Element change

	accumulate (t: TENSOR; g: TENSOR)
			-- gradient (t) += g (or := g when absent).
		require
			tracked_owner: t.is_tracked
			shaped: g.shape ~ t.shape
			same_backend: g.has_same_backend (t)
			untracked_grad: not g.is_tracked
		local
			l_copy: TENSOR_STORAGE
		do
			if attached gradients.item (t.id) as l_existing then
				gradients.force (l_existing + g, t.id)
			else
					-- Copy on first store: never keep the caller's storage (verdict G-A).
				l_copy := g.backend.new_storage (g.shape.count)
				g.backend.copy_values (g.storage, l_copy)
				gradients.force (create {TENSOR}.make_leaf (g.shape, l_copy, g.backend, False), t.id)
				owners.force (t, t.id)
			end
		ensure
			has_it: has (t)
			keys: ids_model |=| (old ids_model & t.id)
			copied: gradient (t).storage /= g.storage
		end

	discard (t: TENSOR)
			-- Drop `t''s gradient (intermediates during the backward sweep), giving
			-- its memory back when no other tensor sees it.
		do
			if attached gradients.item (t.id) as g and then not g.storage_shared and then not g.is_released then
				g.release
			end
			gradients.remove (t.id)
			owners.remove (t.id)
		ensure
			gone: not has (t)
			keys: ids_model |=| (old ids_model / t.id)
		end

	scale_all (a_factor: REAL_32)
			-- Multiply every gradient by `a_factor'.
		require
			not_nan: a_factor = a_factor
		local
			l_keys: ARRAY [INTEGER_64]
		do
			l_keys := gradients.current_keys
			across l_keys as k loop
				if attached gradients.item (k) as g then
					gradients.force (g.scaled (a_factor), k)
				end
			end
		ensure
			keys: ids_model |=| old ids_model
		end

	clip_global_norm (a_max: REAL_64)
			-- Scale every gradient so `global_norm' <= `a_max'; never scale up.
		require
			positive: a_max > 0.0
			finite: not has_non_finite
		local
			l_norm: REAL_64
		do
			l_norm := global_norm
			if l_norm > a_max then
				scale_all ((a_max / l_norm).truncated_to_real)
			end
		ensure
			bounded: global_norm <= a_max * (1.0 + 1.0e-5)
			untouched_when_within: old global_norm <= a_max implies global_norm = old global_norm
			keys: ids_model |=| old ids_model
		end

	merge (o: GRAD_STORE)
			-- Add every gradient of `o' into Current (gradient accumulation).
		require
			not_self: o /= Current
			compatible: common_ids_shaped (o)
		do
			across o.gradients as g loop
				if attached o.owners.item (@g.key) as t then
					accumulate (t, g)
				end
			end
		ensure
			union: ids_model |=| (old ids_model + o.ids_model)
		end

feature {GRAD_STORE} -- Implementation

	gradients: HASH_TABLE [TENSOR, INTEGER_64]
			-- Gradient by tensor id.

	owners: HASH_TABLE [TENSOR, INTEGER_64]
			-- The tensor each gradient belongs to, by id.

invariant
	same_keys_count: owners.count = gradients.count

end
