note
	description: "Immutable tensor dimensions, outermost first (row-major, contiguous)."
	author: "Larry Rix"

class
	TENSOR_SHAPE

inherit
	ANY
		redefine
			is_equal
		end

create
	make, make_scalar

feature {NONE} -- Initialization

	make (a_dims: ARRAY [INTEGER])
			-- Shape with dimensions `a_dims', outermost first.
		require
			at_least_one_dim: a_dims.count >= 1
			dims_positive: across a_dims as d all d >= 1 end
			not_too_large: fits (a_dims)
		local
			i: INTEGER
		do
			create dims.make_empty (a_dims.count)
			count := 1
			from i := a_dims.lower until i > a_dims.upper loop
				dims.extend (a_dims [i])
				count := count * a_dims [i]
				i := i + 1
			end
		ensure
			rank_set: rank = a_dims.count
			dims_copied: across 1 |..| rank as ix all dim (ix) = a_dims [a_dims.lower + ix - 1] end
			count_is_product: count = product_of (a_dims)
		end

	make_scalar
			-- Shape [1].
		do
			create dims.make_filled (1, 1)
			count := 1
		ensure
			scalar: rank = 1 and count = 1
		end

feature -- Access

	rank: INTEGER
			-- Number of dimensions.
		do
			Result := dims.count
		end

	count: INTEGER
			-- Number of elements (product of dims), cached.

	dim (i: INTEGER): INTEGER
			-- Size of dimension `i' (1 = outermost).
		require
			valid: 1 <= i and i <= rank
		do
			Result := dims [i - 1]
		ensure
			positive: Result >= 1
		end

	last: INTEGER
			-- Innermost dimension.
		do
			Result := dim (rank)
		end

	dims_model: MML_SEQUENCE [INTEGER]
			-- Model of the dimensions.
		do
			create Result.from_iterable (to_array)
		ensure
			counted: Result.count = rank
			faithful: across 1 |..| rank as ix all Result [ix] = dim (ix) end
		end

	to_array: ARRAY [INTEGER]
			-- The dimensions as an array (1-based).
		local
			i: INTEGER
		do
			create Result.make_filled (1, 1, rank)
			from i := 1 until i > rank loop
				Result [i] := dims [i - 1]
				i := i + 1
			end
		ensure
			counted: Result.count = rank
			one_based: Result.lower = 1
		end

feature -- Status report

	broadcasts_with (o: TENSOR_SHAPE): BOOLEAN
			-- Numpy rule: trailing-aligned dims are equal or one of them is 1.
		local
			i, a, b: INTEGER
		do
			Result := True
			from i := 0 until not Result or i >= rank.min (o.rank) loop
				a := dims [rank - 1 - i]
				b := o.dims [o.rank - 1 - i]
				Result := a = b or a = 1 or b = 1
				i := i + 1
			end
		end

	is_equal (o: like Current): BOOLEAN
			-- Same rank and dims.
		local
			i: INTEGER
		do
			Result := rank = o.rank
			from i := 0 until not Result or i >= rank loop
				Result := dims [i] = o.dims [i]
				i := i + 1
			end
		ensure then
			exact: Result = (rank = o.rank and then across 1 |..| rank as ix all dim (ix) = o.dim (ix) end)
		end

	is_permutation (a_order: ARRAY [INTEGER]): BOOLEAN
			-- Is `a_order' a permutation of 1 .. rank?
		local
			l_seen: SPECIAL [BOOLEAN]
		do
			if a_order.count = rank then
				create l_seen.make_filled (False, rank + 1)
				Result := True
				across a_order as k until not Result loop
					Result := 1 <= k and k <= rank and then not l_seen [k]
					if Result then
						l_seen [k] := True
					end
				end
			end
		ensure
			sized: Result implies a_order.count = rank
		end

feature -- Derived shapes

	broadcast_with (o: TENSOR_SHAPE): TENSOR_SHAPE
			-- The broadcast of Current and `o'.
		require
			compatible: broadcasts_with (o)
		local
			l_dims: ARRAY [INTEGER]
			r, i, a, b: INTEGER
		do
			r := rank.max (o.rank)
			create l_dims.make_filled (1, 1, r)
			from i := 0 until i = r loop
				if i < rank then a := dims [rank - 1 - i] else a := 1 end
				if i < o.rank then b := o.dims [o.rank - 1 - i] else b := 1 end
				l_dims [r - i] := a.max (b)
				i := i + 1
			end
			create Result.make (l_dims)
		ensure
			rank_is_max: Result.rank = rank.max (o.rank)
		end

	reshaped (a_dims: ARRAY [INTEGER]): TENSOR_SHAPE
			-- Same count, dimensions `a_dims'.
		require
			at_least_one_dim: a_dims.count >= 1
			dims_positive: across a_dims as d all d >= 1 end
			not_too_large: fits (a_dims)
			same_count: product_of (a_dims) = count
		do
			create Result.make (a_dims)
		ensure
			same_count: Result.count = count
			rank_set: Result.rank = a_dims.count
		end

	with_dim (i, a_value: INTEGER): TENSOR_SHAPE
			-- Current with dimension `i' set to `a_value'.
		require
			valid: 1 <= i and i <= rank
			positive: a_value >= 1
		local
			l_dims: ARRAY [INTEGER]
		do
			l_dims := to_array
			l_dims [i] := a_value
			create Result.make (l_dims)
		ensure
			same_rank: Result.rank = rank
			set: Result.dim (i) = a_value
			others_kept: across 1 |..| rank as j all j /= i implies Result.dim (j) = dim (j) end
		end

	without_dim (i: INTEGER): TENSOR_SHAPE
			-- Current without dimension `i'.
		require
			valid: 1 <= i and i <= rank
			rank_above_one: rank >= 2
		local
			l_dims: ARRAY [INTEGER]
			j, k: INTEGER
		do
			create l_dims.make_filled (1, 1, rank - 1)
			k := 1
			from j := 1 until j > rank loop
				if j /= i then
					l_dims [k] := dims [j - 1]
					k := k + 1
				end
				j := j + 1
			end
			create Result.make (l_dims)
		ensure
			one_less: Result.rank = rank - 1
		end

	without_last: TENSOR_SHAPE
			-- Current without the innermost dimension.
		require
			rank_above_one: rank >= 2
		do
			Result := without_dim (rank)
		ensure
			one_less: Result.rank = rank - 1
		end

	appended (a_dim: INTEGER): TENSOR_SHAPE
			-- Current with a new innermost dimension `a_dim'.
		require
			positive: a_dim >= 1
		local
			l_dims: ARRAY [INTEGER]
		do
			l_dims := to_array
			l_dims.force (a_dim, rank + 1)
			create Result.make (l_dims)
		ensure
			one_more: Result.rank = rank + 1
			last_set: Result.last = a_dim
		end

	permuted (a_order: ARRAY [INTEGER]): TENSOR_SHAPE
			-- Dimensions reordered: result dim i = dim (a_order [i]).
		require
			permutation: is_permutation (a_order)
		local
			l_dims: ARRAY [INTEGER]
			k: INTEGER
		do
			create l_dims.make_filled (1, 1, rank)
			from k := 1 until k > rank loop
				l_dims [k] := dims [a_order [a_order.lower + k - 1] - 1]
				k := k + 1
			end
			create Result.make (l_dims)
		ensure
			same_rank: Result.rank = rank
			reordered: across 1 |..| rank as i all Result.dim (i) = dim (a_order [a_order.lower + i - 1]) end
		end

feature -- Measurement

	product_of (a_dims: ARRAY [INTEGER]): INTEGER
			-- Product of `a_dims'.
		require
			not_too_large: fits (a_dims)
		do
			Result := 1
			across a_dims as d loop
				Result := Result * d
			end
		end

	fits (a_dims: ARRAY [INTEGER]): BOOLEAN
			-- Is the product of `a_dims' (computed in INTEGER_64, stopping early)
			-- at most `Max_elements'? (Phase 2 verdict R1-13: INTEGER_32 overflow.)
		local
			l_product: INTEGER_64
		do
			Result := True
			l_product := 1
			across a_dims as d until not Result loop
				l_product := l_product * d.to_integer_64.max (1)
				Result := l_product <= Max_elements
			end
		end

feature -- Constants

	Max_elements: INTEGER = 536_870_911
			-- Largest element count whose 4-byte block fits MANAGED_POINTER's
			-- INTEGER byte count: {INTEGER_32}.max_value // 4.

feature {TENSOR_SHAPE} -- Implementation

	dims: SPECIAL [INTEGER]
			-- The dimensions, 0-based.

invariant
	rank_positive: rank >= 1
	count_positive: count >= 1

end
