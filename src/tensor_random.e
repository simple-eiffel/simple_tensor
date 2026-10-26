note
	description: "[
		Seeded randomness for initialization, dropout masks and batching.
		Commands advance the generator; queries never do (CQS, Phase 2 verdict
		R2-04): `draw_index' then `last_index'; fills write into an untracked
		tensor. Wraps simple_randomizer (ISE RANDOM, Lehmer 16807): its single
		cycle of 2^31-2 draws is spent in ~9 RixGPT dropout iterations, so
		recipe-scale dropout waits on gap G-6 (a long-period generator, R3-03).
		The Gaussian draw is gap G-2. Long fills pass GC points.
	]"
	author: "Larry Rix"

class
	TENSOR_RANDOM

inherit
	GC_POINT

create
	make

feature {NONE} -- Initialization

	make (a_seed: INTEGER)
			-- Generator seeded with `a_seed'.
		require
			seed_valid: a_seed > 0 and a_seed < 2_147_483_647
				-- Seed 0 or 2^31-1 gives a constant stream (verdict R3-01).
		do
			seed := a_seed
			create randomizer.make_with_seed (a_seed)
		ensure
			seed_set: seed = a_seed
			nothing_drawn: last_index = 0
		end

feature -- Access

	seed: INTEGER

	last_index: INTEGER
			-- Index drawn by the last `draw_index'.

feature -- Basic operations

	draw_index (a_upper: INTEGER)
			-- Draw a uniform integer in 0 .. a_upper - 1 into `last_index'.
		require
			positive: a_upper >= 1
		do
			last_index := randomizer.random_integer_in_range (0 |..| (a_upper - 1))
		ensure
			in_range: 0 <= last_index and last_index < a_upper
		end

	fill_uniform (a_tensor: TENSOR; a_low, a_high: REAL_64)
			-- Fill `a_tensor' with values uniform in [a_low, a_high).
		require
			ordered: a_low <= a_high
			untracked: not a_tensor.is_tracked
			live: not a_tensor.is_released
		do
			fill_each (a_tensor, 1, a_low, a_high)
		ensure
			in_range: across 0 |..| (a_tensor.shape.count - 1) as ix all a_low <= a_tensor.item (ix) and a_tensor.item (ix) <= a_high end
		end

	fill_normal (a_tensor: TENSOR; a_mean, a_std: REAL_64)
			-- Gaussian fill (gap G-2: SIMPLE_RANDOMIZER.random_gaussian).
		require
			std_non_negative: a_std >= 0.0
			untracked: not a_tensor.is_tracked
			live: not a_tensor.is_released
		do
			fill_each (a_tensor, 2, a_mean, a_std)
		end

	fill_keep_mask (a_tensor: TENSOR; a_keep: REAL_64)
			-- 1/a_keep with probability `a_keep', else 0 (inverted dropout mask:
			-- dropout = x * mask).
		require
			probability: 0.0 < a_keep and a_keep <= 1.0
			untracked: not a_tensor.is_tracked
			live: not a_tensor.is_released
		do
			fill_each (a_tensor, 3, a_keep, 0.0)
		ensure
			two_values: across 0 |..| (a_tensor.shape.count - 1) as ix all a_tensor.item (ix) = 0.0 or (a_tensor.item (ix) - 1.0 / a_keep).abs <= 1.0e-6 end
		end

feature {NONE} -- Implementation

	randomizer: SIMPLE_RANDOMIZER

	fill_each (a_tensor: TENSOR; a_distribution: INTEGER; p1, p2: REAL_64)
			-- Fill element by element: 1 uniform [p1, p2), 2 normal (p1, p2),
			-- 3 keep mask (keep probability p1). One generator, fixed order.
		local
			i: INTEGER
			v: REAL_64
		do
			from i := 0 until i = a_tensor.shape.count loop
				inspect a_distribution
				when 1 then
					v := p1 + (p2 - p1) * randomizer.random_real
				when 2 then
					v := randomizer.random_gaussian (p1, p2)
				else
					if randomizer.random_real < p1 then v := 1.0 / p1 else v := 0.0 end
				end
				a_tensor.storage.put (v.truncated_to_real, i)
				i := i + 1
				if i \\ 4096 = 0 then
					gc_point
				end
			end
		end

end
