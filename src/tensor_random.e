note
	description: "[
		Seeded randomness for initialization, dropout masks and batching.
		Commands advance the generator; queries never do (CQS, Phase 2 verdict
		R2-04): `draw_index' then `last_index'; fills write into an untracked
		tensor, drawn by its backend (TENSOR_BACKEND.fill_random).

		Generator: Philox4x32-10 (SIMPLE_PHILOX, simple_randomizer 1.2.0; gap G-6),
		keyed by `seed', stream 0. It is counter-based, so every value has a fixed
		address in the stream: a fill that starts at `position' p gives element i
		word p + i (uniform, keep mask) or words p + 2i and p + 2i + 1 (normal,
		Box-Muller cosine branch). A GPU kernel handed (seed, p) reproduces any
		element on its own; that is what lets a GPU dropout mask match the CPU's.
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
			-- Generator keyed by `a_seed', at the start of its stream.
		require
			seed_valid: a_seed >= 0
		do
			seed := a_seed
			create philox.make (a_seed.to_natural_64, 0)
		ensure
			seed_set: seed = a_seed
			nothing_drawn: last_index = 0
			at_start: position = 0
		end

feature -- Access

	seed: INTEGER

	last_index: INTEGER
			-- Index drawn by the last `draw_index'.

	position: NATURAL_64
			-- Address of the next word in the stream.
		do
			Result := philox.position
		end

feature -- Basic operations

	skip (n: NATURAL_64)
			-- Reserve the next `n' words for a caller that draws them itself by address
			-- (TENSOR.dropped regenerates its mask from (seed, position) in backward).
		do
			philox.go_to (philox.position + n)
		ensure
			advanced: position = old position + n
		end

	draw_index (a_upper: INTEGER)
			-- Draw a uniform integer in 0 .. a_upper - 1 into `last_index' (one word).
		require
			positive: a_upper >= 1
		do
			last_index := (philox.unit_item * a_upper).floor.min (a_upper - 1)
			philox.forth
		ensure
			in_range: 0 <= last_index and last_index < a_upper
			one_word: position = old position + 1
		end

	fill_uniform (a_tensor: TENSOR; a_low, a_high: REAL_64)
			-- Fill `a_tensor' with values uniform in [a_low, a_high).
		require
			ordered: a_low <= a_high
			untracked: not a_tensor.is_tracked
			live: not a_tensor.is_released
		do
			fill_each (a_tensor, Uniform, a_low, a_high)
		ensure
			in_range: across 0 |..| (a_tensor.shape.count - 1) as ix all a_low <= a_tensor.item (ix) and a_tensor.item (ix) <= a_high end
			one_word_each: position = old position + a_tensor.shape.count.to_natural_64
		end

	fill_normal (a_tensor: TENSOR; a_mean, a_std: REAL_64)
			-- Gaussian fill: Box-Muller from two words per element.
		require
			std_non_negative: a_std >= 0.0
			untracked: not a_tensor.is_tracked
			live: not a_tensor.is_released
		do
			fill_each (a_tensor, Normal, a_mean, a_std)
		ensure
			two_words_each: position = old position + 2 * a_tensor.shape.count.to_natural_64
		end

	fill_keep_mask (a_tensor: TENSOR; a_keep: REAL_64)
			-- 1/a_keep with probability `a_keep', else 0 (inverted dropout mask:
			-- dropout = x * mask). Element i keeps when unit (word p + i) < a_keep.
		require
			probability: 0.0 < a_keep and a_keep <= 1.0
			untracked: not a_tensor.is_tracked
			live: not a_tensor.is_released
		do
			fill_each (a_tensor, Keep_mask, a_keep, 0.0)
		ensure
			two_values: across 0 |..| (a_tensor.shape.count - 1) as ix all a_tensor.item (ix) = 0.0 or (a_tensor.item (ix) - 1.0 / a_keep).abs <= 1.0e-6 end
			one_word_each: position = old position + a_tensor.shape.count.to_natural_64
		end

feature {NONE} -- Implementation

	philox: SIMPLE_PHILOX

	Uniform: INTEGER = 1
	Normal: INTEGER = 2
	Keep_mask: INTEGER = 3

	fill_each (a_tensor: TENSOR; a_distribution: INTEGER; p1, p2: REAL_64)
			-- Fill on the tensor's own backend (TENSOR_BACKEND.fill_random: on the GPU the
			-- words are drawn by a kernel), starting at `position', then advance past the
			-- words used: one per element, two for normal.
		local
			l_start, n: NATURAL_64
			l_kind: INTEGER
		do
			n := a_tensor.shape.count.to_natural_64
			l_start := philox.position
			inspect a_distribution
			when Uniform then
				l_kind := a_tensor.backend.Uniform_fill_kind
			when Normal then
				l_kind := a_tensor.backend.Normal_fill_kind
			else
				l_kind := a_tensor.backend.Keep_mask_fill_kind
			end
			a_tensor.backend.fill_random (a_tensor.storage, l_kind, p1, p2, seed.to_natural_64, l_start)
			if a_distribution = Normal then
				philox.go_to (l_start + 2 * n)
			else
				philox.go_to (l_start + n)
			end
		end

end
