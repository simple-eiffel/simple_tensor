note
	description: "[
		Random windows over a token sequence (rixgpt.py's get_batch): for each
		of `batch_size' rows, a start s drawn uniformly, inputs = tokens [s ..
		s+block-1], targets = tokens [s+1 .. s+block]. CQS: `forth' draws,
		`batch' reads.
	]"
	author: "Larry Rix"

class
	TOKEN_BATCHER

inherit
	GC_POINT

create
	make

feature {NONE} -- Initialization

	make (a_tokens: INDEX_TENSOR; a_block, a_batch: INTEGER; a_random: TENSOR_RANDOM)
		require
			one_dimensional: a_tokens.shape.rank = 1
			positive: a_block >= 1 and a_batch >= 1
			long_enough: a_tokens.count > a_block + 1
		do
			tokens := a_tokens
			block_size := a_block
			batch_size := a_batch
			random := a_random
		ensure
			tokens_set: tokens = a_tokens
			sizes_set: block_size = a_block and batch_size = a_batch
			no_batch_yet: not has_batch
		end

feature -- Access

	tokens: INDEX_TENSOR
	block_size: INTEGER
	batch_size: INTEGER

	batch: TOKEN_BATCH
			-- The batch drawn by the last `forth'.
		require
			drawn: has_batch
		do
			check attached last_batch as b then
				Result := b
			end
		ensure
			batch_rows: Result.inputs.shape.dim (1) = batch_size
			block_cols: Result.inputs.shape.dim (2) = block_size
		end

feature -- Status report

	has_batch: BOOLEAN
		do
			Result := last_batch /= Void
		end

feature -- Cursor movement

	forth
			-- Draw the next batch (advances the generator). rixgpt.py get_batch:
			-- start s uniform in 0 .. count - block - 2; x = t [s .. s+block-1],
			-- y = t [s+1 .. s+block].
		local
			l_shape: TENSOR_SHAPE
			l_inputs, l_targets: INDEX_TENSOR
			b, j, s: INTEGER
		do
			create l_shape.make (<<batch_size, block_size>>)
			create l_inputs.make (l_shape)
			create l_targets.make (l_shape)
			from b := 0 until b = batch_size loop
				random.draw_index (tokens.count - block_size - 1)
				s := random.last_index
				from j := 0 until j = block_size loop
					l_inputs.put (tokens.item (s + j), b * block_size + j)
					l_targets.put (tokens.item (s + j + 1), b * block_size + j)
					j := j + 1
				end
				b := b + 1
				gc_point
			end
			create last_batch.make (l_inputs, l_targets)
		ensure
			drawn: has_batch
		end

feature {NONE} -- Implementation

	random: TENSOR_RANDOM

	last_batch: detachable TOKEN_BATCH

invariant
	positive: block_size >= 1 and batch_size >= 1

end
