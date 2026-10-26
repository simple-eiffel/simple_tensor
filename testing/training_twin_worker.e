note
	description: "[
		Phase 6 SCOOP probe: trains the same seeded tiny network on its own processor.
		Two of these running at once must each match a run on the root processor bit for bit:
		the per-processor once objects (tensor ids, storage budget) must not leak between
		processors. Only reals cross the boundary.
	]"
	author: "Larry Rix"

class
	TRAINING_TWIN_WORKER

feature -- Access

	last_loss: REAL_32
			-- Loss after the last of the steps run by `train'.

feature -- Basic operations

	train (a_seed, a_steps: INTEGER)
			-- `a_steps' AdamW steps of a 6-16-3 tanh network on seeded data.
		require
			positive: a_steps >= 1
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			x, w1, w2, l_loss: TENSOR
			targets: INDEX_TENSOR
			opt: ADAMW_OPTIMIZER
			k: INTEGER
		do
			create f.make
			create r.make (a_seed)
			x := f.zeros (<<32, 6>>)
			r.fill_uniform (x, -1.0, 1.0)
			w1 := f.zeros (<<6, 16>>)
			r.fill_uniform (w1, -0.5, 0.5)
			w1 := f.parameter (w1)
			w2 := f.zeros (<<16, 3>>)
			r.fill_uniform (w2, -0.5, 0.5)
			w2 := f.parameter (w2)
			targets := f.indices (<<0,1,2,0,1,2,2,1, 0,0,1,2,1,1,0,2, 0,1,2,0,1,2,2,1, 0,0,1,2,1,1,0,2>>, <<32>>)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w1, w2>>), create {CONSTANT_SCHEDULE}.make (0.01), 0.9, 0.99, 1.0e-8, 0.01)
			from k := 1 until k > a_steps loop
				l_loss := x.matmul (w1).hyperbolic_tangent.matmul (w2).cross_entropy (targets, -1)
				last_loss := l_loss.item (0)
				opt.step (l_loss.backward)
				k := k + 1
			end
		end

end
