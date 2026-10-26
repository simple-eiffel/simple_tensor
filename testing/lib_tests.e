note
	description: "[
		simple_tensor tests, one per acceptance criterion of intent-v2.md.
		Phase 1: skeletal. Bodies marked Phase 5 are filled by /eiffel.verify.
	]"
	author: "Larry Rix"

class
	LIB_TESTS

inherit
	TEST_SET_BASE

	STORAGE_BUDGET_ACCESS
		undefine
			default_create
		end

feature -- Shape and storage

	test_shape_basics
			-- Shape stores dims, rank and count.
		local
			s: TENSOR_SHAPE
		do
			create s.make (<<2, 3, 4>>)
			assert ("rank", s.rank = 3)
			assert ("count", s.count = 24)
			assert ("last", s.last = 4)
		end

	test_shape_broadcast
			-- (2,3,4) with (4) and (3,1); (2,3) with (4) fails; equality, permutation, derived shapes.
		local
			a, b, c, r: TENSOR_SHAPE
		do
			create a.make (<<2, 3, 4>>)
			create b.make (<<4>>)
			create c.make (<<3, 1>>)
			assert ("(2,3,4) with (4)", a.broadcasts_with (b) and b.broadcasts_with (a))
			r := a.broadcast_with (b)
			assert ("result (2,3,4)", r ~ a)
			assert ("(2,3,4) with (3,1)", a.broadcasts_with (c))
			assert ("result again (2,3,4)", a.broadcast_with (c) ~ a)
			assert ("(2,3) with (4) fails", not (create {TENSOR_SHAPE}.make (<<2, 3>>)).broadcasts_with (b))
			assert ("(1,3) with (2,1) -> (2,3)", (create {TENSOR_SHAPE}.make (<<1, 3>>)).broadcast_with (create {TENSOR_SHAPE}.make (<<2, 1>>)) ~ create {TENSOR_SHAPE}.make (<<2, 3>>))
			assert ("not equal to a different rank", not (a ~ b))
			assert ("permutation 3,1,2", a.is_permutation (<<3, 1, 2>>))
			assert ("not a permutation 1,1,2", not a.is_permutation (<<1, 1, 2>>))
			assert ("permuted (4,2,3)", a.permuted (<<3, 1, 2>>) ~ create {TENSOR_SHAPE}.make (<<4, 2, 3>>))
			assert ("with_dim", a.with_dim (2, 7) ~ create {TENSOR_SHAPE}.make (<<2, 7, 4>>))
			assert ("without_dim", a.without_dim (2) ~ create {TENSOR_SHAPE}.make (<<2, 4>>))
			assert ("appended", b.appended (5) ~ create {TENSOR_SHAPE}.make (<<4, 5>>))
			assert ("dims_model", a.dims_model.count = 3 and a.dims_model [3] = 4)
			assert ("fits rejects 2^31 elements", not a.fits (<<65536, 32768>>))
		end

	test_storage_budget_collects
			-- Over-budget allocation forces a collection (intent-v2 Q7, spike S-4);
			-- release gives the block back; INDEX_TENSOR range queries.
		local
			b: STORAGE_BUDGET
			s: CPU_STORAGE
			ix: INDEX_TENSOR
		do
			create b.make (1_000)
			b.note_allocation (600)
			assert ("under budget: no collection", b.collections = 0 and b.allocated_since_collect = 600)
			b.note_allocation (600)
			assert ("over budget: one collection, count restarts", b.collections = 1 and b.allocated_since_collect = 600)
			assert ("largest tracked", b.largest_request = 600)
			b.note_allocation (5_000)
			assert ("a single request larger than the budget", b.collections = 2 and b.allocated_since_collect = 5_000 and b.largest_request = 5_000)
			create s.make (1024)
			assert ("live, zero-filled", not s.is_released and s.item (1023) = 0.0)
			assert ("not shared yet", not s.is_shared)
			create ix.make_from_array (<<3, -1, 7, 0>>, create {TENSOR_SHAPE}.make (<<2, 2>>))
			assert ("indices copied", ix.item (0) = 3 and ix.item (1) = -1 and ix.item (3) = 0)
			assert ("in range or ignored", ix.all_in_range_or (0, 9, -1) and not ix.all_in_range (0, 9))
			assert ("has a real target", ix.has_value_other_than (-1))
			assert ("reshaped copy", ix.reshaped (<<4>>).item (2) = 7)
		end

feature -- Ops

	test_gradient_checks_all_primitives
			-- GRADIENT_CHECKER on every primitive, REFERENCE backend, seeded inputs away from kinks.
			-- Loss = sum (op (x) * w) with w random and fixed, so every output element matters.
		local
			r: TENSOR_RANDOM
			x, y, xp, yp, a23, b34, a32, b43, a223, b234, table, logits, pos, neg: TENSOR
			l_failures: ARRAYED_LIST [STRING]
			l_indices, l_targets, l_mask: INDEX_TENSOR
			l_shape_13, l_shape_3: TENSOR_SHAPE
		do
			create r.make (2026)
			create l_failures.make (0)
			l_mask := oracle.causal_mask (3)
			create l_shape_13.make (<<1, 3>>)
			create l_shape_3.make (<<3>>)
			x := uniform (<<2, 3>>, -1.0, 1.0, r)
			y := uniform (<<3>>, 0.5, 1.5, r)
			xp := uniform (<<2, 3>>, 0.5, 2.0, r)
			pos := uniform (<<2, 3>>, 0.2, 1.0, r)
			neg := uniform (<<2, 3>>, -1.0, -0.2, r)
				-- Binary, both operands, with broadcasting (2,3) op (3)
			grad_check ("add/x", agent {TENSOR}.plus (y), x, r, l_failures)
			grad_check ("add/y", agent x.plus, y, r, l_failures)
			grad_check ("sub/x", agent {TENSOR}.minus (y), x, r, l_failures)
			grad_check ("sub/y", agent x.minus, y, r, l_failures)
			grad_check ("mul/x", agent {TENSOR}.product (y), x, r, l_failures)
			grad_check ("mul/y", agent x.product, y, r, l_failures)
			grad_check ("div/x", agent {TENSOR}.quotient (y), x, r, l_failures)
			grad_check ("div/y", agent x.quotient, y, r, l_failures)
				-- The eleven unary kinds
			grad_check ("neg", agent {TENSOR}.negated, x, r, l_failures)
			grad_check ("scale", agent {TENSOR}.scaled (1.7), x, r, l_failures)
			grad_check ("add_scalar", agent {TENSOR}.shifted (0.3), x, r, l_failures)
			grad_check ("exp", agent {TENSOR}.exponential, x, r, l_failures)
			grad_check ("log", agent {TENSOR}.logarithm, xp, r, l_failures)
			grad_check ("sqrt", agent {TENSOR}.square_root, xp, r, l_failures)
			grad_check ("tanh", agent {TENSOR}.hyperbolic_tangent, x, r, l_failures)
			grad_check ("sigmoid", agent {TENSOR}.sigmoid, x, r, l_failures)
			grad_check ("relu/positive", agent {TENSOR}.relu, pos, r, l_failures)
			grad_check ("relu/negative", agent {TENSOR}.relu, neg, r, l_failures)
			grad_check ("gelu", agent {TENSOR}.gelu, x, r, l_failures)
			grad_check ("silu", agent {TENSOR}.silu, x, r, l_failures)
				-- Reductions
			grad_check ("sum/keep", agent {TENSOR}.sum (2, True), x, r, l_failures)
			grad_check ("sum/drop", agent {TENSOR}.sum (1, False), x, r, l_failures)
			grad_check ("mean", agent {TENSOR}.mean (2, False), x, r, l_failures)
			grad_check ("max", agent {TENSOR}.maximum (2, True), x, r, l_failures)
			grad_check ("sum_all", agent {TENSOR}.sum_all, x, r, l_failures)
			grad_check ("mean_all", agent {TENSOR}.mean_all, x, r, l_failures)
			grad_check ("reduced_to (1,3)", agent {TENSOR}.reduced_to (l_shape_13), x, r, l_failures)
			grad_check ("reduced_to (3)", agent {TENSOR}.reduced_to (l_shape_3), x, r, l_failures)
				-- Shape
			grad_check ("reshaped", agent {TENSOR}.reshaped (<<3, 2>>), x, r, l_failures)
			grad_check ("permuted 2D", agent {TENSOR}.permuted (<<2, 1>>), x, r, l_failures)
			grad_check ("permuted 3D", agent {TENSOR}.permuted (<<3, 1, 2>>), uniform (<<2, 3, 2>>, -1.0, 1.0, r), r, l_failures)
			grad_check ("transposed", agent {TENSOR}.transposed, x, r, l_failures)
			grad_check ("sliced", agent {TENSOR}.sliced (2, 2, 2), uniform (<<2, 4>>, -1.0, 1.0, r), r, l_failures)
			yp := uniform (<<2, 2>>, -1.0, 1.0, r)
			grad_check ("concat/x", agent {TENSOR}.concatenated (yp, 2), x, r, l_failures)
			grad_check ("concat/y", agent x.concatenated (?, 2), yp, r, l_failures)
				-- Matmul: plain, both transposes, batched with a shared B, batched both
			a23 := uniform (<<2, 3>>, -1.0, 1.0, r)
			b34 := uniform (<<3, 4>>, -1.0, 1.0, r)
			a32 := uniform (<<3, 2>>, -1.0, 1.0, r)
			b43 := uniform (<<4, 3>>, -1.0, 1.0, r)
			a223 := uniform (<<2, 2, 3>>, -1.0, 1.0, r)
			b234 := uniform (<<2, 3, 4>>, -1.0, 1.0, r)
			grad_check ("matmul/A", agent {TENSOR}.matmul (b34), a23, r, l_failures)
			grad_check ("matmul/B", agent a23.matmul, b34, r, l_failures)
			grad_check ("matmul ta/A", agent {TENSOR}.matmul_transposed (b34, True, False), a32, r, l_failures)
			grad_check ("matmul ta/B", agent a32.matmul_transposed (?, True, False), b34, r, l_failures)
			grad_check ("matmul tb/A", agent {TENSOR}.matmul_transposed (b43, False, True), a23, r, l_failures)
			grad_check ("matmul tb/B", agent a23.matmul_transposed (?, False, True), b43, r, l_failures)
			grad_check ("matmul batched shared/A", agent {TENSOR}.matmul (b34), a223, r, l_failures)
			grad_check ("matmul batched shared/B", agent a223.matmul, b34, r, l_failures)
			grad_check ("matmul batched both/B", agent a223.matmul, b234, r, l_failures)
				-- Neural
			grad_check ("softmax", agent {TENSOR}.softmax, uniform (<<2, 4>>, -2.0, 2.0, r), r, l_failures)
			grad_check ("log_softmax", agent {TENSOR}.log_softmax, uniform (<<2, 4>>, -2.0, 2.0, r), r, l_failures)
			grad_check ("masked_fill", agent {TENSOR}.masked_fill (l_mask, 0.0), uniform (<<3, 3>>, -1.0, 1.0, r), r, l_failures)
			table := uniform (<<5, 3>>, -1.0, 1.0, r)
			l_indices := oracle.indices (<<4, 0, 2, 0>>, <<4>>)
			grad_check ("gathered_rows (repeated index)", agent {TENSOR}.gathered_rows (l_indices), table, r, l_failures)
				-- Losses
			logits := uniform (<<4, 5>>, -2.0, 2.0, r)
			l_targets := oracle.indices (<<1, 4, -1, 0>>, <<4>>)
			grad_check ("cross_entropy (one ignored)", agent {TENSOR}.cross_entropy (l_targets, -1), logits, r, l_failures)
			grad_check ("mse/a", agent {TENSOR}.mean_squared_error (yp), uniform (<<2, 2>>, -1.0, 1.0, r), r, l_failures)
			grad_check ("mse/b", agent yp.mean_squared_error, uniform (<<2, 2>>, -1.0, 1.0, r), r, l_failures)
			print ("      " + Gradient_check_count.out + " gradient checks run, worst |analytic - numeric| = " + worst_cell.item.out + "%N")
			across l_failures as f loop
				print ("      " + f + "%N")
			end
			assert ("all " + Gradient_check_count.out + " gradient checks within tolerance (" + l_failures.count.out + " failed)", l_failures.is_empty)
		end

	test_twin_parity_all_kernels
			-- TWIN_CHECKER: every hot kernel of CPU_BACKEND within 1e-5 relative of
			-- REFERENCE_BACKEND (relative to max (1, |oracle|)), forward and backward.
		local
			t: TWIN_CHECKER
		do
			create t.make (oracle, create {SIMPLE_TENSOR}.make)
			t.compare_named ("binary (same shape and broadcast)", agent twin_binary)
			t.compare_named ("unary (all eleven kinds)", agent twin_unary)
			t.compare_named ("unary_derivative (gradient of all eleven)", agent twin_unary_gradient)
			t.compare_named ("reduce (sum, mean, max)", agent twin_reduce)
			t.compare_named ("matmul plain", agent twin_matmul (?, False, False, False))
			t.compare_named ("matmul ta", agent twin_matmul (?, True, False, False))
			t.compare_named ("matmul tb", agent twin_matmul (?, False, True, False))
			t.compare_named ("matmul batched shared B", agent twin_matmul (?, False, False, True))
			t.compare_named ("softmax", agent twin_softmax (?, False))
			t.compare_named ("log_softmax", agent twin_softmax (?, True))
			t.compare_named ("gather_rows + scatter_add_rows", agent twin_gather_scatter)
			t.compare_named ("cross_entropy (loss)", agent twin_cross_entropy (?, False))
			t.compare_named ("cross_entropy_grad", agent twin_cross_entropy (?, True))
			t.compare_named ("adamw_update (3 steps)", agent twin_adamw)
			t.compare_named ("sum_of_squares + has_non_finite (global norm)", agent twin_norm)
			print ("      twin parity: worst relative error " + t.worst_relative_error.out + " (" + t.last_report + ")%N")
			assert ("CPU kernels match the oracle within 1e-5 (" + t.last_report + ")", t.within (1.0e-5))
		end

feature -- Autograd

	test_tied_weight_accumulates
			-- A tensor used twice gets the sum of both contributions:
			-- loss = sum (w*x + w) gives d/dw = x + 1.
		local
			w, x: TENSOR
			g: ARRAY [REAL_32]
		do
			w := oracle.parameter (oracle.from_array (<<0.5, -1.0, 2.0>>, <<3>>))
			x := oracle.from_array (<<3.0, 4.0, -5.0>>, <<3>>)
			g := ((w * x) + w).sum_all.backward.gradient (w).to_array
			assert ("d/dw = x + 1", g [1] = 4.0 and g [2] = 5.0 and g [3] = -4.0)
		end

	test_frozen_leaf_gets_no_gradient
			-- A leaf without requires_grad is absent from the store; only grad leaves remain.
		local
			w, x: TENSOR
			s: GRAD_STORE
		do
			w := oracle.parameter (oracle.from_array (<<1.0, 2.0>>, <<2>>))
			x := oracle.from_array (<<3.0, 4.0>>, <<2>>)
			s := (w * x).exponential.sum_all.backward
			assert ("parameter present", s.has (w))
			assert ("data leaf absent", not s.has (x))
			assert ("intermediates discarded: one entry", s.count = 1)
		end

feature -- Training

	test_adamw_hand_trace
			-- Three steps against torch.optim.AdamW's documented algorithm, computed
			-- independently (.eiffel-workflow/tools/expected_values.py, Python 3.10):
			-- w0 = [1, -0.5], loss = sum (w * x), x = [0.5, -2], lr 0.1, betas (0.9, 0.99),
			-- eps 1e-8, decay 0.1 -> [0.6732890059402, -0.18813950148505]. Both backends.
		local
			w, x: TENSOR
			opt: ADAMW_OPTIMIZER
			k: INTEGER
		do
			across <<oracle, create {SIMPLE_TENSOR}.make>> as f loop
				w := f.parameter (f.from_array (<<1.0, -0.5>>, <<2>>))
				x := f.from_array (<<0.5, -2.0>>, <<2>>)
				create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w>>), create {CONSTANT_SCHEDULE}.make (0.1), 0.9, 0.99, 1.0e-8, 0.1)
				from k := 1 until k > 3 loop
					opt.step ((w * x).sum_all.backward)
					k := k + 1
				end
				assert (f.backend.name + " w[1]", (w.item (0) - 0.6732890059402).abs < 1.0e-5)
				assert (f.backend.name + " w[2]", (w.item (1) + 0.18813950148505).abs < 1.0e-5)
				assert (f.backend.name + " three steps", opt.step_count = 3)
			end
		end

	test_warmup_cosine_matches_rixgpt
			-- rate at steps 0, 75, 150, 2575, 5000 equals rixgpt.py's lr_at (1e-3, 150, 5000, 0.1),
			-- evaluated verbatim in Python 3.10: 0, 0.0005, 0.001, 0.00055, 0.0001.
		local
			s: WARMUP_COSINE_SCHEDULE
		do
			create s.make (1.0e-3, 150, 5000, 0.1)
			assert ("step 0", s.rate (0) = 0.0)
			assert ("step 75", (s.rate (75) - 0.0005).abs < 1.0e-15)
			assert ("step 150", (s.rate (150) - 0.001).abs < 1.0e-15)
			assert ("step 2575", (s.rate (2575) - 0.00055).abs < 1.0e-15)
			assert ("step 5000", (s.rate (5000) - 0.0001).abs < 1.0e-15)
		end

	test_determinism_same_seed
			-- Same seed, identical loss sequence over 20 steps (REFERENCE and CPU), and
			-- the CPU sequence tracks the oracle's.
		local
			r1, r2, c1, c2: ARRAY [REAL_32]
		do
			r1 := training_losses (oracle, 4242)
			r2 := training_losses (oracle, 4242)
			c1 := training_losses (create {SIMPLE_TENSOR}.make, 4242)
			c2 := training_losses (create {SIMPLE_TENSOR}.make, 4242)
			assert ("reference: bit-identical", r1 ~ r2)
			assert ("cpu: bit-identical", c1 ~ c2)
			assert ("training lowers the loss", c1 [20] < c1 [1])
			assert ("cpu tracks the oracle", across 1 |..| 20 as k all (c1 [k] - r1 [k]).abs <= 1.0e-4 * r1 [k].abs.max (1.0) end)
		end

	test_non_finite_detected
			-- A NaN gradient makes GRAD_STORE.has_non_finite true (sqrt of a negative).
		local
			w: TENSOR
			s: GRAD_STORE
		do
			w := oracle.parameter (oracle.from_array (<<4.0, -1.0>>, <<2>>))
			s := w.square_root.sum_all.backward
			assert ("detected", s.has_non_finite)
			w := oracle.parameter (oracle.from_array (<<4.0, 1.0>>, <<2>>))
			assert ("finite case clean", not w.square_root.sum_all.backward.has_non_finite)
		end

feature -- I/O and data

	test_safetensors_round_trip
			-- F32 save then load is exact; F16/BF16 edge values decode correctly (hand-built
			-- file, with a __metadata__ key that must be skipped); errors are reported.
		local
			st: SAFETENSORS_FILE
			a, b: TENSOR
			l_path: STRING_32
			h: ARRAY [REAL_32]
			c: TENSOR_DTYPE_CODEC
		do
			l_path := temp_path ("st_round_trip.safetensors")
			a := oracle.from_array (<<1.5, -2.25, 3.0e-7, 4.0e9, 0.0, -1.0>>, <<2, 3>>)
			b := oracle.from_array (<<7.0, 8.0, 9.0, 10.0>>, <<4>>)
			create st.make (oracle)
			st.save (l_path, <<"layer.0.weight", "bias">>, <<a, b>>)
			assert ("saved", not st.has_error)
			create st.make (oracle)
			st.load (l_path)
			assert ("loaded", st.is_loaded)
			assert ("two names", st.names_model.count = 2 and st.has ("layer.0.weight") and st.has ("bias"))
			assert ("shape kept", st.tensor ("layer.0.weight").shape ~ a.shape)
			assert ("values exact", st.tensor ("layer.0.weight").to_array ~ a.to_array and st.tensor ("bias").to_array ~ b.to_array)
				-- Hand-built F16 + BF16 file
			write_half_file (temp_path ("st_half.safetensors"))
			create st.make (oracle)
			st.load (temp_path ("st_half.safetensors"))
			assert ({STRING_32} "half file loaded: " + st.last_error, st.is_loaded and st.names_model.count = 2)
			h := st.tensor ("h").to_array
			assert ("f16 1, -2", h [1] = 1.0 and h [2] = -2.0)
			assert ("f16 +inf, -inf, NaN", h [3].is_positive_infinity and h [4].is_negative_infinity and h [5].is_nan)
			assert ("f16 smallest subnormal 2^-24", h [6] = (1.0 / 16777216.0).truncated_to_real)
			assert ("f16 max 65504", h [7] = 65504.0)
			assert ("f16 -0", h [8] = 0.0 and (1.0 / h [8]).is_negative_infinity)
			h := st.tensor ("b").to_array
			assert ("bf16 shape 2x3", st.tensor ("b").shape.rank = 2)
			assert ("bf16 1, -2, +inf, NaN", h [1] = 1.0 and h [2] = -2.0 and h [3].is_positive_infinity and h [4].is_nan)
			assert ("bf16 subnormal 2^-133", ((h [5].to_double - 9.183549615799121e-41) / 9.183549615799121e-41).abs < 1.0e-6)
			assert ("bf16 max", ((h [6].to_double - 3.3895313892515355e38) / 3.3895313892515355e38).abs < 1.0e-7)
				-- Encoder round trips
			create c
			assert ("encode 1.0", c.f32_bits_to_f16 (1.0) = 0x3C00)
			assert ("encode max", c.f32_bits_to_f16 (65504.0) = 0x7BFF)
			assert ("encode overflow to inf", c.f32_bits_to_f16 (1.0e6) = 0x7C00)
			assert ("encode 2^-24", c.f32_bits_to_f16 ((1.0 / 16777216.0).truncated_to_real) = 0x0001)
				-- Error path
			create st.make (oracle)
			st.load (temp_path ("no_such_file.safetensors"))
			assert ("missing file reported", st.has_error and not st.is_loaded and st.names_model.is_empty)
		end

	test_token_batcher_shifted_targets
			-- targets [b, t] = inputs [b, t+1] for every row; windows come from the sequence;
			-- same seed, same batches.
		local
			l_tokens: INDEX_TENSOR
			l_values: ARRAY [INTEGER]
			b1, b2: TOKEN_BATCHER
			x, y: INDEX_TENSOR
			i, j: INTEGER
		do
			create l_values.make_filled (0, 1, 500)
			from i := 1 until i > 500 loop
				l_values [i] := (i * 7) \\ 97
				i := i + 1
			end
			l_tokens := oracle.indices (l_values, <<500>>)
			create b1.make (l_tokens, 16, 8, create {TENSOR_RANDOM}.make (99))
			create b2.make (l_tokens, 16, 8, create {TENSOR_RANDOM}.make (99))
			b1.forth
			b2.forth
			x := b1.batch.inputs
			y := b1.batch.targets
			from i := 0 until i = 8 loop
				from j := 0 until j = 15 loop
					assert ("shifted " + i.out + "," + j.out, y.item (i * 16 + j) = x.item (i * 16 + j + 1))
					j := j + 1
				end
				i := i + 1
			end
			assert ("same seed, same batch", across 0 |..| 127 as k all b2.batch.inputs.item (k) = x.item (k) end)
		end

feature -- Memory and GC law

	test_memory_stays_bounded
			-- AC-8: 1,000 steps of a small model under a 16 MB storage budget. Private
			-- memory is sampled EVERY step before anything else runs (review A-01: never
			-- sample right after a cleanup); its rise over the starting value must stay
			-- within budget + live set + margin (64 MB), while the steps churn far more.
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			x, w1, w2, l_loss: TENSOR
			targets: INDEX_TENSOR
			opt: ADAMW_OPTIMIZER
			k: INTEGER
			l_base, l_peak, l_churn_mb: INTEGER_64
			l_old_budget: INTEGER_64
		do
			l_old_budget := storage_budget.budget
			storage_budget.set_budget (16 * 1024 * 1024)
			(create {MEMORY}).full_collect
			l_base := private_mb
			create f.make
			create r.make (5)
			x := f.zeros (<<64, 128>>)
			r.fill_uniform (x, -1.0, 1.0)
			w1 := f.zeros (<<128, 256>>)
			r.fill_uniform (w1, -0.1, 0.1)
			w1 := f.parameter (w1)
			w2 := f.zeros (<<256, 10>>)
			r.fill_uniform (w2, -0.1, 0.1)
			w2 := f.parameter (w2)
			targets := f.indices (<<0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9, 0,1,2,3>>, <<64>>)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w1, w2>>), create {CONSTANT_SCHEDULE}.make (0.001), 0.9, 0.99, 1.0e-8, 0.0)
			from k := 1 until k > 1000 loop
				l_peak := l_peak.max (private_mb - l_base)
				l_loss := x.matmul (w1).gelu.matmul (w2).cross_entropy (targets, -1)
				opt.step (l_loss.backward)
				k := k + 1
			end
				-- Each step allocates at least the activations and gradients: 64x256 floats
				-- three times over plus the weight gradients, about 1 MB; 1,000 steps churn ~1 GB.
			l_churn_mb := 1000 * ((64 * 256 * 3 + 128 * 256 * 2 + 256 * 10 * 2) * 4) // (1024 * 1024)
			storage_budget.set_budget (l_old_budget)
			print ("      memory: rise over start " + l_peak.out + " MB while churning ~" + l_churn_mb.out + " MB (budget 16 MB)%N")
			assert ("rise " + l_peak.out + " MB within 64 MB", l_peak <= 64)
		end

	test_gc_probe_under_10ms
			-- AC-7: training on a separate processor must not stall the root's allocator.
			-- Root's worst allocation burst (forced full_collect per burst, growing live set:
			-- simple_process / S-3 pattern) with the worker training, minus the same
			-- measurement with no worker, stays under 10 ms.
		local
			l_worker: separate GC_PROBE_WORKER
			l_baseline, l_worst: INTEGER_64
		do
			l_baseline := worst_allocation_burst (20, 50)
			create l_worker
			launch_training (l_worker)
			l_worst := worst_allocation_burst (40, 50)
			print ("      GC probe: worst root allocation " + l_worst.out + " ms with training on another processor (" + worker_steps (l_worker).out + " steps in " + worker_elapsed (l_worker).out + " ms, launched just before a ~" + (40 * 50).out + " ms window), baseline " + l_baseline.out + " ms%N")
			assert ("training ran", worker_steps (l_worker) > 0)
			assert ("extra stall " + (l_worst - l_baseline).out + " ms under 10 ms", l_worst - l_baseline < 10)
		end

feature -- Throughput

	test_cpu_matmul_throughput
			-- AC-9: CPU_BACKEND matmul 256x768x768 >= 20 GFLOP/s single thread. Meaningful
			-- only in a LEAN build (the DbC test build is ~4x slower, S-1): this test
			-- records the figure; the gate runs in the lean benchmark target.
		local
			f: SIMPLE_TENSOR
			a, b, c: TENSOR
			t0, l_ms: INTEGER_64
			k: INTEGER
		do
			create f.make
			a := f.ones (<<256, 768>>)
			b := f.ones (<<768, 768>>)
			c := a.matmul (b)
			t0 := clock_ms
			from k := 1 until k > 5 loop
				c := a.matmul (b)
				k := k + 1
			end
			l_ms := (clock_ms - t0).max (1)
			print ("      CPU matmul (DbC test build): " + ((2.0 * 256 * 768 * 768 * 5) / (l_ms / 1000.0) / 1.0e9).truncated_to_integer.out + " GFLOP/s; the AC-9 gate runs in the lean bench target%N")
			assert ("right answer", c.item (0) = 768.0)
		end

	test_readme_quick_start
			-- The README's quick start, verbatim: a 2-8-1 network learns XOR.
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			x, y, w1, b1, w2, b2, l_loss: TENSOR
			opt: ADAMW_OPTIMIZER
			step: INTEGER
		do
			create f.make
			create r.make (42)
			x := f.from_array (<<0, 0,  0, 1,  1, 0,  1, 1>>, <<4, 2>>)
			y := f.from_array (<<0, 1, 1, 0>>, <<4, 1>>)
			w1 := f.zeros (<<2, 8>>);  r.fill_normal (w1, 0.0, 1.0);  w1 := f.parameter (w1)
			b1 := f.parameter (f.zeros (<<1, 8>>))
			w2 := f.zeros (<<8, 1>>);  r.fill_normal (w2, 0.0, 0.5);  w2 := f.parameter (w2)
			b2 := f.parameter (f.zeros (<<1, 1>>))
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w1, b1, w2, b2>>),
				create {CONSTANT_SCHEDULE}.make (0.05), 0.9, 0.999, 1.0e-8, 0.0)
			from step := 1 until step > 500 loop
				l_loss := ((x.matmul (w1) + b1).hyperbolic_tangent.matmul (w2) + b2).mean_squared_error (y)
				opt.step (l_loss.backward)
				step := step + 1
			end
			l_loss := ((x.matmul (w1) + b1).hyperbolic_tangent.matmul (w2) + b2).mean_squared_error (y)
			print ("      XOR after 500 AdamW steps: loss " + l_loss.item (0).out + "%N")
			assert ("learned XOR", l_loss.item (0) < 0.01)
		end

	test_like_factories_and_backend_copy
			-- zeros_like / ones_like values; to_backend round trip CPU -> REFERENCE -> CPU
			-- keeps every value and ends the graph (postconditions same_shape, moved, untracked).
		local
			f: SIMPLE_TENSOR
			t, z, o, on_ref, back: TENSOR
			i: INTEGER
		do
			create f.make
			t := f.parameter (f.from_array (<<1.5, -2.0, 3.25, 0.0, 7.0, -0.5>>, <<2, 3>>))
			z := t.zeros_like
			o := t.ones_like
			on_ref := t.to_backend (create {REFERENCE_BACKEND})
			back := on_ref.to_backend (f.backend)
			from i := 0 until i = 6 loop
				assert ("zero " + i.out, z.item (i) = 0.0)
				assert ("one " + i.out, o.item (i) = 1.0)
				assert ("round trip " + i.out, back.item (i) = t.item (i))
				i := i + 1
			end
			assert ("reference kind", on_ref.backend.kind = on_ref.backend.Reference_backend_kind)
			assert ("graph ends", not on_ref.is_tracked and not back.requires_grad)
		end

	test_grad_store_scale_clip_merge
			-- scale_all doubles; clip_global_norm brings the norm to the cap and leaves a
			-- store within the cap untouched; merge adds gradients and unions the ids.
		local
			f: SIMPLE_TENSOR
			a, b: TENSOR
			g1, g2, g3: GRAD_STORE
			l_norm: REAL_64
		do
			create f.make
			a := f.parameter (f.from_array (<<1.0, 2.0>>, <<2>>))
			b := f.parameter (f.from_array (<<3.0>>, <<1>>))
				-- d/da sum(a*a) = 2a = [2, 4]; d/db sum(b*b*b) = 3b^2 = [27]
			g1 := (a * a).sum_all.backward
			g1.scale_all (2.0)
			assert ("scaled 0", g1.gradient (a).item (0) = 4.0)
			assert ("scaled 1", g1.gradient (a).item (1) = 8.0)
			l_norm := g1.global_norm
			assert ("norm", (l_norm - {DOUBLE_MATH}.sqrt (80.0)).abs < 1.0e-5)
			g1.clip_global_norm (1.0)
			assert ("clipped to cap", (g1.global_norm - 1.0).abs < 1.0e-5)
			assert ("direction kept", (g1.gradient (a).item (1) / g1.gradient (a).item (0) - 2.0).abs < 1.0e-5)
			g1.clip_global_norm (10.0)
			assert ("within cap untouched", (g1.global_norm - 1.0).abs < 1.0e-5)
			g2 := (a * a).sum_all.backward
			g3 := (b * b * b).sum_all.backward
			g2.merge (g3)
			assert ("union", g2.has (a) and g2.has (b) and g2.count = 2)
			assert ("b gradient", (g2.gradient (b).item (0) - 27.0).abs < 1.0e-4)
			g3 := (a * a).sum_all.backward
			g2.merge (g3)
			assert ("accumulated", g2.gradient (a).item (1) = 8.0)
		end

	test_sgd_hand_trace
			-- SGD on sum(w*w): w := w - 0.1 * 2w = 0.8 w, on both backends.
		local
			f: SIMPLE_TENSOR
			w: TENSOR
			opt: SGD_OPTIMIZER
			l_backends: ARRAY [TENSOR_BACKEND]
		do
			l_backends := <<create {CPU_BACKEND}, create {REFERENCE_BACKEND}>>
			across l_backends as be loop
				create f.make_with_backend (be)
				w := f.parameter (f.from_array (<<1.0, -2.0, 5.0>>, <<3>>))
				create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w>>), create {CONSTANT_SCHEDULE}.make (0.1))
				opt.step ((w * w).sum_all.backward)
				opt.step ((w * w).sum_all.backward)
				assert (be.name + " step count", opt.step_count = 2)
				assert (be.name + " w0", (w.item (0) - 0.64).abs < 1.0e-6)
				assert (be.name + " w1", (w.item (1) + 1.28).abs < 1.0e-6)
				assert (be.name + " w2", (w.item (2) - 3.2).abs < 1.0e-6)
			end
		end

	test_random_index_and_dropout_mask
			-- draw_index stays in range and reaches every value; the keep mask holds
			-- 0 or 1/keep (postcondition) with the kept fraction near `keep'.
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			m: TENSOR
			seen: ARRAY [BOOLEAN]
			i, l_kept: INTEGER
		do
			create f.make
			create r.make (11)
			create seen.make_filled (False, 0, 9)
			from i := 1 until i > 1000 loop
				r.draw_index (10)
				seen [r.last_index] := True
				i := i + 1
			end
			assert ("every index drawn", across seen as s all s end)
			m := f.zeros (<<100, 100>>)
			r.fill_keep_mask (m, 0.75)
			from i := 0 until i = 10_000 loop
				if m.item (i) /= 0.0 then l_kept := l_kept + 1 end
				i := i + 1
			end
			assert ("kept fraction " + l_kept.out, 7_300 <= l_kept and l_kept <= 7_700)
		end

	test_storage_sharing_guards_release
			-- A reshaped view shares storage, so both report storage_shared and neither
			-- may be released; a fresh result is unshared and releases.
		local
			f: SIMPLE_TENSOR
			t, v, c: TENSOR
		do
			create f.make
			t := f.ones (<<2, 3>>)
			assert ("fresh unshared", not t.storage_shared)
			v := t.reshaped (<<3, 2>>)
			assert ("view shares", v.storage = t.storage and t.storage_shared and v.storage_shared)
			c := t.scaled (2.0)
			assert ("copy unshared", not c.storage_shared)
			c.release
			assert ("released", c.is_released and c.storage.is_released)
			assert ("source intact", t.item (5) = 1.0 and v.item (5) = 1.0)
		end

	test_preconditions_reject_misuse
			-- Precondition boundaries: each misuse stops at a precondition, not deeper.
		local
			f: SIMPLE_TENSOR
			a, b, v, p: TENSOR
			r: TENSOR_RANDOM
			g: GRAD_STORE
		do
			create f.make
			create r.make (1)
			a := f.ones (<<2, 3>>)
			b := f.ones (<<2, 3>>)
			v := a.reshaped (<<3, 2>>)
			p := f.parameter (f.ones (<<2>>))
			g := (p * p).sum_all.backward
			assert ("control: a valid call is not flagged", not precondition_fails (agent a.plus (b)))
			assert ("matmul inner dims must agree", precondition_fails (agent a.matmul (b)))
			assert ("no broadcast 2x3 with 3x2", precondition_fails (agent a.plus (v)))
			assert ("shared storage cannot be released", precondition_fails (agent v.release))
			assert ("a parameter cannot be released", precondition_fails (agent p.release))
			assert ("reshape keeps the count", precondition_fails (agent a.reshaped (<<4, 2>>)))
			assert ("slice inside the axis", precondition_fails (agent a.sliced (1, 2, 2)))
			assert ("keep probability in (0, 1]", precondition_fails (agent r.fill_keep_mask (b, 0.0)))
			assert ("random fill on an untracked tensor only", precondition_fails (agent r.fill_uniform (p, 0.0, 1.0)))
			assert ("clip cap positive", precondition_fails (agent g.clip_global_norm (0.0)))
			assert ("index draw upper bound positive", precondition_fails (agent r.draw_index (0)))
		end

feature -- Adversarial (Phase 6)

	test_adversarial_safetensors
			-- Hostile files: every case is refused with an explanation, never an exception
			-- and never a silent load; the same reader then loads a good file.
		local
			f: SIMPLE_TENSOR
			st: SAFETENSORS_FILE
			l_cases: ARRAYED_LIST [TUPLE [label: STRING_8; header: STRING_8; data_bytes: INTEGER; length: NATURAL_64]]
			l_path: STRING_32
			l_tiny: MANAGED_POINTER
			l_file: SIMPLE_FILE
		do
			create f.make
			create st.make (f)
			create l_cases.make (16)
			l_cases.extend (["header length 2^40", "{}", 0, ({NATURAL_64} 1).bit_shift_left (40)])
			l_cases.extend (["header longer than the file", "{}", 0, {NATURAL_64} 500])
			l_cases.extend (["header not JSON", "not json at all", 0, {NATURAL_64} 0])
			l_cases.extend (["header is a JSON array", "[1,2]", 0, {NATURAL_64} 0])
			l_cases.extend (["entry not an object", "{%"a%":5}", 0, {NATURAL_64} 0])
			l_cases.extend (["unsupported dtype", "{%"a%":{%"dtype%":%"I64%",%"shape%":[2],%"data_offsets%":[0,16]}}", 16, {NATURAL_64} 0])
			l_cases.extend (["zero dimension", "{%"a%":{%"dtype%":%"F32%",%"shape%":[0],%"data_offsets%":[0,0]}}", 0, {NATURAL_64} 0])
			l_cases.extend (["string dimension", "{%"a%":{%"dtype%":%"F32%",%"shape%":[%"x%"],%"data_offsets%":[0,4]}}", 4, {NATURAL_64} 0])
			l_cases.extend (["dims overflow", "{%"a%":{%"dtype%":%"F32%",%"shape%":[100000,100000],%"data_offsets%":[0,4]}}", 4, {NATURAL_64} 0])
			l_cases.extend (["offsets disagree with shape", "{%"a%":{%"dtype%":%"F32%",%"shape%":[2,2],%"data_offsets%":[0,8]}}", 16, {NATURAL_64} 0])
			l_cases.extend (["data past end of file", "{%"a%":{%"dtype%":%"F32%",%"shape%":[4],%"data_offsets%":[0,16]}}", 8, {NATURAL_64} 0])
			l_cases.extend (["offset into the header", "{%"a%":{%"dtype%":%"F32%",%"shape%":[2],%"data_offsets%":[-8,0]}}", 8, {NATURAL_64} 0])
			l_cases.extend (["offset before the file", "{%"a%":{%"dtype%":%"F32%",%"shape%":[2],%"data_offsets%":[-1000,-992]}}", 8, {NATURAL_64} 0])
			l_cases.extend (["three offsets", "{%"a%":{%"dtype%":%"F32%",%"shape%":[2],%"data_offsets%":[0,8,8]}}", 8, {NATURAL_64} 0])
			l_path := temp_path ("st_hostile.safetensors")
			across l_cases as c loop
				write_raw_safetensors (l_path, c.header, c.data_bytes, c.length)
				assert (c.label + ": no exception", load_survives (st, l_path))
				assert (c.label + ": refused", st.has_error and not st.is_loaded)
				assert (c.label + ": explained", not st.last_error.is_empty)
				assert (c.label + ": nothing loaded", st.names_model.is_empty)
			end
				-- A 3-byte file: too short for the length prefix.
			create l_tiny.make (3)
			create l_file.make (l_path)
			check written: l_file.write_block (l_tiny, 3, False) end
			assert ("3-byte file: no exception", load_survives (st, l_path))
			assert ("3-byte file: refused", st.has_error)
				-- The same reader recovers on a good file.
			write_raw_safetensors (l_path, "{%"a%":{%"dtype%":%"F32%",%"shape%":[2],%"data_offsets%":[0,8]}}", 8, 0)
			st.load (l_path)
			assert ("recovers on a good file", st.is_loaded and st.has ("a") and st.tensor ("a").item (1) = 0.0)
		end

	test_adversarial_numerics
			-- Extreme inputs stay finite where the mathematics is finite, on both backends;
			-- overflow that is real is flagged by has_non_finite rather than hidden.
		local
			f: SIMPLE_TENSOR
			big, sm, x, p, l_loss: TENSOR
			g: GRAD_STORE
			l_backends: ARRAY [TENSOR_BACKEND]
			l_name: STRING_8
		do
			l_backends := <<create {CPU_BACKEND}, create {REFERENCE_BACKEND}>>
			across l_backends as be loop
				create f.make_with_backend (be)
				l_name := be.name + ": "
				big := f.from_array (<<1000.0, 1001.0, 1002.0>>, <<1, 3>>)
				sm := big.softmax
				assert (l_name + "softmax of large logits finite", is_finite (sm.item (0)) and is_finite (sm.item (2)))
				assert (l_name + "softmax sums to 1", (sm.item (0) + sm.item (1) + sm.item (2) - 1.0).abs < 1.0e-5)
				assert (l_name + "softmax order", sm.item (2) > sm.item (1) and sm.item (1) > sm.item (0))
				sm := f.from_array (<<-10000.0, 0.0>>, <<1, 2>>).log_softmax
				assert (l_name + "log_softmax of -1e4 finite", is_finite (sm.item (0)) and (sm.item (0) + 10000.0).abs < 1.0)
				p := f.parameter (f.from_array (<<1000.0, -1000.0>>, <<1, 2>>))
				l_loss := p.cross_entropy (f.indices (<<1>>, <<1>>), -1)
				assert (l_name + "cross entropy of a confident miss is 2000", (l_loss.item (0) - 2000.0).abs < 1.0)
				g := l_loss.backward
				assert (l_name + "its gradient finite", not g.has_non_finite)
				assert (l_name + "its gradient is softmax - onehot", (g.gradient (p).item (0) - 1.0).abs < 1.0e-5 and (g.gradient (p).item (1) + 1.0).abs < 1.0e-5)
				x := f.from_array (<<-100.0, 100.0, -50.0, 50.0>>, <<4>>)
				assert (l_name + "sigmoid saturates", x.sigmoid.item (0) >= 0.0 and x.sigmoid.item (0) < 1.0e-30 and x.sigmoid.item (1) = 1.0)
				assert (l_name + "tanh saturates", x.hyperbolic_tangent.item (2) = -1.0 and x.hyperbolic_tangent.item (3) = 1.0)
				assert (l_name + "gelu of -50 is 0", x.gelu.item (2).abs < 1.0e-20 and x.gelu.item (3) = 50.0)
				assert (l_name + "silu of -100 near 0", x.silu.item (0).abs < 1.0e-30)
				p := f.parameter (f.from_array (<<100.0>>, <<1>>))
				g := p.exponential.sum_all.backward
				assert (l_name + "exp(100) overflow is flagged", g.has_non_finite)
			end
		end

	test_backward_twice_diamond_and_deep_chain
			-- backward is repeatable (it frees gradients, not the forward graph);
			-- a diamond accumulates; a 5,000-op chain runs (iterative DFS, no recursion).
		local
			f: SIMPLE_TENSOR
			x, y, z, l_loss: TENSOR
			g1, g2: GRAD_STORE
			k: INTEGER
			l_expected: REAL_64
		do
			create f.make
			x := f.parameter (f.from_array (<<2.0, -3.0>>, <<2>>))
			l_loss := (x * x + x.scaled (3.0)).sum_all
			g1 := l_loss.backward
			g2 := l_loss.backward
			assert ("diamond: 2x + 3", g1.gradient (x).item (0) = 7.0 and g1.gradient (x).item (1) = -3.0)
			assert ("second backward equal", g2.gradient (x).item (0) = 7.0 and g2.gradient (x).item (1) = -3.0)
			assert ("first store untouched by the second", g1.gradient (x).item (0) = 7.0)
			y := f.parameter (f.from_array (<<1.0>>, <<1>>))
			z := y
			from k := 1 until k > 5_000 loop
				z := z.scaled (1.0001)
				k := k + 1
			end
			g1 := z.sum_all.backward
			l_expected := (1.0001) ^ 5000
			assert ("deep chain gradient " + g1.gradient (y).item (0).out, (g1.gradient (y).item (0) - l_expected).abs / l_expected < 1.0e-3)
		end

	test_two_trainers_on_separate_processors
			-- Two SCOOP processors train the same seeded network at the same time;
			-- each matches the root's own run bit for bit.
		local
			w1, w2: separate TRAINING_TWIN_WORKER
			l_local: TRAINING_TWIN_WORKER
			l1, l2: REAL_32
		do
			create w1
			create w2
			launch_twin (w1, 9, 60)
			launch_twin (w2, 9, 60)
			create l_local
			l_local.train (9, 60)
			l1 := twin_loss (w1)
			l2 := twin_loss (w2)
			print ("      two processors: losses " + l1.out + ", " + l2.out + "; root " + l_local.last_loss.out + "%N")
			assert ("processor 1 equals root", l1 = l_local.last_loss)
			assert ("processor 2 equals root", l2 = l_local.last_loss)
			assert ("it trained", l_local.last_loss < 1.0)
		end

	test_stress_many_small_steps
			-- 5,000 forward/backward/AdamW steps of a small model under a 4 MB budget:
			-- no failure, the loss falls, private memory stays bounded.
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			x, w, y, l_loss: TENSOR
			opt: ADAMW_OPTIMIZER
			k: INTEGER
			l_first: REAL_32
			l_base, l_peak, l_old_budget: INTEGER_64
		do
			l_old_budget := storage_budget.budget
			storage_budget.set_budget (4 * 1024 * 1024)
			(create {MEMORY}).full_collect
			l_base := private_mb
			create f.make
			create r.make (21)
			x := f.zeros (<<8, 4>>)
			r.fill_uniform (x, -1.0, 1.0)
			y := f.zeros (<<8, 2>>)
			r.fill_uniform (y, -0.5, 0.5)
			w := f.zeros (<<4, 2>>)
			r.fill_uniform (w, -0.1, 0.1)
			w := f.parameter (w)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w>>), create {CONSTANT_SCHEDULE}.make (0.01), 0.9, 0.99, 1.0e-8, 0.0)
			l_loss := x.matmul (w).hyperbolic_tangent.mean_squared_error (y)
			l_first := l_loss.item (0)
			from k := 1 until k > 5_000 loop
				l_peak := l_peak.max (private_mb - l_base)
				l_loss := x.matmul (w).hyperbolic_tangent.mean_squared_error (y)
				opt.step (l_loss.backward)
				k := k + 1
			end
			storage_budget.set_budget (l_old_budget)
			print ("      stress: 5,000 steps, loss " + l_first.out + " -> " + l_loss.item (0).out + ", memory rise " + l_peak.out + " MB%N")
			assert ("loss fell", l_loss.item (0) < l_first)
			assert ("memory bounded", l_peak <= 32)
		end

	test_batcher_tightest_corpus
			-- The shortest corpus the batcher accepts (block + 2 tokens) has one start:
			-- every row is tokens 0..block-1, targets 1..block.
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			bt: TOKEN_BATCHER
			j, b: INTEGER
		do
			create f.make
			create r.make (3)
			create bt.make (f.indices (<<10, 11, 12, 13, 14, 15, 16, 17, 18, 19>>, <<10>>), 8, 3, r)
			bt.forth
			from b := 0 until b = 3 loop
				from j := 0 until j = 8 loop
					assert ("input " + b.out + "," + j.out, bt.batch.inputs.item (b * 8 + j) = 10 + j)
					assert ("target " + b.out + "," + j.out, bt.batch.targets.item (b * 8 + j) = 11 + j)
					j := j + 1
				end
				b := b + 1
			end
			assert ("one token too few is refused", precondition_fails (agent make_batcher (f.indices (<<1, 2, 3, 4, 5, 6, 7, 8, 9>>, <<9>>), 8, r)))
		end

	test_budget_lowered_mid_run
			-- Lowering the budget below the bytes already noted collects at once;
			-- raising it, or lowering it while still above the count, does not.
		local
			b: STORAGE_BUDGET
		do
			create b.make (256 * 1024 * 1024)
			b.note_allocation (10 * 1024 * 1024)
			b.set_budget (512 * 1024 * 1024)
			assert ("raise keeps the count", b.collections = 0 and b.allocated_since_collect = 10 * 1024 * 1024)
			b.set_budget (24 * 1024 * 1024)
			assert ("lower but above the count keeps it", b.collections = 0)
			b.note_allocation (1024)
			b.set_budget (1)
			assert ("lower below the count collects", b.collections = 1 and b.allocated_since_collect = 0)
			b.note_allocation (2)
			assert ("new budget governs", b.collections = 2 and b.allocated_since_collect = 2)
		end

	test_random_addresses_are_fixed
			-- G-6 contract for GPU parity: after any earlier draws, element i of a keep mask
			-- equals an independent SIMPLE_PHILOX's word (p + i) test; a normal fill of
			-- 100,000 has mean 0 and std 1 within 3 sigma; positions advance as specified.
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			m, z: TENSOR
			p: SIMPLE_PHILOX
			l_start: NATURAL_64
			i: INTEGER
			l_expected: REAL_32
			l_sum, l_sum_sq, l_mean, l_std: REAL_64
		do
			create f.make
			create r.make (2026)
			r.draw_index (50)
			r.fill_uniform (f.zeros (<<3, 7>>), 0.0, 1.0)
			l_start := r.position
			assert ("advanced 1 + 21", l_start = 22)
			m := f.zeros (<<64, 64>>)
			r.fill_keep_mask (m, 0.9)
			create p.make (2026, 0)
			from i := 0 until i = 4096 loop
				if p.unit_of (p.word (l_start + i.to_natural_64)) < 0.9 then
					l_expected := (1.0 / 0.9).truncated_to_real
				else
					l_expected := 0.0
				end
				assert ("mask element " + i.out, m.item (i) = l_expected)
				i := i + 1
			end
			z := f.zeros (<<100_000>>)
			r.fill_normal (z, 0.0, 1.0)
			assert ("normal used two words each", r.position = l_start + 4096 + 200_000)
			from i := 0 until i = 100_000 loop
				l_sum := l_sum + z.item (i)
				l_sum_sq := l_sum_sq + z.item (i) * z.item (i)
				i := i + 1
			end
			l_mean := l_sum / 100_000
			l_std := {DOUBLE_MATH}.sqrt (l_sum_sq / 100_000 - l_mean * l_mean)
			print ("      Philox normal fill: mean " + l_mean.out + ", std " + l_std.out + "%N")
			assert ("mean", l_mean.abs < 0.0095)
			assert ("std", (l_std - 1.0).abs < 0.01)
		end

feature {NONE} -- Probes

	make_batcher (a_tokens: INDEX_TENSOR; a_block: INTEGER; r: TENSOR_RANDOM)
		local
			bt: TOKEN_BATCHER
		do
			create bt.make (a_tokens, a_block, 1, r)
		end

	launch_twin (a_worker: separate TRAINING_TWIN_WORKER; a_seed, a_steps: INTEGER)
		do
			a_worker.train (a_seed, a_steps)
		end

	twin_loss (a_worker: separate TRAINING_TWIN_WORKER): REAL_32
		do
			Result := a_worker.last_loss
		end

	load_survives (st: SAFETENSORS_FILE; a_path: STRING_32): BOOLEAN
			-- Does `st.load (a_path)' return without an exception?
		local
			tried: BOOLEAN
		do
			if not tried then
				st.load (a_path)
				Result := True
			end
		rescue
			tried := True
			retry
		end

	is_finite (v: REAL_32): BOOLEAN
		do
			Result := v = v and v.abs <= {REAL_32}.max_value
		end

	write_raw_safetensors (a_path: STRING_32; a_header: STRING_8; a_data_bytes: INTEGER; a_length: NATURAL_64)
			-- 8-byte length (`a_length', or the header's true length when 0), header, zero data.
		local
			l_block: MANAGED_POINTER
			l_file: SIMPLE_FILE
			i: INTEGER
		do
			create l_block.make (8 + a_header.count + a_data_bytes)
			if a_length = 0 then
				l_block.put_natural_64_le (a_header.count.to_natural_64, 0)
			else
				l_block.put_natural_64_le (a_length, 0)
			end
			from i := 1 until i > a_header.count loop
				l_block.put_character (a_header [i], 8 + i - 1)
				i := i + 1
			end
			create l_file.make (a_path)
			check written: l_file.write_block (l_block, l_block.count, False) end
		end

	precondition_fails (a_call: ROUTINE): BOOLEAN
			-- Does calling `a_call' raise a precondition violation?
		local
			tried: BOOLEAN
		do
			if not tried then
				a_call.call (Void)
			end
		rescue
			if attached (create {EXCEPTION_MANAGER}).last_exception as e then
				Result := attached {PRECONDITION_VIOLATION} e or attached {PRECONDITION_VIOLATION} e.original
			end
			tried := True
			retry
		end

	launch_training (a_worker: separate GC_PROBE_WORKER)
		do
			a_worker.train (4)
		end

	worker_elapsed (a_worker: separate GC_PROBE_WORKER): INTEGER_64
		do
			Result := a_worker.elapsed_ms
		end

	worker_steps (a_worker: separate GC_PROBE_WORKER): INTEGER
		do
			Result := a_worker.steps_done
		end

	worst_allocation_burst (a_bursts, a_gap_ms: INTEGER): INTEGER_64
			-- simple_process SCOOP_TEST_APP routine: longest single burst, with a growing
			-- live set and one forced full collection per burst.
		local
			l_env: EXECUTION_ENVIRONMENT
			l_memory: MEMORY
			l_live, l_junk: ARRAYED_LIST [STRING_8]
			i, s: INTEGER
			t0, l_span: INTEGER_64
		do
			create l_env
			create l_memory
			create l_live.make (a_bursts * 200)
			from i := 1 until i > a_bursts loop
				t0 := clock_ms
				create l_junk.make (2_000)
				from s := 1 until s > 2_000 loop
					l_junk.extend (create {STRING_8}.make_filled ('x', 1_024))
					if s <= 200 then l_live.extend (l_junk.last) end
					s := s + 1
				end
				l_memory.full_collect
				l_span := clock_ms - t0
				if l_span > Result then Result := l_span end
				l_env.sleep (a_gap_ms.to_integer_64 * 1_000_000)
				i := i + 1
			end
		end

	clock_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

	private_mb: INTEGER_64
			-- Current private (commit) memory of this process, MB.
		external
			"C inline use <windows.h>, <psapi.h>"
		alias
			"[
				PROCESS_MEMORY_COUNTERS pmc;
				pmc.cb = sizeof(pmc);
				GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
				return (EIF_INTEGER_64)(pmc.PagefileUsage / (1024 * 1024));
			]"
		end

feature {NONE} -- Checkpoint fixtures

	temp_path (a_name: STRING_8): STRING_32
			-- `a_name' in the system temporary folder.
		do
			if attached (create {EXECUTION_ENVIRONMENT}).temporary_directory_path as l_dir then
				Result := l_dir.extended (a_name).name
			else
				Result := a_name
			end
		end

	write_half_file (a_path: STRING_32)
			-- A safetensors file built by hand: "h" F16 [8] edge values, "b" BF16 [2,3],
			-- and a __metadata__ entry.
		local
			l_header: STRING_8
			l_block: MANAGED_POINTER
			l_f16, l_bf16: ARRAY [INTEGER]
			l_file: SIMPLE_FILE
			i, l_start: INTEGER
		do
			l_header := "{%"__metadata__%":{%"format%":%"pt%"},%"h%":{%"dtype%":%"F16%",%"shape%":[8],%"data_offsets%":[0,16]},%"b%":{%"dtype%":%"BF16%",%"shape%":[2,3],%"data_offsets%":[16,28]}}"
			from until l_header.count \\ 8 = 0 loop
				l_header.append_character (' ')
			end
			l_f16 := <<0x3C00, 0xC000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x7BFF, 0x8000>>
			l_bf16 := <<0x3F80, 0xC000, 0x7F80, 0x7FC0, 0x0001, 0x7F7F>>
			create l_block.make (8 + l_header.count + 28)
			l_block.put_natural_64_le (l_header.count.to_natural_64, 0)
			from i := 1 until i > l_header.count loop
				l_block.put_character (l_header [i], 8 + i - 1)
				i := i + 1
			end
			l_start := 8 + l_header.count
			from i := 1 until i > 8 loop
				l_block.put_natural_16_le (l_f16 [i].to_natural_16, l_start + (i - 1) * 2)
				i := i + 1
			end
			from i := 1 until i > 6 loop
				l_block.put_natural_16_le (l_bf16 [i].to_natural_16, l_start + 16 + (i - 1) * 2)
				i := i + 1
			end
			create l_file.make (a_path)
			check written: l_file.write_block (l_block, l_block.count, False) end
		end

feature {NONE} -- Training fixture

	training_losses (f: SIMPLE_TENSOR; a_seed: INTEGER): ARRAY [REAL_32]
			-- Loss after each of 20 AdamW steps of a tiny 2-layer network
			-- (tanh hidden layer, cross-entropy) on fixed random data.
		local
			r: TENSOR_RANDOM
			x, w1, w2, b1, l_loss: TENSOR
			targets: INDEX_TENSOR
			opt: ADAMW_OPTIMIZER
			k: INTEGER
		do
			create r.make (a_seed)
			x := f.zeros (<<16, 6>>)
			r.fill_uniform (x, -1.0, 1.0)
			w1 := f.zeros (<<6, 8>>)
			r.fill_uniform (w1, -0.5, 0.5)
			w1 := f.parameter (w1)
			b1 := f.parameter (f.zeros (<<8>>))
			w2 := f.zeros (<<8, 3>>)
			r.fill_uniform (w2, -0.5, 0.5)
			w2 := f.parameter (w2)
			targets := f.indices (<<0, 1, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 2>>, <<16>>)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w1, b1, w2>>), create {CONSTANT_SCHEDULE}.make (0.05), 0.9, 0.99, 1.0e-8, 0.01)
			create Result.make_filled (0.0, 1, 20)
			from k := 1 until k > 20 loop
				l_loss := (x.matmul (w1) + b1).hyperbolic_tangent.matmul (w2).cross_entropy (targets, -1)
				Result [k] := l_loss.item (0)
				opt.step (l_loss.backward)
				k := k + 1
			end
		end

feature {NONE} -- Twin-parity cases (each builds the same seeded inputs on the facade it is given)

	seeded (f: SIMPLE_TENSOR; a_dims: ARRAY [INTEGER]; a_low, a_high: REAL_64; a_seed: INTEGER): TENSOR
		local
			r: TENSOR_RANDOM
		do
			create r.make (a_seed)
			Result := f.zeros (a_dims)
			r.fill_uniform (Result, a_low, a_high)
		end

	twin_binary (f: SIMPLE_TENSOR): TENSOR
		local
			x, y, z: TENSOR
		do
			x := seeded (f, <<4, 5>>, -2.0, 2.0, 11)
			y := seeded (f, <<5>>, 0.5, 2.0, 12)
			z := seeded (f, <<4, 5>>, -1.0, 1.0, 13)
			Result := ((x + y) * (x - y) / y) + (x * z) - (z / x.shifted (5.0))
		end

	twin_unary (f: SIMPLE_TENSOR): TENSOR
		local
			x, p: TENSOR
		do
			x := seeded (f, <<1, 16>>, -3.0, 3.0, 21)
			p := seeded (f, <<1, 16>>, 0.1, 4.0, 22)
			Result := x.negated.concatenated (x.scaled (1.7), 2).concatenated (x.shifted (0.3), 2)
				.concatenated (x.exponential, 2).concatenated (p.logarithm, 2).concatenated (p.square_root, 2)
				.concatenated (x.hyperbolic_tangent, 2).concatenated (x.sigmoid, 2).concatenated (x.relu, 2)
				.concatenated (x.gelu, 2).concatenated (x.silu, 2)
		end

	twin_unary_gradient (f: SIMPLE_TENSOR): TENSOR
		local
			x, p: TENSOR
			l_loss: TENSOR
		do
			x := f.parameter (seeded (f, <<16>>, -3.0, 3.0, 31))
			p := f.parameter (seeded (f, <<16>>, 0.1, 4.0, 32))
			l_loss := (x.negated + x.scaled (1.7) + x.shifted (0.3) + x.exponential + x.hyperbolic_tangent
				+ x.sigmoid + x.relu + x.gelu + x.silu + p.logarithm + p.square_root).sum_all
			Result := l_loss.backward.gradient (x).concatenated (l_loss.backward.gradient (p), 1)
		end

	twin_reduce (f: SIMPLE_TENSOR): TENSOR
		local
			x: TENSOR
		do
			x := seeded (f, <<3, 4, 5>>, -2.0, 2.0, 41)
			Result := x.sum (2, False).reshaped (<<15>>).concatenated (x.mean (2, False).reshaped (<<15>>), 1)
				.concatenated (x.maximum (2, False).reshaped (<<15>>), 1).concatenated (x.sum (3, True).reshaped (<<12>>), 1)
				.concatenated (x.maximum (1, False).reshaped (<<20>>), 1)
		end

	twin_matmul (f: SIMPLE_TENSOR; a_ta, a_tb, a_batched: BOOLEAN): TENSOR
		local
			a, b: TENSOR
		do
			if a_batched then
				a := seeded (f, <<3, 7, 9>>, -1.0, 1.0, 51)
			elseif a_ta then
				a := seeded (f, <<9, 7>>, -1.0, 1.0, 52)
			else
				a := seeded (f, <<7, 9>>, -1.0, 1.0, 53)
			end
			if a_tb then
				b := seeded (f, <<11, 9>>, -1.0, 1.0, 54)
			else
				b := seeded (f, <<9, 11>>, -1.0, 1.0, 55)
			end
			Result := a.matmul_transposed (b, a_ta, a_tb)
		end

	twin_softmax (f: SIMPLE_TENSOR; a_log: BOOLEAN): TENSOR
		local
			x: TENSOR
		do
			x := seeded (f, <<6, 13>>, -6.0, 6.0, 61)
			if a_log then Result := x.log_softmax else Result := x.softmax end
		end

	twin_gather_scatter (f: SIMPLE_TENSOR): TENSOR
		local
			table, w: TENSOR
			ix: INDEX_TENSOR
			l_loss: TENSOR
		do
			table := f.parameter (seeded (f, <<6, 4>>, -1.0, 1.0, 71))
			ix := f.indices (<<5, 0, 3, 0, 5, 2, 1, 5>>, <<2, 4>>)
			w := seeded (f, <<2, 4, 4>>, -1.0, 1.0, 72)
			l_loss := (table.gathered_rows (ix) * w).sum_all
			Result := table.gathered_rows (ix).reshaped (<<32>>).concatenated (l_loss.backward.gradient (table).reshaped (<<24>>), 1)
		end

	twin_cross_entropy (f: SIMPLE_TENSOR; a_gradient: BOOLEAN): TENSOR
		local
			logits: TENSOR
			targets: INDEX_TENSOR
			l_loss: TENSOR
		do
			logits := f.parameter (seeded (f, <<5, 7>>, -3.0, 3.0, 81))
			targets := f.indices (<<6, -1, 0, 3, 3>>, <<5>>)
			l_loss := logits.cross_entropy (targets, -1)
			if a_gradient then
				Result := l_loss.backward.gradient (logits)
			else
				Result := l_loss
			end
		end

	twin_adamw (f: SIMPLE_TENSOR): TENSOR
		local
			w, x: TENSOR
			opt: ADAMW_OPTIMIZER
			l_params: ARRAYED_LIST [TENSOR]
			k: INTEGER
		do
			w := f.parameter (seeded (f, <<10>>, -1.0, 1.0, 91))
			x := seeded (f, <<10>>, -2.0, 2.0, 92)
			create l_params.make_from_array (<<w>>)
			create opt.make (l_params, create {CONSTANT_SCHEDULE}.make (0.01), 0.9, 0.99, 1.0e-8, 0.1)
			from k := 1 until k > 3 loop
				opt.step (((w * x).exponential).sum_all.backward)
				k := k + 1
			end
			Result := w.detached.scaled (1.0)
		end

	twin_norm (f: SIMPLE_TENSOR): TENSOR
		local
			w: TENSOR
			s: GRAD_STORE
		do
			w := f.parameter (seeded (f, <<50>>, -2.0, 2.0, 101))
			s := (w * w * w).sum_all.backward
			if s.has_non_finite then
				Result := f.scalar (-1.0)
			else
				Result := f.scalar (s.global_norm.truncated_to_real)
			end
		end

feature {NONE} -- Gradient-check harness

	oracle: SIMPLE_TENSOR
			-- Facade on the oracle backend.
		once
			create Result.make_with_backend (create {REFERENCE_BACKEND})
		end

	uniform (a_dims: ARRAY [INTEGER]; a_low, a_high: REAL_64; r: TENSOR_RANDOM): TENSOR
			-- Untracked tensor with values uniform in [a_low, a_high).
		do
			Result := oracle.zeros (a_dims)
			r.fill_uniform (Result, a_low, a_high)
		end

	grad_check (a_name: STRING; a_op: FUNCTION [TENSOR, TENSOR]; a_x: TENSOR; r: TENSOR_RANDOM; a_failures: ARRAYED_LIST [STRING])
			-- Check d/dx of sum (op (x) * w), w random and fixed; record a failure by name.
		local
			w: TENSOR
			l_checker: GRADIENT_CHECKER
		do
			w := uniform (a_op.item ([a_x.detached]).shape.to_array, -1.0, 1.0, r)
			create l_checker.make (oracle)
			l_checker.check_function (agent weighted_sum (?, a_op, w), oracle.parameter (a_x))
			gradient_check_count_cell.put (gradient_check_count_cell.item + 1)
			worst_cell.put (worst_cell.item.max (l_checker.worst_error))
			if not l_checker.passed then
				a_failures.extend (a_name + ": " + l_checker.last_report)
			end
		end

	weighted_sum (x: TENSOR; a_op: FUNCTION [TENSOR, TENSOR]; w: TENSOR): TENSOR
			-- sum (op (x) * w)
		do
			Result := (a_op.item ([x]) * w).sum_all
		end

	Gradient_check_count: INTEGER
			-- Checks run so far.
		do
			Result := gradient_check_count_cell.item
		end

	worst_cell: CELL [REAL_64]
		once
			create Result.put (0.0)
		end

	gradient_check_count_cell: CELL [INTEGER]
		once
			create Result.put (0)
		end

end
