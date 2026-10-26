note
	description: "[
		M2 module tests (oracle backend): values by hand, PyTorch naming and tying,
		gradient checks through each module, attention against a naive per-head
		REAL_64 computation from its own weights, dropout statistics.
	]"
	author: "Larry Rix"

class
	NN_TESTS

inherit
	LIB_TESTS

feature -- Tests

	test_nn_linear_by_hand
			-- y = x W^T + b with known values, 2D and 3D input.
		local
			r: TENSOR_RANDOM
			lin: NN_LINEAR
			y: TENSOR
			v: ARRAY [REAL_32]
		do
			create r.make (1)
			create lin.make_normal (oracle, 3, 2, True, 0.5, r)
			oracle.backend.upload (<<1, 2, 3, 4, 5, 6>>, lin.weight.storage)
			check attached lin.bias as b then
				oracle.backend.upload (<<0.5, -1.0>>, b.storage)
			end
			y := lin.forward (oracle.from_array (<<1, 0, -1, 2, 1, 0>>, <<2, 3>>))
			v := y.to_array
			assert ("2D values", v [1] = -1.5 and v [2] = -3.0 and v [3] = 4.5 and v [4] = 12.0)
			y := lin.forward (oracle.from_array (<<1, 0, -1, 2, 1, 0>>, <<2, 1, 3>>))
			assert ("3D shape", y.shape.rank = 3 and y.shape.dim (1) = 2 and y.shape.dim (2) = 1 and y.shape.dim (3) = 2)
			assert ("3D values", y.to_array [4] = 12.0)
			assert ("rejects a wrong last axis", not lin.accepts (oracle.zeros (<<2, 4>>)))
		end

	test_nn_names_and_tying
			-- PyTorch parameter names and order; a tied weight is listed once.
		local
			r: TENSOR_RANDOM
			att: NN_CAUSAL_SELF_ATTENTION
			pair: TIED_PAIR
			l_names: ARRAYED_LIST [TUPLE [name: STRING_8; tensor: TENSOR]]
		do
			create r.make (2)
			create att.make (oracle, 4, 2, 0.0, 0.02, r)
			l_names := att.named_parameters
			assert ("four parameters", l_names.count = 4)
			assert ("in_proj_weight", l_names [1].name.same_string ("in_proj_weight") and l_names [1].tensor.shape.dim (1) = 12)
			assert ("in_proj_bias", l_names [2].name.same_string ("in_proj_bias"))
			assert ("out_proj.weight", l_names [3].name.same_string ("out_proj.weight"))
			assert ("out_proj.bias", l_names [4].name.same_string ("out_proj.bias"))
			assert ("count", att.parameter_count = 48 + 12 + 16 + 4)
			create pair.make (oracle, r)
			assert ("tied once", pair.named_parameters.count = 1 and pair.named_parameters [1].name.same_string ("tok.weight"))
			assert ("same tensor", pair.head.weight = pair.tok.table)
			assert ("lookup", pair.parameter ("tok.weight") = pair.tok.table and pair.parameter ("head.weight") = Void)
		end

	test_nn_layer_norm
			-- Values of [1, 2, 3, 4] by hand; gradient checks through the composition.
		local
			r: TENSOR_RANDOM
			ln: NN_LAYER_NORM
			v: ARRAY [REAL_32]
			l_failures: ARRAYED_LIST [STRING]
			l_expected: REAL_64
		do
			create r.make (3)
			create ln.make (oracle, 4, 1.0e-5)
			v := ln.forward (oracle.from_array (<<1, 2, 3, 4>>, <<1, 4>>)).to_array
			l_expected := 1.5 / {DOUBLE_MATH}.sqrt (1.25 + 1.0e-5)
			assert ("first", (v [1] + l_expected).abs < 1.0e-5)
			assert ("last", (v [4] - l_expected).abs < 1.0e-5)
			create l_failures.make (0)
			grad_check ("layer_norm", agent ln.forward, uniform (<<3, 4>>, -2.0, 2.0, r), r, l_failures)
			grad_check ("layer_norm 3D", agent ln.forward, uniform (<<2, 2, 4>>, -2.0, 2.0, r), r, l_failures)
			assert ("layer norm gradients (" + l_failures.count.out + " failed)", l_failures.is_empty)
		end

	test_nn_mlp_and_attention_gradients
			-- Gradient checks through the MLP and attention (eval mode: no dropout).
		local
			r: TENSOR_RANDOM
			mlp: NN_MLP
			att: NN_CAUSAL_SELF_ATTENTION
			l_failures: ARRAYED_LIST [STRING]
		do
			create r.make (4)
			create l_failures.make (0)
			create mlp.make (oracle, 4, 8, 0.2, 0.3, r)
			mlp.set_training (False)
			grad_check ("mlp", agent mlp.forward, uniform (<<2, 3, 4>>, -1.0, 1.0, r), r, l_failures)
			create att.make (oracle, 4, 2, 0.2, 0.3, r)
			att.set_training (False)
			grad_check ("attention", agent att.forward, uniform (<<2, 3, 4>>, -1.0, 1.0, r), r, l_failures)
			across l_failures as f loop
				print ("      " + f + "%N")
			end
			assert ("module gradients (" + l_failures.count.out + " failed)", l_failures.is_empty)
		end

	test_nn_attention_matches_naive
			-- B=2, T=3, C=4, H=2: forward equals a naive REAL_64 per-head computation from
			-- the module's own weights (q, k, v row blocks; causal; softmax; concat; out_proj).
		local
			r: TENSOR_RANDOM
			att: NN_CAUSAL_SELF_ATTENTION
			x, y: TENSOR
			xv, w_in, b_in, w_o, b_o, yv: ARRAY [REAL_32]
			bb, t, s, h, i, j, c, d, l_heads: INTEGER
			q, k, v: ARRAY [REAL_64]
			l_scores: ARRAY [REAL_64]
			l_concat: ARRAY [REAL_64]
			l_max, l_sum, l_out, l_worst: REAL_64
		do
			create r.make (5)
			c := 4; l_heads := 2; d := 2
			create att.make (oracle, c, l_heads, 0.0, 0.3, r)
			att.set_training (False)
			x := uniform (<<2, 3, 4>>, -1.0, 1.0, r)
			y := att.forward (x)
			xv := x.to_array; w_in := att.in_proj_weight.to_array; b_in := att.in_proj_bias.to_array
			w_o := att.out_proj.weight.to_array
			check attached att.out_proj.bias as ob then b_o := ob.to_array end
			yv := y.to_array
			from bb := 0 until bb = 2 loop
				create q.make_filled (0.0, 0, 3 * c - 1)
				create k.make_filled (0.0, 0, 3 * c - 1)
				create v.make_filled (0.0, 0, 3 * c - 1)
					-- Projections for every position t: rows 0..C-1 q, C..2C-1 k, 2C..3C-1 v.
				from t := 0 until t = 3 loop
					from i := 0 until i = c loop
						q [t * c + i] := b_in [i + 1]; k [t * c + i] := b_in [c + i + 1]; v [t * c + i] := b_in [2 * c + i + 1]
						from j := 0 until j = c loop
							q [t * c + i] := q [t * c + i] + w_in [i * c + j + 1] * xv [(bb * 3 + t) * c + j + 1]
							k [t * c + i] := k [t * c + i] + w_in [(c + i) * c + j + 1] * xv [(bb * 3 + t) * c + j + 1]
							v [t * c + i] := v [t * c + i] + w_in [(2 * c + i) * c + j + 1] * xv [(bb * 3 + t) * c + j + 1]
							j := j + 1
						end
						i := i + 1
					end
					t := t + 1
				end
				from t := 0 until t = 3 loop
					create l_concat.make_filled (0.0, 0, c - 1)
					from h := 0 until h = l_heads loop
						create l_scores.make_filled (0.0, 0, t)
						l_max := -1.0e300
						from s := 0 until s > t loop
							from i := 0 until i = d loop
								l_scores [s] := l_scores [s] + q [t * c + h * d + i] * k [s * c + h * d + i]
								i := i + 1
							end
							l_scores [s] := l_scores [s] / {DOUBLE_MATH}.sqrt (d)
							l_max := l_max.max (l_scores [s])
							s := s + 1
						end
						l_sum := 0.0
						from s := 0 until s > t loop
							l_scores [s] := {DOUBLE_MATH}.exp (l_scores [s] - l_max); l_sum := l_sum + l_scores [s]
							s := s + 1
						end
						from s := 0 until s > t loop
							from i := 0 until i = d loop
								l_concat [h * d + i] := l_concat [h * d + i] + l_scores [s] / l_sum * v [s * c + h * d + i]
								i := i + 1
							end
							s := s + 1
						end
						h := h + 1
					end
					from i := 0 until i = c loop
						l_out := b_o [i + 1]
						from j := 0 until j = c loop
							l_out := l_out + w_o [i * c + j + 1] * l_concat [j]
							j := j + 1
						end
						l_worst := l_worst.max ((l_out - yv [(bb * 3 + t) * c + i + 1]).abs)
						i := i + 1
					end
					t := t + 1
				end
				bb := bb + 1
			end
			print ("      attention vs naive per-head computation: worst |difference| " + l_worst.out + "%N")
			assert ("attention matches the naive computation", l_worst < 1.0e-5)
		end

	test_nn_dropout
			-- Eval: identity. Training, p = 0.5: values 0 or 2, about half kept, mean near 1.
		local
			r: TENSOR_RANDOM
			dr: NN_DROPOUT
			x, y: TENSOR
			v: ARRAY [REAL_32]
			i, l_kept: INTEGER
		do
			create r.make (6)
			create dr.make (0.5, r)
			x := oracle.ones (<<10_000>>)
			dr.set_training (False)
			assert ("identity in eval", dr.forward (x) = x)
			dr.set_training (True)
			y := dr.forward (x)
			v := y.to_array
			from i := 1 until i > 10_000 loop
				assert ("two values", v [i] = 0.0 or v [i] = 2.0)
				if v [i] = 2.0 then l_kept := l_kept + 1 end
				i := i + 1
			end
			assert ("about half kept: " + l_kept.out, 4_850 <= l_kept and l_kept <= 5_150)
		end

end
