note
	description: "[
		The oracle twin (I-002, D-004): every hot kernel in plain Eiffel on
		C-heap storage, arithmetic and reductions in REAL_64, a `gc_point' in
		every long loop. Slow on purpose; CPU_BACKEND must agree with it.
		IEEE semantics are kept (log of 0 is -inf, sqrt of a negative is NaN),
		so DOUBLE_MATH is used for exp, log and sqrt; tanh, erf and sigmoid come
		from SIMPLE_MATH (gap G-3).
	]"
	author: "Larry Rix"

class
	REFERENCE_BACKEND

inherit
	HOST_BACKEND

create
	default_create

feature -- Identity

	kind: INTEGER
		do
			Result := Reference_backend_kind
		end

	name: STRING_8
		do
			Result := "reference"
		end

feature -- Hot kernels: elementwise

	binary (a_kind: INTEGER; a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b, a_shape_out: TENSOR_SHAPE)
		local
			i: INTEGER
			x, y, z: REAL_64
		do
			from i := 0 until i = a_out.count loop
				x := a.item (source_index (i, a_shape_out, a_shape_a))
				y := b.item (source_index (i, a_shape_out, a_shape_b))
				inspect a_kind
				when Add_kind then z := x + y
				when Sub_kind then z := x - y
				when Mul_kind then z := x * y
				else z := x / y
				end
				a_out.put (z.truncated_to_real, i)
				i := i + 1
				pace (i)
			end
		end

	unary (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_param: REAL_32)
		local
			i: INTEGER
		do
			from i := 0 until i = a.count loop
				a_out.put (unary_value (a_kind, a.item (i), a_param).truncated_to_real, i)
				i := i + 1
				pace (i)
			end
		end

	unary_derivative (a_kind: INTEGER; a, a_y, a_out: TENSOR_STORAGE; a_param: REAL_32)
		local
			i: INTEGER
			x, y, d, s: REAL_64
		do
			from i := 0 until i = a.count loop
				x := a.item (i)
				y := a_y.item (i)
				inspect a_kind
				when Neg_kind then d := -1.0
				when Scale_kind then d := a_param
				when Add_scalar_kind then d := 1.0
				when Exp_kind then d := y
				when Log_kind then d := 1.0 / x
				when Sqrt_kind then d := 0.5 / y
				when Tanh_kind then d := 1.0 - y * y
				when Sigmoid_kind then d := y * (1.0 - y)
				when Relu_kind then
					if x > 0.0 then d := 1.0 else d := 0.0 end
				when Gelu_kind then
					d := 0.5 * (1.0 + math.erf (x / Sqrt_2)) + x * Inv_sqrt_2_pi * {DOUBLE_MATH}.exp (-0.5 * x * x)
				else
						-- Silu
					s := math.sigmoid (x)
					d := s + x * s * (1.0 - s)
				end
				a_out.put (d.truncated_to_real, i)
				i := i + 1
				pace (i)
			end
		end

feature -- Hot kernels: reductions

	reduce (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
		local
			l_outer, n, l_inner, o, i, j: INTEGER
			l_acc, v: REAL_64
		do
			l_outer := outer_size (a_shape_a, a_axis)
			n := a_shape_a.dim (a_axis)
			l_inner := inner_size (a_shape_a, a_axis)
			from o := 0 until o = l_outer loop
				from i := 0 until i = l_inner loop
					if a_kind = Max_kind then
						l_acc := a.item (o * n * l_inner + i)
						from j := 1 until j = n loop
							v := a.item ((o * n + j) * l_inner + i)
							if v > l_acc then l_acc := v end
							j := j + 1
						end
					else
						l_acc := 0.0
						from j := 0 until j = n loop
							l_acc := l_acc + a.item ((o * n + j) * l_inner + i)
							j := j + 1
						end
						if a_kind = Mean_kind then
							l_acc := l_acc / n
						end
					end
					a_out.put (l_acc.truncated_to_real, o * l_inner + i)
					i := i + 1
				end
				o := o + 1
				gc_point
			end
		end

	sum_of_squares (a: TENSOR_STORAGE): REAL_64
		local
			i: INTEGER
			v: REAL_64
		do
			from i := 0 until i = a.count loop
				v := a.item (i)
				Result := Result + v * v
				i := i + 1
				pace (i)
			end
		end

	has_non_finite (a: TENSOR_STORAGE): BOOLEAN
		local
			i: INTEGER
			v: REAL_32
		do
			from i := 0 until Result or i = a.count loop
				v := a.item (i)
				Result := v.is_nan or v.is_positive_infinity or v.is_negative_infinity
				i := i + 1
				pace (i)
			end
		end

feature -- Hot kernels: linear algebra

	matmul (a, b, a_out: TENSOR_STORAGE; a_batch, a_m, a_k, a_n: INTEGER; a_ta, a_tb, a_b_shared: BOOLEAN)
		local
			bi, i, j, p, l_base_a, l_base_b, l_base_out: INTEGER
			l_acc, va, vb: REAL_64
		do
			from bi := 0 until bi = a_batch loop
				l_base_a := bi * a_m * a_k
				if a_b_shared then l_base_b := 0 else l_base_b := bi * a_k * a_n end
				l_base_out := bi * a_m * a_n
				from i := 0 until i = a_m loop
					from j := 0 until j = a_n loop
						l_acc := 0.0
						from p := 0 until p = a_k loop
							if a_ta then va := a.item (l_base_a + p * a_m + i) else va := a.item (l_base_a + i * a_k + p) end
							if a_tb then vb := b.item (l_base_b + j * a_k + p) else vb := b.item (l_base_b + p * a_n + j) end
							l_acc := l_acc + va * vb
							p := p + 1
						end
						a_out.put (l_acc.truncated_to_real, l_base_out + i * a_n + j)
						j := j + 1
					end
					i := i + 1
					gc_point
				end
				bi := bi + 1
			end
		end

feature -- Hot kernels: neural

	softmax (a, a_out: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_log: BOOLEAN)
		local
			r, c, l_base: INTEGER
			l_max, l_sum, l_log_sum: REAL_64
		do
			from r := 0 until r = a_rows loop
				l_base := r * a_cols
				l_max := a.item (l_base)
				from c := 1 until c = a_cols loop
					l_max := l_max.max (a.item (l_base + c))
					c := c + 1
				end
				l_sum := 0.0
				from c := 0 until c = a_cols loop
					l_sum := l_sum + {DOUBLE_MATH}.exp (a.item (l_base + c) - l_max)
					c := c + 1
				end
				l_log_sum := {DOUBLE_MATH}.log (l_sum)
				from c := 0 until c = a_cols loop
					if a_log then
						a_out.put ((a.item (l_base + c) - l_max - l_log_sum).truncated_to_real, l_base + c)
					else
						a_out.put (({DOUBLE_MATH}.exp (a.item (l_base + c) - l_max) / l_sum).truncated_to_real, l_base + c)
					end
					c := c + 1
				end
				r := r + 1
				gc_point
			end
		end

feature -- Hot kernels: indexing

	gather_rows (a_table: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
		local
			i, d, l_row: INTEGER
		do
			from i := 0 until i = a_indices.count loop
				l_row := a_indices.item (i)
				from d := 0 until d = a_dim loop
					a_out.put (a_table.item (l_row * a_dim + d), i * a_dim + d)
					d := d + 1
				end
				i := i + 1
				pace (i)
			end
		end

	scatter_add_rows (a_grad: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
		local
			i, d, l_row, l_target: INTEGER
		do
			from i := 0 until i = a_indices.count loop
				l_row := a_indices.item (i)
				from d := 0 until d = a_dim loop
					l_target := l_row * a_dim + d
					a_out.put (a_out.item (l_target) + a_grad.item (i * a_dim + d), l_target)
					d := d + 1
				end
				i := i + 1
				pace (i)
			end
		end

feature -- Hot kernels: losses

	cross_entropy (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER)
		local
			r, c, t, n_valid, l_base: INTEGER
			l_max, l_sum, l_total: REAL_64
		do
			from r := 0 until r = a_rows loop
				t := a_targets.item (r)
				if t /= a_ignore then
					l_base := r * a_classes
					l_max := a_logits.item (l_base)
					from c := 1 until c = a_classes loop
						l_max := l_max.max (a_logits.item (l_base + c))
						c := c + 1
					end
					l_sum := 0.0
					from c := 0 until c = a_classes loop
						l_sum := l_sum + {DOUBLE_MATH}.exp (a_logits.item (l_base + c) - l_max)
						c := c + 1
					end
					l_total := l_total + (l_max + {DOUBLE_MATH}.log (l_sum) - a_logits.item (l_base + t))
					n_valid := n_valid + 1
				end
				r := r + 1
				gc_point
			end
			a_out.put ((l_total / n_valid).truncated_to_real, 0)
		end

	cross_entropy_grad (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER; a_scale: REAL_32)
		local
			r, c, t, n_valid, l_base: INTEGER
			l_max, l_sum, p: REAL_64
		do
			from r := 0 until r = a_rows loop
				if a_targets.item (r) /= a_ignore then
					n_valid := n_valid + 1
				end
				r := r + 1
			end
			from r := 0 until r = a_rows loop
				t := a_targets.item (r)
				l_base := r * a_classes
				if t = a_ignore then
					from c := 0 until c = a_classes loop
						a_out.put (0.0, l_base + c)
						c := c + 1
					end
				else
					l_max := a_logits.item (l_base)
					from c := 1 until c = a_classes loop
						l_max := l_max.max (a_logits.item (l_base + c))
						c := c + 1
					end
					l_sum := 0.0
					from c := 0 until c = a_classes loop
						l_sum := l_sum + {DOUBLE_MATH}.exp (a_logits.item (l_base + c) - l_max)
						c := c + 1
					end
					from c := 0 until c = a_classes loop
						p := {DOUBLE_MATH}.exp (a_logits.item (l_base + c) - l_max) / l_sum
						if c = t then p := p - 1.0 end
						a_out.put ((a_scale * p / n_valid).truncated_to_real, l_base + c)
						c := c + 1
					end
				end
				r := r + 1
				gc_point
			end
		end

feature -- Hot kernels: optimizers

	adamw_update (a_param, a_grad, a_m, a_v: TENSOR_STORAGE; a_rate, a_beta1, a_beta2, a_eps, a_decay: REAL_64; a_step: INTEGER)
		local
			i: INTEGER
			p, g, m, v, l_bc1, l_bc2: REAL_64
		do
			l_bc1 := 1.0 - a_beta1 ^ a_step.to_double
			l_bc2 := 1.0 - a_beta2 ^ a_step.to_double
			from i := 0 until i = a_param.count loop
				g := a_grad.item (i)
				p := a_param.item (i) * (1.0 - a_rate * a_decay)
				m := a_beta1 * a_m.item (i) + (1.0 - a_beta1) * g
				v := a_beta2 * a_v.item (i) + (1.0 - a_beta2) * g * g
				p := p - a_rate * (m / l_bc1) / ({DOUBLE_MATH}.sqrt (v / l_bc2) + a_eps)
				a_m.put (m.truncated_to_real, i)
				a_v.put (v.truncated_to_real, i)
				a_param.put (p.truncated_to_real, i)
				i := i + 1
				pace (i)
			end
		end

feature {NONE} -- Implementation

	unary_value (a_kind: INTEGER; x: REAL_64; a_param: REAL_32): REAL_64
			-- f (x) for unary kind `a_kind'.
		do
			inspect a_kind
			when Neg_kind then Result := -x
			when Scale_kind then Result := a_param * x
			when Add_scalar_kind then Result := x + a_param
			when Exp_kind then Result := {DOUBLE_MATH}.exp (x)
			when Log_kind then Result := {DOUBLE_MATH}.log (x)
			when Sqrt_kind then Result := {DOUBLE_MATH}.sqrt (x)
			when Tanh_kind then Result := math.tanh (x)
			when Sigmoid_kind then Result := math.sigmoid (x)
			when Relu_kind then Result := x.max (0.0)
			when Gelu_kind then Result := 0.5 * x * (1.0 + math.erf (x / Sqrt_2))
			else
					-- Silu
				Result := x * math.sigmoid (x)
			end
		end

	math: SIMPLE_MATH
			-- tanh, erf, sigmoid (simple_math 0.2.0, gap G-3).
		once
			create Result.make
		end

	Sqrt_2: REAL_64 = 1.4142135623730951
	Inv_sqrt_2_pi: REAL_64 = 0.3989422804014327
			-- 1 / √(2π).

end
