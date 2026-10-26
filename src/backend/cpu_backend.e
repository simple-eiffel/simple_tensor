note
	description: "[
		The fast CPU backend: the hot kernels (`Hot_kernels') as
		`external "C blocking inline"' on CPU_STORAGE.address (C-heap memory,
		safe to mark: D-002, spike S-3). Cold kernels come from HOST_BACKEND,
		shared with REFERENCE_BACKEND (siblings, Phase 2 verdict R2-05).

		Arithmetic matches the oracle's: elementwise and transcendental values
		are computed in double and stored as float; reductions accumulate in
		double. Matmul accumulates rows in float (i-k-j, spike S-1) for speed;
		its twin tolerance is relative to max(1, |reference|).

		Every pointer handed to C is C heap (CPU_STORAGE, INDEX_TENSOR, or a
		MANAGED_POINTER of geometry built here), never Eiffel memory: that is
		what makes `blocking' legal. Measured (spike S-1, lean, one thread):
		plain Eiffel 4.03 GFLOP/s, C inline 26.6 GFLOP/s on 256x768x768.
	]"
	author: "Larry Rix"

class
	CPU_BACKEND

inherit
	HOST_BACKEND

create
	default_create

feature -- Identity

	kind: INTEGER
		do
			Result := Cpu_backend_kind
		end

	name: STRING_8
		do
			Result := "cpu"
		end

feature -- Hot kernels: elementwise

	binary (a_kind: INTEGER; a, b, a_out: TENSOR_STORAGE; a_shape_a, a_shape_b, a_shape_out: TENSOR_SHAPE)
		local
			l_dims, l_sa, l_sb: MANAGED_POINTER
		do
			if a_shape_a ~ a_shape_out and a_shape_b ~ a_shape_out then
				c_binary_same (a_kind, address (a), address (b), address (a_out), a_out.count)
			else
				l_dims := int_array (a_shape_out.to_array)
				l_sa := int_array (broadcast_strides (a_shape_out, a_shape_a))
				l_sb := int_array (broadcast_strides (a_shape_out, a_shape_b))
				c_binary_broadcast (a_kind, address (a), address (b), address (a_out), a_shape_out.rank, l_dims.item, l_sa.item, l_sb.item, a_out.count)
			end
		end

	unary (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_param: REAL_32)
		do
			c_unary (a_kind, address (a), address (a_out), a.count, a_param)
		end

	unary_derivative (a_kind: INTEGER; a, a_y, a_out: TENSOR_STORAGE; a_param: REAL_32)
		do
			c_unary_derivative (a_kind, address (a), address (a_y), address (a_out), a.count, a_param)
		end

feature -- Hot kernels: reductions

	reduce (a_kind: INTEGER; a, a_out: TENSOR_STORAGE; a_shape_a: TENSOR_SHAPE; a_axis: INTEGER)
		do
			c_reduce (a_kind, address (a), address (a_out), outer_size (a_shape_a, a_axis), a_shape_a.dim (a_axis), inner_size (a_shape_a, a_axis))
		end

	sum_of_squares (a: TENSOR_STORAGE): REAL_64
		do
			Result := c_sum_of_squares (address (a), a.count)
		end

	has_non_finite (a: TENSOR_STORAGE): BOOLEAN
		do
			Result := c_has_non_finite (address (a), a.count) /= 0
		end

feature -- Hot kernels: linear algebra

	matmul (a, b, a_out: TENSOR_STORAGE; a_batch, a_m, a_k, a_n: INTEGER; a_ta, a_tb, a_b_shared: BOOLEAN)
		do
			c_matmul (address (a), address (b), address (a_out), a_batch, a_m, a_k, a_n, a_ta.to_integer, a_tb.to_integer, a_b_shared.to_integer)
		end

feature -- Hot kernels: neural

	softmax (a, a_out: TENSOR_STORAGE; a_rows, a_cols: INTEGER; a_log: BOOLEAN)
		do
			c_softmax (address (a), address (a_out), a_rows, a_cols, a_log.to_integer)
		end

feature -- Hot kernels: indexing

	gather_rows (a_table: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
		do
			c_gather_rows (address (a_table), a_indices.address, address (a_out), a_indices.count, a_dim)
		end

	scatter_add_rows (a_grad: TENSOR_STORAGE; a_indices: INDEX_TENSOR; a_out: TENSOR_STORAGE; a_rows, a_dim: INTEGER)
		do
			c_scatter_add_rows (address (a_grad), a_indices.address, address (a_out), a_indices.count, a_dim)
		end

feature -- Hot kernels: losses

	cross_entropy (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER)
		do
			c_cross_entropy (address (a_logits), a_targets.address, a_ignore, address (a_out), a_rows, a_classes, 0, 0.0)
		end

	cross_entropy_grad (a_logits: TENSOR_STORAGE; a_targets: INDEX_TENSOR; a_ignore: INTEGER; a_out: TENSOR_STORAGE; a_rows, a_classes: INTEGER; a_scale: REAL_32)
		do
			c_cross_entropy (address (a_logits), a_targets.address, a_ignore, address (a_out), a_rows, a_classes, 1, a_scale)
		end

feature -- Hot kernels: optimizers

	adamw_update (a_param, a_grad, a_m, a_v: TENSOR_STORAGE; a_rate, a_beta1, a_beta2, a_eps, a_decay: REAL_64; a_step: INTEGER)
		do
			c_adamw (address (a_param), address (a_grad), address (a_m), address (a_v), a_param.count, a_rate, a_beta1, a_beta2, a_eps, a_decay, a_step)
		end

feature {NONE} -- Marshalling

	address (s: TENSOR_STORAGE): POINTER
			-- C-heap address of `s' (a CPU_STORAGE: the backend owns it).
		require
			cpu: attached {CPU_STORAGE} s
		do
			check attached {CPU_STORAGE} s as c then
				Result := c.address
			end
		end

	int_array (a_values: ARRAY [INTEGER]): MANAGED_POINTER
			-- `a_values' as a C int array on the C heap.
		local
			i: INTEGER
		do
			create Result.make (a_values.count.max (1) * {PLATFORM}.integer_32_bytes)
			from i := 0 until i = a_values.count loop
				Result.put_integer_32 (a_values [a_values.lower + i], i * {PLATFORM}.integer_32_bytes)
				i := i + 1
			end
		end

	broadcast_strides (a_out, a_in: TENSOR_SHAPE): ARRAY [INTEGER]
			-- Per dim of `a_out': the stride of `a_in' along it (0 where `a_in'
			-- broadcasts or has no such dim).
		local
			k, k_in, s: INTEGER
		do
			create Result.make_filled (0, 1, a_out.rank)
			s := 1
			from k := a_out.rank until k = 0 loop
				k_in := k - (a_out.rank - a_in.rank)
				if k_in >= 1 then
					if a_in.dim (k_in) /= 1 then
						Result [k] := s
					end
					s := s * a_in.dim (k_in)
				end
				k := k - 1
			end
		end

feature {NONE} -- C kernels (all `blocking': every pointer is C heap)

	c_binary_same (a_kind: INTEGER; a, b, a_out: POINTER; n: INTEGER)
		external
			"C blocking inline"
		alias
			"[
				const float* A = (const float*)$a; const float* B = (const float*)$b; float* O = (float*)$a_out;
				int i, n = $n, k = $a_kind;
				for (i = 0; i < n; i++) {
					double x = A[i], y = B[i], z;
					if (k == 1) z = x + y; else if (k == 2) z = x - y; else if (k == 3) z = x * y; else z = x / y;
					O[i] = (float) z;
				}
			]"
		end

	c_binary_broadcast (a_kind: INTEGER; a, b, a_out: POINTER; a_rank: INTEGER; a_dims, a_sa, a_sb: POINTER; n: INTEGER)
		external
			"C blocking inline"
		alias
			"[
				const float* A = (const float*)$a; const float* B = (const float*)$b; float* O = (float*)$a_out;
				const int* D = (const int*)$a_dims; const int* SA = (const int*)$a_sa; const int* SB = (const int*)$a_sb;
				int f, j, r = $a_rank, n = $n, k = $a_kind;
				for (f = 0; f < n; f++) {
					int rem = f, ia = 0, ib = 0;
					double x, y, z;
					for (j = r - 1; j >= 0; j--) { int c = rem % D[j]; rem /= D[j]; ia += c * SA[j]; ib += c * SB[j]; }
					x = A[ia]; y = B[ib];
					if (k == 1) z = x + y; else if (k == 2) z = x - y; else if (k == 3) z = x * y; else z = x / y;
					O[f] = (float) z;
				}
			]"
		end

	c_unary (a_kind: INTEGER; a, a_out: POINTER; n: INTEGER; a_param: REAL_32)
		external
			"C blocking inline use <math.h>"
		alias
			"[
				const float* A = (const float*)$a; float* O = (float*)$a_out;
				int i, n = $n, k = $a_kind; double p = $a_param;
				for (i = 0; i < n; i++) {
					double x = A[i], z;
					switch (k) {
					case 11: z = -x; break;
					case 12: z = p * x; break;
					case 13: z = x + p; break;
					case 14: z = exp(x); break;
					case 15: z = log(x); break;
					case 16: z = sqrt(x); break;
					case 17: z = tanh(x); break;
					case 18: z = (x >= 0.0) ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); break;
					case 19: z = (x > 0.0) ? x : 0.0; break;
					case 20: z = 0.5 * x * (1.0 + erf(x / 1.4142135623730951)); break;
					default: { double s = (x >= 0.0) ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); z = x * s; }
					}
					O[i] = (float) z;
				}
			]"
		end

	c_unary_derivative (a_kind: INTEGER; a, a_y, a_out: POINTER; n: INTEGER; a_param: REAL_32)
		external
			"C blocking inline use <math.h>"
		alias
			"[
				const float* A = (const float*)$a; const float* Y = (const float*)$a_y; float* O = (float*)$a_out;
				int i, n = $n, k = $a_kind; double p = $a_param;
				for (i = 0; i < n; i++) {
					double x = A[i], y = Y[i], d;
					switch (k) {
					case 11: d = -1.0; break;
					case 12: d = p; break;
					case 13: d = 1.0; break;
					case 14: d = y; break;
					case 15: d = 1.0 / x; break;
					case 16: d = 0.5 / y; break;
					case 17: d = 1.0 - y * y; break;
					case 18: d = y * (1.0 - y); break;
					case 19: d = (x > 0.0) ? 1.0 : 0.0; break;
					case 20: d = 0.5 * (1.0 + erf(x / 1.4142135623730951)) + x * 0.3989422804014327 * exp(-0.5 * x * x); break;
					default: { double s = (x >= 0.0) ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); d = s + x * s * (1.0 - s); }
					}
					O[i] = (float) d;
				}
			]"
		end

	c_reduce (a_kind: INTEGER; a, a_out: POINTER; a_outer, n, a_inner: INTEGER)
		external
			"C blocking inline"
		alias
			"[
				const float* A = (const float*)$a; float* O = (float*)$a_out;
				int o, i, j, outer = $a_outer, nn = $n, inner = $a_inner, k = $a_kind;
				for (o = 0; o < outer; o++) {
					for (i = 0; i < inner; i++) {
						double acc;
						if (k == 33) {
							acc = A[(size_t) o * nn * inner + i];
							for (j = 1; j < nn; j++) { double v = A[((size_t) o * nn + j) * inner + i]; if (v > acc) acc = v; }
						} else {
							acc = 0.0;
							for (j = 0; j < nn; j++) acc += A[((size_t) o * nn + j) * inner + i];
							if (k == 32) acc = acc / nn;
						}
						O[(size_t) o * inner + i] = (float) acc;
					}
				}
			]"
		end

	c_sum_of_squares (a: POINTER; n: INTEGER): REAL_64
		external
			"C blocking inline"
		alias
			"[
				const float* A = (const float*)$a; int i, n = $n; double s = 0.0;
				for (i = 0; i < n; i++) { double v = A[i]; s += v * v; }
				return s;
			]"
		end

	c_has_non_finite (a: POINTER; n: INTEGER): INTEGER
		external
			"C blocking inline use <math.h>"
		alias
			"[
				const float* A = (const float*)$a; int i, n = $n;
				for (i = 0; i < n; i++) { if (!isfinite(A[i])) return 1; }
				return 0;
			]"
		end

	c_matmul (a, b, a_out: POINTER; a_batch, m, k, n, ta, tb, a_shared: INTEGER)
		external
			"C blocking inline use <stdlib.h>"
		alias
			"[
				const float* A = (const float*)$a; const float* B = (const float*)$b; float* C = (float*)$a_out;
				int bi, i, j, p, M = $m, K = $k, N = $n, TA = $ta, TB = $tb, SH = $a_shared, NB = $a_batch;
				for (bi = 0; bi < NB; bi++) {
					const float* Ab = A + (size_t) bi * M * K;
					const float* Bb = B + (SH ? 0 : (size_t) bi * K * N);
					float* Cb = C + (size_t) bi * M * N;
					for (i = 0; i < M * N; i++) Cb[i] = 0.0f;
					if (!TA && !TB) {
						for (i = 0; i < M; i++) {
							float* ci = Cb + (size_t) i * N;
							for (p = 0; p < K; p++) {
								float av = Ab[(size_t) i * K + p];
								const float* bp = Bb + (size_t) p * N;
								for (j = 0; j < N; j++) ci[j] += av * bp[j];
							}
						}
					} else if (!TA && TB) {
							/* Transpose B (N x K) once into scratch (K x N), then the fast i-k-j
							   path: a dot-product loop carries one float sum the compiler will
							   not vectorize (measured 3.7 GFLOP/s). */
						float* Bt = (float*) malloc((size_t) K * N * sizeof(float));
						if (Bt != NULL) {
							for (j = 0; j < N; j++) for (p = 0; p < K; p++) Bt[(size_t) p * N + j] = Bb[(size_t) j * K + p];
							for (i = 0; i < M; i++) {
								float* ci = Cb + (size_t) i * N;
								for (p = 0; p < K; p++) {
									float av = Ab[(size_t) i * K + p];
									const float* bp = Bt + (size_t) p * N;
									for (j = 0; j < N; j++) ci[j] += av * bp[j];
								}
							}
							free(Bt);
						} else {
							for (i = 0; i < M; i++) for (j = 0; j < N; j++) {
								float acc = 0.0f;
								for (p = 0; p < K; p++) acc += Ab[(size_t) i * K + p] * Bb[(size_t) j * K + p];
								Cb[(size_t) i * N + j] = acc;
							}
						}
					} else if (TA && !TB) {
							/* C += outer products: row p of A (as a column) times row p of B. */
						for (p = 0; p < K; p++) {
							const float* ap = Ab + (size_t) p * M;
							const float* bp = Bb + (size_t) p * N;
							for (i = 0; i < M; i++) {
								float av = ap[i];
								float* ci = Cb + (size_t) i * N;
								for (j = 0; j < N; j++) ci[j] += av * bp[j];
							}
						}
					} else {
						for (i = 0; i < M; i++) {
							for (j = 0; j < N; j++) {
								float acc = 0.0f;
								for (p = 0; p < K; p++) acc += Ab[(size_t) p * M + i] * Bb[(size_t) j * K + p];
								Cb[(size_t) i * N + j] = acc;
							}
						}
					}
				}
			]"
		end

	c_softmax (a, a_out: POINTER; a_rows, a_cols, a_log: INTEGER)
		external
			"C blocking inline use <math.h>"
		alias
			"[
				const float* A = (const float*)$a; float* O = (float*)$a_out;
				int r, c, rows = $a_rows, cols = $a_cols, lg = $a_log;
				for (r = 0; r < rows; r++) {
					const float* x = A + (size_t) r * cols; float* y = O + (size_t) r * cols;
					double mx = x[0], s = 0.0, ls;
					for (c = 1; c < cols; c++) if (x[c] > mx) mx = x[c];
					for (c = 0; c < cols; c++) s += exp(x[c] - mx);
					ls = log(s);
					for (c = 0; c < cols; c++) y[c] = lg ? (float)(x[c] - mx - ls) : (float)(exp(x[c] - mx) / s);
				}
			]"
		end

	c_gather_rows (a_table, a_indices, a_out: POINTER; a_count, a_dim: INTEGER)
		external
			"C blocking inline"
		alias
			"[
				const float* T = (const float*)$a_table; const int* I = (const int*)$a_indices; float* O = (float*)$a_out;
				int i, d, cnt = $a_count, dim = $a_dim;
				for (i = 0; i < cnt; i++) {
					const float* row = T + (size_t) I[i] * dim; float* o = O + (size_t) i * dim;
					for (d = 0; d < dim; d++) o[d] = row[d];
				}
			]"
		end

	c_scatter_add_rows (a_grad, a_indices, a_out: POINTER; a_count, a_dim: INTEGER)
		external
			"C blocking inline"
		alias
			"[
				const float* G = (const float*)$a_grad; const int* I = (const int*)$a_indices; float* O = (float*)$a_out;
				int i, d, cnt = $a_count, dim = $a_dim;
				for (i = 0; i < cnt; i++) {
					const float* g = G + (size_t) i * dim; float* o = O + (size_t) I[i] * dim;
					for (d = 0; d < dim; d++) o[d] += g[d];
				}
			]"
		end

	c_cross_entropy (a_logits, a_targets: POINTER; a_ignore: INTEGER; a_out: POINTER; a_rows, a_classes, a_grad: INTEGER; a_scale: REAL_32)
			-- Loss (a_grad = 0: out [0]) or its gradient (a_grad = 1: out = scale (softmax - onehot) / n_valid).
		external
			"C blocking inline use <math.h>"
		alias
			"[
				const float* L = (const float*)$a_logits; const int* T = (const int*)$a_targets; float* O = (float*)$a_out;
				int r, c, rows = $a_rows, cls = $a_classes, ign = $a_ignore, nv = 0;
				double total = 0.0, sc = $a_scale;
				for (r = 0; r < rows; r++) if (T[r] != ign) nv++;
				for (r = 0; r < rows; r++) {
					const float* x = L + (size_t) r * cls; int t = T[r];
					double mx, s = 0.0;
					if (t == ign) {
						if ($a_grad) for (c = 0; c < cls; c++) O[(size_t) r * cls + c] = 0.0f;
						continue;
					}
					mx = x[0];
					for (c = 1; c < cls; c++) if (x[c] > mx) mx = x[c];
					for (c = 0; c < cls; c++) s += exp(x[c] - mx);
					if ($a_grad) {
						for (c = 0; c < cls; c++) {
							double p = exp(x[c] - mx) / s;
							if (c == t) p -= 1.0;
							O[(size_t) r * cls + c] = (float)(sc * p / nv);
						}
					} else {
						total += mx + log(s) - x[t];
					}
				}
				if (!$a_grad) O[0] = (float)(total / nv);
			]"
		end

	c_adamw (a_param, a_grad, a_m, a_v: POINTER; n: INTEGER; a_rate, a_beta1, a_beta2, a_eps, a_decay: REAL_64; a_step: INTEGER)
		external
			"C blocking inline use <math.h>"
		alias
			"[
				float* P = (float*)$a_param; const float* G = (const float*)$a_grad; float* Mo = (float*)$a_m; float* V = (float*)$a_v;
				int i, cnt = $n;
				double lr = $a_rate, b1 = $a_beta1, b2 = $a_beta2, eps = $a_eps, wd = $a_decay;
				double bc1 = 1.0 - pow(b1, (double) $a_step), bc2 = 1.0 - pow(b2, (double) $a_step);
				for (i = 0; i < cnt; i++) {
					double g = G[i];
					double p = P[i] * (1.0 - lr * wd);
					double m = b1 * Mo[i] + (1.0 - b1) * g;
					double v = b2 * V[i] + (1.0 - b2) * g * g;
					p = p - lr * (m / bc1) / (sqrt(v / bc2) + eps);
					Mo[i] = (float) m; V[i] = (float) v; P[i] = (float) p;
				}
			]"
		end

end
