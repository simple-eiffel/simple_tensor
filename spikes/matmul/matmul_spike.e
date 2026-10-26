note
	description: "[
		SPIKE (research, 2026-10-25): plain-Eiffel REAL_32 matmul throughput,
		finalized, single thread. Shape = one Needle 3 projection over 256
		tokens: (256 x 768) * (768 x 768) = 302 MFLOP per multiply.
		Variants: naive i-j-k, cache-friendly i-k-j, i-k-j over 4 output rows
		at once. Prints GFLOP/s and a checksum so the loops cannot be elided.
	]"

class
	MATMUL_SPIKE

create
	make

feature {NONE} -- Initialization

	make
		local
			a, b, c: SPECIAL [REAL_32]
			i: INTEGER
		do
			create a.make_filled (0.0, M * K)
			create b.make_filled (0.0, K * N)
			create c.make_filled (0.0, M * N)
			from i := 0 until i = a.count loop
				a [i] := ((i \\ 97) - 48).to_real / 97.0
				i := i + 1
			end
			from i := 0 until i = b.count loop
				b [i] := ((i \\ 89) - 44).to_real / 89.0
				i := i + 1
			end
			time ("naive i-j-k ", agent naive (a, b, c), c)
			time ("i-k-j       ", agent ikj (a, b, c), c)
			time ("i-k-j x4 row", agent ikj4 (a, b, c), c)
			time ("C inline ikj", agent c_ikj_call (a, b, c), c)
		end

	c_ikj_call (a, b, c: SPECIAL [REAL_32])
		do
			c_ikj (a.base_address, b.base_address, c.base_address, M, K, N)
		end

	c_ikj (a, b, c: POINTER; m_, k_, n_: INTEGER)
			-- Same i-k-j loop in C on the SPECIAL areas (unmarked: touches Eiffel memory).
		external
			"C inline"
		alias
			"[
				float* A = (float*)$a; float* B = (float*)$b; float* C = (float*)$c;
				int M = $m_, K = $k_, N = $n_, i, p, j;
				for (i = 0; i < M * N; i++) C[i] = 0.0f;
				for (i = 0; i < M; i++) {
					float* ci = C + (size_t)i * N;
					for (p = 0; p < K; p++) {
						float av = A[(size_t)i * K + p];
						const float* bp = B + (size_t)p * N;
						for (j = 0; j < N; j++) ci[j] += av * bp[j];
					}
				}
			]"
		end

feature {NONE} -- Kernels (row-major: a[i*K+p], b[p*N+j], c[i*N+j])

	naive (a, b, c: SPECIAL [REAL_32])
		local
			i, j, p: INTEGER
			s: REAL_32
		do
			from i := 0 until i = M loop
				from j := 0 until j = N loop
					s := 0.0
					from p := 0 until p = K loop
						s := s + a [i * K + p] * b [p * N + j]
						p := p + 1
					end
					c [i * N + j] := s
					j := j + 1
				end
				i := i + 1
			end
		end

	ikj (a, b, c: SPECIAL [REAL_32])
		local
			i, j, p, ci, bp: INTEGER
			av: REAL_32
		do
			c.fill_with (0.0, 0, c.count - 1)
			from i := 0 until i = M loop
				ci := i * N
				from p := 0 until p = K loop
					av := a [i * K + p]
					bp := p * N
					from j := 0 until j = N loop
						c [ci + j] := c [ci + j] + av * b [bp + j]
						j := j + 1
					end
					p := p + 1
				end
				i := i + 1
			end
		end

	ikj4 (a, b, c: SPECIAL [REAL_32])
			-- Four output rows share each load of b.
		local
			i, j, p, c0, c1, c2, c3, bp: INTEGER
			a0, a1, a2, a3, bv: REAL_32
		do
			c.fill_with (0.0, 0, c.count - 1)
			from i := 0 until i = M loop
				c0 := i * N; c1 := c0 + N; c2 := c1 + N; c3 := c2 + N
				from p := 0 until p = K loop
					a0 := a [i * K + p]; a1 := a [(i + 1) * K + p]
					a2 := a [(i + 2) * K + p]; a3 := a [(i + 3) * K + p]
					bp := p * N
					from j := 0 until j = N loop
						bv := b [bp + j]
						c [c0 + j] := c [c0 + j] + a0 * bv
						c [c1 + j] := c [c1 + j] + a1 * bv
						c [c2 + j] := c [c2 + j] + a2 * bv
						c [c3 + j] := c [c3 + j] + a3 * bv
						j := j + 1
					end
					p := p + 1
				end
				i := i + 4
			end
		end

feature {NONE} -- Timing

	time (a_label: STRING; a_kernel: PROCEDURE; c: SPECIAL [REAL_32])
		local
			t0, t1: INTEGER_64
			r, l_reps: INTEGER
			l_sum: REAL_64
			l_seconds: REAL_64
			i: INTEGER
		do
			l_reps := 5
			a_kernel.call (Void)
			t0 := c_ticks
			from r := 1 until r > l_reps loop
				a_kernel.call (Void)
				r := r + 1
			end
			t1 := c_ticks
			from i := 0 until i = c.count loop
				l_sum := l_sum + c [i]
				i := i + 1
			end
			l_seconds := (t1 - t0) / c_frequency
			print (a_label + ": " + (l_seconds * 1000 / l_reps).rounded.out + " ms/multiply, "
				+ ((2.0 * M * K * N * l_reps / l_seconds / 1.0e7).rounded / 100.0).out + " GFLOP/s, checksum "
				+ l_sum.truncated_to_integer_64.out + "%N")
		end

	c_ticks: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER t; QueryPerformanceCounter(&t); return (EIF_INTEGER_64)t.QuadPart;"
		end

	c_frequency: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER f; QueryPerformanceFrequency(&f); return (EIF_INTEGER_64)f.QuadPart;"
		end

feature {NONE} -- Shape

	M: INTEGER = 256
	K: INTEGER = 768
	N: INTEGER = 768

end
