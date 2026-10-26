note
	description: "[
		S-3 probe processor: runs ~3 s of matmul kernels three ways while the
		root measures its own allocation latency.

		  `run_unmarked_special'  ONE long unmarked C call on SPECIAL memory
		                          (the only legal way to run C on Eiffel memory)
		  `run_blocking_managed'  ONE long `C blocking' call on C-heap memory
		                          (D-002 option 2)
		  `run_chunked_special'   many SHORT unmarked calls, one matmul each,
		                          on SPECIAL memory (D-002 option 1, chunked)
	]"

class
	KERNEL_RUNNER

create
	make

feature {NONE} -- Initialization

	make
		local
			i: INTEGER
		do
			create sa.make_filled (0.0, M * K)
			create sb.make_filled (0.0, K * N)
			create sc.make_filled (0.0, M * N)
			create ma.make (M * K * 4)
			create mb.make (K * N * 4)
			create mc.make (M * N * 4)
			from i := 0 until i = M * K loop
				sa [i] := ((i \\ 97) - 48).to_real / 97.0
				ma.put_real_32 (sa [i], i * 4)
				i := i + 1
			end
			from i := 0 until i = K * N loop
				sb [i] := ((i \\ 89) - 44).to_real / 89.0
				mb.put_real_32 (sb [i], i * 4)
				i := i + 1
			end
		end

feature -- Access

	runs_done: INTEGER
	elapsed_ms: INTEGER_64

feature -- Runs

	run_unmarked_special (a_reps: INTEGER)
		local
			t0: INTEGER_64
		do
			t0 := now_ms
			c_matmul_reps_unmarked (sa.base_address, sb.base_address, sc.base_address, M, K, N, a_reps)
			elapsed_ms := now_ms - t0
			runs_done := runs_done + 1
		end

	run_blocking_managed (a_reps: INTEGER)
		local
			t0: INTEGER_64
		do
			t0 := now_ms
			c_matmul_reps_blocking (ma.item, mb.item, mc.item, M, K, N, a_reps)
			elapsed_ms := now_ms - t0
			runs_done := runs_done + 1
		end

	run_chunked_special (a_reps: INTEGER)
		local
			t0: INTEGER_64
			r: INTEGER
		do
			t0 := now_ms
			from r := 1 until r > a_reps loop
				c_matmul_reps_unmarked (sa.base_address, sb.base_address, sc.base_address, M, K, N, 1)
				r := r + 1
			end
			elapsed_ms := now_ms - t0
			runs_done := runs_done + 1
		end

	run_plain_eiffel (a_reps: INTEGER)
			-- Plain-Eiffel ikj4 matmuls on SPECIAL, no allocation inside.
		local
			t0: INTEGER_64
			r, i, j, p, c0, c1, c2, c3, bp: INTEGER
			a0, a1, a2, a3, bv: REAL_32
		do
			t0 := now_ms
			from r := 1 until r > a_reps loop
				sc.fill_with (0.0, 0, sc.count - 1)
				from i := 0 until i = M loop
					c0 := i * N; c1 := c0 + N; c2 := c1 + N; c3 := c2 + N
					from p := 0 until p = K loop
						a0 := sa [i * K + p]; a1 := sa [(i + 1) * K + p]; a2 := sa [(i + 2) * K + p]; a3 := sa [(i + 3) * K + p]
						bp := p * N
						from j := 0 until j = N loop
							bv := sb [bp + j]
							sc [c0 + j] := sc [c0 + j] + a0 * bv
							sc [c1 + j] := sc [c1 + j] + a1 * bv
							sc [c2 + j] := sc [c2 + j] + a2 * bv
							sc [c3 + j] := sc [c3 + j] + a3 * bv
							j := j + 1
						end
						p := p + 1
					end
					i := i + 4
					if yields then c_yield end
				end
				r := r + 1
			end
			elapsed_ms := now_ms - t0
			runs_done := runs_done + 1
		end

	yields: BOOLEAN
			-- Should `run_plain_eiffel' pass a GC point once per 4 output rows?

	set_yields
		do
			yields := True
		end

	c_yield
			-- An empty `blocking' external: entering and leaving it lets a
			-- pending collection proceed past this thread.
		external
			"C blocking inline"
		alias
			"/* GC point */"
		end

feature {NONE} -- Data

	sa, sb, sc: SPECIAL [REAL_32]
	ma, mb, mc: MANAGED_POINTER

	M: INTEGER = 256
	K: INTEGER = 768
	N: INTEGER = 768

feature {NONE} -- Externals

	c_matmul_reps_unmarked (a, b, c: POINTER; m_, k_, n_, reps: INTEGER)
		external
			"C inline"
		alias
			"[
				float* A = (float*)$a; float* B = (float*)$b; float* C = (float*)$c;
				int M = $m_, K = $k_, N = $n_, r, i, p, j;
				for (r = 0; r < $reps; r++) {
					for (i = 0; i < M * N; i++) C[i] = 0.0f;
					for (i = 0; i < M; i++) {
						float* ci = C + (size_t)i * N;
						for (p = 0; p < K; p++) {
							float av = A[(size_t)i * K + p];
							const float* bp = B + (size_t)p * N;
							for (j = 0; j < N; j++) ci[j] += av * bp[j];
						}
					}
				}
			]"
		end

	c_matmul_reps_blocking (a, b, c: POINTER; m_, k_, n_, reps: INTEGER)
			-- Safe to mark: all three are MANAGED_POINTER blocks on the C heap.
		external
			"C blocking inline"
		alias
			"[
				float* A = (float*)$a; float* B = (float*)$b; float* C = (float*)$c;
				int M = $m_, K = $k_, N = $n_, r, i, p, j;
				for (r = 0; r < $reps; r++) {
					for (i = 0; i < M * N; i++) C[i] = 0.0f;
					for (i = 0; i < M; i++) {
						float* ci = C + (size_t)i * N;
						for (p = 0; p < K; p++) {
							float av = A[(size_t)i * K + p];
							const float* bp = B + (size_t)p * N;
							for (j = 0; j < N; j++) ci[j] += av * bp[j];
						}
					}
				}
			]"
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

end
