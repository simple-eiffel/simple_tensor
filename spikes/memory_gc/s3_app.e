note
	description: "[
		SPIKE S-3 (simple_tensor D-002): where should tensor memory live?
		A. Plain-Eiffel loop throughput over SPECIAL vs MANAGED_POINTER (lean build).
		B. GC law: root allocation latency while another processor runs ~3 s of
		   matmul kernels unmarked-on-SPECIAL, blocking-on-C-heap, or chunked.
		Burst routine copied from simple_process SCOOP_TEST_APP (forced
		full_collect per burst, growing live set).
	]"

class
	S3_APP

create
	make

feature {NONE} -- Initialization

	make
		do
			print ("=== S-3 A: plain-Eiffel ikj4 throughput (256x768x768) ===%N")
			throughput
			print ("%N=== S-3 B: root allocation latency during ~" + Reps.out + " matmuls on another processor ===%N")
			probe_unmarked
			probe_blocking
			probe_chunked
			probe_plain_eiffel
			probe_plain_eiffel_yielding
		end

feature {NONE} -- Part A

	throughput
		local
			sa, sb, sc: SPECIAL [REAL_32]
			ma, mb, mc: MANAGED_POINTER
			i: INTEGER
			t0: INTEGER_64
		do
			create sa.make_filled (0.0, M * K); create sb.make_filled (0.0, K * N); create sc.make_filled (0.0, M * N)
			create ma.make (M * K * 4); create mb.make (K * N * 4); create mc.make (M * N * 4)
			from i := 0 until i = M * K loop
				sa [i] := ((i \\ 97) - 48).to_real / 97.0; ma.put_real_32 (sa [i], i * 4); i := i + 1
			end
			from i := 0 until i = K * N loop
				sb [i] := ((i \\ 89) - 44).to_real / 89.0; mb.put_real_32 (sb [i], i * 4); i := i + 1
			end
			ikj4_special (sa, sb, sc)
			t0 := now_ms
			from i := 1 until i > 3 loop ikj4_special (sa, sb, sc); i := i + 1 end
			print ("SPECIAL         : " + gflops (now_ms - t0, 3) + " GFLOP/s%N")
			ikj4_managed (ma, mb, mc)
			t0 := now_ms
			from i := 1 until i > 3 loop ikj4_managed (ma, mb, mc); i := i + 1 end
			print ("MANAGED_POINTER : " + gflops (now_ms - t0, 3) + " GFLOP/s%N")
			print ("same result     : " + (sc [1000] = mc.read_real_32 (4000) and sc [M * N - 1] = mc.read_real_32 ((M * N - 1) * 4)).out + "%N")
		end

	ikj4_special (a, b, c: SPECIAL [REAL_32])
		local
			i, j, p, c0, c1, c2, c3, bp: INTEGER
			a0, a1, a2, a3, bv: REAL_32
		do
			c.fill_with (0.0, 0, c.count - 1)
			from i := 0 until i = M loop
				c0 := i * N; c1 := c0 + N; c2 := c1 + N; c3 := c2 + N
				from p := 0 until p = K loop
					a0 := a [i * K + p]; a1 := a [(i + 1) * K + p]; a2 := a [(i + 2) * K + p]; a3 := a [(i + 3) * K + p]
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

	ikj4_managed (a, b, c: MANAGED_POINTER)
		local
			i, j, p, c0, c1, c2, c3, bp: INTEGER
			a0, a1, a2, a3, bv: REAL_32
		do
			from i := 0 until i = M * N loop c.put_real_32 (0.0, i * 4); i := i + 1 end
			from i := 0 until i = M loop
				c0 := i * N * 4; c1 := c0 + N * 4; c2 := c1 + N * 4; c3 := c2 + N * 4
				from p := 0 until p = K loop
					a0 := a.read_real_32 ((i * K + p) * 4); a1 := a.read_real_32 (((i + 1) * K + p) * 4)
					a2 := a.read_real_32 (((i + 2) * K + p) * 4); a3 := a.read_real_32 (((i + 3) * K + p) * 4)
					bp := p * N * 4
					from j := 0 until j = N * 4 loop
						bv := b.read_real_32 (bp + j)
						c.put_real_32 (c.read_real_32 (c0 + j) + a0 * bv, c0 + j)
						c.put_real_32 (c.read_real_32 (c1 + j) + a1 * bv, c1 + j)
						c.put_real_32 (c.read_real_32 (c2 + j) + a2 * bv, c2 + j)
						c.put_real_32 (c.read_real_32 (c3 + j) + a3 * bv, c3 + j)
						j := j + 4
					end
					p := p + 1
				end
				i := i + 4
			end
		end

feature {NONE} -- Part B

	probe_unmarked
		local
			r: separate KERNEL_RUNNER
			w: INTEGER_64
		do
			create r.make
			start_unmarked (r)
			w := worst_allocation_burst (Bursts, Gap_ms)
			print ("unmarked ONE long call on SPECIAL  : kernel " + kernel_ms (r).out + " ms, worst root allocation " + w.out + " ms%N")
		end

	probe_blocking
		local
			r: separate KERNEL_RUNNER
			w: INTEGER_64
		do
			create r.make
			start_blocking (r)
			w := worst_allocation_burst (Bursts, Gap_ms)
			print ("BLOCKING one long call on C heap    : kernel " + kernel_ms (r).out + " ms, worst root allocation " + w.out + " ms%N")
		end

	probe_chunked
		local
			r: separate KERNEL_RUNNER
			w: INTEGER_64
		do
			create r.make
			start_chunked (r)
			w := worst_allocation_burst (Bursts, Gap_ms)
			print ("unmarked CHUNKED (1 matmul/call)    : kernel " + kernel_ms (r).out + " ms, worst root allocation " + w.out + " ms%N")
		end

	probe_plain_eiffel
		local
			r: separate KERNEL_RUNNER
			w: INTEGER_64
		do
			create r.make
			start_plain (r)
			w := worst_allocation_burst (Bursts, Gap_ms)
			print ("PLAIN EIFFEL loop (no allocation)   : kernel " + kernel_ms (r).out + " ms, worst root allocation " + w.out + " ms%N")
		end

	start_plain (r: separate KERNEL_RUNNER) do r.run_plain_eiffel (Plain_reps) end

	probe_plain_eiffel_yielding
		local
			r: separate KERNEL_RUNNER
			w: INTEGER_64
		do
			create r.make
			start_plain_yielding (r)
			w := worst_allocation_burst (Bursts, Gap_ms)
			print ("PLAIN EIFFEL + blocking no-op/4 rows: kernel " + kernel_ms (r).out + " ms, worst root allocation " + w.out + " ms%N")
		end

	start_plain_yielding (r: separate KERNEL_RUNNER)
		do
			r.set_yields
			r.run_plain_eiffel (Plain_reps)
		end

	Plain_reps: INTEGER = 35

	start_unmarked (r: separate KERNEL_RUNNER) do r.run_unmarked_special (Reps) end
	start_blocking (r: separate KERNEL_RUNNER) do r.run_blocking_managed (Reps) end
	start_chunked (r: separate KERNEL_RUNNER) do r.run_chunked_special (Reps) end

	kernel_ms (r: separate KERNEL_RUNNER): INTEGER_64
		require
			finished: r.runs_done = 1
		do
			Result := r.elapsed_ms
		end

	worst_allocation_burst (a_bursts, a_gap_ms: INTEGER): INTEGER_64
			-- simple_process SCOOP_TEST_APP routine: longest single burst, with a
			-- growing live set and one forced full collection per burst.
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
				t0 := now_ms
				create l_junk.make (2_000)
				from s := 1 until s > 2_000 loop
					l_junk.extend (create {STRING_8}.make_filled ('x', 1_024))
					if s <= 200 then l_live.extend (l_junk.last) end
					s := s + 1
				end
				l_memory.full_collect
				l_span := now_ms - t0
				if l_span > Result then Result := l_span end
				l_env.sleep (a_gap_ms.to_integer_64 * 1_000_000)
				i := i + 1
			end
		end

feature {NONE} -- Helpers

	gflops (a_ms: INTEGER_64; a_reps: INTEGER): STRING
		do
			Result := ((2.0 * M * K * N * a_reps / (a_ms / 1000.0) / 1.0e7).rounded / 100.0).out
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

	M: INTEGER = 256
	K: INTEGER = 768
	N: INTEGER = 768
	Reps: INTEGER = 250
	Bursts: INTEGER = 60
	Gap_ms: INTEGER = 100

end
