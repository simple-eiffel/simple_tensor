note
	description: "[
		AC-9 benchmark (lean build only: `ec.sh release -target simple_tensor_bench',
		run F_code/simple_tensor_bench_lean.exe). CPU_BACKEND matmul throughput at the
		spike-S-1 shape (256x768 · 768x768, single thread) for each transpose flag
		combination, plus one full training step of that layer, so the backward's
		transposed matmuls are measured, not guessed.
	]"
	author: "Larry Rix"

class
	BENCH_APP

create
	make

feature {NONE} -- Initialization

	make
		local
			f: SIMPLE_TENSOR
			a, b, at, bt: TENSOR
			r: TENSOR_RANDOM
		do
			create f.make
			create r.make (3)
			a := f.zeros (<<256, 768>>)
			r.fill_uniform (a, -1.0, 1.0)
			b := f.zeros (<<768, 768>>)
			r.fill_uniform (b, -1.0, 1.0)
			at := a.transposed
			bt := b.transposed
			print ("simple_tensor bench (lean), CPU_BACKEND, one thread%N")
			report ("matmul 256x768 . 768x768", agent a.matmul_transposed (b, False, False))
			report ("matmul transpose A", agent at.matmul_transposed (b, True, False))
			report ("matmul transpose B", agent a.matmul_transposed (bt, False, True))
			train_step_report (f, a, b)
		end

feature {NONE} -- Measurement

	report (a_label: STRING; a_op: FUNCTION [TENSOR])
			-- Five timed runs after one warm-up; GFLOP/s at 2·256·768·768 per run.
		local
			t0, l_ms: INTEGER_64
			k: INTEGER
			l_gflops: REAL_64
		do
			a_op.call (Void)
			t0 := now_ms
			from k := 1 until k > 5 loop
				a_op.call (Void)
				k := k + 1
			end
			l_ms := (now_ms - t0).max (1)
			l_gflops := 2.0 * 256 * 768 * 768 * 5 / (l_ms / 1000.0) / 1.0e9
			print (a_label + ": " + (l_ms // 5).out + " ms, " + ((l_gflops * 10).rounded / 10).out + " GFLOP/s%N")
		end

	train_step_report (f: SIMPLE_TENSOR; x, w0: TENSOR)
			-- Forward + backward + AdamW of x·W, tanh, MSE: three matmuls per step.
		local
			w, y, l_loss: TENSOR
			opt: ADAMW_OPTIMIZER
			t0: INTEGER_64
			k: INTEGER
		do
			w := f.parameter (w0)
			y := f.zeros (<<256, 768>>)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w>>), create {CONSTANT_SCHEDULE}.make (0.001), 0.9, 0.99, 1.0e-8, 0.0)
			l_loss := x.matmul (w).hyperbolic_tangent.mean_squared_error (y)
			opt.step (l_loss.backward)
			t0 := now_ms
			from k := 1 until k > 5 loop
				l_loss := x.matmul (w).hyperbolic_tangent.mean_squared_error (y)
				opt.step (l_loss.backward)
				k := k + 1
			end
			print ("training step (fwd + bwd + AdamW, 3 matmuls): " + ((now_ms - t0) // 5).out + " ms%N")
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

end
