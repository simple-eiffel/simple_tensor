note
	description: "[
		AC-7 probe processor: runs real training work on the CPU backend (forward,
		backward, AdamW on a 256x768 x 768x768 layer) for about two seconds while
		the root measures its own allocation latency. Builds every tensor on its
		own processor; only integers and reals cross the boundary (R2-13).
	]"
	author: "Larry Rix"

class
	GC_PROBE_WORKER

feature -- Access

	steps_done: INTEGER
	elapsed_ms: INTEGER_64

feature -- Basic operations

	train (a_steps: INTEGER)
			-- `a_steps' training steps of one linear layer with tanh and MSE.
		require
			positive: a_steps >= 1
		local
			f: SIMPLE_TENSOR
			r: TENSOR_RANDOM
			x, w, y, l_loss: TENSOR
			opt: ADAMW_OPTIMIZER
			k: INTEGER
			t0: INTEGER_64
		do
			t0 := now_ms
			create f.make
			create r.make (7)
			x := f.zeros (<<256, 768>>)
			r.fill_uniform (x, -1.0, 1.0)
			w := f.zeros (<<768, 768>>)
			r.fill_uniform (w, -0.05, 0.05)
			w := f.parameter (w)
			y := f.zeros (<<256, 768>>)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w>>), create {CONSTANT_SCHEDULE}.make (0.001), 0.9, 0.99, 1.0e-8, 0.0)
			from k := 1 until k > a_steps loop
				l_loss := x.matmul (w).hyperbolic_tangent.mean_squared_error (y)
				opt.step (l_loss.backward)
				k := k + 1
			end
			steps_done := a_steps
			elapsed_ms := now_ms - t0
		end

feature {NONE} -- Clock

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

end
