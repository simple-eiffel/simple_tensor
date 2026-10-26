note
	description: "[
		M3b gate 4 bench (lean: ec.sh release -config simple_tensor_vulkan.ecf
		-target simple_tensor_vulkan_bench). The same training steps on CPU_BACKEND
		and VULKAN_BACKEND: forward, backward, AdamW, ending with the loss read back
		(which waits for the GPU), so the GPU time is complete work, not recording.
	]"
	author: "Larry Rix"

class
	GPU_BENCH_APP

create
	make

feature {NONE} -- Initialization

	make
		local
			cpu, gpu: SIMPLE_TENSOR
		do
			create cpu.make
			create gpu.make_with_backend (create {VULKAN_BACKEND}.make)
			print ("simple_tensor GPU bench (lean)%N")
			if attached {VULKAN_BACKEND} gpu.backend as v then
				print ("GPU: " + v.device_name + "%N")
			end
			layer_step (cpu, "CPU", 256, 768, 768, 5)
			layer_step (gpu, "GPU", 256, 768, 768, 20)
			mlp_step (cpu, "CPU", 4096, 768, 3072, 2)
			mlp_step (gpu, "GPU", 4096, 768, 3072, 10)
			matmul_only (gpu, "GPU", 2048, 2048, 2048, 20)
			if attached {VULKAN_BACKEND} gpu.backend as v then
					-- A/B of the matmul tiles: 64 x 64 against 128 x 128, square and at the
					-- RixGPT MLP shape (one micro-batch of 32 x 256 tokens, 384 -> 1536).
				v.set_wide_tiles (False)
				matmul_only (gpu, "GPU 64-tile", 2048, 2048, 2048, 20)
				matmul_only (gpu, "GPU 64-tile", 8192, 384, 1536, 20)
				v.set_wide_tiles (True)
				matmul_only (gpu, "GPU 128-tile", 2048, 2048, 2048, 20)
				matmul_only (gpu, "GPU 128-tile", 8192, 384, 1536, 20)
					-- Wall time against GPU kernel time for the RixGPT shape.
				v.set_profiling (True)
				v.set_wide_tiles (False)
				v.reset_profile
				matmul_only (gpu, "GPU 64-tile profiled", 8192, 384, 1536, 20)
				matmul_only (gpu, "GPU 64-tile profiled", 2048, 2048, 2048, 20)
				print (v.profile_report)
				v.set_wide_tiles (True)
				v.reset_profile
				matmul_only (gpu, "GPU 128-tile profiled", 8192, 384, 1536, 20)
				matmul_only (gpu, "GPU 128-tile profiled", 2048, 2048, 2048, 20)
				print (v.profile_report)
					-- Kernel throughput by shape: the RixGPT trainer's products for one
					-- micro-batch of 32 x 256 tokens (C 384, 6 heads of 64, B*H = 192).
				print ("RixGPT matmul shapes, GPU kernel time (batch, M x K x N, flags):%N")
				shape_profile (gpu, v, 1, 8192, 384, 1152, False, True)
				shape_profile (gpu, v, 1, 8192, 384, 1536, False, True)
				shape_profile (gpu, v, 1, 8192, 1536, 384, False, True)
				shape_profile (gpu, v, 1, 8192, 384, 316, False, True)
				shape_profile (gpu, v, 1, 8192, 1152, 384, False, False)
				shape_profile (gpu, v, 1, 1152, 8192, 384, True, False)
				shape_profile (gpu, v, 1, 1536, 8192, 384, True, False)
				shape_profile (gpu, v, 1, 384, 8192, 1536, True, False)
				shape_profile (gpu, v, 1, 316, 8192, 384, True, False)
				shape_profile (gpu, v, 192, 256, 64, 256, False, True)
				shape_profile (gpu, v, 192, 256, 256, 64, False, False)
				shape_profile (gpu, v, 192, 256, 256, 64, True, False)
				shape_profile (gpu, v, 192, 256, 64, 256, False, False)
				v.set_profiling (False)
				print ("GPU buffer pool: " + v.context.pool_hits.out + " allocations served from the pool, " + (v.context.pooled_bytes // (1024 * 1024)).out + " MB pooled%N")
			end
		end

feature {NONE} -- Steps

	layer_step (f: SIMPLE_TENSOR; a_label: STRING; m, k, n, reps: INTEGER)
			-- x (m x k) . W (k x n), tanh, MSE to zeros, backward, AdamW.
		local
			r: TENSOR_RANDOM
			x, w, y, l_loss: TENSOR
			opt: ADAMW_OPTIMIZER
			t0: INTEGER_64
			i: INTEGER
			l_last: REAL_32
		do
			create r.make (3)
			x := f.zeros (<<m, k>>); r.fill_uniform (x, -1.0, 1.0)
			w := f.zeros (<<k, n>>); r.fill_uniform (w, -0.05, 0.05); w := f.parameter (w)
			y := f.zeros (<<m, n>>)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w>>), create {CONSTANT_SCHEDULE}.make (0.001), 0.9, 0.99, 1.0e-8, 0.0)
			l_loss := x.matmul (w).hyperbolic_tangent.mean_squared_error (y)
			opt.step (l_loss.backward)
			l_last := l_loss.item (0)
			t0 := now_ms
			from i := 1 until i > reps loop
				l_loss := x.matmul (w).hyperbolic_tangent.mean_squared_error (y)
				opt.step (l_loss.backward)
				l_last := l_loss.item (0)
				i := i + 1
			end
			print (a_label + " layer step " + m.out + "x" + k.out + " . " + k.out + "x" + n.out + " (fwd+bwd+AdamW): "
				+ ((now_ms - t0) / reps).truncated_to_integer.out + " ms, loss " + l_last.out + "%N")
		end

	mlp_step (f: SIMPLE_TENSOR; a_label: STRING; m, d, h, reps: INTEGER)
			-- Transformer-MLP shape: x (m x d) . W1 (d x h), GELU, . W2 (h x d), MSE, backward, AdamW.
		local
			r: TENSOR_RANDOM
			x, w1, w2, y, l_loss: TENSOR
			opt: ADAMW_OPTIMIZER
			t0: INTEGER_64
			i: INTEGER
			l_last: REAL_32
			l_flops: REAL_64
		do
			create r.make (5)
			x := f.zeros (<<m, d>>); r.fill_uniform (x, -1.0, 1.0)
			y := f.zeros (<<m, d>>); r.fill_uniform (y, -0.1, 0.1)
			w1 := f.zeros (<<d, h>>); r.fill_uniform (w1, -0.03, 0.03); w1 := f.parameter (w1)
			w2 := f.zeros (<<h, d>>); r.fill_uniform (w2, -0.03, 0.03); w2 := f.parameter (w2)
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w1, w2>>), create {CONSTANT_SCHEDULE}.make (0.001), 0.9, 0.99, 1.0e-8, 0.0)
			l_loss := x.matmul (w1).gelu.matmul (w2).mean_squared_error (y)
			opt.step (l_loss.backward)
			l_last := l_loss.item (0)
			t0 := now_ms
			from i := 1 until i > reps loop
				l_loss := x.matmul (w1).gelu.matmul (w2).mean_squared_error (y)
				opt.step (l_loss.backward)
				l_last := l_loss.item (0)
				i := i + 1
			end
				-- Two matmuls forward, four backward (the input needs no gradient: three).
			l_flops := 2.0 * m * d * h * 5
			print (a_label + " MLP step " + m.out + "x" + d.out + " -> " + h.out + " -> " + d.out + " (fwd+bwd+AdamW): "
				+ ((now_ms - t0) / reps).truncated_to_integer.out + " ms, ~"
				+ (l_flops * reps / ((now_ms - t0).max (1) / 1000.0) / 1.0e9).truncated_to_integer.out + " GFLOP/s in matmuls, loss " + l_last.out + "%N")
		end

	shape_profile (f: SIMPLE_TENSOR; v: VULKAN_BACKEND; nb, m, k, n: INTEGER; ta, tb: BOOLEAN)
			-- Ten untracked products of the given shape; print GPU kernel ms per call and TFLOP/s.
		local
			a, b, c: TENSOR
			i, reps: INTEGER
			l_ms: REAL_64
			l_any: REAL_32
			fd: FORMAT_DOUBLE
		do
			reps := 10
			if ta then a := f.ones (<<nb, k, m>>) else a := f.ones (<<nb, m, k>>) end
			if tb then b := f.ones (<<nb, n, k>>) else b := f.ones (<<nb, k, n>>) end
			c := a.matmul_transposed (b, ta, tb)
			l_any := c.item (0)
			v.reset_profile
			from i := 1 until i > reps loop
				c := a.matmul_transposed (b, ta, tb)
				i := i + 1
			end
			l_any := c.item (0)
			l_ms := (v.profiled_ms ("matmul") + v.profiled_ms ("matmul_128")) / reps
			create fd.make (8, 3)
			print ("  " + nb.out + " x " + m.out + "x" + k.out + "x" + n.out + " ta=" + ta.out + " tb=" + tb.out
				+ ": " + fd.formatted (l_ms) + " ms, " + fd.formatted (2.0 * nb * m * k * n / (l_ms / 1000.0) / 1.0e12)
				+ " TFLOP/s%N")
		end

	matmul_only (f: SIMPLE_TENSOR; a_label: STRING; m, k, n, reps: INTEGER)
			-- Untracked matmuls back to back, one read at the end.
		local
			a, b, c: TENSOR
			t0: INTEGER_64
			i: INTEGER
			l_any: REAL_32
		do
			a := f.ones (<<m, k>>)
			b := f.ones (<<k, n>>)
			c := a.matmul (b)
			l_any := c.item (0)
			t0 := now_ms
			from i := 1 until i > reps loop
				c := a.matmul (b)
				i := i + 1
			end
			l_any := c.item (0)
			print (a_label + " matmul " + m.out + "x" + k.out + "x" + n.out + " through TENSOR: "
				+ (2.0 * m * k * n * reps / ((now_ms - t0).max (1) / 1000.0) / 1.0e9).truncated_to_integer.out
				+ " GFLOP/s (c[0] = " + l_any.out + ")%N")
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

end
