note
	description: "[
		SPIKE S-2 (simple_tensor M0): REAL_32 matmul on the RTX 5070 Ti through
		simple_vulkan + simple_shaderc AS THEY STAND (host-visible buffers, one
		submit + vkQueueWaitIdle per dispatch). Two GLSL kernels, naive and
		16x16 shared-memory tiled, at a Needle-sized shape and a large one.
		Checks GPU results against a C reference on the CPU.
	]"

class
	S2_APP

create
	make

feature {NONE} -- Initialization

	make
		local
			l_ctx: VULKAN_CONTEXT
		do
			create l_ctx.make
			if l_ctx.is_valid then
				print ("GPU: " + l_ctx.device_name + " (discrete: " + l_ctx.is_discrete_gpu.out + ")%N")
				run_shape (l_ctx, 256, 768, 768, 20)
				run_shape (l_ctx, 2048, 2048, 2048, 5)
			else
				print ("no Vulkan context%N")
			end
		end

feature {NONE} -- Runs

	run_shape (a_ctx: VULKAN_CONTEXT; a_m, a_k, a_n, a_reps: INTEGER)
		local
			l_a, l_b, l_ref, l_out: MANAGED_POINTER
			l_ga, l_gb, l_gc: VULKAN_BUFFER
			l_push: MANAGED_POINTER
			i: INTEGER
			l_ok: BOOLEAN
			t0: INTEGER_64
		do
			print ("%N--- " + a_m.out + " x " + a_k.out + " x " + a_n.out + " (" + (2.0 * a_m * a_k * a_n / 1.0e9).out + " GFLOP per multiply) ---%N")
			create l_a.make (a_m * a_k * 4)
			create l_b.make (a_k * a_n * 4)
			create l_ref.make (a_m * a_n * 4)
			create l_out.make (a_m * a_n * 4)
			from i := 0 until i = a_m * a_k loop l_a.put_real_32 (((i \\ 97) - 48).to_real / 97.0, i * 4); i := i + 1 end
			from i := 0 until i = a_k * a_n loop l_b.put_real_32 (((i \\ 89) - 44).to_real / 89.0, i * 4); i := i + 1 end
			t0 := now_ms
			c_matmul (l_a.item, l_b.item, l_ref.item, a_m, a_k, a_n)
			print ("CPU C reference : " + (now_ms - t0).out + " ms%N")

			create l_ga.make (a_ctx, l_a.count.to_integer_64, {VULKAN_BUFFER}.Buffer_storage)
			create l_gb.make (a_ctx, l_b.count.to_integer_64, {VULKAN_BUFFER}.Buffer_storage)
			create l_gc.make (a_ctx, l_out.count.to_integer_64, {VULKAN_BUFFER}.Buffer_storage)
			t0 := now_ms
			l_ok := l_ga.upload (l_a.item, l_a.count.to_integer_64, 0) and l_gb.upload (l_b.item, l_b.count.to_integer_64, 0)
			print ("upload A+B      : " + (now_ms - t0).out + " ms (ok " + l_ok.out + ")%N")

			create l_push.make (12)
			l_push.put_natural_32 (a_m.to_natural_32, 0)
			l_push.put_natural_32 (a_k.to_natural_32, 4)
			l_push.put_natural_32 (a_n.to_natural_32, 8)

			run_kernel (a_ctx, "naive ", Naive_glsl, l_ga, l_gb, l_gc, l_push, l_ref, l_out, a_m, a_k, a_n, a_reps)
			run_kernel (a_ctx, "tiled ", Tiled_glsl, l_ga, l_gb, l_gc, l_push, l_ref, l_out, a_m, a_k, a_n, a_reps)
		end

	run_kernel (a_ctx: VULKAN_CONTEXT; a_label, a_glsl: STRING; a_ga, a_gb, a_gc: VULKAN_BUFFER;
			a_push, a_ref, a_out: MANAGED_POINTER; a_m, a_k, a_n, a_reps: INTEGER)
		local
			l_shaderc: SIMPLE_SHADERC
			l_shader: VULKAN_SHADER
			l_pipe: VULKAN_PIPELINE
			l_ok: BOOLEAN
			r, gx, gy: INTEGER
			t0, l_ms, t1: INTEGER_64
		do
			create l_shaderc.make
			if attached l_shaderc.compile_compute (a_glsl) as l_spirv then
				create l_shader.make_from_memory (a_ctx, l_spirv.item, l_spirv.count.to_integer_64)
				create l_pipe.make (a_ctx, l_shader)
				l_ok := l_pipe.bind_buffer (0, a_ga) and l_pipe.bind_buffer (1, a_gb) and l_pipe.bind_buffer (2, a_gc)
					and l_pipe.set_push_constants (a_push.item, 12)
				gx := (a_n + 15) // 16
				gy := (a_m + 15) // 16
				l_ok := l_ok and l_pipe.dispatch (a_ctx, gx, gy, 1)
				t0 := now_ms
				from r := 1 until r > a_reps loop
					l_ok := l_ok and l_pipe.dispatch (a_ctx, gx, gy, 1)
					r := r + 1
				end
				l_ms := now_ms - t0
				t1 := now_ms
				l_ok := l_ok and a_gc.download (a_out.item, a_out.count.to_integer_64, 0)
				print (a_label + "  : " + (l_ms / a_reps).out + " ms/multiply, "
					+ ((2.0 * a_m * a_k * a_n * a_reps / (l_ms / 1000.0) / 1.0e7).rounded / 100.0).out
					+ " GFLOP/s, download " + (now_ms - t1).out + " ms, max rel err "
					+ c_max_rel_err (a_ref.item, a_out.item, a_m * a_n).out + ", ok " + l_ok.out + "%N")
			else
				print (a_label + "  : shader compile failed: " + l_shaderc.last_error + "%N")
			end
		end

feature {NONE} -- Kernels

	Naive_glsl: STRING = "[
		#version 450
		layout(local_size_x = 16, local_size_y = 16) in;
		layout(std430, binding = 0) readonly buffer BA { float a[]; };
		layout(std430, binding = 1) readonly buffer BB { float b[]; };
		layout(std430, binding = 2) writeonly buffer BC { float c[]; };
		layout(push_constant) uniform P { uint M; uint K; uint N; } p;
		void main() {
			uint row = gl_GlobalInvocationID.y;
			uint col = gl_GlobalInvocationID.x;
			if (row >= p.M || col >= p.N) return;
			float acc = 0.0;
			for (uint k = 0; k < p.K; k++) acc += a[row * p.K + k] * b[k * p.N + col];
			c[row * p.N + col] = acc;
		}
	]"

	Tiled_glsl: STRING = "[
		#version 450
		layout(local_size_x = 16, local_size_y = 16) in;
		layout(std430, binding = 0) readonly buffer BA { float a[]; };
		layout(std430, binding = 1) readonly buffer BB { float b[]; };
		layout(std430, binding = 2) writeonly buffer BC { float c[]; };
		layout(push_constant) uniform P { uint M; uint K; uint N; } p;
		shared float As[16][16];
		shared float Bs[16][16];
		void main() {
			uint row = gl_GlobalInvocationID.y;
			uint col = gl_GlobalInvocationID.x;
			uint ty = gl_LocalInvocationID.y;
			uint tx = gl_LocalInvocationID.x;
			float acc = 0.0;
			for (uint t = 0; t < p.K; t += 16) {
				As[ty][tx] = (row < p.M && t + tx < p.K) ? a[row * p.K + t + tx] : 0.0;
				Bs[ty][tx] = (col < p.N && t + ty < p.K) ? b[(t + ty) * p.N + col] : 0.0;
				barrier();
				for (uint k = 0; k < 16; k++) acc += As[ty][k] * Bs[k][tx];
				barrier();
			}
			if (row < p.M && col < p.N) c[row * p.N + col] = acc;
		}
	]"

feature {NONE} -- C helpers

	c_matmul (a, b, c: POINTER; m_, k_, n_: INTEGER)
		external
			"C blocking inline"
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

	c_max_rel_err (a_ref, a_got: POINTER; n: INTEGER): REAL_64
		external
			"C blocking inline use <math.h>"
		alias
			"[
				float* R = (float*)$a_ref; float* G = (float*)$a_got; double worst = 0.0; int i;
				for (i = 0; i < $n; i++) {
					double d = fabs((double)R[i] - (double)G[i]) / (fabs((double)R[i]) + 1e-3);
					if (d > worst) worst = d;
				}
				return worst;
			]"
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

end
