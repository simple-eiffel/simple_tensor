note
	description: "[
		Spike S-5 (M3a): what simple_vulkan 0.2.0's compute batches buy for training.
		(a) dispatch overhead: 1,000 small dispatches (x := x + 1 over 64K floats),
		    legacy (submit + vkQueueWaitIdle each) vs recorded in one batch + one flush;
		(b) the S-2 tiled matmul on device-local buffers through an address pipeline,
		    20 dispatches recorded per flush, at 256x768x768 and 2048^3 (S-2 host-visible:
		    431 and 332 GFLOP/s).
	]"
	author: "Larry Rix"

class
	S5_APP

create
	make

feature {NONE} -- Initialization

	make
		local
			ctx: VULKAN_CONTEXT
		do
			create ctx.make
			if ctx.is_valid and then ctx.has_device_address then
				print ("S-5 on " + ctx.device_name + "%N")
				overhead (ctx)
				matmul (ctx, 256, 768, 768, 20)
				matmul (ctx, 2048, 2048, 2048, 20)
				ctx.dispose
			else
				print ("no Vulkan device with buffer device addresses%N")
			end
		end

feature {NONE} -- (a) dispatch overhead

	overhead (ctx: VULKAN_CONTEXT)
		local
			legacy_buf: VULKAN_BUFFER
			legacy_pipe: VULKAN_PIPELINE
			dev: VULKAN_DEVICE_BUFFER
			addr_pipe: VULKAN_ADDRESS_PIPELINE
			l_push, l_host: MANAGED_POINTER
			t0, l_legacy, l_batched: INTEGER_64
			i, n: INTEGER
			ok: BOOLEAN
		do
			n := 65_536
			create l_host.make (n * 4)
				-- Legacy: descriptor pipeline, host-visible buffer, one submit + wait per dispatch.
			create legacy_buf.make (ctx, n * 4, {VULKAN_BUFFER}.Buffer_storage)
			ok := legacy_buf.upload (l_host.item, n * 4, 0)
			if attached shader (ctx, Legacy_increment_glsl) as s then
				create legacy_pipe.make (ctx, s)
				create l_push.make (4)
				l_push.put_natural_32 (n.to_natural_32, 0)
				ok := ok and legacy_pipe.bind_buffer (0, legacy_buf) and legacy_pipe.set_push_constants (l_push.item, 4)
				t0 := now_ms
				from i := 1 until i > 1000 loop
					ok := ok and legacy_pipe.dispatch (ctx, n // 256, 1, 1)
					i := i + 1
				end
				l_legacy := now_ms - t0
				ok := ok and legacy_buf.download (l_host.item, n * 4, 0)
				print ("(a) legacy  1,000 dispatches: " + l_legacy.out + " ms, x[0] = " + l_host.read_real_32 (0).out + " (ok " + ok.out + ")%N")
				legacy_pipe.dispose
				s.dispose
			end
			legacy_buf.dispose
				-- Batched: address pipeline, device-local buffer, record 1,000 then one flush.
			create dev.make (ctx, n * 4)
			if attached shader (ctx, Increment_glsl) as s then
				create addr_pipe.make (ctx, s)
				create l_push.make (12)
				l_push.put_natural_64 (dev.address, 0)
				l_push.put_natural_32 (n.to_natural_32, 8)
				ok := dev.record_fill (0, n * 4, 0)
				ctx.flush
				t0 := now_ms
				from i := 1 until i > 1000 loop
					ok := ok and addr_pipe.record (l_push.item, 12, n // 256, 1, 1)
					i := i + 1
				end
				ctx.flush
				l_batched := now_ms - t0
				ok := ok and dev.read (l_host.item, n * 4, 0)
				print ("(a) batched 1,000 dispatches: " + l_batched.out + " ms, x[0] = " + l_host.read_real_32 (0).out + " (ok " + ok.out + ")%N")
				addr_pipe.dispose
				s.dispose
			end
			dev.dispose
		end

feature {NONE} -- (b) device-local matmul

	matmul (ctx: VULKAN_CONTEXT; m, k, n, reps: INTEGER)
		local
			a, b, c: VULKAN_DEVICE_BUFFER
			p: VULKAN_ADDRESS_PIPELINE
			l_a, l_b, l_c, l_push: MANAGED_POINTER
			t0, l_ms: INTEGER_64
			i, j, r, col: INTEGER
			ok: BOOLEAN
			l_ref, l_err, l_worst: REAL_64
		do
			create l_a.make (m * k * 4); create l_b.make (k * n * 4); create l_c.make (m * n * 4)
			from i := 0 until i = m * k loop l_a.put_real_32 (((i * 37) \\ 101 - 50).to_real / 50.0, i * 4); i := i + 1 end
			from i := 0 until i = k * n loop l_b.put_real_32 (((i * 53) \\ 97 - 48).to_real / 48.0, i * 4); i := i + 1 end
			create a.make (ctx, m * k * 4); create b.make (ctx, k * n * 4); create c.make (ctx, m * n * 4)
			ok := a.write (l_a.item, m * k * 4, 0) and b.write (l_b.item, k * n * 4, 0)
			if attached shader (ctx, Tiled_glsl) as s then
				create p.make (ctx, s)
				create l_push.make (36)
				l_push.put_natural_64 (a.address, 0)
				l_push.put_natural_64 (b.address, 8)
				l_push.put_natural_64 (c.address, 16)
				l_push.put_natural_32 (m.to_natural_32, 24)
				l_push.put_natural_32 (k.to_natural_32, 28)
				l_push.put_natural_32 (n.to_natural_32, 32)
				ok := ok and p.record (l_push.item, 36, (n + 15) // 16, (m + 15) // 16, 1)
				ctx.flush
				t0 := now_ms
				from i := 1 until i > reps loop
					ok := ok and p.record (l_push.item, 36, (n + 15) // 16, (m + 15) // 16, 1)
					i := i + 1
				end
				ctx.flush
				l_ms := (now_ms - t0).max (1)
				ok := ok and c.read (l_c.item, m * n * 4, 0)
					-- Check 64 sampled entries against a REAL_64 dot product.
				from i := 0 until i = 64 loop
					r := (i * 7919) \\ m
					col := (i * 104729) \\ n
					l_ref := 0.0
					from j := 0 until j = k loop
						l_ref := l_ref + l_a.read_real_32 ((r * k + j) * 4) * l_b.read_real_32 ((j * n + col) * 4)
						j := j + 1
					end
					l_err := (l_c.read_real_32 ((r * n + col) * 4) - l_ref).abs / l_ref.abs.max (1.0)
					l_worst := l_worst.max (l_err)
					i := i + 1
				end
				print ("(b) matmul " + m.out + "x" + k.out + "x" + n.out + ": " + (l_ms // reps).out + " ms/dispatch avg, "
					+ ((2.0 * m * k * n * reps) / (l_ms / 1000.0) / 1.0e9).truncated_to_integer.out + " GFLOP/s, worst rel err " + l_worst.out + " (ok " + ok.out + ")%N")
				p.dispose
				s.dispose
			end
			a.dispose; b.dispose; c.dispose
		end

feature {NONE} -- Helpers

	shader (ctx: VULKAN_CONTEXT; a_glsl: STRING): detachable VULKAN_SHADER
		do
			if attached (create {SIMPLE_SHADERC}.make).compile_compute (a_glsl) as l_code then
				create Result.make_from_memory (ctx, l_code.item, l_code.count.to_integer_64)
			else
				print ("shader compile failed%N")
			end
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

feature {NONE} -- Shaders

	Legacy_increment_glsl: STRING = "[
		#version 450
		layout(local_size_x = 256) in;
		layout(std430, binding = 0) buffer B { float x[]; };
		layout(push_constant) uniform P { uint n; } p;
		void main() { uint i = gl_GlobalInvocationID.x; if (i < p.n) x[i] = x[i] + 1.0; }
	]"

	Increment_glsl: STRING = "[
		#version 460
		#extension GL_EXT_buffer_reference : require
		layout(local_size_x = 256) in;
		layout(buffer_reference, std430) buffer Floats { float v[]; };
		layout(push_constant) uniform P { Floats x; uint n; } p;
		void main() { uint i = gl_GlobalInvocationID.x; if (i < p.n) p.x.v[i] = p.x.v[i] + 1.0; }
	]"

	Tiled_glsl: STRING = "[
		#version 460
		#extension GL_EXT_buffer_reference : require
		layout(local_size_x = 16, local_size_y = 16) in;
		layout(buffer_reference, std430) readonly buffer RO { float v[]; };
		layout(buffer_reference, std430) writeonly buffer WO { float v[]; };
		layout(push_constant) uniform P { RO a; RO b; WO c; uint M; uint K; uint N; } p;
		shared float As[16][16];
		shared float Bs[16][16];
		void main() {
			uint row = gl_GlobalInvocationID.y;
			uint col = gl_GlobalInvocationID.x;
			uint ty = gl_LocalInvocationID.y;
			uint tx = gl_LocalInvocationID.x;
			float acc = 0.0;
			for (uint t = 0; t < p.K; t += 16) {
				As[ty][tx] = (row < p.M && t + tx < p.K) ? p.a.v[row * p.K + t + tx] : 0.0;
				Bs[ty][tx] = (col < p.N && t + ty < p.K) ? p.b.v[(t + ty) * p.N + col] : 0.0;
				barrier();
				for (uint k = 0; k < 16; k++) acc += As[ty][k] * Bs[k][tx];
				barrier();
			}
			if (row < p.M && col < p.N) p.c.v[row * p.N + col] = acc;
		}
	]"

end
