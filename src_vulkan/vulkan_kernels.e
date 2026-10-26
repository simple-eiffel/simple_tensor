note
	description: "[
		GLSL sources of VULKAN_BACKEND's kernels. Each reads its buffers through device
		addresses in push constants (GL_EXT_buffer_reference): addresses first, then
		32-bit fields, in the order VULKAN_BACKEND packs them (VULKAN_PUSH_BLOCK).
		Elementwise kernels run 256 threads per group over a 2-D group grid, flat id
		`gid ()'. Semantics follow CPU_BACKEND's C kernels (which compute in double);
		these compute in float, within twin parity's 1e-4.
	]"
	author: "Larry Rix"

class
	VULKAN_KERNELS

feature -- Source

	source (a_name: STRING_8): STRING_8
			-- Full GLSL of kernel `a_name'.
		require
			known: has (a_name)
		do
			if a_name.same_string ("matmul") then
				Result := Matmul
			else
				check attached bodies.item (a_name) as b then
					Result := Header + b
				end
			end
		end

	has (a_name: STRING_8): BOOLEAN
		do
			Result := a_name.same_string ("matmul") or bodies.has (a_name)
		end

feature {NONE} -- Shared GLSL

	Header: STRING_8 = "[
		#version 460
		#extension GL_EXT_buffer_reference : require
		layout(local_size_x = 256) in;
		layout(buffer_reference, std430, buffer_reference_align = 4) buffer F { float v[]; };
		layout(buffer_reference, std430, buffer_reference_align = 4) buffer I { int v[]; };
		uint gid() { return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 256u + gl_LocalInvocationID.x; }
		float stable_sigmoid(float x) { if (x >= 0.0) return 1.0 / (1.0 + exp(-x)); float e = exp(x); return e / (1.0 + e); }
		float safe_tanh(float x) { return tanh(clamp(x, -15.0, 15.0)); }
		float erf_as(float x) {
			float s = sign(x); float a = abs(x);
			float t = 1.0 / (1.0 + 0.3275911 * a);
			float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
			return s * y;
		}
		bool non_finite(float x) { return (floatBitsToUint(x) & 0x7F800000u) == 0x7F800000u; }

	]"

	Matmul: STRING_8 = "[
		#version 460
		#extension GL_EXT_buffer_reference : require
		layout(local_size_x = 16, local_size_y = 16) in;
		layout(buffer_reference, std430, buffer_reference_align = 4) buffer F { float v[]; };
		layout(push_constant) uniform P { F a; F b; F c; uint M; uint K; uint N; uint ta; uint tb; uint sh; } p;
		shared float As[16][16];
		shared float Bs[16][16];
		void main() {
			uint z = gl_WorkGroupID.z;
			uint ba = z * p.M * p.K;
			uint bb = (p.sh != 0u) ? 0u : z * p.K * p.N;
			uint bc = z * p.M * p.N;
			uint row = gl_GlobalInvocationID.y;
			uint col = gl_GlobalInvocationID.x;
			uint ty = gl_LocalInvocationID.y;
			uint tx = gl_LocalInvocationID.x;
			float acc = 0.0;
			for (uint t = 0u; t < p.K; t += 16u) {
				uint ka = t + tx;
				uint kb = t + ty;
				float av = 0.0;
				float bv = 0.0;
				if (row < p.M && ka < p.K) av = (p.ta != 0u) ? p.a.v[ba + ka * p.M + row] : p.a.v[ba + row * p.K + ka];
				if (col < p.N && kb < p.K) bv = (p.tb != 0u) ? p.b.v[bb + col * p.K + kb] : p.b.v[bb + kb * p.N + col];
				As[ty][tx] = av;
				Bs[ty][tx] = bv;
				barrier();
				for (uint k = 0u; k < 16u; k++) acc += As[ty][k] * Bs[k][tx];
				barrier();
			}
			if (row < p.M && col < p.N) p.c.v[bc + row * p.N + col] = acc;
		}
	]"

feature {NONE} -- Kernel bodies

	bodies: HASH_TABLE [STRING_8, STRING_8]
		once
			create Result.make (32)
			Result.put (Binary_same, "binary_same")
			Result.put (Binary_broadcast, "binary_broadcast")
			Result.put (Unary, "unary")
			Result.put (Unary_derivative, "unary_derivative")
			Result.put (Reduce, "reduce")
			Result.put (Norm_partials, "norm_partials")
			Result.put (Softmax, "softmax")
			Result.put (Gather_rows, "gather_rows")
			Result.put (Scatter_add_rows, "scatter_add_rows")
			Result.put (Ce_rows, "ce_rows")
			Result.put (Ce_sum, "ce_sum")
			Result.put (Ce_grad, "ce_grad")
			Result.put (Adamw, "adamw")
			Result.put (Sgd, "sgd")
			Result.put (Masked_fill, "masked_fill")
			Result.put (Permute, "permute")
			Result.put (Slice, "slice")
			Result.put (Add_slice, "add_slice")
			Result.put (Concat, "concat")
			Result.put (Argmax_mask, "argmax_mask")
			Result.put (Reduce_to, "reduce_to")
		end

	Binary_same: STRING_8 = "[
		layout(push_constant) uniform P { F a; F b; F o; uint n; int kind; } p;
		void main() {
			uint i = gid(); if (i >= p.n) return;
			float x = p.a.v[i]; float y = p.b.v[i]; float z;
			if (p.kind == 1) z = x + y; else if (p.kind == 2) z = x - y; else if (p.kind == 3) z = x * y; else z = x / y;
			p.o.v[i] = z;
		}
	]"

	Binary_broadcast: STRING_8 = "[
		layout(push_constant) uniform P { F a; F b; F o; uint n; int kind; int rank; int dims[6]; int sa[6]; int sb[6]; } p;
		void main() {
			uint f = gid(); if (f >= p.n) return;
			uint rem = f; int ia = 0; int ib = 0;
			for (int j = p.rank - 1; j >= 0; j--) {
				uint d = uint(p.dims[j]); int c = int(rem % d); rem /= d;
				ia += c * p.sa[j]; ib += c * p.sb[j];
			}
			float x = p.a.v[ia]; float y = p.b.v[ib]; float z;
			if (p.kind == 1) z = x + y; else if (p.kind == 2) z = x - y; else if (p.kind == 3) z = x * y; else z = x / y;
			p.o.v[f] = z;
		}
	]"

	Unary: STRING_8 = "[
		layout(push_constant) uniform P { F a; F o; uint n; int kind; float param; } p;
		void main() {
			uint i = gid(); if (i >= p.n) return;
			float x = p.a.v[i]; float z;
			switch (p.kind) {
			case 11: z = -x; break;
			case 12: z = p.param * x; break;
			case 13: z = x + p.param; break;
			case 14: z = exp(x); break;
			case 15: z = log(x); break;
			case 16: z = sqrt(x); break;
			case 17: z = safe_tanh(x); break;
			case 18: z = stable_sigmoid(x); break;
			case 19: z = (x > 0.0) ? x : 0.0; break;
			case 20: z = 0.5 * x * (1.0 + erf_as(x * 0.70710678118654752)); break;
			default: z = x * stable_sigmoid(x); break;
			}
			p.o.v[i] = z;
		}
	]"

	Unary_derivative: STRING_8 = "[
		layout(push_constant) uniform P { F a; F y; F o; uint n; int kind; float param; } p;
		void main() {
			uint i = gid(); if (i >= p.n) return;
			float x = p.a.v[i]; float y = p.y.v[i]; float d;
			switch (p.kind) {
			case 11: d = -1.0; break;
			case 12: d = p.param; break;
			case 13: d = 1.0; break;
			case 14: d = y; break;
			case 15: d = 1.0 / x; break;
			case 16: d = 0.5 / y; break;
			case 17: d = 1.0 - y * y; break;
			case 18: d = y * (1.0 - y); break;
			case 19: d = (x > 0.0) ? 1.0 : 0.0; break;
			case 20: d = 0.5 * (1.0 + erf_as(x * 0.70710678118654752)) + x * 0.3989422804014327 * exp(-0.5 * x * x); break;
			default: { float s = stable_sigmoid(x); d = s + x * s * (1.0 - s); } break;
			}
			p.o.v[i] = d;
		}
	]"

	Reduce: STRING_8 = "[
		layout(push_constant) uniform P { F a; F o; uint outer; uint nn; uint inner; int kind; } p;
		void main() {
			uint t = gid(); if (t >= p.outer * p.inner) return;
			uint o = t / p.inner; uint i = t % p.inner;
			uint base = o * p.nn * p.inner + i;
			float acc;
			if (p.kind == 33) {
				acc = p.a.v[base];
				for (uint j = 1u; j < p.nn; j++) { float v = p.a.v[base + j * p.inner]; if (v > acc) acc = v; }
			} else {
				acc = 0.0;
				for (uint j = 0u; j < p.nn; j++) acc += p.a.v[base + j * p.inner];
				if (p.kind == 32) acc = acc / float(p.nn);
			}
			p.o.v[t] = acc;
		}
	]"

	Norm_partials: STRING_8 = "[
		layout(push_constant) uniform P { F a; F part; uint n; } p;
		shared float ss[256];
		shared float nf[256];
		void main() {
			uint lid = gl_LocalInvocationID.x;
			float s = 0.0; float bad = 0.0;
			for (uint i = gl_WorkGroupID.x * 256u + lid; i < p.n; i += 65536u) {
				float v = p.a.v[i];
				if (non_finite(v)) bad = 1.0; else s += v * v;
			}
			ss[lid] = s; nf[lid] = bad;
			barrier();
			for (uint w = 128u; w > 0u; w >>= 1u) {
				if (lid < w) { ss[lid] += ss[lid + w]; nf[lid] = max(nf[lid], nf[lid + w]); }
				barrier();
			}
			if (lid == 0u) { p.part.v[gl_WorkGroupID.x * 2u] = ss[0]; p.part.v[gl_WorkGroupID.x * 2u + 1u] = nf[0]; }
		}
	]"

	Softmax: STRING_8 = "[
		layout(push_constant) uniform P { F a; F o; uint rows; uint cols; uint lg; } p;
		void main() {
			uint r = gid(); if (r >= p.rows) return;
			uint b = r * p.cols;
			float mx = p.a.v[b];
			for (uint c = 1u; c < p.cols; c++) mx = max(mx, p.a.v[b + c]);
			float s = 0.0;
			for (uint c = 0u; c < p.cols; c++) s += exp(p.a.v[b + c] - mx);
			float ls = log(s);
			for (uint c = 0u; c < p.cols; c++) {
				float x = p.a.v[b + c];
				p.o.v[b + c] = (p.lg != 0u) ? (x - mx - ls) : (exp(x - mx) / s);
			}
		}
	]"

	Gather_rows: STRING_8 = "[
		layout(push_constant) uniform P { F t; I ix; F o; uint count; uint dim; } p;
		void main() {
			uint f = gid(); if (f >= p.count * p.dim) return;
			uint i = f / p.dim; uint d = f % p.dim;
			p.o.v[f] = p.t.v[uint(p.ix.v[i]) * p.dim + d];
		}
	]"

	Scatter_add_rows: STRING_8 = "[
		layout(push_constant) uniform P { F g; I ix; F o; uint count; uint dim; uint rows; } p;
		void main() {
			uint f = gid(); if (f >= p.rows * p.dim) return;
			uint r = f / p.dim; uint d = f % p.dim;
			float acc = p.o.v[f];
			for (uint i = 0u; i < p.count; i++) if (uint(p.ix.v[i]) == r) acc += p.g.v[i * p.dim + d];
			p.o.v[f] = acc;
		}
	]"

	Ce_rows: STRING_8 = "[
		layout(push_constant) uniform P { F l; I t; F rl; uint rows; uint cls; int ign; } p;
		void main() {
			uint r = gid(); if (r >= p.rows) return;
			int t = p.t.v[r];
			if (t == p.ign) { p.rl.v[r] = 0.0; return; }
			uint b = r * p.cls;
			float mx = p.l.v[b];
			for (uint c = 1u; c < p.cls; c++) mx = max(mx, p.l.v[b + c]);
			float s = 0.0;
			for (uint c = 0u; c < p.cls; c++) s += exp(p.l.v[b + c] - mx);
			p.rl.v[r] = mx + log(s) - p.l.v[b + uint(t)];
		}
	]"

	Ce_sum: STRING_8 = "[
		layout(push_constant) uniform P { F rl; F o; uint rows; float inv_nv; } p;
		shared float acc[256];
		void main() {
			uint lid = gl_LocalInvocationID.x;
			float s = 0.0;
			for (uint i = lid; i < p.rows; i += 256u) s += p.rl.v[i];
			acc[lid] = s;
			barrier();
			for (uint w = 128u; w > 0u; w >>= 1u) { if (lid < w) acc[lid] += acc[lid + w]; barrier(); }
			if (lid == 0u) p.o.v[0] = acc[0] * p.inv_nv;
		}
	]"

	Ce_grad: STRING_8 = "[
		layout(push_constant) uniform P { F l; I t; F o; uint rows; uint cls; int ign; float scale_nv; } p;
		void main() {
			uint r = gid(); if (r >= p.rows) return;
			int t = p.t.v[r];
			uint b = r * p.cls;
			if (t == p.ign) { for (uint c = 0u; c < p.cls; c++) p.o.v[b + c] = 0.0; return; }
			float mx = p.l.v[b];
			for (uint c = 1u; c < p.cls; c++) mx = max(mx, p.l.v[b + c]);
			float s = 0.0;
			for (uint c = 0u; c < p.cls; c++) s += exp(p.l.v[b + c] - mx);
			for (uint c = 0u; c < p.cls; c++) {
				float q = exp(p.l.v[b + c] - mx) / s;
				if (int(c) == t) q -= 1.0;
				p.o.v[b + c] = p.scale_nv * q;
			}
		}
	]"

	Adamw: STRING_8 = "[
		layout(push_constant) uniform P { F p_; F g; F m; F v; uint n; float lr; float b1; float b2; float eps; float wd; float bc1; float bc2; } p;
		void main() {
			uint i = gid(); if (i >= p.n) return;
			float g = p.g.v[i];
			float w = p.p_.v[i] * (1.0 - p.lr * p.wd);
			float m = p.b1 * p.m.v[i] + (1.0 - p.b1) * g;
			float v = p.b2 * p.v.v[i] + (1.0 - p.b2) * g * g;
			w = w - p.lr * (m / p.bc1) / (sqrt(v / p.bc2) + p.eps);
			p.m.v[i] = m; p.v.v[i] = v; p.p_.v[i] = w;
		}
	]"

	Sgd: STRING_8 = "[
		layout(push_constant) uniform P { F p_; F g; uint n; float lr; } p;
		void main() {
			uint i = gid(); if (i >= p.n) return;
			p.p_.v[i] = p.p_.v[i] - p.lr * p.g.v[i];
		}
	]"

	Masked_fill: STRING_8 = "[
		layout(push_constant) uniform P { F a; I mask; F o; uint n; float value; int rank; int dims[6]; int ms[6]; } p;
		void main() {
			uint f = gid(); if (f >= p.n) return;
			uint rem = f; int im = 0;
			for (int j = p.rank - 1; j >= 0; j--) { uint d = uint(p.dims[j]); int c = int(rem % d); rem /= d; im += c * p.ms[j]; }
			p.o.v[f] = (p.mask.v[im] != 0) ? p.value : p.a.v[f];
		}
	]"

	Permute: STRING_8 = "[
		layout(push_constant) uniform P { F a; F o; uint n; int rank; int dims[6]; int st[6]; } p;
		void main() {
			uint f = gid(); if (f >= p.n) return;
			uint rem = f; int src = 0;
			for (int j = p.rank - 1; j >= 0; j--) { uint d = uint(p.dims[j]); int c = int(rem % d); rem /= d; src += c * p.st[j]; }
			p.o.v[f] = p.a.v[src];
		}
	]"

	Slice: STRING_8 = "[
		layout(push_constant) uniform P { F a; F o; uint n; uint nn; uint inner; uint start0; uint len; } p;
		void main() {
			uint f = gid(); if (f >= p.n) return;
			uint i = f % p.inner; uint j = (f / p.inner) % p.len; uint o = f / (p.inner * p.len);
			p.o.v[f] = p.a.v[(o * p.nn + p.start0 + j) * p.inner + i];
		}
	]"

	Add_slice: STRING_8 = "[
		layout(push_constant) uniform P { F part; F o; uint n; uint nn; uint inner; uint start0; uint len; } p;
		void main() {
			uint f = gid(); if (f >= p.n) return;
			uint i = f % p.inner; uint j = (f / p.inner) % p.len; uint o = f / (p.inner * p.len);
			uint t = (o * p.nn + p.start0 + j) * p.inner + i;
			p.o.v[t] = p.o.v[t] + p.part.v[f];
		}
	]"

	Concat: STRING_8 = "[
		layout(push_constant) uniform P { F a; F b; F o; uint n; uint na; uint nb; uint inner; } p;
		void main() {
			uint f = gid(); if (f >= p.n) return;
			uint w = p.na + p.nb;
			uint i = f % p.inner; uint j = (f / p.inner) % w; uint o = f / (p.inner * w);
			p.o.v[f] = (j < p.na) ? p.a.v[(o * p.na + j) * p.inner + i] : p.b.v[(o * p.nb + j - p.na) * p.inner + i];
		}
	]"

	Argmax_mask: STRING_8 = "[
		layout(push_constant) uniform P { F a; F o; uint outer; uint nn; uint inner; } p;
		void main() {
			uint t = gid(); if (t >= p.outer * p.inner) return;
			uint o = t / p.inner; uint i = t % p.inner;
			uint base = o * p.nn * p.inner + i;
			uint best = 0u;
			for (uint j = 1u; j < p.nn; j++) if (p.a.v[base + j * p.inner] > p.a.v[base + best * p.inner]) best = j;
			for (uint j = 0u; j < p.nn; j++) p.o.v[base + j * p.inner] = (j == best) ? 1.0 : 0.0;
		}
	]"

	Reduce_to: STRING_8 = "[
		layout(push_constant) uniform P { F a; F o; uint n_out; int rank; int kept; int dims[6]; } p;
		void main() {
			uint t = gid(); if (t >= p.n_out) return;
			int st[6]; int s = 1;
			for (int k = p.rank - 1; k >= 0; k--) { st[k] = s; s *= p.dims[k]; }
			uint rem = t; int base = 0; uint red = 1u;
			for (int k = p.rank - 1; k >= 0; k--) {
				if ((p.kept & (1 << k)) != 0) { uint d = uint(p.dims[k]); base += int(rem % d) * st[k]; rem /= d; }
				else red *= uint(p.dims[k]);
			}
			float acc = 0.0;
			for (uint r = 0u; r < red; r++) {
				uint rr = r; int idx = base;
				for (int k = p.rank - 1; k >= 0; k--) {
					if ((p.kept & (1 << k)) == 0) { uint d = uint(p.dims[k]); idx += int(rr % d) * st[k]; rr /= d; }
				}
				acc += p.a.v[idx];
			}
			p.o.v[t] = acc;
		}
	]"

end
