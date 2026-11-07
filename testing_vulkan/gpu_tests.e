note
	description: "[
		M3b GPU suite: the M1 twin cases, unchanged, with VULKAN_BACKEND as the candidate,
		plus parity cases for the COLD kernels (on CPU they are HOST_BACKEND code shared
		with the oracle, so M1 never twin-checked them; on the GPU they are new GLSL).
		Gradient checks stay on the oracle by design (spec 03 A-S04): autograd is
		backend-independent, and every forward and backward kernel it calls is checked
		here against REFERENCE_BACKEND. Training end to end must track the oracle.
	]"
	author: "Larry Rix"

class
	GPU_TESTS

inherit
	LIB_TESTS

feature -- GPU tests

	test_gpu_storage_basics
			-- from_array/to_array round trip, mirror reads, upload invalidating the mirror,
			-- fill, zeros, copy (parameter), release.
		local
			t, c: TENSOR
			l_values: ARRAY [REAL_32]
		do
			t := gpu.from_array (<<1.5, -2.0, 3.25, 0.0, 7.0, -0.5>>, <<2, 3>>)
			assert ("on the GPU", t.backend.kind = t.backend.Vulkan_backend_kind)
			l_values := t.to_array
			assert ("round trip", l_values [1] = 1.5 and l_values [3] = 3.25 and l_values [6] = -0.5)
			assert ("mirror item", t.item (4) = 7.0)
			t.backend.upload (<<1.5, -2.0, 3.25, 0.0, 9.0, -0.5>>, t.storage)
			assert ("upload invalidates the mirror", t.item (4) = 9.0 and t.to_array [5] = 9.0)
			c := gpu.filled (<<1000>>, 0.25)
			assert ("fill", c.item (0) = 0.25 and c.item (999) = 0.25)
			c := gpu.zeros (<<4>>)
			assert ("zeros", c.item (3) = 0.0)
			c := gpu.parameter (t)
			assert ("copy (parameter)", c.item (4) = 9.0 and c.storage /= t.storage)
			c := t.scaled (2.0)
			c.release
			assert ("released", c.is_released)
		end

	test_gpu_twin_parity
			-- Every hot kernel on the GPU within 1e-4 relative of REFERENCE_BACKEND.
		local
			t: TWIN_CHECKER
		do
			create t.make (oracle, gpu)
			t.compare_named ("binary (same shape and broadcast)", agent twin_binary)
			t.compare_named ("unary (all eleven kinds)", agent twin_unary)
			t.compare_named ("unary_derivative (gradient of all eleven)", agent twin_unary_gradient)
			t.compare_named ("reduce (sum, mean, max)", agent twin_reduce)
			t.compare_named ("matmul plain", agent twin_matmul (?, False, False, False))
			t.compare_named ("matmul ta", agent twin_matmul (?, True, False, False))
			t.compare_named ("matmul tb", agent twin_matmul (?, False, True, False))
			t.compare_named ("matmul batched shared B", agent twin_matmul (?, False, False, True))
			t.compare_named ("matmul multi-tile, ragged edges, all flags, batched", agent twin_matmul_large)
			t.compare_named ("matmul 128-wide tiles, ragged edges, all flags, batched", agent twin_matmul_wide)
			t.compare_named ("matmul split-K, ragged slices, all flags", agent twin_matmul_split_k)
			t.compare_named ("softmax", agent twin_softmax (?, False))
			t.compare_named ("log_softmax", agent twin_softmax (?, True))
			t.compare_named ("gather_rows + scatter_add_rows", agent twin_gather_scatter)
			t.compare_named ("cross_entropy (loss)", agent twin_cross_entropy (?, False))
			t.compare_named ("cross_entropy_grad", agent twin_cross_entropy (?, True))
			t.compare_named ("adamw_update (3 steps)", agent twin_adamw)
			t.compare_named ("sum_of_squares + has_non_finite (global norm)", agent twin_norm)
			t.compare_named ("global norm over several gradients (one read)", agent twin_global_norm)
			print ("      GPU twin parity (hot): worst relative error " + t.worst_relative_error.out + " (" + t.last_report + ")%N")
			assert ("GPU hot kernels match the oracle within 1e-4 (" + t.last_report + ")", t.within (1.0e-4))
		end

	test_gpu_cold_parity
			-- The cold kernels on the GPU within 1e-4 relative of REFERENCE_BACKEND.
		local
			t: TWIN_CHECKER
		do
			create t.make (oracle, gpu)
			t.compare_named ("permute 2D and 3D", agent twin_permute)
			t.compare_named ("slice + add_slice (slice backward)", agent twin_slice)
			t.compare_named ("concat + its backward", agent twin_concat)
			t.compare_named ("masked_fill (causal) + backward", agent twin_masked_fill)
			t.compare_named ("reduce_to_shape (broadcast backward)", agent twin_reduce_to)
			t.compare_named ("argmax_mask (max backward)", agent twin_argmax)
			t.compare_named ("sgd_update (2 steps)", agent twin_sgd)
			t.compare_named ("fill + copy", agent twin_fill_copy)
			t.compare_named ("fill_random (uniform, normal, keep mask)", agent twin_random)
			t.compare_named ("causal_softmax (fused) + backward", agent twin_causal_softmax)
			t.compare_named ("layer_norm (fused) + three gradients", agent twin_layer_norm)
			t.compare_named ("linear (fused bias) + three gradients", agent twin_linear)
			t.compare_named ("dropped (regenerated mask) + gradient", agent twin_dropped)
			print ("      GPU twin parity (cold): worst relative error " + t.worst_relative_error.out + " (" + t.last_report + ")%N")
			assert ("GPU cold kernels match the oracle within 1e-4 (" + t.last_report + ")", t.within (1.0e-4))
		end

	test_gpu_mixed_precision_matmul
			-- With mixed precision on, products run on the tensor cores (bf16 operands, f32
			-- sums): the ragged multi-tile, 128-wide and split-K product sets, every transpose
			-- flag and batching, each within 1% of the oracle by norm (bf16 keeps 8 bits).
			-- Skipped without bf16 cooperative matrices. Mixed precision is off again after.
		local
			l_err, l_worst: REAL_64
		do
			if attached {VULKAN_BACKEND} gpu.backend as v and then v.can_mix_precision then
				v.set_mixed_precision (True)
				v.reset_profile
				v.set_profiling (True)
				l_worst := norm_relative_error (twin_matmul_large (gpu), twin_matmul_large (oracle))
				l_err := norm_relative_error (twin_matmul_wide (gpu), twin_matmul_wide (oracle))
				l_worst := l_worst.max (l_err)
				l_err := norm_relative_error (twin_matmul_split_k (gpu), twin_matmul_split_k (oracle))
				l_worst := l_worst.max (l_err)
				v.set_profiling (False)
				v.set_mixed_precision (False)
				print ("      GPU mixed precision (tensor cores): worst norm-relative error " + l_worst.out
					+ ", tensor-core kernel time " + v.profiled_ms ("matmul_coop").out + " ms%N")
				assert ("ran on the tensor cores", v.profiled_ms ("matmul_coop") > 0.0)
				assert ("within 1%% of the oracle by norm", l_worst <= 0.01)
				assert ("bf16 really rounds (not f32)", l_worst > 1.0e-5)
			else
				print ("      GPU mixed precision: SKIP (no bf16 cooperative matrices)%N")
			end
		end

	norm_relative_error (a_candidate, a_oracle: TENSOR): REAL_64
			-- ||candidate - oracle|| / ||oracle||.
		local
			c, o: ARRAY [REAL_32]
			i: INTEGER
			l_diff, l_norm: REAL_64
		do
			c := a_candidate.to_array
			o := a_oracle.to_array
			from i := o.lower until i > o.upper loop
				l_diff := l_diff + (c [i] - o [i]).to_double ^ 2
				l_norm := l_norm + o [i].to_double ^ 2
				i := i + 1
			end
			Result := (l_diff / l_norm.max (1.0e-30)) ^ 0.5
		end

	test_gpu_keep_mask_bit_identical
			-- A 100,000-element dropout mask drawn on the GPU equals the oracle's element for
			-- element (the keep test is integer-exact), after 1,000 words were already drawn.
		local
			rg, ro: TENSOR_RANDOM
			mg, mo: TENSOR
			g, o: ARRAY [REAL_32]
			i, l_kept: INTEGER
		do
			create rg.make (9001)
			create ro.make (9001)
			rg.fill_uniform (gpu.zeros (<<1000>>), 0.0, 1.0)
			ro.fill_uniform (oracle.zeros (<<1000>>), 0.0, 1.0)
			mg := gpu.zeros (<<100_000>>)
			mo := oracle.zeros (<<100_000>>)
			rg.fill_keep_mask (mg, 0.8)
			ro.fill_keep_mask (mo, 0.8)
			g := mg.to_array
			o := mo.to_array
			from i := 1 until i > 100_000 loop
				assert ("mask element " + i.out, g [i] = o [i])
				if g [i] /= 0.0 then l_kept := l_kept + 1 end
				i := i + 1
			end
			print ("      GPU keep mask: 100,000 elements identical to the oracle's, " + l_kept.out + " kept (p = 0.8)%N")
			assert ("kept fraction", 79_500 <= l_kept and l_kept <= 80_500)
		end

	test_gpu_training_tracks_oracle
			-- 20 AdamW steps of the M1 training fixture: GPU losses within 1e-3 of the oracle's.
		local
			g, o: ARRAY [REAL_32]
		do
			g := training_losses (gpu, 4242)
			o := training_losses (oracle, 4242)
			print ("      GPU training: loss " + g [1].out + " -> " + g [20].out + " (oracle " + o [20].out + ")%N")
			assert ("GPU tracks the oracle", across 1 |..| 20 as k all (g [k] - o [k]).abs <= 1.0e-3 * o [k].abs.max (1.0) end)
		end

	test_gpu_fallback_ledger
			-- Report which kernels ran as host fallbacks during this suite (M3b gate 3 reads it).
		do
			if attached {VULKAN_BACKEND} gpu.backend as v then
				print ("      GPU device: " + v.device_name + "; host fallbacks so far: " + v.fallback_total.out + "%N")
				across v.fallback_counts as c loop
					print ("        " + @c.key + ": " + c.out + "%N")
				end
			end
		end

feature {NONE} -- Cold-kernel twin cases

	twin_permute (f: SIMPLE_TENSOR): TENSOR
		local
			x: TENSOR
		do
			x := seeded (f, <<2, 3, 4>>, -1.0, 1.0, 81)
			Result := x.permuted (<<3, 1, 2>>).reshaped (<<24>>)
				.concatenated (x.permuted (<<2, 3, 1>>).reshaped (<<24>>), 1)
				.concatenated (seeded (f, <<5, 7>>, -1.0, 1.0, 82).transposed.reshaped (<<35>>), 1)
		end

	twin_slice (f: SIMPLE_TENSOR): TENSOR
		local
			x, w: TENSOR
		do
			x := f.parameter (seeded (f, <<3, 6, 2>>, -1.0, 1.0, 83))
			w := seeded (f, <<3, 3, 2>>, -1.0, 1.0, 84)
			Result := x.sliced (2, 3, 3).reshaped (<<18>>)
				.concatenated ((x.sliced (2, 3, 3) * w).sum_all.backward.gradient (x).reshaped (<<36>>), 1)
		end

	twin_concat (f: SIMPLE_TENSOR): TENSOR
		local
			a, b, w: TENSOR
		do
			a := f.parameter (seeded (f, <<2, 3, 2>>, -1.0, 1.0, 85))
			b := f.parameter (seeded (f, <<2, 1, 2>>, -1.0, 1.0, 86))
			w := seeded (f, <<2, 4, 2>>, -1.0, 1.0, 87)
			Result := a.concatenated (b, 2).reshaped (<<16>>)
				.concatenated ((a.concatenated (b, 2) * w).sum_all.backward.gradient (b).reshaped (<<4>>), 1)
		end

	twin_masked_fill (f: SIMPLE_TENSOR): TENSOR
		local
			x, w: TENSOR
		do
			x := f.parameter (seeded (f, <<2, 4, 4>>, -1.0, 1.0, 88))
			w := seeded (f, <<2, 4, 4>>, -1.0, 1.0, 89)
			Result := x.masked_fill (f.causal_mask (4), -1.0e9).softmax.reshaped (<<32>>)
				.concatenated ((x.masked_fill (f.causal_mask (4), 0.0) * w).sum_all.backward.gradient (x).reshaped (<<32>>), 1)
		end

	twin_reduce_to (f: SIMPLE_TENSOR): TENSOR
		local
			x, b, w: TENSOR
		do
			x := seeded (f, <<3, 4, 5>>, -1.0, 1.0, 90)
			b := f.parameter (seeded (f, <<4, 1>>, -1.0, 1.0, 91))
			w := seeded (f, <<3, 4, 5>>, -1.0, 1.0, 92)
			Result := ((x + b) * w).sum_all.backward.gradient (b).reshaped (<<4>>)
				.concatenated (x.reduced_to (create {TENSOR_SHAPE}.make (<<1, 5>>)).reshaped (<<5>>), 1)
		end

	twin_argmax (f: SIMPLE_TENSOR): TENSOR
		local
			x: TENSOR
		do
			x := f.parameter (seeded (f, <<3, 7>>, -1.0, 1.0, 93))
			Result := x.maximum (2, False).sum_all.backward.gradient (x).reshaped (<<21>>)
				.concatenated (x.maximum (1, False).sum_all.backward.gradient (x).reshaped (<<21>>), 1)
		end

	twin_sgd (f: SIMPLE_TENSOR): TENSOR
		local
			w: TENSOR
			opt: SGD_OPTIMIZER
		do
			w := f.parameter (seeded (f, <<4, 5>>, -1.0, 1.0, 94))
			create opt.make (create {ARRAYED_LIST [TENSOR]}.make_from_array (<<w>>), create {CONSTANT_SCHEDULE}.make (0.1))
			opt.step ((w * w * w).sum_all.backward)
			opt.step ((w * w * w).sum_all.backward)
			Result := w.detached
		end

	twin_fill_copy (f: SIMPLE_TENSOR): TENSOR
		local
			x: TENSOR
		do
			x := seeded (f, <<3, 5>>, -1.0, 1.0, 95)
			Result := f.filled (<<7>>, -0.375).concatenated (f.parameter (x).reshaped (<<15>>), 1)
				.concatenated (x.zeros_like.reshaped (<<15>>), 1).concatenated (x.ones_like.reshaped (<<15>>), 1)
		end

	twin_random (f: SIMPLE_TENSOR): TENSOR
		local
			r: TENSOR_RANDOM
			u, n, m: TENSOR
		do
			create r.make (77)
			r.draw_index (50)
			u := f.zeros (<<3, 5>>); r.fill_uniform (u, -2.0, 3.0)
			n := f.zeros (<<4, 4>>); r.fill_normal (n, 0.5, 2.0)
			m := f.zeros (<<64>>); r.fill_keep_mask (m, 0.8)
			Result := u.reshaped (<<15>>).concatenated (n.reshaped (<<16>>), 1).concatenated (m, 1)
		end

	twin_matmul_large (f: SIMPLE_TENSOR): TENSOR
			-- Sizes that cross 64-wide tiles with ragged edges (70, 130, 90), every transpose
			-- flag, batched with two operands and with a shared right-hand side.
		local
			a, b, at, bt, ab, bb: TENSOR
		do
			a := seeded (f, <<70, 130>>, -1.0, 1.0, 110)
			b := seeded (f, <<130, 90>>, -1.0, 1.0, 111)
			at := seeded (f, <<130, 70>>, -1.0, 1.0, 112)
			bt := seeded (f, <<90, 130>>, -1.0, 1.0, 113)
			ab := seeded (f, <<3, 70, 130>>, -1.0, 1.0, 114)
			bb := seeded (f, <<3, 130, 90>>, -1.0, 1.0, 115)
			Result := a.matmul (b).reshaped (<<6300>>)
				.concatenated (at.matmul_transposed (b, True, False).reshaped (<<6300>>), 1)
				.concatenated (a.matmul_transposed (bt, False, True).reshaped (<<6300>>), 1)
				.concatenated (at.matmul_transposed (bt, True, True).reshaped (<<6300>>), 1)
				.concatenated (ab.matmul (bb).reshaped (<<18900>>), 1)
				.concatenated (ab.matmul (b).reshaped (<<18900>>), 1)
		end

	twin_matmul_wide (f: SIMPLE_TENSOR): TENSOR
			-- Sizes that take the 128 x 128 kernel (M, N >= 128) and cross its tiles raggedly
			-- (260, 200), with K = 75 ragged against its 8-deep K steps; every transpose flag,
			-- batched with two operands and with a shared right-hand side.
		local
			a, b, at, bt, ab, bb: TENSOR
		do
			a := seeded (f, <<260, 75>>, -1.0, 1.0, 120)
			b := seeded (f, <<75, 200>>, -1.0, 1.0, 121)
			at := seeded (f, <<75, 260>>, -1.0, 1.0, 122)
			bt := seeded (f, <<200, 75>>, -1.0, 1.0, 123)
			ab := seeded (f, <<2, 260, 75>>, -1.0, 1.0, 124)
			bb := seeded (f, <<2, 75, 200>>, -1.0, 1.0, 125)
			Result := a.matmul (b).reshaped (<<52000>>)
				.concatenated (at.matmul_transposed (b, True, False).reshaped (<<52000>>), 1)
				.concatenated (a.matmul_transposed (bt, False, True).reshaped (<<52000>>), 1)
				.concatenated (at.matmul_transposed (bt, True, True).reshaped (<<52000>>), 1)
				.concatenated (ab.matmul (bb).reshaped (<<104000>>), 1)
				.concatenated (ab.matmul (b).reshaped (<<104000>>), 1)
		end

	twin_matmul_split_k (f: SIMPLE_TENSOR): TENSOR
			-- Long-K products that VULKAN_BACKEND splits across K slices: 40 x 1100 x 70
			-- (4 slices of 288, the last 236 deep) under every transpose flag, and a weight-
			-- gradient shape, 316 x 2000 x 384 with A transposed (7 slices).
		local
			a, b, at, bt, g, x: TENSOR
		do
			a := seeded (f, <<40, 1100>>, -1.0, 1.0, 130)
			b := seeded (f, <<1100, 70>>, -1.0, 1.0, 131)
			at := seeded (f, <<1100, 40>>, -1.0, 1.0, 132)
			bt := seeded (f, <<70, 1100>>, -1.0, 1.0, 133)
			g := seeded (f, <<2000, 316>>, -1.0, 1.0, 134)
			x := seeded (f, <<2000, 384>>, -1.0, 1.0, 135)
			Result := a.matmul (b).reshaped (<<2800>>)
				.concatenated (at.matmul_transposed (b, True, False).reshaped (<<2800>>), 1)
				.concatenated (a.matmul_transposed (bt, False, True).reshaped (<<2800>>), 1)
				.concatenated (at.matmul_transposed (bt, True, True).reshaped (<<2800>>), 1)
				.concatenated (g.matmul_transposed (x, True, False).reshaped (<<121344>>), 1)
		end

	twin_global_norm (f: SIMPLE_TENSOR): TENSOR
		local
			a, b, c: TENSOR
			g: GRAD_STORE
		do
			a := f.parameter (seeded (f, <<300, 7>>, -1.0, 1.0, 130))
			b := f.parameter (seeded (f, <<13>>, -1.0, 1.0, 131))
			c := f.parameter (seeded (f, <<50, 50>>, -1.0, 1.0, 132))
			g := ((a * a).sum_all + (b * b * b).sum_all + (c * a.sliced (1, 1, 50).sum (2, True)).sum_all).backward
			Result := f.from_array (<<g.global_norm.truncated_to_real>>, <<1>>)
		end

	twin_linear (f: SIMPLE_TENSOR): TENSOR
		local
			x, w, b, g, y: TENSOR
			s: GRAD_STORE
		do
			x := f.parameter (seeded (f, <<70, 33>>, -1.0, 1.0, 120))
			w := f.parameter (seeded (f, <<65, 33>>, -1.0, 1.0, 121))
			b := f.parameter (seeded (f, <<65>>, -1.0, 1.0, 122))
			g := seeded (f, <<70, 65>>, -1.0, 1.0, 123)
			y := x.linear (w, b)
			s := (y * g).sum_all.backward
			Result := y.reshaped (<<4550>>).concatenated (s.gradient (x).reshaped (<<2310>>), 1)
				.concatenated (s.gradient (w).reshaped (<<2145>>), 1).concatenated (s.gradient (b), 1)
		end

	twin_dropped (f: SIMPLE_TENSOR): TENSOR
		local
			x, y: TENSOR
		do
			x := f.parameter (seeded (f, <<9, 11>>, -1.0, 1.0, 124))
			y := x.dropped (0.75, 2027, 12345)
			Result := y.reshaped (<<99>>).concatenated ((y * y).sum_all.backward.gradient (x).reshaped (<<99>>), 1)
		end

	twin_causal_softmax (f: SIMPLE_TENSOR): TENSOR
		local
			x, w, a: TENSOR
		do
			x := f.parameter (seeded (f, <<2, 3, 5, 5>>, -2.0, 2.0, 96))
			w := seeded (f, <<2, 3, 5, 5>>, -1.0, 1.0, 97)
			a := x.causal_softmax (0.6)
			Result := a.reshaped (<<150>>).concatenated ((a * w).sum_all.backward.gradient (x).reshaped (<<150>>), 1)
		end

	twin_layer_norm (f: SIMPLE_TENSOR): TENSOR
		local
			x, w, b, g, y: TENSOR
			s: GRAD_STORE
		do
			x := f.parameter (seeded (f, <<4, 6>>, -2.0, 2.0, 98))
			w := f.parameter (seeded (f, <<6>>, 0.5, 1.5, 99))
			b := f.parameter (seeded (f, <<6>>, -0.5, 0.5, 100))
			g := seeded (f, <<4, 6>>, -1.0, 1.0, 101)
			y := x.layer_norm (w, b, 1.0e-5)
			s := (y * g).sum_all.backward
			Result := y.reshaped (<<24>>).concatenated (s.gradient (x).reshaped (<<24>>), 1)
				.concatenated (s.gradient (w), 1).concatenated (s.gradient (b), 1)
		end

feature -- Custom kernels

	test_gpu_custom_kernel
			-- A client kernel (y = 2 x + c over n elements) added, run on a 2-D group grid,
			-- its output read back fresh (the run made the host mirror stale).
		local
			x, y: TENSOR
			p: VULKAN_PUSH_BLOCK
			i, n: INTEGER
			ok: BOOLEAN
		do
			check attached {VULKAN_BACKEND} gpu.backend as v then
				if not v.has_kernel ("test_affine") then
					assert ("built-ins known", v.has_kernel ("matmul") and v.has_kernel ("adamw"))
					v.add_kernel ("test_affine", v.kernel_header + "[
						layout(push_constant) uniform P { F x; F y; uint n; float c; } p;
						void main() { uint i = gid(); if (i < p.n) p.y.v[i] = 2.0 * p.x.v[i] + p.c; }
					]")
				end
				assert ("added", v.has_kernel ("test_affine"))
				n := 70000
				x := gpu.from_array (seeded_values (n), <<n>>)
				y := gpu.zeros (<<n>>)
				assert ("mirror filled", y.item (5) = 0.0)
				create p.make
				p.address (v.storage_address (x.storage)); p.address (v.storage_address (y.storage))
				p.natural (n); p.real (0.5)
				v.run_kernel ("test_affine", p, 137, 2, 1, <<y.storage>>)
				ok := True
				from i := 0 until i = n loop
					ok := ok and y.item (i) = 2.0 * x.item (i) + 0.5
					i := i + 1
				end
				assert ("every element", ok)
			end
		end

feature {NONE} -- Custom kernel data

	seeded_values (n: INTEGER): ARRAY [REAL_32]
			-- `n' values in [-8, 8) with exact doubles (multiples of 1/64).
		local
			i: INTEGER
		do
			create Result.make_filled (0.0, 1, n)
			from i := 1 until i > n loop
				Result [i] := ((i * 7919) \\ 1024 - 512).to_real / 64.0
				i := i + 1
			end
		end

feature {NONE} -- Subject

	gpu: SIMPLE_TENSOR
			-- Facade on this processor's GPU backend.
		once
			create Result.make_with_backend (create {VULKAN_BACKEND}.make)
		end

end
