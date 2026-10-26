note
	description: "Test runner for simple_tensor."
	author: "Larry Rix"

class
	TEST_APP

create
	make

feature {NONE} -- Initialization

	make
			-- Run every test.
		local
			t: LIB_TESTS
			n: NN_TESTS
		do
			create t
			print ("simple_tensor tests%N===================%N")
			-- Shape and storage
			run (agent t.test_shape_basics, "shape basics")
			run (agent t.test_shape_broadcast, "shape broadcast rules")
			run (agent t.test_storage_budget_collects, "storage budget forces a collection")
			-- Ops: gradient checks (REFERENCE) and twin parity (CPU vs REFERENCE)
			run (agent t.test_gradient_checks_all_primitives, "gradient check: every primitive")
			run (agent t.test_twin_parity_all_kernels, "twin parity: every CPU kernel")
			-- Autograd
			run (agent t.test_tied_weight_accumulates, "tied weight accumulates both gradients")
			run (agent t.test_frozen_leaf_gets_no_gradient, "frozen leaf gets no gradient")
			-- Training
			run (agent t.test_adamw_hand_trace, "AdamW 3-step hand trace")
			run (agent t.test_warmup_cosine_matches_rixgpt, "schedule equals rixgpt.py lr_at")
			run (agent t.test_determinism_same_seed, "same seed, same losses")
			run (agent t.test_non_finite_detected, "non-finite gradients detected")
			-- I/O and data
			run (agent t.test_safetensors_round_trip, "safetensors round trip")
			run (agent t.test_token_batcher_shifted_targets, "batcher targets are inputs shifted by one")
			-- Memory and GC law
			run (agent t.test_memory_stays_bounded, "1,000 steps stay within the memory budget")
			run (agent t.test_gc_probe_under_10ms, "GC probe: other processor's worst allocation under 10 ms")
			run (agent t.test_cpu_matmul_throughput, "CPU matmul throughput recorded (gate in lean target)")
			run (agent t.test_readme_quick_start, "README quick start learns XOR")
			run (agent t.test_like_factories_and_backend_copy, "zeros_like, ones_like, to_backend round trip")
			run (agent t.test_grad_store_scale_clip_merge, "GRAD_STORE scale, clip, merge")
			run (agent t.test_sgd_hand_trace, "SGD hand trace on both backends")
			run (agent t.test_random_index_and_dropout_mask, "random index range and dropout mask")
			run (agent t.test_storage_sharing_guards_release, "shared storage guards release")
			run (agent t.test_preconditions_reject_misuse, "preconditions reject misuse")
			run (agent t.test_adversarial_safetensors, "hostile safetensors files refused")
			run (agent t.test_adversarial_numerics, "extreme values stay finite or are flagged")
			run (agent t.test_backward_twice_diamond_and_deep_chain, "backward twice, diamond, 5,000-op chain")
			run (agent t.test_two_trainers_on_separate_processors, "two trainers on separate processors")
			run (agent t.test_stress_many_small_steps, "stress: 5,000 small steps")
			run (agent t.test_batcher_tightest_corpus, "batcher on the tightest corpus")
			run (agent t.test_budget_lowered_mid_run, "budget lowered mid-run collects")
			run (agent t.test_random_addresses_are_fixed, "random values have fixed stream addresses (G-6)")
			-- SCOOP consumer (compile-time compatibility of the main types)
			run (agent (create {TEST_SCOOP_CONSUMER}).test_scoop_compatibility, "SCOOP consumer compiles and creates the main types")
			create n
			-- M2 modules
			run (agent n.test_nn_linear_by_hand, "NN_LINEAR by hand (2D and 3D)")
			run (agent n.test_nn_names_and_tying, "module names (PyTorch layout) and tied weights")
			run (agent n.test_nn_layer_norm, "NN_LAYER_NORM values and gradients")
			run (agent n.test_nn_mlp_and_attention_gradients, "MLP and attention gradient checks")
			run (agent n.test_nn_attention_matches_naive, "attention equals a naive per-head computation")
			run (agent n.test_nn_dropout, "NN_DROPOUT eval identity and training statistics")
			print ("%N===================%NResults: " + passed.out + " passed, " + failed.out + " failed%N")
		end

feature {NONE} -- Runner

	passed, failed: INTEGER

	run (a_test: PROCEDURE; a_name: STRING)
		local
			l_retried: BOOLEAN
		do
			if not l_retried then
				a_test.call (Void)
				print ("  PASS: " + a_name + "%N")
				passed := passed + 1
			end
		rescue
			print ("  FAIL: " + a_name + "%N")
			if attached (create {EXCEPTION_MANAGER}).last_exception as e then
				print ("        " + e.generating_type.name + " in " + if attached e.type_name as tn then tn.to_string_8 else "?" end + "." + if attached e.recipient_name as rn then rn.to_string_8 else "?" end + ": " + if attached e.description as d then d.to_string_8 else "" end + "%N")
			end
			failed := failed + 1
			l_retried := True
			retry
		end

end
