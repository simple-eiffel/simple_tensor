note
	description: "Test runner for simple_tensor's GPU backend (target simple_tensor_vulkan_tests)."
	author: "Larry Rix"

class
	GPU_TEST_APP

create
	make

feature {NONE} -- Initialization

	make
		local
			t: GPU_TESTS
		do
			create t
			print ("simple_tensor GPU tests%N=======================%N")
			run (agent t.test_gpu_storage_basics, "GPU storage basics")
			run (agent t.test_gpu_twin_parity, "GPU twin parity: every hot kernel")
			run (agent t.test_gpu_cold_parity, "GPU twin parity: every cold kernel")
			run (agent t.test_gpu_training_tracks_oracle, "GPU training tracks the oracle")
			run (agent t.test_gpu_fallback_ledger, "GPU host-fallback ledger")
			print ("=======================%NResults: " + passed.out + " passed, " + failed.out + " failed%N")
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
