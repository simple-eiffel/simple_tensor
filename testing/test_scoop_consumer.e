note
	description: "SCOOP consumer compatibility: the library's main types used from a SCOOP-enabled system."
	author: "Larry Rix"

class
	TEST_SCOOP_CONSUMER

feature -- Test

	test_scoop_compatibility
			-- Declare and create the main types (compile-time check under concurrency=scoop).
		local
			l_facade: SIMPLE_TENSOR
			l_reference: SIMPLE_TENSOR
			l_shape: TENSOR_SHAPE
			l_store: GRAD_STORE
			l_schedule: WARMUP_COSINE_SCHEDULE
			l_random: TENSOR_RANDOM
			l_file: SAFETENSORS_FILE
		do
			create l_facade.make
			create l_reference.make_with_backend (create {REFERENCE_BACKEND})
			create l_shape.make (<<2, 2>>)
			create l_store.make
			create l_schedule.make (1.0e-3, 150, 5000, 0.1)
			create l_random.make (1337)
			create l_file.make (l_facade)
		end

end
