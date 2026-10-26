note
	description: "Test module: an embedding and a linear head sharing its table (RixGPT's tying)."
	author: "Larry Rix"

class
	TIED_PAIR

inherit
	NN_MODULE

create
	make

feature {NONE} -- Initialization

	make (f: SIMPLE_TENSOR; r: TENSOR_RANDOM)
		do
			make_module
			create tok.make_normal (f, 5, 3, 0.02, r)
			create head.make_tied (f, tok.table, False)
			register_child ("tok", tok)
			register_child ("head", head)
		end

feature -- Access

	tok: NN_EMBEDDING
	head: NN_LINEAR

end
