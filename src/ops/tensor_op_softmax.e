note
	description: "[
		Softmax or log-softmax over the last axis.
		Backward: softmax y·(g − Σ(g·y)); log-softmax g − softmax·Σg.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_SOFTMAX

inherit
	TENSOR_OP
		redefine
			saved_live
		end

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a: TENSOR; a_output: TENSOR_STORAGE; a_log: BOOLEAN)
		require
			output_sized: a_output.count = a.shape.count
		do
			inputs := <<a>>
			output_values := a_output
			is_log := a_log
			output_shape := a.shape
		ensure
			input_set: inputs [1] = a
			output_saved: output_values = a_output
			log_set: is_log = a_log
		end

feature -- Access

	output_values: TENSOR_STORAGE
	is_log: BOOLEAN

	name: STRING_8
		do
			if is_log then Result := "log_softmax" else Result := "softmax" end
		end

feature -- Status report

	saved_live: BOOLEAN
			-- Inputs and the saved output are still in memory.
		do
			Result := Precursor and not output_values.is_released
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
			-- softmax: y·(g − Σ(g·y)); log-softmax: g − exp(y)·Σg (sums over the last axis).
		local
			a, y: TENSOR
			r: INTEGER
		do
			a := inputs [1]
			if a.is_tracked then
				y := leaf_over (output_values, a.shape, a.backend)
				r := a.shape.rank
				if is_log then
					a_store.accumulate (a, a_output_grad - y.exponential * a_output_grad.sum (r, True))
				else
					a_store.accumulate (a, y * (a_output_grad - (a_output_grad * y).sum (r, True)))
				end
			end
		end

end
