note
	description: "[
		The eleven one-input ops (neg, scale, add_scalar, exp, log, sqrt, tanh,
		sigmoid, relu, gelu, silu). They share one backward rule,
		g * f'(a), with f' computed by the backend's `unary_derivative' from
		the input and the saved output, so one class serves all eleven.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_UNARY

inherit
	TENSOR_OP
		redefine
			saved_live
		end

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (a_kind: INTEGER; a: TENSOR; a_output: TENSOR_STORAGE; a_param: REAL_32)
		require
			valid_kind: is_unary_kind (a_kind)
			output_sized: a_output.count = a.shape.count
		do
			kind := a_kind
			inputs := <<a>>
			output_shape := a.shape
			output_values := a_output
			param := a_param
		ensure
			kind_set: kind = a_kind
			input_set: inputs [1] = a
			output_saved: output_values = a_output
			param_set: param = a_param
		end

feature -- Access

	kind: INTEGER
	param: REAL_32

	output_values: TENSOR_STORAGE
			-- The forward output y (the exp, sqrt, tanh and sigmoid derivatives use it).

	name: STRING_8
		do
			Result := "unary"
		end

feature -- Status report

	saved_live: BOOLEAN
			-- Inputs and the saved output are still in memory.
		do
			Result := Precursor and not output_values.is_released
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
			-- g · f'(a), f' from the backend's derivative table and the saved output.
		local
			a: TENSOR
			l_derivative: TENSOR_STORAGE
		do
			a := inputs [1]
			if a.is_tracked then
				l_derivative := a.backend.new_storage (a.shape.count)
				a.backend.unary_derivative (kind, a.storage, output_values, l_derivative, param)
				a_store.accumulate (a, a_output_grad * leaf_over (l_derivative, a.shape, a.backend))
			end
		end

invariant
	valid_kind: is_unary_kind (kind)
	output_sized: output_values.count = output_shape.count

end
