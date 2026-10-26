note
	description: "[
		Fused linear layer y = x W^T + b (TENSOR.linear): the bias is added in place on the
		matmul's fresh output, so no second full-size tensor. x is (rows x in), W (out x in),
		b (out). Backward: dx = g W; dW = g^T x; db = sum of g over rows.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_LINEAR

inherit
	TENSOR_OP

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (x, a_weight, a_bias: TENSOR; a_shape: TENSOR_SHAPE)
		do
			inputs := <<x, a_weight, a_bias>>
			output_shape := a_shape
		ensure
			inputs_set: inputs [1] = x and inputs [2] = a_weight and inputs [3] = a_bias
		end

feature -- Access

	name: STRING_8
		do
			Result := "linear"
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			x, w, b: TENSOR
		do
			x := inputs [1]; w := inputs [2]; b := inputs [3]
			if x.is_tracked then
				a_store.accumulate (x, a_output_grad.matmul (w.detached))
			end
			if w.is_tracked then
				a_store.accumulate (w, a_output_grad.matmul_transposed (x.detached, True, False))
			end
			if b.is_tracked then
				a_store.accumulate (b, a_output_grad.reduced_to (b.shape))
			end
		end

end
