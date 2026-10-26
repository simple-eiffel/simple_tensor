note
	description: "[
		Op kind codes shared by TENSOR, the ops and backend kernels (split from
		backend kinds, Phase 2 verdict R2-09). Public: they appear in public
		preconditions (`is_unary_kind'), so a {NONE} inheritance would be a VAPE.
	]"
	author: "Larry Rix"

class
	TENSOR_OP_KINDS

feature -- Binary kinds

	Add_kind: INTEGER = 1
	Sub_kind: INTEGER = 2
	Mul_kind: INTEGER = 3
	Div_kind: INTEGER = 4

	is_binary_kind (k: INTEGER): BOOLEAN
		do
			Result := Add_kind <= k and k <= Div_kind
		end

feature -- Unary kinds (TENSOR_OP_UNARY; `param' is used by the scale and add-scalar kinds only)

	Neg_kind: INTEGER = 11
	Scale_kind: INTEGER = 12
	Add_scalar_kind: INTEGER = 13
	Exp_kind: INTEGER = 14
	Log_kind: INTEGER = 15
	Sqrt_kind: INTEGER = 16
	Tanh_kind: INTEGER = 17
	Sigmoid_kind: INTEGER = 18
	Relu_kind: INTEGER = 19
	Gelu_kind: INTEGER = 20
	Silu_kind: INTEGER = 21

	is_unary_kind (k: INTEGER): BOOLEAN
		do
			Result := Neg_kind <= k and k <= Silu_kind
		end

	uses_param (k: INTEGER): BOOLEAN
			-- Does unary kind `k' read its `param'? (verdict R2-07)
		do
			Result := k = Scale_kind or k = Add_scalar_kind
		end

feature -- Reduction kinds

	Sum_kind: INTEGER = 31
	Mean_kind: INTEGER = 32
	Max_kind: INTEGER = 33

	is_reduce_kind (k: INTEGER): BOOLEAN
		do
			Result := Sum_kind <= k and k <= Max_kind
		end

end
