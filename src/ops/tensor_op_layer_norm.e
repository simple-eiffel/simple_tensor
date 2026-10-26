note
	description: "[
		Fused layer normalization over the last axis (TENSOR.layer_norm): one output plus
		per-row mean and rstd, instead of the composition's five temporaries.
		Backward, with x^ = (x - mean) * rstd recomputed and gw = g * weight:
			dx = rstd * (gw - mean (gw) - x^ * mean (gw * x^))     (means over the last axis)
			dweight = sum over rows of g * x^;  dbias = sum over rows of g
	]"
	author: "Larry Rix"

class
	TENSOR_OP_LAYER_NORM

inherit
	TENSOR_OP
		redefine
			saved_live
		end

create {TENSOR}
	make

feature {NONE} -- Initialization

	make (x, a_weight, a_bias: TENSOR; a_mean, a_rstd: TENSOR_STORAGE)
		require
			stats_sized: a_mean.count = x.shape.count // x.shape.last and a_rstd.count = a_mean.count
		do
			inputs := <<x, a_weight, a_bias>>
			mean := a_mean
			rstd := a_rstd
			output_shape := x.shape
		ensure
			inputs_set: inputs [1] = x and inputs [2] = a_weight and inputs [3] = a_bias
		end

feature -- Access

	mean, rstd: TENSOR_STORAGE
			-- Per-row statistics from the forward pass.

	name: STRING_8
		do
			Result := "layer_norm"
		end

feature -- Status report

	saved_live: BOOLEAN
		do
			Result := Precursor and not mean.is_released and not rstd.is_released
		end

feature -- Autograd

	backward (a_output_grad: TENSOR; a_store: GRAD_STORE)
		local
			x, w, b, m, rs, xhat, gw: TENSOR
			r: INTEGER
			l_stats: TENSOR_SHAPE
		do
			x := inputs [1]; w := inputs [2]; b := inputs [3]
			r := x.shape.rank
			l_stats := x.shape.with_dim (r, 1)
			m := leaf_over (mean, l_stats, x.backend)
			rs := leaf_over (rstd, l_stats, x.backend)
			xhat := (x.detached - m) * rs
			if x.is_tracked then
				gw := a_output_grad * w.detached
				a_store.accumulate (x, rs * (gw - gw.mean (r, True) - xhat * (gw * xhat).mean (r, True)))
			end
			if w.is_tracked then
				a_store.accumulate (w, shaped_for (a_output_grad * xhat, w))
			end
			if b.is_tracked then
				a_store.accumulate (b, shaped_for (a_output_grad, b))
			end
		end

end
