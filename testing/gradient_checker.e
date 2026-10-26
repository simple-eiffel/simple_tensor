note
	description: "[
		Central-difference gradient check of a TENSOR -> scalar function on the
		REFERENCE backend (spec 03 A-S04): relative step 1e-2·max(1,|x|),
		tolerance 1e-2 absolute + 1e-3 relative. A failure is re-checked under
		a REAL_64 storage build before it counts as a defect (intent-v2 Q9).
	]"
	author: "Larry Rix"

class
	GRADIENT_CHECKER

create
	make

feature {NONE} -- Initialization

	make (a_facade: SIMPLE_TENSOR)
		require
			on_reference_backend: a_facade.backend.kind = a_facade.backend.Reference_backend_kind
		do
			facade := a_facade
			create last_report.make_empty
		ensure
			facade_set: facade = a_facade
		end

feature -- Access

	facade: SIMPLE_TENSOR

	worst_error: REAL_64
			-- Largest mismatch (beyond tolerance scale) of the last `check_function'.

	last_report: STRING_8

feature -- Basic operations

	check_function (a_function: FUNCTION [TENSOR, TENSOR]; a_input: TENSOR)
			-- Compare backward's gradient of `a_function' at `a_input' with central differences.
		require
			leaf_input: a_input.requires_grad
		local
			l_loss: TENSOR
			l_analytic: ARRAY [REAL_32]
			l_values, l_plus, l_minus: ARRAY [REAL_32]
			l_dims: ARRAY [INTEGER]
			i: INTEGER
			x, h, f_plus, f_minus, l_numeric, l_error: REAL_64
		do
			worst_error := 0.0
			l_ok := True
			l_loss := a_function.item ([a_input])
			check scalar_loss: l_loss.is_scalar end
			l_analytic := l_loss.backward.gradient (a_input).to_array
			l_values := a_input.to_array
			l_dims := a_input.shape.to_array
			from i := l_values.lower until i > l_values.upper loop
				x := l_values [i]
				h := 1.0e-2 * x.abs.max (1.0)
				l_plus := l_values.twin
				l_plus [i] := (x + h).truncated_to_real
				l_minus := l_values.twin
				l_minus [i] := (x - h).truncated_to_real
				f_plus := a_function.item ([facade.from_array (l_plus, l_dims)]).item (0)
				f_minus := a_function.item ([facade.from_array (l_minus, l_dims)]).item (0)
					-- Use the steps actually taken after REAL_32 rounding.
				l_numeric := (f_plus - f_minus) / (l_plus [i].to_double - l_minus [i].to_double)
				l_error := (l_analytic [i] - l_numeric).abs - (Tolerance_absolute + Tolerance_relative * l_numeric.abs)
				if l_error > 0.0 then
					l_ok := False
				end
				worst_error := worst_error.max ((l_analytic [i] - l_numeric).abs)
				i := i + 1
			end
			last_report := "worst |analytic - numeric| = " + worst_error.out + " over " + l_values.count.out + " elements"
		ensure
			reported: not last_report.is_empty
		end

	passed: BOOLEAN
			-- Did the last check stay within tolerance?
		do
			Result := l_ok
		end

feature {NONE} -- Implementation

	l_ok: BOOLEAN

	Tolerance_absolute: REAL_64 = 1.0e-2
	Tolerance_relative: REAL_64 = 1.0e-3

end
