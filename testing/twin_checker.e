note
	description: "[
		Twin parity (I-002): run the same op on two backends from the same seeded
		inputs and report the worst relative error, |candidate - oracle| /
		max (1, |oracle|). The executable form of the semantic postcondition
		every TENSOR_BACKEND shares. The worst error accumulates over calls.
	]"
	author: "Larry Rix"

class
	TWIN_CHECKER

create
	make

feature {NONE} -- Initialization

	make (a_oracle, a_candidate: SIMPLE_TENSOR)
		require
			distinct_kinds: a_oracle.backend.kind /= a_candidate.backend.kind
		do
			oracle := a_oracle
			candidate := a_candidate
			create last_report.make_empty
		ensure
			set: oracle = a_oracle and candidate = a_candidate
		end

feature -- Access

	oracle, candidate: SIMPLE_TENSOR

	worst_relative_error: REAL_64

	last_report: STRING_8
			-- Name and error of the worst comparison so far.

feature -- Basic operations

	compare (a_op: FUNCTION [SIMPLE_TENSOR, TENSOR])
			-- Run `a_op' on both facades (same seed inside `a_op') and record the worst error.
		do
			compare_named ("op", a_op)
		ensure
			non_negative: worst_relative_error >= 0.0
		end

	compare_named (a_name: STRING_8; a_op: FUNCTION [SIMPLE_TENSOR, TENSOR])
			-- `compare', remembering `a_name' when it sets a new worst.
		local
			l_ref, l_got: ARRAY [REAL_32]
			i: INTEGER
			l_error: REAL_64
		do
			l_ref := a_op.item ([oracle]).to_array
			l_got := a_op.item ([candidate]).to_array
			check same_count: l_ref.count = l_got.count end
			from i := l_ref.lower until i > l_ref.upper loop
				l_error := (l_got [i] - l_ref [i]).abs / (l_ref [i].abs.max (1.0))
				if l_error > worst_relative_error or l_error /= l_error then
					worst_relative_error := l_error
					last_report := a_name + " element " + i.out + ": oracle " + l_ref [i].out + ", candidate " + l_got [i].out
				end
				i := i + 1
			end
		ensure
			non_negative: worst_relative_error >= 0.0 or worst_relative_error /= worst_relative_error
		end

	within (a_tolerance: REAL_64): BOOLEAN
		require
			positive: a_tolerance > 0.0
		do
			Result := worst_relative_error <= a_tolerance
		end

end
