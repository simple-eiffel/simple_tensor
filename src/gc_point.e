note
	description: "[
		An empty `C blocking' external. A long Eiffel loop calls `gc_point'
		every few milliseconds so a collection pending on another processor
		can proceed. Measured (spike S-3, 2026-10-25): a 3.2 s plain-Eiffel
		loop stalled another processor's allocation 3,129 ms without it,
		6 ms with it, at no speed cost.
	]"
	author: "Larry Rix"

class
	GC_POINT

feature {NONE} -- GC

	gc_point
			-- Let a pending collection pass this thread.
		external
			"C blocking inline"
		alias
			"/* GC point */"
		end

end
