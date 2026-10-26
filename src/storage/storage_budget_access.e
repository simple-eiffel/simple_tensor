note
	description: "[
		Shared access to this thread's STORAGE_BUDGET (Phase 2 verdict G-B).
		Every class that allocates C-heap memory (CPU_STORAGE, INDEX_TENSOR)
		inherits this, so all of them note into ONE budget per thread. A
		`once' routine is per thread, hence per SCOOP processor (review spike
		R-1: two processors each saw only their own 1,000 increments).
	]"
	author: "Larry Rix"

class
	STORAGE_BUDGET_ACCESS

feature {NONE} -- Access

	storage_budget: STORAGE_BUDGET
			-- This thread's byte budget.
		once
			create Result.make (Default_storage_budget)
		end

	Default_storage_budget: INTEGER_64 = 268_435_456
			-- 256 MB between forced collections. Peak memory under a budget is
			-- about the live set plus the budget (review A-01: 221 MB peak at
			-- 2,000 x 4 MB churn with a collection every 50).

end
