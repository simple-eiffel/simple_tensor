note
	description: "[
		Byte budget for C-heap tensor memory on this thread (intent-v2 Q7).
		The collector sees only the small Eiffel wrapper of a MANAGED_POINTER,
		not the C-heap block behind it: spike S-4 churned 8 GB of dead 4 MB
		blocks and peaked at 8,040 MB with no collection at all, 221 MB with a
		forced collection every 50 (re-measured after review A-01). Every
		allocation is noted here; once the bytes allocated since the last
		collection would exceed `budget', a full collection runs first.
	]"
	author: "Larry Rix"

class
	STORAGE_BUDGET

create
	make

feature {NONE} -- Initialization

	make (a_budget: INTEGER_64)
			-- Budget of `a_budget' bytes between forced collections.
		require
			positive: a_budget > 0
		do
			budget := a_budget
		ensure
			budget_set: budget = a_budget
			fresh: allocated_since_collect = 0 and collections = 0 and largest_request = 0
		end

feature -- Access

	budget: INTEGER_64
			-- Bytes allowed between forced collections.

	allocated_since_collect: INTEGER_64
			-- Bytes noted since the last forced collection.

	collections: INTEGER
			-- Forced collections so far.

	largest_request: INTEGER_64
			-- Largest single allocation noted so far.

feature -- Element change

	note_allocation (a_bytes: INTEGER_64)
			-- Record an allocation of `a_bytes'; collect first when it would go over budget.
		require
			non_negative: a_bytes >= 0
		do
			if allocated_since_collect + a_bytes > budget then
				(create {MEMORY}).full_collect
				collections := collections + 1
				allocated_since_collect := a_bytes
			else
				allocated_since_collect := allocated_since_collect + a_bytes
			end
			largest_request := largest_request.max (a_bytes)
		ensure
			collected_when_over: (old allocated_since_collect + a_bytes > budget) implies
				(collections = old collections + 1 and allocated_since_collect = a_bytes)
			counted_when_under: (old allocated_since_collect + a_bytes <= budget) implies
				(collections = old collections and allocated_since_collect = old allocated_since_collect + a_bytes)
			largest_tracked: largest_request = (old largest_request).max (a_bytes)
			budget_unchanged: budget = old budget
		end

	set_budget (a_budget: INTEGER_64)
			-- Budget of `a_budget' bytes from now on; when the bytes already noted exceed
			-- it, collect first (Phase 6: a caller lowering the budget mid-run cannot be
			-- expected to know `allocated_since_collect').
		require
			positive: a_budget > 0
		do
			if allocated_since_collect > a_budget + largest_request then
				(create {MEMORY}).full_collect
				collections := collections + 1
				allocated_since_collect := 0
			end
			budget := a_budget
		ensure
			set: budget = a_budget
			count_kept_when_within: (old allocated_since_collect <= a_budget + old largest_request) implies
				(allocated_since_collect = old allocated_since_collect and collections = old collections)
			collected_when_over: (old allocated_since_collect > a_budget + old largest_request) implies
				(allocated_since_collect = 0 and collections = old collections + 1)
			largest_unchanged: largest_request = old largest_request
		end

invariant
	budget_positive: budget > 0
	non_negative: allocated_since_collect >= 0 and collections >= 0 and largest_request >= 0
	bounded: allocated_since_collect <= budget + largest_request

end
