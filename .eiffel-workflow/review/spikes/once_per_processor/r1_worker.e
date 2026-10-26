note
	description: "R-1 worker: bumps its own view of a `once' counter."

class
	R1_WORKER

feature

	final_count: INTEGER_64

	bump (n: INTEGER)
		local
			i: INTEGER
		do
			from i := 1 until i > n loop
				counter.put (counter.item + 1)
				i := i + 1
			end
			final_count := counter.item
		end

	counter: CELL [INTEGER_64]
		once
			create Result.put (0)
		end

end
