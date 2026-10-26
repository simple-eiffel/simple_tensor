note
	description: "[
		Review spike R-1 (claim CL-2): is a `once' function per SCOOP processor?
		Two separate workers each bump a `once' CELL counter 1,000 times; the
		root bumps its own 10 times. If `once' is per processor, the workers
		report 1000 and 1000 and the root 10. If it were process-wide, totals
		would interleave (e.g. 2010 at the end).
	]"

class
	R1_APP

create
	make

feature {NONE} -- Initialization

	make
		local
			a, b: separate R1_WORKER
			i: INTEGER
		do
			create a
			create b
			start (a)
			start (b)
			from i := 1 until i > 10 loop
				counter.put (counter.item + 1)
				i := i + 1
			end
			print ("root counter: " + counter.item.out + "%N")
			print ("worker A counter: " + result_of (a).out + "%N")
			print ("worker B counter: " + result_of (b).out + "%N")
		end

	start (w: separate R1_WORKER)
		do
			w.bump (1000)
		end

	result_of (w: separate R1_WORKER): INTEGER_64
		do
			Result := w.final_count
		end

	counter: CELL [INTEGER_64]
		once
			create Result.put (0)
		end

end
