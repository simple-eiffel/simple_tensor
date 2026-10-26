note
	description: "[
		SPIKE S-4 (intent review Q7): does C-heap memory behind MANAGED_POINTER
		pile up because the collector only sees the small Eiffel wrappers?
		Allocates 2,000 dead 4 MB MANAGED_POINTERs (8 GB total) three ways and
		reports the process's peak private bytes:
		  A. drop them, let the GC decide;
		  B. drop them, MEMORY.full_collect every 50;
		  C. free each one explicitly (a `release' that frees the block now).
	]"

class
	S4_APP

create
	make

feature {NONE} -- Initialization

	make
		do
			print ("start private MB: " + private_mb.out + "%N")
			run ("A. GC decides            ", False, False)
			run ("B. full_collect every 50 ", True, False)
			run ("C. explicit release      ", False, True)
		end

	run (a_label: STRING; a_collect, a_release: BOOLEAN)
		local
			i: INTEGER
			l_block: MANAGED_POINTER
			l_peak: INTEGER_64
			l_mem: MEMORY
			t0: INTEGER_64
		do
			create l_mem
			l_mem.full_collect
			t0 := now_ms
			from i := 1 until i > 2_000 loop
				create l_block.make (4 * 1024 * 1024)
				l_block.put_real_32 (1.0, 0)
				if a_release then
					c_free_block (l_block)
				end
				-- Sample EVERY iteration, BEFORE any collection (review A-01: sampling
				-- only right after a collect measured the floor, not the peak).
				l_peak := l_peak.max (private_mb)
				if a_collect and i \\ 50 = 0 then
					l_mem.full_collect
				end
				i := i + 1
			end
			l_mem.full_collect
			print (a_label + ": peak private MB " + l_peak.out + ", after collect " + private_mb.out + ", " + (now_ms - t0).out + " ms%N")
		end

	c_free_block (a_block: MANAGED_POINTER)
			-- What a TENSOR_STORAGE.release would do: give the 4 MB back now and
			-- leave the wrapper pointing at a 1-byte block.
		do
			a_block.resize (1)
		end

	private_mb: INTEGER_64
		external
			"C inline use <windows.h>, <psapi.h>"
		alias
			"[
				PROCESS_MEMORY_COUNTERS pmc;
				pmc.cb = sizeof(pmc);
				GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
				return (EIF_INTEGER_64)(pmc.PagefileUsage / (1024 * 1024));
			]"
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

end
