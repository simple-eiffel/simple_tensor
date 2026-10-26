note
	description: "[
		Linear warmup to `peak', then cosine decay to `floor_ratio' · peak at
		`total' — exactly rixgpt.py's `lr_at' (RixGPT recipe: peak 1e-3,
		warmup 150, total 5000, floor ratio 0.1).
	]"
	author: "Larry Rix"

class
	WARMUP_COSINE_SCHEDULE

inherit
	LR_SCHEDULE

create
	make

feature {NONE} -- Initialization

	make (a_peak: REAL_64; a_warmup, a_total: INTEGER; a_floor_ratio: REAL_64)
		require
			peak_positive: a_peak > 0.0
			warmup_non_negative: a_warmup >= 0
			total_after_warmup: a_total > a_warmup
			floor_ratio_valid: 0.0 <= a_floor_ratio and a_floor_ratio <= 1.0
		do
			peak := a_peak
			warmup := a_warmup
			total := a_total
			floor_ratio := a_floor_ratio
		ensure
			peak_set: peak = a_peak
			warmup_set: warmup = a_warmup
			total_set: total = a_total
			floor_set: floor_ratio = a_floor_ratio
		end

feature -- Access

	peak: REAL_64
	warmup: INTEGER
	total: INTEGER
	floor_ratio: REAL_64

	rate (a_step: INTEGER): REAL_64
			-- rixgpt.py `lr_at', exactly.
		local
			r: REAL_64
		do
			if a_step < warmup then
				Result := peak * a_step / warmup.max (1)
			else
				r := ((a_step - warmup) / (total - warmup)).min (1.0)
				Result := peak * floor_ratio + 0.5 * (peak - peak * floor_ratio) * (1.0 + math.cos (math.pi * r))
			end
		ensure then
			warmup_linear: a_step < warmup implies Result = peak * a_step / warmup.max (1)
			never_below_floor: a_step >= warmup implies Result >= peak * floor_ratio - 1.0e-12
			never_above_peak: Result <= peak + 1.0e-12
			floor_after_total: a_step >= total implies (Result - peak * floor_ratio).abs <= 1.0e-12
		end

feature {NONE} -- Implementation

	math: SIMPLE_MATH
		once
			create Result.make
		end

invariant
	ordered: total > warmup and warmup >= 0
	floor_ratio_valid: 0.0 <= floor_ratio and floor_ratio <= 1.0

end
