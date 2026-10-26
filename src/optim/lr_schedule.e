note
	description: "Learning rate as a function of the step."
	author: "Larry Rix"

deferred class
	LR_SCHEDULE

feature -- Access

	rate (a_step: INTEGER): REAL_64
			-- Learning rate at step `a_step'.
		require
			non_negative: a_step >= 0
		deferred
		ensure
			non_negative: Result >= 0.0
		end

end
