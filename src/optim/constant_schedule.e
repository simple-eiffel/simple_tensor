note
	description: "The same learning rate at every step."
	author: "Larry Rix"

class
	CONSTANT_SCHEDULE

inherit
	LR_SCHEDULE

create
	make

feature {NONE} -- Initialization

	make (a_rate: REAL_64)
		require
			non_negative: a_rate >= 0.0
		do
			value := a_rate
		ensure
			set: value = a_rate
		end

feature -- Access

	value: REAL_64

	rate (a_step: INTEGER): REAL_64
		do
			Result := value
		ensure then
			constant: Result = value
		end

invariant
	non_negative: value >= 0.0

end
