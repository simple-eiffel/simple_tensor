note
	description: "Backend kind codes (split from op kinds, Phase 2 verdict R2-09)."
	author: "Larry Rix"

class
	BACKEND_KINDS

feature -- Backend kinds

	Reference_backend_kind: INTEGER = 1
	Cpu_backend_kind: INTEGER = 2
	Vulkan_backend_kind: INTEGER = 3

end
