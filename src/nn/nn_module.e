note
	description: "[
		A trainable building block (spec 04: NN_MODULE). Owns named parameter leaves and
		named child modules; `named_parameters' walks them depth first with dotted names
		("blocks.0.attn.in_proj_weight"), PyTorch's state_dict naming, so checkpoints map
		one to one. A tensor shared by two modules (tied weights) is listed once, under
		its first name. Training mode switches dropout on; it propagates to children.
		Modules that map a TENSOR to a TENSOR inherit NN_LAYER; NN_EMBEDDING maps indices
		and inherits this class only (spec 04, Liskov).
	]"
	author: "Larry Rix"

deferred class
	NN_MODULE

feature {NONE} -- Initialization

	make_module
			-- Empty parameter and child lists; training mode on (PyTorch's default).
		do
			create own_parameters.make (4)
			create children.make (4)
			is_training := True
		ensure
			empty: own_parameters.is_empty and children.is_empty
			training: is_training
		end

feature -- Access

	own_parameters: ARRAYED_LIST [TUPLE [name: STRING_8; tensor: TENSOR]]
			-- Parameters registered by this module itself.

	children: ARRAYED_LIST [TUPLE [name: STRING_8; module: NN_MODULE]]
			-- Child modules, in registration order.

	named_parameters: ARRAYED_LIST [TUPLE [name: STRING_8; tensor: TENSOR]]
			-- Every parameter of this module and its descendants, depth first, dotted names,
			-- each tensor once (by id).
		local
			l_seen: HASH_TABLE [BOOLEAN, INTEGER_64]
		do
			create Result.make (16)
			create l_seen.make (16)
			collect ("", Result, l_seen)
		ensure
			distinct: across Result as a all across Result as b all (@a.cursor_index /= @b.cursor_index) implies a.tensor /= b.tensor end end
		end

	parameters: ARRAYED_LIST [TENSOR]
			-- The tensors of `named_parameters', same order.
		do
			create Result.make (16)
			across named_parameters as p loop
				Result.extend (p.tensor)
			end
		ensure
			same_count: Result.count = named_parameters.count
		end

	parameter_count: INTEGER_64
			-- Trainable scalars.
		do
			across named_parameters as p loop
				Result := Result + p.tensor.shape.count
			end
		end

	parameter (a_name: STRING_8): detachable TENSOR
			-- The parameter named `a_name' (dotted), if any.
		do
			across named_parameters as p until attached Result loop
				if p.name.same_string (a_name) then
					Result := p.tensor
				end
			end
		end

feature -- Status report

	is_training: BOOLEAN
			-- Dropout active?

feature -- Mode

	set_training (a_on: BOOLEAN)
			-- Training mode `a_on' for this module and every descendant.
		do
			is_training := a_on
			across children as c loop
				c.module.set_training (a_on)
			end
		ensure
			set: is_training = a_on
		end

feature {NONE} -- Registration

	register_parameter (a_name: STRING_8; t: TENSOR)
		require
			trainable: t.requires_grad
			fresh_name: across own_parameters as p all not p.name.same_string (a_name) end
		do
			own_parameters.extend ([a_name, t])
		ensure
			registered: own_parameters.count = old own_parameters.count + 1
		end

	register_child (a_name: STRING_8; m: NN_MODULE)
		require
			fresh_name: across children as c all not c.name.same_string (a_name) end
			not_self: m /= Current
		do
			children.extend ([a_name, m])
		ensure
			registered: children.count = old children.count + 1
		end

feature {NN_MODULE} -- Implementation

	collect (a_prefix: STRING_8; a_into: ARRAYED_LIST [TUPLE [name: STRING_8; tensor: TENSOR]]; a_seen: HASH_TABLE [BOOLEAN, INTEGER_64])
			-- Append this module's parameters under `a_prefix', then each child's.
		do
			across own_parameters as p loop
				if not a_seen.has (p.tensor.id) then
					a_seen.put (True, p.tensor.id)
					a_into.extend ([a_prefix + p.name, p.tensor])
				end
			end
			across children as c loop
				c.module.collect (a_prefix + c.name + ".", a_into, a_seen)
			end
		end

end
