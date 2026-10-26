note
	description: "[
		safetensors checkpoints: 8-byte little-endian header length, a JSON
		header (simple_json) mapping names to {dtype, shape, data_offsets},
		then raw little-endian data. The header key `__metadata__' (a string map,
		not a tensor) is skipped on load. Loads F32, F16 and BF16 into REAL_32;
		saves F32. Environmental errors use `has_error' / `last_error'.

		Format read raw from the primary source (huggingface/safetensors
		README, 2026-10-25): offsets are relative to the start of the byte
		buffer; the header may be padded with spaces; shape [] is a scalar
		(loaded as [1]); empty tensors (a 0 dim) are legal in the format but
		not representable here, so they are reported as an error.
		Bulk reads and writes go through simple_file's read_block/write_block
		(C heap, `blocking', 64-bit offsets: gap G-5).
	]"
	author: "Larry Rix"

class
	SAFETENSORS_FILE

inherit
	GC_POINT

create
	make

feature {NONE} -- Initialization

	make (a_facade: SIMPLE_TENSOR)
			-- Reader/writer creating tensors through `a_facade'.
		do
			facade := a_facade
			create last_error.make_empty
			create tensors.make (32)
		ensure
			facade_set: facade = a_facade
			nothing_loaded: not is_loaded and not has_error
		end

feature -- Access

	facade: SIMPLE_TENSOR

	last_error: STRING_32

	names_model: MML_SET [STRING_8]
			-- Names of the loaded tensors.
		do
			create Result
			across tensors.current_keys as k loop
				Result := Result & k
			end
		end

	tensor (a_name: STRING_8): TENSOR
			-- The loaded tensor `a_name' (an untracked leaf).
		require
			loaded: is_loaded
			known: has (a_name)
		do
			check attached tensors.item (a_name) as t then
				Result := t
			end
		ensure
			untracked: not Result.is_tracked
		end

feature -- Status report

	is_loaded: BOOLEAN
	has_error: BOOLEAN

	has (a_name: STRING_8): BOOLEAN
		do
			Result := tensors.has (a_name)
		end

feature -- Basic operations

	load (a_path: READABLE_STRING_GENERAL)
			-- Read every tensor of the file at `a_path'.
		local
			l_file: SIMPLE_FILE
			l_prefix, l_header: MANAGED_POINTER
			l_length: NATURAL_64
		do
			reset
			create l_file.make (a_path)
			create l_prefix.make (8)
			if not l_file.read_block (l_prefix, 0, 0, 8) then
				fail ({STRING_32} "Cannot read the 8-byte header length: " + l_file.last_error)
			else
				l_length := l_prefix.read_natural_64_le (0)
				if l_length = 0 or l_length > Max_header_bytes then
					fail ({STRING_32} "Implausible header length " + l_length.out)
				else
					create l_header.make (l_length.to_integer_32)
					if not l_file.read_block (l_header, 0, 8, l_length.to_integer_32) then
						fail ({STRING_32} "Cannot read the header: " + l_file.last_error)
					else
						load_tensors (l_file, header_text (l_header), 8 + l_length.to_integer_64)
					end
				end
			end
			if has_error then
				tensors.wipe_out
			else
				is_loaded := True
			end
		ensure
			loaded_xor_error: is_loaded xor has_error
			error_explained: has_error implies not last_error.is_empty
			nothing_on_error: has_error implies names_model.is_empty
		end

	save (a_path: READABLE_STRING_GENERAL; a_names: ARRAY [STRING_8]; a_tensors: ARRAY [TENSOR])
			-- Write `a_tensors' under `a_names' as F32.
		require
			paired: a_names.count = a_tensors.count
			live: across a_tensors as t all not t.is_released end
			distinct_names: across a_names.lower |..| a_names.upper as i all
				across a_names.lower |..| a_names.upper as j all i /= j implies not a_names [i].same_string (a_names [j]) end end
			no_reserved_name: across a_names as n all not n.same_string ("__metadata__") end
		local
			l_file: SIMPLE_FILE
			l_header: STRING_8
			l_block: MANAGED_POINTER
			l_offset: INTEGER_64
			k, l_bytes: INTEGER
		do
			has_error := False
			create last_error.make_empty
			create l_header.make (256)
			l_header.append_character ('{')
			from k := a_names.lower until k > a_names.upper loop
				if k > a_names.lower then
					l_header.append_character (',')
				end
				l_bytes := a_tensors [a_tensors.lower + k - a_names.lower].shape.count * {PLATFORM}.real_32_bytes
				l_header.append (json_string (a_names [k]))
				l_header.append (":{%"dtype%":%"F32%",%"shape%":[")
				append_dims (l_header, a_tensors [a_tensors.lower + k - a_names.lower].shape)
				l_header.append ("],%"data_offsets%":[" + l_offset.out + "," + (l_offset + l_bytes).out + "]}")
				l_offset := l_offset + l_bytes
				k := k + 1
			end
			l_header.append_character ('}')
			from until l_header.count \\ 8 = 0 loop
				l_header.append_character (' ')
			end
			create l_block.make (8 + l_header.count)
			l_block.put_natural_64_le (l_header.count.to_natural_64, 0)
			(l_block.item + 8).memory_copy (l_header.area.base_address, l_header.count)
			create l_file.make (a_path)
			if not l_file.write_block (l_block, l_block.count, False) then
				fail ({STRING_32} "Cannot write the header: " + l_file.last_error)
			end
			from k := a_tensors.lower until has_error or k > a_tensors.upper loop
				if attached {CPU_STORAGE} a_tensors [k].storage as l_storage then
					create l_block.share_from_pointer (l_storage.address, l_storage.count * {PLATFORM}.real_32_bytes)
					if not l_file.write_block (l_block, l_block.count, True) then
						fail ({STRING_32} "Cannot write tensor data: " + l_file.last_error)
					end
				else
					fail ({STRING_32} "Only host (CPU) tensors can be saved in M1")
				end
				k := k + 1
			end
		ensure
			error_explained: has_error implies not last_error.is_empty
		end

feature {NONE} -- Implementation

	tensors: HASH_TABLE [TENSOR, STRING_8]

	Max_header_bytes: NATURAL_64 = 100_000_000
			-- The reference implementation's limit (100 MB).

	reset
		do
			is_loaded := False
			has_error := False
			create last_error.make_empty
			tensors.wipe_out
		ensure
			clean: not is_loaded and not has_error and tensors.is_empty
		end

	fail (a_message: STRING_32)
		do
			if not has_error then
				has_error := True
				last_error := a_message
			end
		ensure
			flagged: has_error and not last_error.is_empty
		end

	header_text (a_header: MANAGED_POINTER): STRING_32
			-- The header bytes decoded from UTF-8, trailing padding kept (JSON ignores it).
		local
			l_bytes: STRING_8
		do
			create l_bytes.make_filled (' ', a_header.count)
			l_bytes.area.base_address.memory_copy (a_header.item, a_header.count)
			Result := {UTF_CONVERTER}.utf_8_string_8_to_string_32 (l_bytes)
		end

	load_tensors (a_file: SIMPLE_FILE; a_header: STRING_32; a_buffer_start: INTEGER_64)
			-- Every tensor described by `a_header', its data at `a_buffer_start' + offsets.
		local
			l_json: SIMPLE_JSON
			l_dtype: STRING_32
			l_dims: ARRAY [INTEGER]
			l_begin, l_end: INTEGER_64
			l_count, l_width: INTEGER
		do
			create l_json
			if attached l_json.parse (a_header) as l_value and then l_value.is_object then
				across l_value.as_object.keys as k until has_error loop
					if not k.same_string ({STRING_32} "__metadata__") then
						if attached l_value.as_object.object_item (k) as l_entry and then
							attached l_entry.string_item ({STRING_32} "dtype") as l_type and then
							attached l_entry.array_item ({STRING_32} "shape") as l_shape and then
							attached l_entry.array_item ({STRING_32} "data_offsets") as l_offsets and then l_offsets.count = 2
						then
							l_dtype := l_type
							l_dims := dims_of (l_shape)
							l_begin := l_offsets.integer_item (1)
							l_end := l_offsets.integer_item (2)
							l_width := width_of (l_dtype)
							if l_width = 0 then
								fail ({STRING_32} "Unsupported dtype " + l_dtype + " for " + k)
							elseif l_begin < 0 or l_end < l_begin then
									-- Offsets are relative to the data buffer: a negative one would
									-- read the header (or before the file) as tensor data.
								fail ({STRING_32} "data_offsets out of the data buffer: " + k)
							elseif l_dims.is_empty then
								fail ({STRING_32} "Empty tensor (a 0 dimension) not representable: " + k)
							elseif not (create {TENSOR_SHAPE}.make_scalar).fits (l_dims) then
								fail ({STRING_32} "Tensor too large: " + k)
							else
								l_count := (create {TENSOR_SHAPE}.make_scalar).product_of (l_dims)
								if l_end - l_begin /= l_count.to_integer_64 * l_width then
									fail ({STRING_32} "data_offsets disagree with shape and dtype: " + k)
								else
									read_tensor (a_file, k.to_string_8, l_dtype, l_dims, a_buffer_start + l_begin, l_count, l_width)
								end
							end
						else
							fail ({STRING_32} "Malformed entry: " + k)
						end
					end
				end
			else
				fail ({STRING_32} "Header is not a JSON object: " + l_json.errors_as_string)
			end
		end

	read_tensor (a_file: SIMPLE_FILE; a_name: STRING_8; a_dtype: STRING_32; a_dims: ARRAY [INTEGER]; a_offset: INTEGER_64; a_count, a_width: INTEGER)
			-- Read and decode one tensor into a new storage of the facade's backend.
		local
			l_raw: MANAGED_POINTER
			l_storage: TENSOR_STORAGE
			l_codec: TENSOR_DTYPE_CODEC
		do
			create l_raw.make (a_count * a_width)
			if not a_file.read_block (l_raw, 0, a_offset, a_count * a_width) then
				fail ({STRING_32} "Cannot read data of " + a_name + ": " + a_file.last_error)
			else
				l_storage := facade.backend.new_storage (a_count)
				if attached {CPU_STORAGE} l_storage as l_cpu then
					create l_codec
					if a_dtype.same_string ({STRING_32} "F32") then
						l_codec.f32_copy (l_raw.item, a_count, l_cpu)
					elseif a_dtype.same_string ({STRING_32} "F16") then
						l_codec.f16_to_f32 (l_raw.item, a_count, l_cpu)
					else
						l_codec.bf16_to_f32 (l_raw.item, a_count, l_cpu)
					end
					tensors.force (create {TENSOR}.make_leaf (create {TENSOR_SHAPE}.make (a_dims), l_storage, facade.backend, False), a_name)
				else
					fail ({STRING_32} "Only host (CPU) backends can load in M1")
				end
			end
		end

	dims_of (a_shape: SIMPLE_JSON_ARRAY): ARRAY [INTEGER]
			-- The dims; [] (a scalar) becomes [1]; empty when any dim is 0 or negative.
		local
			i: INTEGER
			l_dim: INTEGER_64
		do
			if a_shape.count = 0 then
				Result := <<1>>
			else
				create Result.make_filled (1, 1, a_shape.count)
				from i := 1 until i > a_shape.count loop
					l_dim := a_shape.integer_item (i)
					if l_dim < 1 or l_dim > {INTEGER_32}.max_value then
						create Result.make_empty
						i := a_shape.count + 1
					else
						Result [i] := l_dim.to_integer_32
						i := i + 1
					end
				end
			end
		end

	width_of (a_dtype: STRING_32): INTEGER
			-- Bytes per element, 0 for an unsupported dtype.
		do
			if a_dtype.same_string ({STRING_32} "F32") then
				Result := 4
			elseif a_dtype.same_string ({STRING_32} "F16") or a_dtype.same_string ({STRING_32} "BF16") then
				Result := 2
			end
		end

	json_string (a_text: STRING_8): STRING_8
			-- `a_text' as a JSON string literal.
		do
			create Result.make (a_text.count + 2)
			Result.append_character ('"')
			across a_text as c loop
				if c = '"' or c = '\' then
					Result.append_character ('\')
				end
				Result.append_character (c)
			end
			Result.append_character ('"')
		end

	append_dims (a_header: STRING_8; a_shape: TENSOR_SHAPE)
		local
			k: INTEGER
		do
			from k := 1 until k > a_shape.rank loop
				if k > 1 then
					a_header.append_character (',')
				end
				a_header.append (a_shape.dim (k).out)
				k := k + 1
			end
		end

invariant
	loaded_means_no_error: is_loaded implies not has_error

end
