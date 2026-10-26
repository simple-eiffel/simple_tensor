note
	description: "[
		fp16 / bf16 <-> REAL_32 conversion of whole blocks (checkpoint I/O,
		D-005). Kernels are `C blocking' on C-heap memory. IEEE half floats keep
		signed zero, subnormals, infinities and NaN; bfloat16 is the top half of
		a float (a shift).
	]"
	author: "Larry Rix"

class
	TENSOR_DTYPE_CODEC

feature -- Conversion

	f16_to_f32 (a_source: POINTER; a_count: INTEGER; a_out: CPU_STORAGE)
			-- Decode `a_count' IEEE half floats at `a_source' into `a_out'.
		require
			source: a_source /= default_pointer
			sized: a_out.count = a_count
			live: not a_out.is_released
		do
			c_f16_to_f32 (a_source, a_out.address, a_count)
		end

	bf16_to_f32 (a_source: POINTER; a_count: INTEGER; a_out: CPU_STORAGE)
			-- Decode `a_count' bfloat16 values at `a_source' into `a_out'.
		require
			source: a_source /= default_pointer
			sized: a_out.count = a_count
			live: not a_out.is_released
		do
			c_bf16_to_f32 (a_source, a_out.address, a_count)
		end

	f32_copy (a_source: POINTER; a_count: INTEGER; a_out: CPU_STORAGE)
			-- Copy `a_count' little-endian floats at `a_source' into `a_out'.
		require
			source: a_source /= default_pointer
			sized: a_out.count = a_count
			live: not a_out.is_released
		do
			c_copy (a_source, a_out.address, a_count * {PLATFORM}.real_32_bytes)
		end

	f32_bits_to_f16 (a_value: REAL_32): NATURAL_16
			-- Nearest half float (round to nearest even), for tests.
		do
			Result := c_f32_to_f16 (a_value).to_natural_16
		end

feature {NONE} -- C kernels (blocking: C-heap buffers only)

	c_f16_to_f32 (a_source, a_out: POINTER; n: INTEGER)
		external
			"C blocking inline use <string.h>"
		alias
			"[
				const unsigned short* H = (const unsigned short*)$a_source; float* F = (float*)$a_out;
				int i, cnt = $n;
				for (i = 0; i < cnt; i++) {
					unsigned int h = H[i], s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, f;
					if (e == 0) {
						if (m == 0) { f = s; }
						else {
							int ex = 1;
							while ((m & 0x400u) == 0) { m <<= 1; ex--; }
							m &= 0x3FFu;
							f = s | ((unsigned int)(ex + 112) << 23) | (m << 13);
						}
					} else if (e == 31) {
						f = s | 0x7F800000u | (m << 13);
					} else {
						f = s | ((e + 112u) << 23) | (m << 13);
					}
					memcpy(&F[i], &f, 4);
				}
			]"
		end

	c_bf16_to_f32 (a_source, a_out: POINTER; n: INTEGER)
		external
			"C blocking inline use <string.h>"
		alias
			"[
				const unsigned short* H = (const unsigned short*)$a_source; float* F = (float*)$a_out;
				int i, cnt = $n;
				for (i = 0; i < cnt; i++) { unsigned int f = ((unsigned int)H[i]) << 16; memcpy(&F[i], &f, 4); }
			]"
		end

	c_copy (a_source, a_out: POINTER; a_bytes: INTEGER)
		external
			"C blocking inline use <string.h>"
		alias
			"memcpy((void*)$a_out, (const void*)$a_source, (size_t)$a_bytes);"
		end

	c_f32_to_f16 (a_value: REAL_32): INTEGER
		external
			"C inline use <string.h>"
		alias
			"[
				float fv = $a_value; unsigned int x, sign, m; int e;
				memcpy(&x, &fv, 4);
				sign = (x >> 16) & 0x8000u; e = (int)((x >> 23) & 0xFFu) - 127 + 15; m = x & 0x7FFFFFu;
				if (((x >> 23) & 0xFFu) == 0xFFu) return (EIF_INTEGER)(sign | 0x7C00u | (m ? 0x200u : 0u));
				if (e >= 31) return (EIF_INTEGER)(sign | 0x7C00u);
				if (e <= 0) {
					unsigned int hm, rem, half; int shift;
					if (e < -10) return (EIF_INTEGER)sign;
					m |= 0x800000u; shift = 14 - e;
					hm = m >> shift; rem = m & ((1u << shift) - 1u); half = 1u << (shift - 1);
					if (rem > half || (rem == half && (hm & 1u))) hm++;
					return (EIF_INTEGER)(sign | hm);
				} else {
					unsigned int hm = m >> 13, rem = m & 0x1FFFu, r = sign | ((unsigned int)e << 10) | hm;
					if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) r++;
					return (EIF_INTEGER)r;
				}
			]"
		end

end
