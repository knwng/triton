	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	kernel_unified_attention_2d     ; -- Begin function kernel_unified_attention_2d
	.p2align	8
	.type	kernel_unified_attention_2d,@function
kernel_unified_attention_2d:            ; @kernel_unified_attention_2d
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.18:
	.file	1 "/app/OAI-triton/unified_attn_ubench" "unified_attention_aiter.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.19:
.LBB0_0:
	s_load_dwordx2 s[24:25], s[0:1], 0xb8
	s_load_dwordx4 s[20:23], s[0:1], 0xa8
	s_load_dwordx8 s[68:75], s[0:1], 0x88
	s_load_dword s19, s[0:1], 0xc0
	s_load_dwordx2 s[80:81], s[0:1], 0x78
	s_load_dwordx8 s[60:67], s[0:1], 0x58
	s_mov_b32 s18, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s19, 1
	s_cbranch_scc1 .LBB0_2
.LBB0_1:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	s_add_i32 s21, s19, s18
	s_lshr_b32 s26, s21, 31
	s_add_i32 s21, s21, s26
	s_ashr_i32 s26, s21, 1
	s_ashr_i32 s27, s26, 31
	s_lshl_b64 s[28:29], s[26:27], 2
	s_add_u32 s28, s24, s28
	s_addc_u32 s29, s25, s29
	s_load_dword s21, s[28:29], 0x0
	s_add_i32 s27, s26, 1
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s28, s21, 31
	s_lshr_b32 s28, s28, 28
	s_add_i32 s21, s21, s28
	s_ashr_i32 s21, s21, 4
	s_add_i32 s21, s21, s26
	s_cmp_gt_i32 s21, s17
	s_cselect_b32 s19, s26, s19
	s_cselect_b32 s18, s18, s27
	s_cmp_lt_i32 s18, s19
	s_cbranch_scc1 .LBB0_1
.LBB0_2:                                ; %Flow226
	s_ashr_i32 s19, s18, 31
	s_lshl_b64 s[26:27], s[18:19], 2
	s_load_dword s33, s[0:1], 0x38
	s_add_u32 s0, s24, s26
	s_addc_u32 s1, s25, s27
	s_add_u32 s0, s0, -4
	s_addc_u32 s1, s1, -1
	s_load_dwordx2 s[82:83], s[0:1], 0x0
	s_sub_i32 s0, s17, s18
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s1, s82, 31
	s_lshr_b32 s1, s1, 28
	s_add_i32 s1, s82, s1
	s_lshr_b32 s1, s1, 4
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s19, s0, 4
	s_sub_i32 s17, s83, s82
	s_add_i32 s19, s19, 16
	s_cmp_lt_i32 s19, s17
	s_cbranch_scc0 .LBB0_17
; %bb.3:
	v_lshrrev_b32_e32 v1, 3, v0
	v_lshrrev_b32_e32 v11, 6, v0
	s_lshl_b32 s24, s16, 3
	v_or_b32_e32 v10, s19, v11
	v_and_or_b32 v2, v1, 7, s24
	v_lshlrev_b32_e32 v19, 3, v0
	v_cmp_gt_i32_e64 s[0:1], s17, v10
	v_cmp_gt_i32_e32 vcc, 64, v2
	v_and_b32_e32 v18, 56, v19
	s_and_b64 s[28:29], vcc, s[0:1]
	v_mad_u64_u32 v[20:21], s[0:1], s64, v2, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_add_u32_e32 v3, s82, v10
	v_mul_lo_u32 v3, s62, v3
	v_add3_u32 v4, v3, v20, v18
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	global_load_dwordx4 v[6:9], v[4:5], off nt
.LBB0_5:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v3, 32, v1
	v_lshrrev_b32_e32 v3, 3, v3
	v_or_b32_e32 v12, s19, v3
	v_cmp_gt_i32_e64 s[0:1], s17, v12
	s_and_b64 s[28:29], vcc, s[0:1]
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB0_7
; %bb.6:
	v_add_u32_e32 v2, s82, v12
	v_mul_lo_u32 v2, s62, v2
	v_add3_u32 v2, v2, v20, v18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[4:5]
	global_load_dwordx4 v[2:5], v[2:3], off nt
.LBB0_7:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v12, 8, v10
	v_cmp_gt_i32_e64 s[0:1], s17, v12
	s_and_b64 s[28:29], vcc, s[0:1]
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB0_9
; %bb.8:
	v_add_u32_e32 v12, s82, v12
	v_mul_lo_u32 v12, s62, v12
	v_add3_u32 v12, v12, v20, v18
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[4:5]
	global_load_dwordx4 v[14:17], v[12:13], off nt
.LBB0_9:
	s_or_b64 exec, exec, s[0:1]
	v_or3_b32 v21, v11, s19, 12
	v_cmp_gt_i32_e64 s[0:1], s17, v21
	s_and_b64 s[28:29], vcc, s[0:1]
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	s_and_saveexec_b64 s[0:1], s[28:29]
	s_cbranch_execz .LBB0_11
; %bb.10:
	v_add_u32_e32 v10, s82, v21
	v_mul_lo_u32 v10, s62, v10
	v_add3_u32 v10, v10, v20, v18
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[4:5]
	global_load_dwordx4 v[10:13], v[10:11], off nt
.LBB0_11:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v21, 0xc0, v0
	v_lshrrev_b32_e32 v20, 4, v21
	v_bfe_u32 v22, v0, 3, 2
	v_or3_b32 v72, v20, v22, s19
	v_and_b32_e32 v22, 7, v0
	v_or_b32_e32 v73, s24, v22
	v_cmp_gt_i32_e64 s[0:1], s17, v72
	v_cmp_gt_i32_e32 vcc, 64, v73
	s_and_b64 s[84:85], vcc, s[0:1]
	v_lshlrev_b32_e32 v20, 4, v0
	s_movk_i32 s0, 0x70
	v_bitop3_b32 v23, v20, v0, s0 bitop3:0x78
	s_add_u32 s0, s14, s26
	s_addc_u32 s1, s15, s27
	s_add_u32 s0, s0, -4
	v_add_u32_e32 v75, 0, v23
	s_addc_u32 s1, s1, -1
	s_waitcnt vmcnt(0)
	ds_write_b128 v75, v[6:9]
	ds_write_b128 v75, v[2:5] offset:4096
	ds_write_b128 v75, v[14:17] offset:8192
	ds_write_b128 v75, v[10:13] offset:12288
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_load_dword s1, s[0:1], 0x0
	v_and_b32_e32 v74, 32, v0
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s0, s1, s17
	s_add_i32 s4, s19, s0
	s_add_i32 s4, s4, 16
	s_min_i32 s1, s4, s1
	s_add_i32 s1, s1, 63
	s_cmp_lt_i32 s1, 64
	s_cbranch_scc1 .LBB0_14
; %bb.12:                               ; %.lr.ph19
	s_add_i32 s4, s18, -1
	s_ashr_i32 s5, s4, 31
	s_mul_i32 s5, s60, s5
	s_mul_hi_u32 s14, s60, s4
	s_add_i32 s5, s14, s5
	s_mul_i32 s14, s61, s4
	s_add_i32 s5, s5, s14
	s_ashr_i32 s14, s1, 31
	s_lshr_b32 s14, s14, 26
	s_mul_i32 s4, s60, s4
	s_add_i32 s1, s1, s14
	s_ashr_i32 s67, s1, 6
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s86, s12, s4
	s_addc_u32 s87, s13, s5
	s_ashr_i32 s1, s16, 31
	s_mul_i32 s5, s22, s1
	s_mul_hi_u32 s12, s72, s16
	s_mul_i32 s1, s72, s1
	s_mul_hi_u32 s4, s22, s16
	s_add_i32 s1, s12, s1
	s_mul_i32 s12, s73, s16
	s_add_i32 s4, s4, s5
	s_mul_i32 s5, s23, s16
	s_add_i32 s13, s1, s12
	s_mul_i32 s12, s72, s16
	s_add_i32 s5, s4, s5
	s_lshl_b64 s[12:13], s[12:13], 1
	s_mul_i32 s4, s22, s16
	s_add_u32 s71, s6, s12
	s_addc_u32 s72, s7, s13
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s73, s8, s4
	s_addc_u32 s81, s9, s5
	s_ashr_i32 s25, s24, 31
	s_lshl_b64 s[4:5], s[24:25], 1
	s_add_u32 s76, s10, s4
	s_addc_u32 s1, s11, s5
	v_lshlrev_b32_e32 v2, 1, v22
	v_bfrev_b32_e32 v3, 1
	s_and_b32 s77, s1, 0xffff
	s_mov_b32 s79, 0x27000
	s_mov_b32 s78, 0x7ffffffe
	v_cndmask_b32_e32 v2, v3, v2, vcc
	buffer_load_ushort v6, v2, s[76:79], 0 offen
	v_and_b32_e32 v7, 31, v0
	v_lshlrev_b32_e32 v9, 6, v21
	v_and_b32_e32 v10, 0x70, v19
	v_lshrrev_b32_e32 v11, 1, v74
	v_lshlrev_b32_e32 v7, 7, v7
	v_bitop3_b32 v9, v10, v9, v11 bitop3:0xde
	s_movk_i32 s4, 0x60
	v_or_b32_e32 v13, v9, v7
	v_bitop3_b32 v9, v9, s4, v7 bitop3:0x36
	v_add_u32_e32 v28, 0, v13
	v_add3_u32 v76, v72, s0, 1
	v_mad_u64_u32 v[2:3], s[0:1], v1, s70, v[18:19]
	v_mad_u64_u32 v[4:5], s[0:1], v1, s20, v[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	v_xad_u32 v29, v13, 32, 0
	v_xad_u32 v30, v13, 64, 0
	v_add_u32_e32 v31, 0, v9
	ds_read_b128 v[48:51], v28
	ds_read_b128 v[52:55], v29
	ds_read_b128 v[56:59], v30
	ds_read_b128 v[60:63], v31
	v_bfe_i32 v3, v0, 3, 1
	v_mov_b32_e32 v12, 0x108
	s_movk_i32 s5, 0x84
	v_and_b32_e32 v1, 12, v1
	v_cmp_eq_u32_e64 s[0:1], 0, v74
	v_and_b32_e32 v8, 0x70, v0
	v_and_b32_e32 v5, 16, v0
	v_cndmask_b32_e64 v12, v12, 0, s[0:1]
	v_bitop3_b32 v1, v3, v1, s5 bitop3:0x6c
	v_lshrrev_b32_e32 v8, 1, v8
	v_and_or_b32 v5, v20, 64, v5
	v_xor_b32_e32 v1, v1, v12
	v_bitop3_b32 v81, v7, v11, v10 bitop3:0x36
	v_xor_b32_e32 v82, v20, v8
	v_or_b32_e32 v10, v5, v1
	v_bitop3_b32 v1, v5, 16, v1 bitop3:0x36
	v_mov_b32_e32 v78, 0xff800000
	v_mov_b32_e32 v0, 0
	v_xor_b32_e32 v3, 32, v81
	v_xor_b32_e32 v7, 64, v81
	v_xor_b32_e32 v8, 0x60, v81
	v_xor_b32_e32 v9, 8, v82
	v_lshl_add_u32 v87, v1, 1, 0
	s_lshl_b32 s0, s70, 5
	s_lshl_b32 s1, s20, 5
	v_lshrrev_b32_e32 v77, 3, v74
	s_mov_b32 s83, 0xff800000
	v_mov_b32_e32 v93, 1.0
	v_lshlrev_b32_e32 v79, 1, v2
	v_lshlrev_b32_e32 v80, 1, v4
	v_add_u32_e32 v83, 0, v3
	v_add_u32_e32 v84, 0, v7
	v_add_u32_e32 v85, 0, v8
	v_lshl_add_u32 v86, v10, 1, 0
	v_add_lshl_u32 v88, v2, s0, 1
	v_add_lshl_u32 v89, v4, s1, 1
	s_mov_b32 s70, 0xc2fc0000
	v_add_u32_e32 v90, 0, v9
	v_mov_b32_e32 v91, 0x42800000
	v_not_b32_e32 v92, 63
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v1, 16, v6
	v_cndmask_b32_e32 v94, v78, v1, vcc
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
.LBB0_13:                               ; =>This Inner Loop Header: Depth=1
	s_load_dword s1, s[86:87], 0x0
	v_add_u32_e32 v111, 0, v81
	v_cmp_lt_i32_e64 s[64:65], v77, v76
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s1, 31
	s_mul_hi_u32 s6, s74, s1
	s_mul_i32 s7, s75, s1
	s_mul_i32 s0, s74, s1
	s_mul_hi_u32 s8, s68, s1
	s_mul_i32 s9, s69, s1
	s_mul_i32 s4, s68, s1
	s_mul_i32 s1, s74, s5
	s_mul_i32 s5, s68, s5
	s_add_i32 s5, s8, s5
	s_add_i32 s1, s6, s1
	s_add_i32 s5, s5, s9
	s_add_i32 s1, s1, s7
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s76, s71, s4
	s_addc_u32 s4, s72, s5
	s_and_b32 s77, s4, 0xffff
	buffer_load_dwordx4 v[32:35], v79, s[76:79], 0 offen
	buffer_load_dwordx4 v[36:39], v88, s[76:79], 0 offen
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s76, s73, s0
	s_addc_u32 s0, s81, s1
	s_and_b32 s77, s0, 0xffff
	buffer_load_dwordx4 v[64:67], v80, s[76:79], 0 offen
	buffer_load_dwordx4 v[68:71], v89, s[76:79], 0 offen
	s_barrier
	s_and_b64 s[64:65], s[84:85], s[64:65]
	s_add_i32 s67, s67, -1
	s_waitcnt vmcnt(3)
	ds_write_b128 v75, v[32:35]
	s_waitcnt vmcnt(2)
	ds_write_b128 v75, v[36:39] offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[32:35], v111
	ds_read_b128 v[96:99], v83
	ds_read_b128 v[112:115], v83 offset:4096
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[32:35], v[48:51], 0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[96:99], v[52:55], v[32:47]
	ds_read_b128 v[96:99], v84
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[96:99], v[56:59], v[32:47]
	ds_read_b128 v[96:99], v85
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[96:99], v[60:63], v[32:47]
	s_nop 7
	s_nop 3
	v_fma_f32 v95, s33, v32, 0
	v_fma_f32 v96, s33, v33, 0
	v_fma_f32 v97, s33, v34, 0
	v_fma_f32 v98, s33, v35, 0
	ds_read_b128 v[32:35], v111 offset:4096
	v_fma_f32 v99, s33, v36, 0
	v_fma_f32 v100, s33, v37, 0
	v_fma_f32 v101, s33, v38, 0
	v_fma_f32 v102, s33, v39, 0
	v_fma_f32 v103, s33, v40, 0
	v_fma_f32 v104, s33, v41, 0
	v_fma_f32 v105, s33, v42, 0
	v_fma_f32 v106, s33, v43, 0
	v_fma_f32 v107, s33, v44, 0
	v_fma_f32 v108, s33, v45, 0
	v_fma_f32 v109, s33, v46, 0
	v_fma_f32 v110, s33, v47, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[32:35], v[48:51], 0
	v_add_u32_e32 v111, 0, v82
	v_mfma_f32_32x32x16_bf16 v[32:47], v[112:115], v[52:55], v[32:47]
	ds_read_b128 v[112:115], v84 offset:4096
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[112:115], v[56:59], v[32:47]
	ds_read_b128 v[112:115], v85 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write2st64_b64 v111, v[64:65], v[68:69] offset1:8
	ds_write2st64_b64 v90, v[66:67], v[70:71] offset1:8
	v_add_u32_e32 v64, 1, v77
	v_cmp_lt_i32_e32 vcc, v64, v76
	v_add_u32_e32 v64, 2, v77
	v_cmp_lt_i32_e64 s[0:1], v64, v76
	v_add_u32_e32 v64, 3, v77
	v_cmp_lt_i32_e64 s[4:5], v64, v76
	v_add_u32_e32 v64, 8, v77
	v_cmp_lt_i32_e64 s[6:7], v64, v76
	v_add_u32_e32 v64, 9, v77
	v_cmp_lt_i32_e64 s[8:9], v64, v76
	v_add_u32_e32 v64, 10, v77
	v_cmp_lt_i32_e64 s[10:11], v64, v76
	v_add_u32_e32 v64, 11, v77
	v_cmp_lt_i32_e64 s[12:13], v64, v76
	v_add_u32_e32 v64, 16, v77
	v_cmp_lt_i32_e64 s[14:15], v64, v76
	v_add_u32_e32 v64, 17, v77
	v_cmp_lt_i32_e64 s[16:17], v64, v76
	v_add_u32_e32 v64, 18, v77
	v_cmp_lt_i32_e64 s[18:19], v64, v76
	v_add_u32_e32 v64, 19, v77
	v_cmp_lt_i32_e64 s[20:21], v64, v76
	v_add_u32_e32 v64, 24, v77
	v_cmp_lt_i32_e64 s[22:23], v64, v76
	v_add_u32_e32 v64, 25, v77
	v_cmp_lt_i32_e64 s[24:25], v64, v76
	v_add_u32_e32 v64, 26, v77
	v_cmp_lt_i32_e64 s[26:27], v64, v76
	v_add_u32_e32 v64, 27, v77
	v_cmp_lt_i32_e64 s[28:29], v64, v76
	v_add_u32_e32 v64, 32, v77
	v_cmp_lt_i32_e64 s[30:31], v64, v76
	v_add_u32_e32 v64, 33, v77
	v_cmp_lt_i32_e64 s[34:35], v64, v76
	v_add_u32_e32 v64, 34, v77
	v_cmp_lt_i32_e64 s[36:37], v64, v76
	v_add_u32_e32 v64, 35, v77
	v_cmp_lt_i32_e64 s[38:39], v64, v76
	v_add_u32_e32 v64, 40, v77
	v_cmp_lt_i32_e64 s[40:41], v64, v76
	v_add_u32_e32 v64, 41, v77
	v_cmp_lt_i32_e64 s[42:43], v64, v76
	v_add_u32_e32 v64, 42, v77
	v_cmp_lt_i32_e64 s[44:45], v64, v76
	v_add_u32_e32 v64, 43, v77
	v_mfma_f32_32x32x16_bf16 v[32:47], v[112:115], v[60:63], v[32:47]
	v_cmp_lt_i32_e64 s[46:47], v64, v76
	v_add_u32_e32 v64, 48, v77
	v_cmp_lt_i32_e64 s[48:49], v64, v76
	v_add_u32_e32 v64, 49, v77
	v_cmp_lt_i32_e64 s[50:51], v64, v76
	v_add_u32_e32 v64, 50, v77
	v_cmp_lt_i32_e64 s[52:53], v64, v76
	v_add_u32_e32 v64, 51, v77
	v_cmp_lt_i32_e64 s[54:55], v64, v76
	v_add_u32_e32 v64, 56, v77
	v_cmp_lt_i32_e64 s[56:57], v64, v76
	v_add_u32_e32 v64, 57, v77
	v_cmp_lt_i32_e64 s[58:59], v64, v76
	v_add_u32_e32 v64, 58, v77
	v_cmp_lt_i32_e64 s[60:61], v64, v76
	v_add_u32_e32 v64, 59, v77
	s_and_b64 vcc, s[84:85], vcc
	v_fma_f32 v32, s33, v32, 0
	v_cmp_lt_i32_e64 s[62:63], v64, v76
	s_and_b64 s[0:1], s[84:85], s[0:1]
	s_and_b64 s[4:5], s[84:85], s[4:5]
	s_and_b64 s[30:31], s[84:85], s[30:31]
	v_cndmask_b32_e64 v64, v78, v95, s[64:65]
	v_cndmask_b32_e32 v65, v78, v96, vcc
	s_and_b64 s[6:7], s[84:85], s[6:7]
	s_and_b64 s[8:9], s[84:85], s[8:9]
	v_cndmask_b32_e64 v97, v78, v97, s[0:1]
	v_cndmask_b32_e64 v98, v78, v98, s[4:5]
	v_cndmask_b32_e64 v66, v78, v32, s[30:31]
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v32, v64, v65
	s_and_b64 s[10:11], s[84:85], s[10:11]
	s_and_b64 s[12:13], s[84:85], s[12:13]
	v_cndmask_b32_e64 v99, v78, v99, s[6:7]
	v_cndmask_b32_e64 v100, v78, v100, s[8:9]
	v_max3_f32 v32, v32, v97, v98
	s_and_b64 s[14:15], s[84:85], s[14:15]
	s_and_b64 s[16:17], s[84:85], s[16:17]
	v_cndmask_b32_e64 v101, v78, v101, s[10:11]
	v_cndmask_b32_e64 v102, v78, v102, s[12:13]
	v_max3_f32 v32, v32, v99, v100
	s_and_b64 s[18:19], s[84:85], s[18:19]
	s_and_b64 s[20:21], s[84:85], s[20:21]
	v_cndmask_b32_e64 v111, v78, v103, s[14:15]
	v_cndmask_b32_e64 v112, v78, v104, s[16:17]
	v_max3_f32 v32, v32, v101, v102
	s_and_b64 s[22:23], s[84:85], s[22:23]
	s_and_b64 s[24:25], s[84:85], s[24:25]
	v_cndmask_b32_e64 v113, v78, v105, s[18:19]
	v_cndmask_b32_e64 v106, v78, v106, s[20:21]
	v_max3_f32 v32, v32, v111, v112
	s_and_b64 s[26:27], s[84:85], s[26:27]
	s_and_b64 s[28:29], s[84:85], s[28:29]
	v_cndmask_b32_e64 v107, v78, v107, s[22:23]
	v_cndmask_b32_e64 v108, v78, v108, s[24:25]
	v_max3_f32 v32, v32, v113, v106
	v_fma_f32 v33, s33, v33, 0
	s_and_b64 s[34:35], s[84:85], s[34:35]
	v_cndmask_b32_e64 v109, v78, v109, s[26:27]
	v_cndmask_b32_e64 v110, v78, v110, s[28:29]
	v_max3_f32 v32, v32, v107, v108
	v_fma_f32 v34, s33, v34, 0
	v_fma_f32 v35, s33, v35, 0
	s_and_b64 s[36:37], s[84:85], s[36:37]
	s_and_b64 s[38:39], s[84:85], s[38:39]
	v_cndmask_b32_e64 v96, v78, v33, s[34:35]
	v_max3_f32 v32, v32, v109, v110
	v_fma_f32 v36, s33, v36, 0
	v_fma_f32 v37, s33, v37, 0
	s_and_b64 s[40:41], s[84:85], s[40:41]
	s_and_b64 s[42:43], s[84:85], s[42:43]
	v_cndmask_b32_e64 v95, v78, v34, s[36:37]
	v_cndmask_b32_e64 v71, v78, v35, s[38:39]
	v_max3_f32 v32, v32, v66, v96
	v_fma_f32 v38, s33, v38, 0
	v_fma_f32 v39, s33, v39, 0
	s_and_b64 s[44:45], s[84:85], s[44:45]
	s_and_b64 s[46:47], s[84:85], s[46:47]
	v_cndmask_b32_e64 v70, v78, v36, s[40:41]
	v_cndmask_b32_e64 v69, v78, v37, s[42:43]
	v_max3_f32 v32, v32, v95, v71
	v_fma_f32 v40, s33, v40, 0
	v_fma_f32 v41, s33, v41, 0
	s_and_b64 s[48:49], s[84:85], s[48:49]
	s_and_b64 s[50:51], s[84:85], s[50:51]
	v_cndmask_b32_e64 v68, v78, v38, s[44:45]
	v_cndmask_b32_e64 v67, v78, v39, s[46:47]
	v_max3_f32 v32, v32, v70, v69
	v_fma_f32 v42, s33, v42, 0
	v_fma_f32 v43, s33, v43, 0
	s_and_b64 s[52:53], s[84:85], s[52:53]
	s_and_b64 s[54:55], s[84:85], s[54:55]
	v_cndmask_b32_e64 v34, v78, v40, s[48:49]
	v_cndmask_b32_e64 v41, v78, v41, s[50:51]
	v_max3_f32 v32, v32, v68, v67
	v_fma_f32 v44, s33, v44, 0
	v_fma_f32 v45, s33, v45, 0
	s_and_b64 s[56:57], s[84:85], s[56:57]
	s_and_b64 s[58:59], s[84:85], s[58:59]
	v_cndmask_b32_e64 v40, v78, v42, s[52:53]
	v_cndmask_b32_e64 v39, v78, v43, s[54:55]
	v_max3_f32 v32, v32, v34, v41
	v_fma_f32 v46, s33, v46, 0
	v_fma_f32 v47, s33, v47, 0
	s_and_b64 s[60:61], s[84:85], s[60:61]
	s_and_b64 s[62:63], s[84:85], s[62:63]
	v_cndmask_b32_e64 v38, v78, v44, s[56:57]
	v_cndmask_b32_e64 v37, v78, v45, s[58:59]
	v_max3_f32 v32, v32, v40, v39
	v_cndmask_b32_e64 v36, v78, v46, s[60:61]
	v_cndmask_b32_e64 v35, v78, v47, s[62:63]
	v_max3_f32 v32, v32, v38, v37
	v_max3_f32 v32, v32, v36, v35
	v_mov_b32_e32 v33, v32
	s_nop 1
	v_permlane32_swap_b32_e32 v32, v33
	v_max3_f32 v32, v94, v32, v33
	v_cmp_lg_f32_e32 vcc, s83, v32
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cndmask_b32_e32 v33, 0, v32, vcc
	v_sub_f32_e32 v32, v65, v33
	v_mul_f32_e32 v42, 0x3fb8aa3b, v32
	v_cmp_gt_f32_e64 s[0:1], s70, v42
	v_sub_f32_e32 v42, v97, v33
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_cmp_gt_f32_e64 s[4:5], s70, v43
	v_sub_f32_e32 v43, v98, v33
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_cmp_gt_f32_e64 s[6:7], s70, v44
	v_sub_f32_e32 v44, v99, v33
	v_mul_f32_e32 v45, 0x3fb8aa3b, v44
	v_cmp_gt_f32_e64 s[8:9], s70, v45
	v_sub_f32_e32 v45, v100, v33
	v_mul_f32_e32 v46, 0x3fb8aa3b, v45
	v_cmp_gt_f32_e64 s[10:11], s70, v46
	v_sub_f32_e32 v46, v101, v33
	v_mul_f32_e32 v47, 0x3fb8aa3b, v46
	v_cmp_gt_f32_e64 s[12:13], s70, v47
	v_sub_f32_e32 v47, v102, v33
	v_mul_f32_e32 v65, 0x3fb8aa3b, v47
	v_cmp_gt_f32_e64 s[14:15], s70, v65
	v_sub_f32_e32 v65, v94, v33
	v_mul_f32_e32 v94, 0x3fb8aa3b, v65
	v_sub_f32_e32 v64, v64, v33
	v_cmp_gt_f32_e32 vcc, s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v64
	v_cmp_gt_f32_e64 s[16:17], s70, v94
	v_cndmask_b32_e64 v97, 0, v91, s[6:7]
	v_fmac_f32_e32 v97, 0x3fb8aa3b, v43
	v_cndmask_b32_e64 v94, 0, v91, s[16:17]
	v_fmac_f32_e32 v94, 0x3fb8aa3b, v64
	v_cndmask_b32_e64 v64, 0, v91, s[0:1]
	v_fmac_f32_e32 v64, 0x3fb8aa3b, v32
	v_cndmask_b32_e64 v32, 0, v91, s[4:5]
	v_fmac_f32_e32 v32, 0x3fb8aa3b, v42
	v_exp_f32_e32 v42, v94
	v_cndmask_b32_e64 v43, 0, v92, s[16:17]
	v_exp_f32_e32 v32, v32
	v_cndmask_b32_e64 v98, 0, v91, s[8:9]
	v_ldexp_f32 v42, v42, v43
	v_exp_f32_e32 v43, v64
	v_fmac_f32_e32 v98, 0x3fb8aa3b, v44
	v_cndmask_b32_e64 v44, 0, v92, s[0:1]
	v_cndmask_b32_e64 v99, 0, v91, s[10:11]
	v_ldexp_f32 v43, v43, v44
	v_cndmask_b32_e64 v44, 0, v92, s[4:5]
	v_ldexp_f32 v44, v32, v44
	v_exp_f32_e32 v32, v97
	v_fmac_f32_e32 v99, 0x3fb8aa3b, v45
	v_cndmask_b32_e64 v45, 0, v92, s[6:7]
	v_cndmask_b32_e64 v100, 0, v91, s[12:13]
	v_ldexp_f32 v45, v32, v45
	v_exp_f32_e32 v32, v98
	v_fmac_f32_e32 v100, 0x3fb8aa3b, v46
	v_cndmask_b32_e64 v46, 0, v92, s[8:9]
	v_cndmask_b32_e64 v101, 0, v91, s[14:15]
	v_ldexp_f32 v46, v32, v46
	v_exp_f32_e32 v32, v99
	v_fmac_f32_e32 v101, 0x3fb8aa3b, v47
	v_cndmask_b32_e64 v47, 0, v92, s[10:11]
	v_cndmask_b32_e64 v64, 0, v92, s[12:13]
	v_ldexp_f32 v47, v32, v47
	v_exp_f32_e32 v32, v100
	v_cndmask_b32_e32 v102, 0, v91, vcc
	v_fmac_f32_e32 v102, 0x3fb8aa3b, v65
	v_cndmask_b32_e64 v65, 0, v92, s[14:15]
	v_ldexp_f32 v64, v32, v64
	v_exp_f32_e32 v32, v101
	ds_read_b64_tr_b16 v[98:99], v86
	ds_read_b64_tr_b16 v[100:101], v87 offset:1024
	v_cndmask_b32_e32 v94, 0, v92, vcc
	v_ldexp_f32 v65, v32, v65
	v_exp_f32_e32 v32, v102
	v_cvt_pk_bf16_f32 v102, v42, v43
	v_cvt_pk_bf16_f32 v103, v44, v45
	v_cvt_pk_bf16_f32 v104, v46, v47
	v_ldexp_f32 v32, v32, v94
	v_cvt_pk_bf16_f32 v105, v64, v65
	v_pk_mul_f32 v[30:31], v[30:31], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[32:33] op_sel_hi:[1,0]
	v_sub_f32_e32 v94, v112, v33
	v_mul_f32_e32 v97, 0x3fb8aa3b, v94
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[98:101], v[102:105], v[16:31]
	ds_read_b64_tr_b16 v[98:99], v86 offset:64
	ds_read_b64_tr_b16 v[100:101], v87 offset:1088
	v_pk_mul_f32 v[14:15], v[14:15], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[32:33] op_sel_hi:[1,0]
	v_cmp_gt_f32_e32 vcc, s70, v97
	v_sub_f32_e32 v97, v113, v33
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[98:101], v[102:105], v[0:15]
	v_mul_f32_e32 v98, 0x3fb8aa3b, v97
	v_cmp_gt_f32_e64 s[0:1], s70, v98
	v_sub_f32_e32 v98, v106, v33
	v_mul_f32_e32 v99, 0x3fb8aa3b, v98
	v_cmp_gt_f32_e64 s[4:5], s70, v99
	v_sub_f32_e32 v99, v107, v33
	v_mul_f32_e32 v100, 0x3fb8aa3b, v99
	v_cmp_gt_f32_e64 s[6:7], s70, v100
	v_sub_f32_e32 v100, v108, v33
	v_mul_f32_e32 v101, 0x3fb8aa3b, v100
	v_cmp_gt_f32_e64 s[8:9], s70, v101
	v_sub_f32_e32 v101, v109, v33
	v_mul_f32_e32 v102, 0x3fb8aa3b, v101
	v_cmp_gt_f32_e64 s[10:11], s70, v102
	v_sub_f32_e32 v102, v110, v33
	v_mul_f32_e32 v103, 0x3fb8aa3b, v102
	v_cmp_gt_f32_e64 s[12:13], s70, v103
	v_sub_f32_e32 v103, v111, v33
	v_mul_f32_e32 v104, 0x3fb8aa3b, v103
	v_cmp_gt_f32_e64 s[14:15], s70, v104
	v_sub_f32_e32 v95, v95, v33
	v_sub_f32_e32 v71, v71, v33
	v_cndmask_b32_e64 v104, 0, v91, s[14:15]
	v_fmac_f32_e32 v104, 0x3fb8aa3b, v103
	v_cndmask_b32_e32 v103, 0, v91, vcc
	v_fmac_f32_e32 v103, 0x3fb8aa3b, v94
	v_cndmask_b32_e64 v94, 0, v91, s[0:1]
	v_fmac_f32_e32 v94, 0x3fb8aa3b, v97
	v_cndmask_b32_e64 v97, 0, v91, s[4:5]
	v_fmac_f32_e32 v97, 0x3fb8aa3b, v98
	v_cndmask_b32_e64 v98, 0, v91, s[6:7]
	v_fmac_f32_e32 v98, 0x3fb8aa3b, v99
	v_cndmask_b32_e64 v99, 0, v91, s[8:9]
	v_fmac_f32_e32 v99, 0x3fb8aa3b, v100
	v_cndmask_b32_e64 v100, 0, v91, s[10:11]
	v_fmac_f32_e32 v100, 0x3fb8aa3b, v101
	v_cndmask_b32_e64 v101, 0, v91, s[12:13]
	v_fmac_f32_e32 v101, 0x3fb8aa3b, v102
	v_exp_f32_e32 v102, v104
	v_cndmask_b32_e64 v104, 0, v92, s[14:15]
	v_exp_f32_e32 v94, v94
	v_sub_f32_e32 v70, v70, v33
	v_ldexp_f32 v106, v102, v104
	v_exp_f32_e32 v102, v103
	v_cndmask_b32_e32 v103, 0, v92, vcc
	v_sub_f32_e32 v69, v69, v33
	v_sub_f32_e32 v68, v68, v33
	v_ldexp_f32 v107, v102, v103
	v_cndmask_b32_e64 v102, 0, v92, s[0:1]
	v_ldexp_f32 v108, v94, v102
	v_exp_f32_e32 v94, v97
	v_cndmask_b32_e64 v97, 0, v92, s[4:5]
	v_cvt_pk_bf16_f32 v102, v106, v107
	v_sub_f32_e32 v67, v67, v33
	v_ldexp_f32 v109, v94, v97
	v_exp_f32_e32 v94, v98
	v_cndmask_b32_e64 v97, 0, v92, s[6:7]
	v_cvt_pk_bf16_f32 v103, v108, v109
	v_sub_f32_e32 v66, v66, v33
	v_ldexp_f32 v110, v94, v97
	v_exp_f32_e32 v94, v99
	v_cndmask_b32_e64 v97, 0, v92, s[8:9]
	v_sub_f32_e32 v41, v41, v33
	v_sub_f32_e32 v40, v40, v33
	v_ldexp_f32 v111, v94, v97
	v_exp_f32_e32 v94, v100
	v_cndmask_b32_e64 v97, 0, v92, s[10:11]
	v_cvt_pk_bf16_f32 v104, v110, v111
	v_sub_f32_e32 v39, v39, v33
	v_ldexp_f32 v112, v94, v97
	v_exp_f32_e32 v94, v101
	v_cndmask_b32_e64 v97, 0, v92, s[12:13]
	ds_read_b64_tr_b16 v[98:99], v86 offset:2048
	ds_read_b64_tr_b16 v[100:101], v87 offset:3072
	v_sub_f32_e32 v38, v38, v33
	v_ldexp_f32 v113, v94, v97
	v_sub_f32_e32 v94, v96, v33
	v_mul_f32_e32 v96, 0x3fb8aa3b, v94
	v_cmp_gt_f32_e32 vcc, s70, v96
	v_mul_f32_e32 v96, 0x3fb8aa3b, v95
	v_cmp_gt_f32_e64 s[0:1], s70, v96
	v_mul_f32_e32 v96, 0x3fb8aa3b, v71
	v_cvt_pk_bf16_f32 v105, v112, v113
	v_cmp_gt_f32_e64 s[4:5], s70, v96
	v_mul_f32_e32 v96, 0x3fb8aa3b, v70
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[98:101], v[102:105], v[16:31]
	ds_read_b64_tr_b16 v[100:101], v87 offset:3136
	ds_read_b64_tr_b16 v[98:99], v86 offset:2112
	v_cmp_gt_f32_e64 s[6:7], s70, v96
	v_mul_f32_e32 v96, 0x3fb8aa3b, v69
	v_cmp_gt_f32_e64 s[8:9], s70, v96
	v_mul_f32_e32 v96, 0x3fb8aa3b, v68
	v_cmp_gt_f32_e64 s[10:11], s70, v96
	v_mul_f32_e32 v96, 0x3fb8aa3b, v67
	v_cmp_gt_f32_e64 s[12:13], s70, v96
	v_mul_f32_e32 v96, 0x3fb8aa3b, v66
	v_cmp_gt_f32_e64 s[14:15], s70, v96
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[98:101], v[102:105], v[0:15]
	v_cndmask_b32_e64 v98, 0, v91, s[10:11]
	v_cndmask_b32_e64 v96, 0, v91, s[14:15]
	v_fmac_f32_e32 v96, 0x3fb8aa3b, v66
	v_cndmask_b32_e32 v66, 0, v91, vcc
	v_fmac_f32_e32 v98, 0x3fb8aa3b, v68
	v_cndmask_b32_e64 v68, 0, v91, s[12:13]
	v_fmac_f32_e32 v66, 0x3fb8aa3b, v94
	v_fmac_f32_e32 v68, 0x3fb8aa3b, v67
	v_exp_f32_e32 v67, v96
	v_exp_f32_e32 v66, v66
	v_cndmask_b32_e64 v97, 0, v91, s[8:9]
	v_cndmask_b32_e64 v94, 0, v91, s[0:1]
	v_fmac_f32_e32 v97, 0x3fb8aa3b, v69
	v_cndmask_b32_e64 v69, 0, v92, s[14:15]
	v_fmac_f32_e32 v94, 0x3fb8aa3b, v95
	v_ldexp_f32 v102, v67, v69
	v_cndmask_b32_e32 v67, 0, v92, vcc
	v_ldexp_f32 v103, v66, v67
	v_exp_f32_e32 v66, v94
	v_cndmask_b32_e64 v95, 0, v91, s[4:5]
	v_fmac_f32_e32 v95, 0x3fb8aa3b, v71
	v_cndmask_b32_e64 v67, 0, v92, s[0:1]
	v_ldexp_f32 v67, v66, v67
	v_exp_f32_e32 v66, v95
	v_cndmask_b32_e64 v71, 0, v91, s[6:7]
	v_fmac_f32_e32 v71, 0x3fb8aa3b, v70
	v_cndmask_b32_e64 v69, 0, v92, s[4:5]
	v_ldexp_f32 v69, v66, v69
	v_exp_f32_e32 v66, v71
	v_cndmask_b32_e64 v70, 0, v92, s[6:7]
	v_cndmask_b32_e64 v71, 0, v92, s[8:9]
	v_exp_f32_e32 v68, v68
	v_ldexp_f32 v70, v66, v70
	v_exp_f32_e32 v66, v97
	v_cndmask_b32_e64 v94, 0, v92, s[10:11]
	v_cvt_pk_bf16_f32 v99, v67, v69
	v_sub_f32_e32 v37, v37, v33
	v_ldexp_f32 v71, v66, v71
	v_exp_f32_e32 v66, v98
	v_cvt_pk_bf16_f32 v98, v102, v103
	v_cvt_pk_bf16_f32 v100, v70, v71
	v_sub_f32_e32 v36, v36, v33
	v_ldexp_f32 v66, v66, v94
	v_cndmask_b32_e64 v94, 0, v92, s[12:13]
	v_ldexp_f32 v68, v68, v94
	ds_read_b64_tr_b16 v[94:95], v86 offset:4096
	ds_read_b64_tr_b16 v[96:97], v87 offset:5120
	v_cvt_pk_bf16_f32 v101, v66, v68
	v_sub_f32_e32 v35, v35, v33
	v_sub_f32_e32 v34, v34, v33
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[94:97], v[98:101], v[16:31]
	ds_read_b64_tr_b16 v[94:95], v86 offset:4160
	ds_read_b64_tr_b16 v[96:97], v87 offset:5184
	s_add_u32 s86, s86, 4
	s_addc_u32 s87, s87, 0
	v_add_u32_e32 v77, 64, v77
	s_cmp_lg_u32 s67, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[94:97], v[98:101], v[0:15]
	v_mul_f32_e32 v94, 0x3fb8aa3b, v41
	v_cmp_gt_f32_e32 vcc, s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v40
	v_cmp_gt_f32_e64 s[0:1], s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v39
	v_cmp_gt_f32_e64 s[4:5], s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v38
	v_cmp_gt_f32_e64 s[6:7], s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v37
	v_cmp_gt_f32_e64 s[8:9], s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v36
	v_cmp_gt_f32_e64 s[10:11], s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v35
	v_cmp_gt_f32_e64 s[12:13], s70, v94
	v_mul_f32_e32 v94, 0x3fb8aa3b, v34
	v_cmp_gt_f32_e64 s[14:15], s70, v94
	v_cndmask_b32_e64 v97, 0, v91, s[12:13]
	v_fmac_f32_e32 v97, 0x3fb8aa3b, v35
	v_cndmask_b32_e64 v94, 0, v91, s[14:15]
	v_fmac_f32_e32 v94, 0x3fb8aa3b, v34
	v_cndmask_b32_e32 v34, 0, v91, vcc
	v_fmac_f32_e32 v34, 0x3fb8aa3b, v41
	v_exp_f32_e32 v35, v94
	v_exp_f32_e32 v34, v34
	v_cndmask_b32_e64 v96, 0, v91, s[10:11]
	v_cndmask_b32_e64 v41, 0, v91, s[0:1]
	v_fmac_f32_e32 v96, 0x3fb8aa3b, v36
	v_cndmask_b32_e64 v36, 0, v92, s[14:15]
	v_fmac_f32_e32 v41, 0x3fb8aa3b, v40
	v_ldexp_f32 v40, v35, v36
	v_cndmask_b32_e32 v35, 0, v92, vcc
	v_ldexp_f32 v94, v34, v35
	v_exp_f32_e32 v34, v41
	v_cndmask_b32_e64 v95, 0, v91, s[4:5]
	v_fmac_f32_e32 v95, 0x3fb8aa3b, v39
	v_cndmask_b32_e64 v35, 0, v92, s[0:1]
	v_ldexp_f32 v34, v34, v35
	v_exp_f32_e32 v35, v95
	v_cndmask_b32_e64 v39, 0, v91, s[6:7]
	v_fmac_f32_e32 v39, 0x3fb8aa3b, v38
	v_cndmask_b32_e64 v36, 0, v92, s[4:5]
	v_ldexp_f32 v35, v35, v36
	v_exp_f32_e32 v36, v39
	v_cndmask_b32_e64 v38, 0, v91, s[8:9]
	v_fmac_f32_e32 v38, 0x3fb8aa3b, v37
	v_cndmask_b32_e64 v37, 0, v92, s[6:7]
	v_ldexp_f32 v36, v36, v37
	v_exp_f32_e32 v37, v38
	v_cndmask_b32_e64 v38, 0, v92, s[8:9]
	v_cndmask_b32_e64 v39, 0, v92, s[10:11]
	v_cndmask_b32_e64 v41, 0, v92, s[12:13]
	v_ldexp_f32 v37, v37, v38
	v_exp_f32_e32 v38, v96
	s_nop 0
	v_ldexp_f32 v38, v38, v39
	v_exp_f32_e32 v39, v97
	s_nop 0
	v_ldexp_f32 v39, v39, v41
	v_add_f32_e32 v41, v42, v43
	v_add_f32_e32 v41, v44, v41
	v_add_f32_e32 v41, v45, v41
	v_add_f32_e32 v41, v46, v41
	v_add_f32_e32 v41, v47, v41
	v_add_f32_e32 v41, v64, v41
	v_add_f32_e32 v41, v65, v41
	v_add_f32_e32 v41, v106, v41
	v_add_f32_e32 v41, v107, v41
	v_add_f32_e32 v41, v108, v41
	v_add_f32_e32 v41, v109, v41
	v_add_f32_e32 v41, v110, v41
	v_add_f32_e32 v41, v111, v41
	v_add_f32_e32 v41, v112, v41
	v_add_f32_e32 v41, v113, v41
	v_add_f32_e32 v41, v102, v41
	v_add_f32_e32 v41, v103, v41
	v_add_f32_e32 v41, v67, v41
	v_add_f32_e32 v41, v69, v41
	v_add_f32_e32 v41, v70, v41
	ds_read_b64_tr_b16 v[42:43], v86 offset:6144
	ds_read_b64_tr_b16 v[44:45], v87 offset:7168
	v_add_f32_e32 v41, v71, v41
	v_add_f32_e32 v41, v66, v41
	v_add_f32_e32 v41, v68, v41
	v_add_f32_e32 v41, v40, v41
	v_cvt_pk_bf16_f32 v64, v40, v94
	v_cvt_pk_bf16_f32 v65, v34, v35
	v_cvt_pk_bf16_f32 v66, v36, v37
	v_cvt_pk_bf16_f32 v67, v38, v39
	v_add_f32_e32 v47, v94, v41
	v_mov_b32_e32 v94, v33
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[42:45], v[64:67], v[16:31]
	ds_read_b64_tr_b16 v[42:43], v87 offset:7232
	ds_read_b64_tr_b16 v[40:41], v86 offset:6208
	v_add_f32_e32 v33, v34, v47
	v_add_f32_e32 v33, v35, v33
	v_add_f32_e32 v33, v36, v33
	v_add_f32_e32 v33, v37, v33
	v_add_f32_e32 v33, v38, v33
	v_add_f32_e32 v33, v39, v33
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[40:43], v[64:67], v[0:15]
	v_mov_b32_e32 v34, v33
	s_nop 1
	v_permlane32_swap_b32_e32 v33, v34
	v_mov_b32_e32 v46, v93
	v_add_f32_e32 v93, v33, v34
	v_fmac_f32_e32 v93, v46, v32
	s_cbranch_scc1 .LBB0_13
	s_branch .LBB0_15
.LBB0_14:
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v93, 1.0
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v21, v17
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v23, v17
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v25, v17
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v27, v17
	v_mov_b32_e32 v26, v17
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v28, v17
	v_mov_b32_e32 v31, v17
	v_mov_b32_e32 v30, v17
	v_mov_b32_e32 v1, v17
	v_mov_b32_e32 v0, v17
	v_mov_b32_e32 v3, v17
	v_mov_b32_e32 v2, v17
	v_mov_b32_e32 v5, v17
	v_mov_b32_e32 v4, v17
	v_mov_b32_e32 v7, v17
	v_mov_b32_e32 v6, v17
	v_mov_b32_e32 v9, v17
	v_mov_b32_e32 v8, v17
	v_mov_b32_e32 v11, v17
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v13, v17
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v15, v17
	v_mov_b32_e32 v14, v17
.LBB0_15:                               ; %._crit_edge20
	v_div_scale_f32 v32, s[0:1], v93, v93, 1.0
	v_rcp_f32_e32 v33, v32
	v_div_scale_f32 v34, vcc, 1.0, v93, 1.0
	v_fma_f32 v35, -v32, v33, 1.0
	v_fmac_f32_e32 v33, v35, v33
	v_mul_f32_e32 v35, v34, v33
	v_fma_f32 v36, -v32, v35, v34
	v_fmac_f32_e32 v35, v36, v33
	v_fma_f32 v32, -v32, v35, v34
	v_div_fmas_f32 v32, v32, v33, v35
	v_div_fixup_f32 v32, v32, v93, 1.0
	v_pk_mul_f32 v[16:17], v[32:33], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[32:33], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[32:33], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[32:33], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[32:33], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[32:33], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[32:33], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[32:33], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[32:33], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[32:33], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[32:33], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[32:33], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[32:33], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[32:33], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[32:33], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[32:33], v[14:15] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v12, v16, v17
	v_cvt_pk_bf16_f32 v13, v18, v19
	v_cvt_pk_bf16_f32 v14, v20, v21
	v_cvt_pk_bf16_f32 v15, v22, v23
	v_cvt_pk_bf16_f32 v8, v24, v25
	v_cvt_pk_bf16_f32 v9, v26, v27
	v_cvt_pk_bf16_f32 v10, v28, v29
	v_cvt_pk_bf16_f32 v11, v30, v31
	v_cvt_pk_bf16_f32 v4, v0, v1
	v_cvt_pk_bf16_f32 v5, v2, v3
	v_cvt_pk_bf16_f32 v6, v34, v35
	v_cvt_pk_bf16_f32 v7, v36, v37
	v_cvt_pk_bf16_f32 v0, v38, v39
	v_cvt_pk_bf16_f32 v1, v40, v41
	v_cvt_pk_bf16_f32 v2, v42, v43
	v_cvt_pk_bf16_f32 v3, v32, v33
	v_permlane32_swap_b32_e32 v12, v14
	v_permlane32_swap_b32_e32 v13, v15
	v_permlane32_swap_b32_e32 v8, v10
	v_permlane32_swap_b32_e32 v9, v11
	v_permlane32_swap_b32_e32 v4, v6
	v_permlane32_swap_b32_e32 v5, v7
	v_permlane32_swap_b32_e32 v0, v2
	v_permlane32_swap_b32_e32 v1, v3
	s_and_saveexec_b64 s[0:1], s[84:85]
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_17
; %bb.16:                               ; %.critedge
	v_add_u32_e32 v17, s82, v72
	v_lshrrev_b32_e32 v16, 2, v74
	v_mul_lo_u32 v17, s66, v17
	v_mul_lo_u32 v18, s80, v73
	v_add3_u32 v16, v17, v18, v16
	v_add_u32_e32 v18, 48, v16
	v_add_u32_e32 v20, 32, v16
	v_add_u32_e32 v22, 16, v16
	v_ashrrev_i32_e32 v17, 31, v16
	v_ashrrev_i32_e32 v19, 31, v18
	v_ashrrev_i32_e32 v21, 31, v20
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[2:3]
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[2:3]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[2:3]
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[2:3]
	global_store_dwordx4 v[16:17], v[12:15], off
	global_store_dwordx4 v[22:23], v[8:11], off
	global_store_dwordx4 v[20:21], v[4:7], off
	global_store_dwordx4 v[18:19], v[0:3], off
.LBB0_17:                               ; %.critedge10
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernel_unified_attention_2d
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 216
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 116
		.amdhsa_next_free_sgpr 88
		.amdhsa_accum_offset 116
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	kernel_unified_attention_2d, .Lfunc_end0-kernel_unified_attention_2d
	.cfi_endproc
                                        ; -- End function
	.set kernel_unified_attention_2d.num_vgpr, 116
	.set kernel_unified_attention_2d.num_agpr, 0
	.set kernel_unified_attention_2d.numbered_sgpr, 88
	.set kernel_unified_attention_2d.private_seg_size, 0
	.set kernel_unified_attention_2d.uses_vcc, 1
	.set kernel_unified_attention_2d.uses_flat_scratch, 0
	.set kernel_unified_attention_2d.has_dyn_sized_stack, 0
	.set kernel_unified_attention_2d.has_recursion, 0
	.set kernel_unified_attention_2d.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6396
; TotalNumSgprs: 94
; NumVgprs: 116
; NumAgprs: 0
; TotalNumVgprs: 116
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 11
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 94
; NumVGPRsForWavesPerEU: 116
; AccumOffset: 116
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 28
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x6a DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x44 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	116                             ; DW_AT_call_line
	.byte	68                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	208                             ; DW_AT_call_line
	.byte	45                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x59:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	310                             ; DW_AT_call_line
	.byte	35                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x66:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	319                             ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"unified_attention_aiter.py"    ; string offset=7
.Linfo_string2:
	.asciz	"/app/OAI-triton/unified_attn_ubench" ; string offset=34
.Linfo_string3:
	.asciz	"kernel_unified_attention_2d"   ; string offset=70
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           8
        .value_kind:     by_value
      - .offset:         96
        .size:           8
        .value_kind:     by_value
      - .offset:         104
        .size:           8
        .value_kind:     by_value
      - .offset:         112
        .size:           8
        .value_kind:     by_value
      - .offset:         120
        .size:           8
        .value_kind:     by_value
      - .offset:         128
        .size:           8
        .value_kind:     by_value
      - .offset:         136
        .size:           8
        .value_kind:     by_value
      - .offset:         144
        .size:           8
        .value_kind:     by_value
      - .offset:         152
        .size:           8
        .value_kind:     by_value
      - .offset:         160
        .size:           8
        .value_kind:     by_value
      - .offset:         168
        .size:           8
        .value_kind:     by_value
      - .offset:         176
        .size:           8
        .value_kind:     by_value
      - .address_space:  global
        .offset:         184
        .size:           8
        .value_kind:     global_buffer
      - .offset:         192
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         200
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         208
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 216
    .max_flat_workgroup_size: 256
    .name:           kernel_unified_attention_2d
    .private_segment_fixed_size: 0
    .sgpr_count:     94
    .sgpr_spill_count: 0
    .symbol:         kernel_unified_attention_2d.kd
    .uses_dynamic_stack: false
    .vgpr_count:     116
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
