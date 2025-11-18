	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.globl	compute_sum                     # -- Begin function compute_sum
	.p2align	1
	.type	compute_sum,@function
compute_sum:                            # @compute_sum
	.cfi_startproc
# %bb.0:
	li	a7, 0
	li	a2, 0
	fmv.d.x	fa0, zero
	li	a6, 146
	li	a4, 49
	j	.LBB0_2
.LBB0_1:                                #   in Loop: Header=BB0_2 Depth=1
	addi	a2, a2, 1
	addi	a7, a7, 400
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	blt	a6, a2, .LBB0_5
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=1
	li	a5, 0
	mv	a3, a7
	bltz	a4, .LBB0_1
.LBB0_4:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a0, a1, a3
	fld	fa5, 0(a0)
	fadd.d	fa0, fa0, fa5
	addi	a5, a5, 1
	addi	a3, a3, 8
	bge	a4, a5, .LBB0_4
	j	.LBB0_1
.LBB0_5:
	ret
.Lfunc_end0:
	.size	compute_sum, .Lfunc_end0-compute_sum
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function sparse_dense_matmul
.LCPI1_0:
	.byte	16                              # 0x10
	.byte	17                              # 0x11
	.byte	18                              # 0x12
	.byte	19                              # 0x13
	.byte	20                              # 0x14
	.byte	21                              # 0x15
	.byte	22                              # 0x16
	.byte	23                              # 0x17
	.byte	24                              # 0x18
	.byte	25                              # 0x19
	.byte	26                              # 0x1a
	.byte	27                              # 0x1b
	.byte	28                              # 0x1c
	.byte	29                              # 0x1d
	.byte	30                              # 0x1e
	.byte	31                              # 0x1f
	.text
	.globl	sparse_dense_matmul
	.p2align	1
	.type	sparse_dense_matmul,@function
sparse_dense_matmul:                    # @sparse_dense_matmul
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -112
	.cfi_def_cfa_offset 112
	sd	s0, 104(sp)                     # 8-byte Folded Spill
	sd	s1, 96(sp)                      # 8-byte Folded Spill
	sd	s2, 88(sp)                      # 8-byte Folded Spill
	sd	s3, 80(sp)                      # 8-byte Folded Spill
	sd	s4, 72(sp)                      # 8-byte Folded Spill
	sd	s5, 64(sp)                      # 8-byte Folded Spill
	sd	s6, 56(sp)                      # 8-byte Folded Spill
	sd	s7, 48(sp)                      # 8-byte Folded Spill
	sd	s8, 40(sp)                      # 8-byte Folded Spill
	sd	s9, 32(sp)                      # 8-byte Folded Spill
	sd	s10, 24(sp)                     # 8-byte Folded Spill
	sd	s11, 16(sp)                     # 8-byte Folded Spill
	.cfi_offset s0, -8
	.cfi_offset s1, -16
	.cfi_offset s2, -24
	.cfi_offset s3, -32
	.cfi_offset s4, -40
	.cfi_offset s5, -48
	.cfi_offset s6, -56
	.cfi_offset s7, -64
	.cfi_offset s8, -72
	.cfi_offset s9, -80
	.cfi_offset s10, -88
	.cfi_offset s11, -96
	csrr	a1, vlenb
	slli	a1, a1, 4
	sub	sp, sp, a1
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xf0, 0x00, 0x22, 0x11, 0x10, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 112 + 16 * vlenb
	li	s4, 0
	li	t5, 0
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	a6, 320(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	t0, 312(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	t1, 304(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	t2, 296(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	t3, 288(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	s8, 280(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	t4, 272(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	s9, 224(a1)
	csrr	a1, vlenb
	slli	a1, a1, 4
	add	a1, a1, sp
	ld	s3, 144(a1)
	lui	a1, %hi(.LCPI1_0)
	addi	a1, a1, %lo(.LCPI1_0)
	vsetivli	zero, 16, e32, m4, ta, ma
	vle8.v	v8, (a1)
	li	t6, 146
	li	s2, 400
	li	s7, 49
	vsext.vf4	v4, v8
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.v.i	v8, 0
	addi	a1, sp, 16
	vs8r.v	v8, (a1)                        # Unknown-size Folded Spill
	j	.LBB1_2
.LBB1_1:                                #   in Loop: Header=BB1_2 Depth=1
	addi	t5, t5, 1
	addi	s4, s4, 400
.LBB1_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_5 Depth 2
                                        #       Child Loop BB1_8 Depth 3
	blt	t6, t5, .LBB1_11
# %bb.3:                                #   in Loop: Header=BB1_2 Depth=1
	slli	a1, t5, 3
	add	a1, a1, a2
	lwu	a3, 4(a1)
	lwu	a5, 0(a1)
	lwu	s0, 12(a1)
	lwu	a1, 8(a1)
	slli	a3, a3, 32
	or	s6, a3, a5
	slli	s0, s0, 32
	or	s5, s0, a1
	j	.LBB1_5
.LBB1_4:                                #   in Loop: Header=BB1_5 Depth=2
	addi	s6, s6, 1
.LBB1_5:                                #   Parent Loop BB1_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_8 Depth 3
	bge	s6, s5, .LBB1_1
# %bb.6:                                #   in Loop: Header=BB1_5 Depth=2
	slli	a1, s6, 3
	add	a3, a7, a1
	lwu	s0, 4(a3)
	li	s10, 0
	lwu	a3, 0(a3)
	slli	s0, s0, 32
	add	a1, a1, s3
	fld	fa5, 0(a1)
	or	a3, a3, s0
	mul	s11, a3, s2
	li	a1, 50
	mv	s1, s4
	j	.LBB1_8
.LBB1_7:                                #   in Loop: Header=BB1_8 Depth=3
	vsetvli	zero, zero, e32, m4, ta, ma
	vmslt.vx	v3, v4, s0
	vid.v	v8
	vmslt.vx	v2, v8, s0
	add	a4, s8, s1
	addi	a3, sp, 16
	vl8r.v	v8, (a3)                        # Unknown-size Folded Reload
	vmv8r.v	v16, v8
	vmv1r.v	v0, v2
	vsetvli	zero, zero, e64, m8, ta, mu
	vle64.v	v16, (a4), v0.t
	csrr	a3, vlenb
	slli	a3, a3, 3
	add	a3, a3, sp
	addi	a3, a3, 16
	vs8r.v	v16, (a3)                       # Unknown-size Folded Spill
	addi	s0, a4, 128
	vmv8r.v	v24, v8
	vmv1r.v	v0, v3
	vle64.v	v24, (s0), v0.t
	add	a3, s9, s11
	addi	a5, a3, 128
	vmv8r.v	v16, v8
	vle64.v	v16, (a5), v0.t
	vmv1r.v	v0, v2
	vle64.v	v8, (a3), v0.t
	vfmul.vf	v16, v16, fa5
	vfadd.vv	v16, v24, v16
	vfmul.vf	v8, v8, fa5
	csrr	a3, vlenb
	slli	a3, a3, 3
	add	a3, a3, sp
	addi	a3, a3, 16
	vl8r.v	v24, (a3)                       # Unknown-size Folded Reload
	vfadd.vv	v8, v24, v8
	vse64.v	v8, (a4), v0.t
	vmv1r.v	v0, v3
	vse64.v	v16, (s0), v0.t
	addi	s10, s10, 32
	addi	s1, s1, 256
	addi	s11, s11, 256
	addi	a1, a1, -32
.LBB1_8:                                #   Parent Loop BB1_2 Depth=1
                                        #     Parent Loop BB1_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	blt	s7, s10, .LBB1_4
# %bb.9:                                #   in Loop: Header=BB1_8 Depth=3
	li	a4, 32
	mv	s0, a1
	blt	a1, a4, .LBB1_7
# %bb.10:                               #   in Loop: Header=BB1_8 Depth=3
	li	s0, 32
	j	.LBB1_7
.LBB1_11:
	sd	t4, 0(a0)
	sd	s8, 8(a0)
	sd	t3, 16(a0)
	sd	t2, 24(a0)
	sd	t1, 32(a0)
	sd	t0, 40(a0)
	sd	a6, 48(a0)
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	sp, sp, a0
	ld	s0, 104(sp)                     # 8-byte Folded Reload
	ld	s1, 96(sp)                      # 8-byte Folded Reload
	ld	s2, 88(sp)                      # 8-byte Folded Reload
	ld	s3, 80(sp)                      # 8-byte Folded Reload
	ld	s4, 72(sp)                      # 8-byte Folded Reload
	ld	s5, 64(sp)                      # 8-byte Folded Reload
	ld	s6, 56(sp)                      # 8-byte Folded Reload
	ld	s7, 48(sp)                      # 8-byte Folded Reload
	ld	s8, 40(sp)                      # 8-byte Folded Reload
	ld	s9, 32(sp)                      # 8-byte Folded Reload
	ld	s10, 24(sp)                     # 8-byte Folded Reload
	ld	s11, 16(sp)                     # 8-byte Folded Reload
	addi	sp, sp, 112
	ret
.Lfunc_end1:
	.size	sparse_dense_matmul, .Lfunc_end1-sparse_dense_matmul
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0                          # -- Begin function main
.LCPI2_0:
	.quad	147                             # 0x93
	.quad	50                              # 0x32
	.quad	50                              # 0x32
	.quad	1                               # 0x1
.LCPI2_1:
	.quad	0                               # 0x0
	.quad	147                             # 0x93
	.quad	50                              # 0x32
	.quad	50                              # 0x32
	.section	.srodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI2_2:
	.quad	0x4290e8ddde08a94c              # double 4648085193258.3242
	.text
	.globl	main
	.p2align	1
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -464
	.cfi_def_cfa_offset 464
	sd	ra, 456(sp)                     # 8-byte Folded Spill
	sd	s0, 448(sp)                     # 8-byte Folded Spill
	sd	s1, 440(sp)                     # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	lui	a0, 14
	addiw	a0, a0, 1520
	call	malloc
	mv	s0, a0
	addi	a0, a0, 63
	andi	s1, a0, -64
	addi	a0, sp, 280
	call	assemble_sparse_tensor
	addi	a0, sp, 400
	vsetivli	zero, 4, e64, m2, ta, ma
	vle64.v	v8, (a0)
	addi	a0, sp, 336
	vsetivli	zero, 8, e64, m4, ta, ma
	vle64.v	v12, (a0)
	ld	a7, 328(sp)
	ld	a6, 320(sp)
	ld	a5, 312(sp)
	ld	a4, 304(sp)
	ld	a3, 296(sp)
	ld	a2, 288(sp)
	ld	a1, 280(sp)
	ld	a0, 432(sp)
	sd	zero, 176(sp)
	sd	s1, 168(sp)
	sd	s0, 160(sp)
	li	s1, 1
	sd	s1, 152(sp)
	lui	s1, %hi(.L__constant_147x50xf64)
	addi	s1, s1, %lo(.L__constant_147x50xf64)
	sd	s1, 112(sp)
	lui	s1, 228023
	slli	s1, s1, 2
	addi	s1, s1, -273
	sd	s1, 104(sp)
	sd	a0, 96(sp)
	vse64.v	v12, (sp)
	lui	a0, %hi(.LCPI2_0)
	addi	a0, a0, %lo(.LCPI2_0)
	vsetivli	zero, 4, e64, m2, ta, ma
	vle64.v	v10, (a0)
	lui	a0, %hi(.LCPI2_1)
	addi	a0, a0, %lo(.LCPI2_1)
	vle64.v	v12, (a0)
	addi	a0, sp, 184
	vse64.v	v10, (a0)
	addi	a0, sp, 120
	vse64.v	v12, (a0)
	addi	s1, sp, 64
	addi	a0, sp, 224
	vse64.v	v8, (s1)
	call	sparse_dense_matmul
	ld	a6, 272(sp)
	ld	a5, 264(sp)
	ld	a4, 256(sp)
	ld	a3, 248(sp)
	ld	a2, 240(sp)
	ld	a1, 232(sp)
	ld	a0, 224(sp)
	call	compute_sum
	lui	a0, %hi(.LCPI2_2)
	fld	fa5, %lo(.LCPI2_2)(a0)
	feq.d	a0, fa0, fa5
	xori	a0, a0, 1
	ld	ra, 456(sp)                     # 8-byte Folded Reload
	ld	s0, 448(sp)                     # 8-byte Folded Reload
	ld	s1, 440(sp)                     # 8-byte Folded Reload
	addi	sp, sp, 464
	ret
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0                          # -- Begin function assemble_sparse_tensor
.LCPI3_0:
	.quad	0                               # 0x0
	.quad	2449                            # 0x991
	.quad	1                               # 0x1
	.quad	147                             # 0x93
.LCPI3_1:
	.quad	0                               # 0x0
	.quad	2449                            # 0x991
	.quad	1                               # 0x1
	.quad	3735928559                      # 0xdeadbeef
.LCPI3_2:
	.quad	0                               # 0x0
	.quad	148                             # 0x94
	.quad	1                               # 0x1
	.quad	3735928559                      # 0xdeadbeef
	.text
	.globl	assemble_sparse_tensor
	.p2align	1
	.type	assemble_sparse_tensor,@function
assemble_sparse_tensor:                 # @assemble_sparse_tensor
	.cfi_startproc
# %bb.0:
	lui	a1, %hi(.L__constant_148xindex)
	addi	a1, a1, %lo(.L__constant_148xindex)
	ld	a2, 1176(a1)
	lui	a3, %hi(.L__constant_2449xindex)
	addi	a3, a3, %lo(.L__constant_2449xindex)
	lui	a4, %hi(.L__constant_2449xf64)
	addi	a4, a4, %lo(.L__constant_2449xf64)
	sd	a2, 152(a0)
	sd	a2, 144(a0)
	li	a2, 148
	sd	a2, 136(a0)
	li	a2, 147
	sd	a2, 128(a0)
	sd	a4, 88(a0)
	sd	a3, 48(a0)
	sd	a1, 8(a0)
	lui	a1, 228023
	slli	a1, a1, 2
	lui	a2, %hi(.LCPI3_0)
	addi	a2, a2, %lo(.LCPI3_0)
	vsetivli	zero, 4, e64, m2, ta, ma
	vle64.v	v8, (a2)
	addi	a1, a1, -273
	sd	a1, 0(a0)
	addi	a1, a0, 96
	vse64.v	v8, (a1)
	lui	a1, %hi(.LCPI3_1)
	addi	a1, a1, %lo(.LCPI3_1)
	vle64.v	v8, (a1)
	lui	a1, %hi(.LCPI3_2)
	addi	a1, a1, %lo(.LCPI3_2)
	vle64.v	v10, (a1)
	addi	a1, a0, 56
	vse64.v	v8, (a1)
	addi	a0, a0, 16
	vse64.v	v10, (a0)
	ret
.Lfunc_end3:
	.size	assemble_sparse_tensor, .Lfunc_end3-assemble_sparse_tensor
	.cfi_endproc
                                        # -- End function
	.type	.L__constant_147x50xf64,@object # @__constant_147x50xf64
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.L__constant_147x50xf64:
	.quad	0x40078e1a3f4666ed              # double 2.9443860000000002
	.quad	0x401f17518b0d0edc              # double 7.7727719999999998
	.quad	0x40105806af46aa08              # double 4.0859629999999996
	.quad	0x3ffe6aa087ca643d              # double 1.9010320000000001
	.quad	0x40231a135daad602              # double 9.550929
	.quad	0x40157bf1e8e60807              # double 5.3710399999999998
	.quad	0x4013af77af64063a              # double 4.9213550000000001
	.quad	0x3ff038ca8198f1d4              # double 1.013865
	.quad	0x400865d7cb2d905c              # double 3.049728
	.quad	0x40103f84449dbec2              # double 4.0620279999999998
	.quad	0x40131a6a875d56f3              # double 4.7757969999999998
	.quad	0x4015e10f94c87981              # double 5.469786
	.quad	0x3ffc3dfa00e27e0f              # double 1.765131
	.quad	0x40205c4a6223e187              # double 8.1802550000000007
	.quad	0x3fb22ea0fd3b29a2              # double 0.071024000000000004
	.quad	0x4020a01f75104d55              # double 8.3127399999999998
	.quad	0x401def9d2bf55150              # double 7.4839979999999997
	.quad	0x401268f7121ab4b7              # double 4.6025049999999998
	.quad	0x401d4898b2e9ccb8              # double 7.3208950000000001
	.quad	0x4021aed3b6cbd988              # double 8.8414590000000004
	.quad	0x40189ac6cdaf4adc              # double 6.1511490000000002
	.quad	0x401edea2d2fe3f36              # double 7.7174180000000003
	.quad	0x401877481b21c476              # double 6.1164860000000001
	.quad	0x401808d823422468              # double 6.0086370000000002
	.quad	0x401fb549b62c7757              # double 7.9270389999999997
	.quad	0x3feec21187e7c06e              # double 0.96118999999999999
	.quad	0x401d812556d19ded              # double 7.3761190000000001
	.quad	0x4017616bdb1a6d6a              # double 5.8451380000000004
	.quad	0x3ff4a586013660e5              # double 1.290411
	.quad	0x4011d2dcb1465e89              # double 4.4559199999999999
	.quad	0x4019d57eed45e918              # double 6.4584919999999997
	.quad	0x40133ca1cef240fb              # double 4.8092110000000003
	.quad	0x401c5b723ee1bd1f              # double 7.0893030000000001
	.quad	0x4022265c0b999136              # double 9.0749209999999998
	.quad	0x400fca01abd1aa82              # double 3.9736359999999999
	.quad	0x3ffa3c5154866a12              # double 1.639726
	.quad	0x401200e991ff7165              # double 4.5008910000000002
	.quad	0x4021c4df8b157258              # double 8.8845179999999999
	.quad	0x40171f911cf355ce              # double 5.7808270000000004
	.quad	0x400287a8d64d7f0f              # double 2.3162400000000001
	.quad	0x4015273c4f3ba776              # double 5.288316
	.quad	0x401b049a133c1ce7              # double 6.7544940000000002
	.quad	0x4021c0535c9e6688              # double 8.8756360000000001
	.quad	0x3fd0d1f601797cc4              # double 0.26281500000000002
	.quad	0x3fd36317268d3283              # double 0.302923
	.quad	0x3ff961b089a02752              # double 1.5863499999999999
	.quad	0x4012b1e4f765fd8b              # double 4.6737250000000001
	.quad	0x4013ced0f627392a              # double 4.9519690000000001
	.quad	0x401469563a9f383f              # double 5.102868
	.quad	0x4006ac0f01fb82c3              # double 2.8340130000000001
	.quad	0x400f89aa1d755bcd              # double 3.9422190000000001
	.quad	0x401e3e8fb00bcbe6              # double 7.5610949999999999
	.quad	0x402140dbad3a604e              # double 8.6266759999999998
	.quad	0x401ad090f733a8a4              # double 6.703678
	.quad	0x4012191d57ff9b56              # double 4.5245259999999998
	.quad	0x3fe457470eb24a6b              # double 0.63565400000000005
	.quad	0x401b1fae3608d089              # double 6.7809379999999999
	.quad	0x3ff69daceee0f3cb              # double 1.4134949999999999
	.quad	0x4023f51266341180              # double 9.9786560000000009
	.quad	0x40223bb5e0f7fcfc              # double 9.1166219999999995
	.quad	0x3fd7c011d3671ac1              # double 0.37109799999999998
	.quad	0x4000b74c4cdfaca3              # double 2.0895009999999998
	.quad	0x40039d0fa58f7122              # double 2.4516900000000001
	.quad	0x400e16a619da9c99              # double 3.7610589999999999
	.quad	0x4016db691212513b              # double 5.7142679999999997
	.quad	0x400c2c0053e2d624              # double 3.5214850000000002
	.quad	0x3ffe7136a400fba9              # double 1.9026400000000001
	.quad	0x401eb10ba6266fd6              # double 7.6728959999999997
	.quad	0x4009548ba83f4eca              # double 3.1662819999999998
	.quad	0x4015ef66a5508701              # double 5.4837899999999999
	.quad	0x400fb41aef6f8f04              # double 3.962942
	.quad	0x3ff218116ebd4cfd              # double 1.130876
	.quad	0x3fe07f92f2b66b61              # double 0.51557299999999995
	.quad	0x3ff49331a08bfc22              # double 1.285936
	.quad	0x401a3fb58d1526d9              # double 6.5622160000000003
	.quad	0x4005490257c914b1              # double 2.6606489999999998
	.quad	0x401dd796bfca85cb              # double 7.4605360000000003
	.quad	0x40103915bdd76684              # double 4.0557470000000002
	.quad	0x3ff2fa3db3bfb58d              # double 1.186094
	.quad	0x3fe1932d6ece13f5              # double 0.54921600000000004
	.quad	0x400770b524399b2c              # double 2.9300329999999999
	.quad	0x40218ee807bbb624              # double 8.7791139999999998
	.quad	0x3fe4a708ede54b49              # double 0.64539000000000002
	.quad	0x401984b05b7cfe58              # double 6.3795789999999997
	.quad	0x40226dc5d6388659              # double 9.2143999999999995
	.quad	0x4016d57214f0520d              # double 5.7084429999999999
	.quad	0x3ffc775b81301648              # double 1.7791399999999999
	.quad	0x3fe8d5a5b9628cbd              # double 0.77607999999999999
	.quad	0x4023d71348b22079              # double 9.9200689999999998
	.quad	0x402369bcba301217              # double 9.7065180000000008
	.quad	0x4018ab9b66f9335d              # double 6.1675849999999999
	.quad	0x3fdf8ceb356da017              # double 0.49297600000000003
	.quad	0x400b3815a07b352b              # double 3.4023850000000002
	.quad	0x400053b9f127f5e8              # double 2.0408819999999999
	.quad	0x401db0b5ed8d36b5              # double 7.4225690000000002
	.quad	0x3fe7c8adeebb341e              # double 0.74324699999999999
	.quad	0x4001e78fabda0dfe              # double 2.238067
	.quad	0x40010ea4228997c4              # double 2.1321490000000001
	.quad	0x401bf92146a1a501              # double 6.9932910000000001
	.quad	0x401b73eee5258927              # double 6.8632160000000004
	.quad	0x401864b48d3ae686              # double 6.0983450000000001
	.quad	0x4014404b3c3e74b0              # double 5.0627870000000001
	.quad	0x4014dc9c90c4dec2              # double 5.2154410000000002
	.quad	0x4018c7f8ca8198f2              # double 6.1952850000000002
	.quad	0x401993cf70153bd1              # double 6.3943459999999996
	.quad	0x402118a3d70a3d71              # double 8.5481250000000006
	.quad	0x401c18dc55000c95              # double 7.0242779999999998
	.quad	0x40110ab191dde376              # double 4.2604430000000004
	.quad	0x4021a0e7ddca4b12              # double 8.8142689999999995
	.quad	0x401b8030c23fab11              # double 6.8751860000000002
	.quad	0x4022437110e453d2              # double 9.1317219999999999
	.quad	0x3ffb7d09e12a51e3              # double 1.718027
	.quad	0x40198aeb3dd11be7              # double 6.3856630000000001
	.quad	0x40237c046412cf10              # double 9.7422210000000007
	.quad	0x402175441355475a              # double 8.7290349999999997
	.quad	0x4023aea4ebdd334c              # double 9.8411019999999993
	.quad	0x401d2fd6d7e889c2              # double 7.2967180000000003
	.quad	0x401e80d413122b7c              # double 7.6258090000000003
	.quad	0x3fc5de37585be1a8              # double 0.170844
	.quad	0x401ce5ca29845dc8              # double 7.2244039999999998
	.quad	0x3fe1ae48e8a71de7              # double 0.55252500000000004
	.quad	0x401f5405f6ba0621              # double 7.8320540000000002
	.quad	0x400a6a747d805e5f              # double 3.3019799999999999
	.quad	0x40092db61bb05faf              # double 3.1473200000000001
	.quad	0x4021b5b8fde2ef4e              # double 8.854927
	.quad	0x4022db422467be55              # double 9.4282389999999995
	.quad	0x401cdd66adb402d1              # double 7.2162119999999996
	.quad	0x4021cf0ef566490a              # double 8.9044109999999996
	.quad	0x401cd71476af89c6              # double 7.2100390000000001
	.quad	0x3ff50df8b1572581              # double 1.3159110000000001
	.quad	0x3ffd4a9cdc443915              # double 1.830716
	.quad	0x4023bdde7a743a64              # double 9.8708379999999991
	.quad	0x401d26f3f52fc265              # double 7.2880399999999996
	.quad	0x4004695421c04428              # double 2.5514299999999999
	.quad	0x400924b018611fd6              # double 3.1429140000000002
	.quad	0x401b609528f190d1              # double 6.8443189999999996
	.quad	0x4019993d5347a5b1              # double 6.399648
	.quad	0x4000b9e16d6dc1a4              # double 2.0907619999999998
	.quad	0x3ff9854cdb7ae579              # double 1.5950439999999999
	.quad	0x4013c027d88c1db0              # double 4.9376519999999999
	.quad	0x4013f23fab10ba62              # double 4.9865709999999996
	.quad	0x3fe0c1aac53b0814              # double 0.52364100000000002
	.quad	0x4023cde679463cfb              # double 9.9021489999999996
	.quad	0x3ff36b27cbf6e3f8              # double 1.2136610000000001
	.quad	0x401154a7f8012dfd              # double 4.3326719999999996
	.quad	0x4023486705c896dd              # double 9.6414109999999997
	.quad	0x400cc6bf01322f27              # double 3.5970439999999999
	.quad	0x402065ab9f559b3d              # double 8.1985749999999999
	.quad	0x400109f6a93f290b              # double 2.1298650000000001
	.quad	0x3fdbb2e1693c03bc              # double 0.43279299999999998
	.quad	0x4023d02c0a4a05de              # double 9.9065860000000008
	.quad	0x4023c8f7db6e5040              # double 9.8925160000000005
	.quad	0x4016ea75cd0bb6ed              # double 5.7289649999999996
	.quad	0x401c3a76534373f3              # double 7.0570919999999999
	.quad	0x401e549774256bca              # double 7.5826089999999997
	.quad	0x40113740c4156e26              # double 4.3039579999999997
	.quad	0x4022545564b662fe              # double 9.164714
	.quad	0x40131ed56b00ffda              # double 4.7801109999999998
	.quad	0x4014932e3821af7d              # double 5.1437309999999998
	.quad	0x4013a55ef1fddebe              # double 4.9114950000000004
	.quad	0x401a03126e978d50              # double 6.5030000000000001
	.quad	0x3ff507820a30db6a              # double 1.314333
	.quad	0x40235ee17a02fb5d              # double 9.6853139999999999
	.quad	0x400b16177ea1c68f              # double 3.3857870000000001
	.quad	0x40131916ca46e08f              # double 4.7745009999999999
	.quad	0x3ff9c9635610adcd              # double 1.611667
	.quad	0x400315d35653c9ac              # double 2.3856570000000001
	.quad	0x40181f40a2877ee5              # double 6.0305200000000001
	.quad	0x4021306ec17ebaf1              # double 8.594595
	.quad	0x401bbc8ffb8b2639              # double 6.9341429999999997
	.quad	0x4009ce10cf5b1c86              # double 3.2256179999999999
	.quad	0x4000d31ca7d6733f              # double 2.1030820000000001
	.quad	0x400c2d373affb04f              # double 3.522078
	.quad	0x4014efa1e3eaf683              # double 5.2340159999999996
	.quad	0x4013395c42203600              # double 4.8060159999999996
	.quad	0x4007dfb69984a0e4              # double 2.984235
	.quad	0x3f913f8db4890929              # double 0.016844000000000001
	.quad	0x4004e9835158b828              # double 2.61402
	.quad	0x401a03dee78183f9              # double 6.5037799999999999
	.quad	0x4012ed73c9257860              # double 4.7318870000000004
	.quad	0x40124c929aa1d756              # double 4.5747780000000002
	.quad	0x40175209246bf013              # double 5.8301129999999999
	.quad	0x4009bf3300de4c51              # double 3.218359
	.quad	0x401df80c308feac4              # double 7.4922339999999998
	.quad	0x40159903ea704bc2              # double 5.3994289999999996
	.quad	0x40238beee5258927              # double 9.7733070000000008
	.quad	0x3ff4f2b66b6177ea              # double 1.309256
	.quad	0x401da641f644955b              # double 7.4123609999999998
	.quad	0x4010f6a2b1704ff4              # double 4.2408549999999998
	.quad	0x3ffdbe5d9e40c847              # double 1.858976
	.quad	0x401393892ee84ad8              # double 4.8940780000000004
	.quad	0x3fdeb76f6d762520              # double 0.47994599999999998
	.quad	0x40107d759ee88df3              # double 4.1225189999999996
	.quad	0x4022db912dba4d6e              # double 9.4288419999999995
	.quad	0x401e90d3cff64cf9              # double 7.6414330000000001
	.quad	0x401338eb8950763a              # double 4.8055859999999999
	.quad	0x4004670867ad8e43              # double 2.5503089999999999
	.quad	0x40041cd3e0bd449a              # double 2.5140760000000002
	.quad	0x400a40ae10492360              # double 3.2815820000000002
	.quad	0x3ffd4bddb9841aac              # double 1.8310219999999999
	.quad	0x401b569a0ad8a116              # double 6.8345719999999996
	.quad	0x401fd1aef6f8f041              # double 7.9547689999999997
	.quad	0x401f5f190d173fb8              # double 7.8428690000000003
	.quad	0x3fff728954a7f801              # double 1.965463
	.quad	0x4023c9da16616b55              # double 9.8942420000000002
	.quad	0x4016d2bf11f926c8              # double 5.7058070000000001
	.quad	0x4002f1f4f50a02b8              # double 2.3681429999999999
	.quad	0x401d7d0cc35ce182              # double 7.3721189999999996
	.quad	0x4020c38a19c9d5a2              # double 8.3819130000000008
	.quad	0x4018ca06e9ff0cbb              # double 6.197292
	.quad	0x400373e4ef0281ba              # double 2.4315889999999998
	.quad	0x401cc429e0a41a26              # double 7.1915659999999999
	.quad	0x3ffb9daad601ffb5              # double 1.7259930000000001
	.quad	0x4008a53eb399f5e0              # double 3.080686
	.quad	0x4020a786c226809d              # double 8.3271999999999995
	.quad	0x401850a4a05dd8f9              # double 6.0787529999999999
	.quad	0x40118564f97edc7f              # double 4.3802680000000001
	.quad	0x3fea519dece57109              # double 0.82246300000000006
	.quad	0x40152fd60e94ee39              # double 5.2967149999999998
	.quad	0x401104d2f5dbb9d0              # double 4.2547110000000004
	.quad	0x400a8d4c33b53932              # double 3.3189929999999999
	.quad	0x4008a0d90e23af32              # double 3.0785390000000001
	.quad	0x4021fcd50ebaade6              # double 8.9938129999999994
	.quad	0x3fe4793ff25e56cd              # double 0.63980099999999995
	.quad	0x401481d68c692f6f              # double 5.1267950000000004
	.quad	0x4003c1f4b1ee2435              # double 2.4697049999999998
	.quad	0x40231f192ea52ef9              # double 9.5607389999999998
	.quad	0x3ff0c8637bd05af7              # double 1.048923
	.quad	0x4011fd1287c200c1              # double 4.4971410000000001
	.quad	0x4016a7ac81d3aa37              # double 5.6637440000000003
	.quad	0x401b2640e9d51b50              # double 6.7873570000000001
	.quad	0x401c9a7935760392              # double 7.1508529999999997
	.quad	0x4023b2eea63b688c              # double 9.8494770000000002
	.quad	0x3fe82543f1c75819              # double 0.75454900000000003
	.quad	0x400462f2f9873ffb              # double 2.5483150000000001
	.quad	0x401041bfbdf090f7              # double 4.0642079999999998
	.quad	0x401190b49e01de27              # double 4.3913140000000004
	.quad	0x401cd29670196d8f              # double 7.2056519999999997
	.quad	0x3ff7104d551d68c7              # double 1.4414800000000001
	.quad	0x4003f2b3461309c8              # double 2.493506
	.quad	0x4009c49dbec2480f              # double 3.2210040000000002
	.quad	0x3ff46c8605681ecd              # double 1.2764949999999999
	.quad	0x4000901754b05b7d              # double 2.070357
	.quad	0x400a054ef459d990              # double 3.2525919999999999
	.quad	0x40231248d7e02646              # double 9.5357120000000002
	.quad	0x3ffc56b873797460              # double 1.771172
	.quad	0x4015d258d5842b73              # double 5.4554169999999997
	.quad	0x400b7f7d73c92578              # double 3.4372509999999998
	.quad	0x40177ceaaf35e311              # double 5.8719890000000001
	.quad	0x4014c4f61672324d              # double 5.1923450000000004
	.quad	0x400f7b003686a4ca              # double 3.9350589999999999
	.quad	0x3ffac743201040c0              # double 1.673648
	.quad	0x402351bb2788db05              # double 9.6596309999999991
	.quad	0x3ff14b8f14db5958              # double 1.0809470000000001
	.quad	0x401da67a0f9096bc              # double 7.4125750000000004
	.quad	0x40205aad3e920c07              # double 8.1771030000000007
	.quad	0x3fec2fd5454152b1              # double 0.88083900000000004
	.quad	0x4011744e50c5eb31              # double 4.3635799999999998
	.quad	0x401183fe9fadafd1              # double 4.3789009999999999
	.quad	0x3fe1a915379fa97e              # double 0.55188999999999999
	.quad	0x3ff9fc968943e100              # double 1.6241669999999999
	.quad	0x3ff1db7e0ac7da1f              # double 1.1160870000000001
	.quad	0x40211a75ee99a62f              # double 8.5516810000000003
	.quad	0x4021c9e279dd3bb0              # double 8.8943060000000003
	.quad	0x401fb5e9185cee18              # double 7.9276470000000003
	.quad	0x402371a2a4db163c              # double 9.7219440000000005
	.quad	0x3ff525785f8d2e51              # double 1.3216479999999999
	.quad	0x3fe3f37329c347e9              # double 0.62346800000000002
	.quad	0x401c67b2031ceaf2              # double 7.1012649999999997
	.quad	0x3fe02f42784a9479              # double 0.50576900000000002
	.quad	0x4018968c692f6e83              # double 6.1470200000000004
	.quad	0x401a51769a91105e              # double 6.5795539999999999
	.quad	0x401be07ee0b0af60              # double 6.9692340000000002
	.quad	0x4001cf262888b546              # double 2.2261470000000001
	.quad	0x40142944241c3efb              # double 5.0402990000000001
	.quad	0x3ffb94e2b063e07a              # double 1.723849
	.quad	0x40191b7a5f41aef7              # double 6.276834
	.quad	0x4023cc36b8f9b131              # double 9.8988549999999992
	.quad	0x40013a92a3055326              # double 2.1536
	.quad	0x4023d5d2d01c0ca6              # double 9.917624
	.quad	0x3fea004d98394749              # double 0.81253699999999995
	.quad	0x401e927aa68f4b62              # double 7.643046
	.quad	0x401f4bbe878fabda              # double 7.823969
	.quad	0x3ffa6dddf43c7d5f              # double 1.651823
	.quad	0x3f99129888f861a6              # double 0.024485
	.quad	0x40214906cca2db62              # double 8.6426300000000005
	.quad	0x4015423076c050be              # double 5.3146380000000004
	.quad	0x3fee2007dd441355              # double 0.94140999999999997
	.quad	0x400724a6a875d56f              # double 2.8928959999999999
	.quad	0x4014ce0eb67c2870              # double 5.2012280000000004
	.quad	0x4001f9e59f2ba9d2              # double 2.24702
	.quad	0x4014d36006d0d499              # double 5.2064209999999997
	.quad	0x3ff201a47a9e2bd0              # double 1.1254010000000001
	.quad	0x40162b25b317efe1              # double 5.5421360000000002
	.quad	0x4020555821294574              # double 8.1666880000000006
	.quad	0x4002c72d1aeb3dd1              # double 2.347254
	.quad	0x3febdcf67b1c0011              # double 0.87072300000000002
	.quad	0x3ffa0714fce746cc              # double 1.6267290000000001
	.quad	0x3ffe27ad8e432442              # double 1.884687
	.quad	0x401907d8cf398e97              # double 6.257663
	.quad	0x4017a94f26aec072              # double 5.9153409999999997
	.quad	0x4021b8e1f7d73c92              # double 8.8610989999999994
	.quad	0x40128c197e564734              # double 4.6368159999999996
	.quad	0x401003b4f6167232              # double 4.0036199999999997
	.quad	0x40159db0574b4070              # double 5.4039929999999998
	.quad	0x3fd7b723ee1bd1ee              # double 0.37055300000000002
	.quad	0x400b0b5781c714fd              # double 3.380538
	.quad	0x4015286a8fc0d2c4              # double 5.2894690000000004
	.quad	0x3ff09781c714fce7              # double 1.0369889999999999
	.quad	0x3f8463497b7414a5              # double 0.0099550000000000003
	.quad	0x4022ad2c9d16fc9c              # double 9.3382310000000003
	.quad	0x400ef316e3715400              # double 3.8686959999999999
	.quad	0x4016e0fbeb9e492c              # double 5.7197110000000002
	.quad	0x400e8edbb59ddc1e              # double 3.8197549999999998
	.quad	0x4010f1f9acffa7eb              # double 4.2363039999999996
	.quad	0x400bee4c942d490e              # double 3.4913569999999998
	.quad	0x40126abb44e50c5f              # double 4.6042300000000003
	.quad	0x401ee8d972cd7cf6              # double 7.727392
	.quad	0x40148e0eb67c2870              # double 5.1387280000000004
	.quad	0x4011ac4b09e98dce              # double 4.4182550000000003
	.quad	0x4019187df5cf2496              # double 6.2739180000000001
	.quad	0x3fe4cfb1183b6028              # double 0.65035299999999996
	.quad	0x4010100ffda4052d              # double 4.0156859999999996
	.quad	0x3ff507570c564f98              # double 1.314292
	.quad	0x40197b7fe08aefb3              # double 6.3706050000000003
	.quad	0x4023dbecaab8a5ce              # double 9.9295399999999994
	.quad	0x40204abd7f51efb7              # double 8.1459770000000002
	.quad	0x4023b7fef39085f5              # double 9.8593670000000006
	.quad	0x4020523a29c779a7              # double 8.1606000000000005
	.quad	0x3ff6d53ac4f7ec35              # double 1.4270579999999999
	.quad	0x4021557b2031ceaf              # double 8.6669549999999997
	.quad	0x401845e02ea960b7              # double 6.068238
	.quad	0x401fdcb20fb224ab              # double 7.9655230000000001
	.quad	0x3fbe4f54d1e96c40              # double 0.118398
	.quad	0x4016ef8ceb356da0              # double 5.7339359999999999
	.quad	0x3fec4810a569b175              # double 0.88379700000000005
	.quad	0x401158ee28672757              # double 4.3368460000000004
	.quad	0x401d6f81626b2f23              # double 7.358892
	.quad	0x402241b0ab2e1694              # double 9.1283010000000004
	.quad	0x4021387fcb923a2a              # double 8.6103500000000004
	.quad	0x3fa38194c0160525              # double 0.038098
	.quad	0x40153b90ea9e6eeb              # double 5.3081699999999996
	.quad	0x401744ef0281ba80              # double 5.8173180000000002
	.quad	0x4020cefd438d1d8a              # double 8.4042759999999994
	.quad	0x40207f1d60631727              # double 8.2482710000000008
	.quad	0x4008eeb67c286f8b              # double 3.1165590000000001
	.quad	0x4007df0f5a1016ce              # double 2.9839159999999998
	.quad	0x4011913a92a30553              # double 4.3918249999999999
	.quad	0x401128c2e770bd01              # double 4.2898059999999996
	.quad	0x4015780ecfa69be1              # double 5.3672440000000003
	.quad	0x4009a8ee6b8305d9              # double 3.2074859999999998
	.quad	0x40114b43d89ce4a8              # double 4.3235010000000003
	.quad	0x4004608637bd05af              # double 2.5471309999999998
	.quad	0x40160d4bf0995aaf              # double 5.5129849999999996
	.quad	0x400a077dd872f33d              # double 3.2536580000000002
	.quad	0x400055da272862f6              # double 2.0419200000000002
	.quad	0x4020a03f4eca686e              # double 8.3129829999999991
	.quad	0x401375a858793dd9              # double 4.8648999999999996
	.quad	0x3fca4790b8498809              # double 0.20530899999999999
	.quad	0x3fe88f12c27a6373              # double 0.76746499999999995
	.quad	0x3ff13721d53cddd7              # double 1.07596
	.quad	0x400e7563a9f383f1              # double 3.8073190000000001
	.quad	0x401a3bcdbbe0157f              # double 6.5584020000000001
	.quad	0x400d8c532a497fa6              # double 3.6935180000000001
	.quad	0x4023d649f51697f2              # double 9.918533
	.quad	0x401f38b0079a2835              # double 7.8053590000000001
	.quad	0x3fe9b50d2806af47              # double 0.80335100000000004
	.quad	0x400d479b34a44c7b              # double 3.659964
	.quad	0x3ff8bd7d3910c2c6              # double 1.546262
	.quad	0x4015b86fcdee34fc              # double 5.4301139999999997
	.quad	0x3fe106ee30caa327              # double 0.53209600000000001
	.quad	0x401ad6fe2e6ea854              # double 6.7099539999999998
	.quad	0x4018eaec4a4095f2              # double 6.2294169999999998
	.quad	0x402171ad85dfa872              # double 8.7220270000000006
	.quad	0x3fe90bed740c4157              # double 0.78270600000000001
	.quad	0x401208f25a250f84              # double 4.508737
	.quad	0x400c81d19157abb9              # double 3.5633880000000002
	.quad	0x401798c4bd33d295              # double 5.8991879999999997
	.quad	0x401baa28fe260b2d              # double 6.9161720000000004
	.quad	0x4017c8d002e23290              # double 5.9461060000000003
	.quad	0x4021d7b5aea3161a              # double 8.9213079999999998
	.quad	0x4013f5119ce075f7              # double 4.989325
	.quad	0x4021b9ca7d6733ec              # double 8.8628730000000004
	.quad	0x40176faeff5c6c12              # double 5.8590660000000003
	.quad	0x4020b6b6390c9108              # double 8.356859
	.quad	0x40213465625a682b              # double 8.6023359999999993
	.quad	0x400b5ecb31c219eb              # double 3.4212859999999998
	.quad	0x3ff01db3397dd00f              # double 1.0072509999999999
	.quad	0x400bf70f7b9e0610              # double 3.495635
	.quad	0x401a3e2b8c75c4a8              # double 6.5607129999999998
	.quad	0x3fd822d5171e29b7              # double 0.37712600000000002
	.quad	0x3fe8d1d25aab4741              # double 0.775613
	.quad	0x4005adf04ddb5526              # double 2.7099310000000001
	.quad	0x40229637de939eae              # double 9.2933950000000003
	.quad	0x3fcfea4ebdd334c6              # double 0.249338
	.quad	0x40186c9f2fdb8fde              # double 6.1060759999999998
	.quad	0x402082f449129889              # double 8.2557700000000001
	.quad	0x3f91a8ef77f27fe5              # double 0.017246000000000001
	.quad	0x40103e71cda2b5a2              # double 4.060981
	.quad	0x401adcec84f8f8a5              # double 6.7157460000000002
	.quad	0x4005422bbecaab8a              # double 2.6573099999999998
	.quad	0x4010d6eefa1e3eaf              # double 4.2098959999999996
	.quad	0x4023ce4345cfede9              # double 9.9028569999999991
	.quad	0x3fdb304039abf338              # double 0.42481999999999998
	.quad	0x401b50dfdef8487c              # double 6.8289790000000004
	.quad	0x401e07af20ea5b53              # double 7.5075039999999999
	.quad	0x4017815a07b352a8              # double 5.8763199999999998
	.quad	0x4005f00713f077cd              # double 2.7422010000000001
	.quad	0x4021002ababead4f              # double 8.5003259999999994
	.quad	0x401b77f84449dbec              # double 6.8671579999999999
	.quad	0x3ff4b442c7fbacb4              # double 1.294009
	.quad	0x40219780fdc1615f              # double 8.7959060000000004
	.quad	0x3fe12f20ea5b530d              # double 0.53700300000000001
	.quad	0x3fc9f090f733a8a4              # double 0.202654
	.quad	0x40203b41e14bdfd2              # double 8.1157369999999993
	.quad	0x401aec6540cc78ea              # double 6.730855
	.quad	0x3ff94bd44998d046              # double 1.581013
	.quad	0x401215a5332acfb7              # double 4.5211379999999997
	.quad	0x4019ab7bf1e8e608              # double 6.417465
	.quad	0x4001b9fb1e18efbb              # double 2.215811
	.quad	0x3ffa512ec6bce853              # double 1.6448199999999999
	.quad	0x3fff00dd3fe1975f              # double 1.937711
	.quad	0x40147f3bea91d9b2              # double 5.1242520000000003
	.quad	0x402349cffeb074a7              # double 9.6441649999999992
	.quad	0x40129b54195cc858              # double 4.651688
	.quad	0x400ee76b7ed41b76              # double 3.8629980000000002
	.quad	0x4016b07d912556d2              # double 5.6723540000000003
	.quad	0x40015fcf3dc054ef              # double 2.1717819999999999
	.quad	0x4010f03dddb1209f              # double 4.2346110000000001
	.quad	0x40194b6a619da9ca              # double 6.3236480000000004
	.quad	0x4021d884a515ce9e              # double 8.9228869999999993
	.quad	0x3fef0ccff21b3aef              # double 0.970314
	.quad	0x40226aad601ffb48              # double 9.2083539999999999
	.quad	0x40147d8c6d612c6b              # double 5.1226060000000002
	.quad	0x40230316e3715400              # double 9.5060339999999997
	.quad	0x4011ea0dbad3a605              # double 4.4785680000000001
	.quad	0x3ff9860029f16b12              # double 1.595215
	.quad	0x401ef81cf7878b7a              # double 7.7422979999999999
	.quad	0x3fe52d362e903644              # double 0.66176900000000005
	.quad	0x401e45c6c11a1123              # double 7.5681409999999997
	.quad	0x4000aa8c9b845565              # double 2.0832760000000001
	.quad	0x4001517720c8cd64              # double 2.1647780000000001
	.quad	0x4016f7a17f4128bf              # double 5.7418269999999998
	.quad	0x4020b67f2c3009b3              # double 8.356439
	.quad	0x3ffe1ede1198aeb8              # double 1.882536
	.quad	0x40123bf0dc768dfc              # double 4.5585360000000001
	.quad	0x401d72817763e4ac              # double 7.3618220000000001
	.quad	0x4023c8f1b25f633d              # double 9.8924690000000001
	.quad	0x40226ca5375c8da0              # double 9.2121980000000008
	.quad	0x3ffaa4fa8b4bf8fd              # double 1.665278
	.quad	0x3fdd15a8deb0fadf              # double 0.45444699999999999
	.quad	0x4008cfeef5ec80c7              # double 3.1015299999999999
	.quad	0x401c7ea140570825              # double 7.1236620000000004
	.quad	0x3ffc438b04ab606b              # double 1.7664899999999999
	.quad	0x3ffdce71476af89c              # double 1.8629009999999999
	.quad	0x3ff2bc664d3bf2f5              # double 1.1709959999999999
	.quad	0x4014e0257c914b16              # double 5.2188929999999996
	.quad	0x40212b7718655193              # double 8.5848929999999992
	.quad	0x401f9548ecd8dbcf              # double 7.8957860000000001
	.quad	0x401092263d816acf              # double 4.1427240000000003
	.quad	0x400a793dd97f62b7              # double 3.3092000000000001
	.quad	0x4010bd05f2884838              # double 4.1845929999999996
	.quad	0x4015c443d46b26c0              # double 5.4416650000000004
	.quad	0x401ad6c93a711516              # double 6.7097519999999999
	.quad	0x4014bb1290257c91              # double 5.1826879999999997
	.quad	0x401f3f9b13165d3a              # double 7.8121150000000004
	.quad	0x40119caff6d33094              # double 4.4030149999999999
	.quad	0x3ff001d4b6a619db              # double 1.0004470000000001
	.quad	0x4017d2f123c42a67              # double 5.9559980000000001
	.quad	0x40171790752da986              # double 5.7730119999999996
	.quad	0x401c984b1ab0856e              # double 7.1487239999999996
	.quad	0x402367a743a647fe              # double 9.7024480000000004
	.quad	0x3ffb1a3c21187e7c              # double 1.693905
	.quad	0x4022ab4bb5e0f7fd              # double 9.334562
	.quad	0x4009068661ae70c1              # double 3.1281859999999999
	.quad	0x402252bf76a2f48c              # double 9.1616169999999996
	.quad	0x40034d2ed783dff4              # double 2.412687
	.quad	0x40153b76f6d76252              # double 5.308071
	.quad	0x4012d74538ef34d7              # double 4.7102250000000003
	.quad	0x40234dd76683c298              # double 9.6520340000000004
	.quad	0x400ff3c925785f8d              # double 3.9940359999999999
	.quad	0x4019f955b4677b39              # double 6.4934909999999997
	.quad	0x4020e3b96af038e3              # double 8.4447740000000007
	.quad	0x40134da09cc319c6              # double 4.8258080000000003
	.quad	0x3fc223fab10ba626              # double 0.14172299999999999
	.quad	0x4018d9fcf3dc054f              # double 6.212879
	.quad	0x401363c8e25c810a              # double 4.8474459999999997
	.quad	0x4022a8f68be2f7b1              # double 9.3300059999999991
	.quad	0x3ffa7267c6b8b695              # double 1.6529309999999999
	.quad	0x3fdf74299d883ba3              # double 0.49146499999999999
	.quad	0x4022011a11233df3              # double 9.0021520000000006
	.quad	0x401d33ac929aa1d7              # double 7.3004629999999997
	.quad	0x40235cc48adeebb3              # double 9.6811869999999995
	.quad	0x400a880f12c27a63              # double 3.3164349999999998
	.quad	0x3ff9bc148344c37e              # double 1.6084179999999999
	.quad	0x4006090de093532e              # double 2.7544209999999998
	.quad	0x40231bfbbd7b2032              # double 9.5546550000000003
	.quad	0x3ffb2b90a7829069              # double 1.6981360000000001
	.quad	0x400b23f572de43ee              # double 3.3925580000000002
	.quad	0x3fdd4bcad9ad85e0              # double 0.45775100000000002
	.quad	0x4000f89acbc8c0cf              # double 2.1213890000000002
	.quad	0x4009eda22f6a50d7              # double 3.2410320000000001
	.quad	0x3feb2e5796bfca86              # double 0.84940700000000002
	.quad	0x3ffdd495182a9931              # double 1.8644000000000001
	.quad	0x40023f79c842fa51              # double 2.2809940000000002
	.quad	0x402262bc947064ed              # double 9.1928450000000002
	.quad	0x401f0c78e9f6a93f              # double 7.7621799999999998
	.quad	0x3ff59de9c0229a5f              # double 1.3510530000000001
	.quad	0x401f2cc6c54bcf0b              # double 7.7937269999999996
	.quad	0x4021e374bc6a7efa              # double 8.9442500000000003
	.quad	0x3fc63265add9c27f              # double 0.17341300000000001
	.quad	0x40060ea0fd3b29a2              # double 2.7571430000000001
	.quad	0x3ff3ea5d6bebe165              # double 1.244718
	.quad	0x4014e5208e150119              # double 5.223757
	.quad	0x3ffbb9096bb98c7e              # double 1.732675
	.quad	0x401c6783dff3f0fe              # double 7.101089
	.quad	0x400bd4ef459d9903              # double 3.4789720000000002
	.quad	0x401f9a1cef240faa              # double 7.9005010000000002
	.quad	0x4009072b020c49ba              # double 3.1284999999999998
	.quad	0x401d62031ceaf252              # double 7.3457150000000002
	.quad	0x40027d9018e75793              # double 2.3113100000000002
	.quad	0x40229d9cae21101b              # double 9.307836
	.quad	0x4023dccaf709b739              # double 9.9312360000000001
	.quad	0x400f73ade225fa66              # double 3.9314840000000002
	.quad	0x4006412a51e321a3              # double 2.781819
	.quad	0x400be46173b85e81              # double 3.4865140000000001
	.quad	0x401a0ab9b23dd54e              # double 6.5104740000000003
	.quad	0x402048d9f90539fc              # double 8.1422880000000006
	.quad	0x4023ce295208e150              # double 9.9026589999999999
	.quad	0x40130e38eb0318b9              # double 4.7638889999999998
	.quad	0x401eb9bae46cfc83              # double 7.6813770000000003
	.quad	0x401b691fb3fa6df0              # double 6.8526600000000002
	.quad	0x400f6ff7dfa00e28              # double 3.9296720000000001
	.quad	0x4023e9815a07b353              # double 9.9560650000000006
	.quad	0x40200a5870da5daf              # double 8.0202059999999999
	.quad	0x40162836eb4e9814              # double 5.5392720000000004
	.quad	0x4021b9947cfa26a2              # double 8.8624609999999997
	.quad	0x4010d6ad2dcb1466              # double 4.2096450000000001
	.quad	0x3ff9309203a322af              # double 1.5743579999999999
	.quad	0x401bc5c3dee78184              # double 6.94313
	.quad	0x401836bb1290257d              # double 6.0534480000000004
	.quad	0x3ffe36f9fcb0c027              # double 1.888422
	.quad	0x401c3b441bb8c32b              # double 7.0578770000000004
	.quad	0x400a10b30b5aa716              # double 3.2581540000000002
	.quad	0x4006d7460242d05f              # double 2.8551139999999999
	.quad	0x3fe9d5347a5b0ff1              # double 0.80727599999999999
	.quad	0x3ff7fc60029f16b1              # double 1.499115
	.quad	0x40142bbe878fabda              # double 5.042719
	.quad	0x4002a1a1db877ab3              # double 2.3289219999999999
	.quad	0x4011f25f202107b8              # double 4.4866910000000004
	.quad	0x40076f112fd32c62              # double 2.9292319999999998
	.quad	0x4003efebd09e12a5              # double 2.4921489999999999
	.quad	0x40208ba7fc32ebe6              # double 8.2727660000000007
	.quad	0x40208071c53f39d2              # double 8.2508680000000005
	.quad	0x401b35ba6efc371e              # double 6.8024690000000003
	.quad	0x401b070329802c0a              # double 6.7568479999999997
	.quad	0x402377268d32830a              # double 9.7327159999999999
	.quad	0x4000f7603925bb7b              # double 2.1207889999999998
	.quad	0x4017261e4f765fd9              # double 5.7872250000000003
	.quad	0x401d579a2834d270              # double 7.3355490000000003
	.quad	0x40061b66f9335d25              # double 2.7633800000000002
	.quad	0x401a0bd76ee73e68              # double 6.5115639999999999
	.quad	0x402228d0aaa7ded7              # double 9.0797170000000005
	.quad	0x401ca60cbf2b239a              # double 7.1621579999999998
	.quad	0x401ae0abb44e50c6              # double 6.7194050000000001
	.quad	0x401f7f2a5a469d73              # double 7.8741849999999997
	.quad	0x40221530ced4e4c9              # double 9.0413879999999995
	.quad	0x4010629e0a41a262              # double 4.0963060000000002
	.quad	0x3ffa2d83c6c97d8d              # double 1.636112
	.quad	0x40121899392189be              # double 4.5240220000000004
	.quad	0x3fe2d32c625e99e9              # double 0.58827799999999997
	.quad	0x401e2c9ff92f2b67              # double 7.5435790000000002
	.quad	0x40216b24851a8694              # double 8.7092629999999999
	.quad	0x401e0f99c38b04ab              # double 7.5152349999999996
	.quad	0x401e5ccccccccccd              # double 7.5906250000000001
	.quad	0x3fc068f4b61fe21e              # double 0.12820300000000001
	.quad	0x4001e858793dd97f              # double 2.2384499999999998
	.quad	0x4009ac404a72ead9              # double 3.2091069999999999
	.quad	0x400d94080f98fa37              # double 3.6972809999999998
	.quad	0x40154dcc20d5629e              # double 5.3259740000000004
	.quad	0x400b250507a6bd6f              # double 3.3930760000000002
	.quad	0x3ffa3c9ff92f2b67              # double 1.6398010000000001
	.quad	0x4008d099e0e73605              # double 3.1018560000000002
	.quad	0x40124a7b9170d62c              # double 4.5727370000000001
	.quad	0x3ff420d887ebf22c              # double 1.258019
	.quad	0x40086f5b1c864884              # double 3.054373
	.quad	0x40212948344c37e7              # double 8.5806290000000001
	.quad	0x401c8d887ebf22c0              # double 7.1382159999999999
	.quad	0x40108421c044284e              # double 4.129035
	.quad	0x3feb3fa6defc7a3a              # double 0.85152000000000005
	.quad	0x4016d71715c63ae2              # double 5.7100489999999997
	.quad	0x3ff134e54f7a9197              # double 1.0754140000000001
	.quad	0x4011f31d712a0ec7              # double 4.4874169999999998
	.quad	0x3fe33621fafc8b00              # double 0.60035799999999995
	.quad	0x4005fd4d834091c1              # double 2.7486830000000002
	.quad	0x4013cbd72bcb5fe5              # double 4.9490629999999998
	.quad	0x400c22d72ffd1dcd              # double 3.5170119999999998
	.quad	0x40000f77f27fe4bd              # double 2.0075530000000001
	.quad	0x3ff28bac710cb296              # double 1.1591
	.quad	0x40233e4b662fdfc2              # double 9.6216690000000007
	.quad	0x3ff5c9635610adcd              # double 1.361667
	.quad	0x40195e2de8709742              # double 6.3419720000000002
	.quad	0x402109491f2dc2b1              # double 8.5181360000000002
	.quad	0x40104f251c193b3a              # double 4.0772899999999996
	.quad	0x402310509bf9c62a              # double 9.5318649999999998
	.quad	0x3ffb9b2e59af9ebf              # double 1.7253860000000001
	.quad	0x4008a49d388a8b09              # double 3.0803780000000001
	.quad	0x3fe0f80346dc5d64              # double 0.53027500000000005
	.quad	0x3ff848d6d3b6cbda              # double 1.5177830000000001
	.quad	0x401f3f151e75360d              # double 7.8116039999999999
	.quad	0x40219e47baa9b49a              # double 8.8091410000000003
	.quad	0x40133b51bd61f5be              # double 4.8079289999999997
	.quad	0x3fe9149c6f36ef80              # double 0.78376599999999996
	.quad	0x40021b03e20ccff2              # double 2.263191
	.quad	0x4023ecb9cb6848bf              # double 9.9623550000000005
	.quad	0x3ff683914f483cb0              # double 1.4071210000000001
	.quad	0x4011ae2f37fbefd0              # double 4.420102
	.quad	0x3fe0c35efa615a8e              # double 0.52384900000000001
	.quad	0x401c02e946801712              # double 7.0028430000000004
	.quad	0x4012acd74927913f              # double 4.6687900000000004
	.quad	0x4000205ab3aabcd8              # double 2.0157980000000002
	.quad	0x40227426dce39b45              # double 9.2268589999999993
	.quad	0x3fe4afff36ac6477              # double 0.64648399999999995
	.quad	0x40227fe7e1fc08fa              # double 9.2498159999999991
	.quad	0x402296672fba01ef              # double 9.2937560000000001
	.quad	0x40217c59d99029ae              # double 8.7428729999999995
	.quad	0x40104cc7d1bb4917              # double 4.0749810000000002
	.quad	0x3fe222a2c2374795              # double 0.56672800000000001
	.quad	0x4020cb3c1ce6c094              # double 8.3969430000000002
	.quad	0x401271e29b6b2af1              # double 4.6112159999999998
	.quad	0x3ff5102ff8ec0f88              # double 1.316452
	.quad	0x402351a52263d817              # double 9.6594630000000005
	.quad	0x401b39a7c17a8933              # double 6.8063039999999999
	.quad	0x40149549f94855da              # double 5.1457899999999999
	.quad	0x40157245f5ad96a7              # double 5.3615950000000003
	.quad	0x4012e87379746024              # double 4.7270029999999998
	.quad	0x400fd549731098d4              # double 3.9791439999999998
	.quad	0x402177b43d89ce4a              # double 8.7337969999999991
	.quad	0x40117faebc408d8f              # double 4.3746900000000002
	.quad	0x40204eeb702602c9              # double 8.1541399999999999
	.quad	0x401e981450efdc9c              # double 7.6485149999999997
	.quad	0x401044b5568e820e              # double 4.0670979999999997
	.quad	0x401a23e07a28bb0a              # double 6.5350359999999998
	.quad	0x401f377bbf93ff26              # double 7.8041830000000001
	.quad	0x3fdc374bc6a7ef9e              # double 0.44087500000000002
	.quad	0x402177a6defc7a3a              # double 8.7336950000000009
	.quad	0x3ffea034f3fd933e              # double 1.914113
	.quad	0x3ffe16191148fda0              # double 1.880395
	.quad	0x4010bec892ab68cf              # double 4.186312
	.quad	0x4013e94ab1d4f9c2              # double 4.977824
	.quad	0x4013e532e7b3d8e0              # double 4.973827
	.quad	0x4016c739b024f65a              # double 5.6945560000000004
	.quad	0x3ff01fe21d96e9bc              # double 1.007784
	.quad	0x4023788e581cf788              # double 9.7354610000000008
	.quad	0x400ac31a4bdba0a5              # double 3.3452649999999999
	.quad	0x400321dc93ea2d30              # double 2.391534
	.quad	0x401ae699c7bcc294              # double 6.7251960000000004
	.quad	0x3ffe0b1ab0856e69              # double 1.8777109999999999
	.quad	0x40176ebd9018e758              # double 5.8581450000000004
	.quad	0x400bd2e1693c03bc              # double 3.4779689999999999
	.quad	0x401cc5881a1554fc              # double 7.1929020000000001
	.quad	0x400e6bedfa43fe5d              # double 3.8027000000000002
	.quad	0x400bbc286f8ad256              # double 3.4668739999999998
	.quad	0x401518ba40d90e24              # double 5.2741480000000003
	.quad	0x400803486f049a99              # double 3.0016029999999998
	.quad	0x40150ab10ba62670              # double 5.2604410000000001
	.quad	0x401a542917507e9e              # double 6.5821880000000004
	.quad	0x3ff968c79f666235              # double 1.5880810000000001
	.quad	0x40160345cfede97d              # double 5.503196
	.quad	0x4018fa393ee5eedd              # double 6.2443590000000002
	.quad	0x401f02ff4ba51a00              # double 7.7529269999999997
	.quad	0x400995f5f0b28523              # double 3.198223
	.quad	0x4002ca619da9c993              # double 2.3488190000000002
	.quad	0x40064d0bb6ed6777              # double 2.78762
	.quad	0x4003b805e5f30e80              # double 2.464855
	.quad	0x40018822ff08893b              # double 2.1914729999999998
	.quad	0x40006a5a469d7343              # double 2.05193
	.quad	0x400ea34117f8444a              # double 3.8297140000000001
	.quad	0x401851bc98a222d5              # double 6.0798209999999999
	.quad	0x3ffe50b84988094e              # double 1.8947069999999999
	.quad	0x3ff900f238972043              # double 1.5627310000000001
	.quad	0x4000dcf0b6b6e0d2              # double 2.1078809999999999
	.quad	0x40232ae3c536d656              # double 9.5837690000000002
	.quad	0x401e0deb52c9d170              # double 7.5135930000000002
	.quad	0x400d8e9d51b4fe7a              # double 3.694636
	.quad	0x400bb7d6b65a9a80              # double 3.4647649999999999
	.quad	0x4023da2ab25b317f              # double 9.926107
	.quad	0x3ff29c5931ca7d67              # double 1.163171
	.quad	0x400eb075b3e1437c              # double 3.8361619999999998
	.quad	0x3fe1004102ff8ec1              # double 0.531281
	.quad	0x4002542917507e9e              # double 2.2910940000000002
	.quad	0x400bbb4f61672325              # double 3.4664600000000001
	.quad	0x40088a4228997c3d              # double 3.0675089999999998
	.quad	0x400d8f544bb1af3a              # double 3.694985
	.quad	0x3fcc28e4fb97bb73              # double 0.219998
	.quad	0x4000f54da4ce8102              # double 2.119777
	.quad	0x4006fcc4ac6cdaf5              # double 2.8734220000000001
	.quad	0x3fe09d3cbc48f10b              # double 0.51919400000000004
	.quad	0x4018fcdaf4adbc66              # double 6.2469289999999997
	.quad	0x4015d6006d0d4995              # double 5.4589860000000003
	.quad	0x40230089e34330d7              # double 9.5010519999999996
	.quad	0x402181774256bca5              # double 8.7528629999999996
	.quad	0x4002ca48f96e1587              # double 2.3487719999999999
	.quad	0x401d34f3ba775fb3              # double 7.3017110000000001
	.quad	0x401b938088509bfa              # double 6.8940450000000002
	.quad	0x3ff5c0746455eaee              # double 1.359486
	.quad	0x40000fb09203a323              # double 2.0076610000000001
	.quad	0x3fdd9b90ea9e6eeb              # double 0.46261999999999998
	.quad	0x3fc491b823c85c25              # double 0.16069700000000001
	.quad	0x400384873365881a              # double 2.439711
	.quad	0x401182f512663411              # double 4.3778879999999996
	.quad	0x3ffe63bddfc9ff93              # double 1.899351
	.quad	0x40200fd36f7e3d1d              # double 8.0309100000000004
	.quad	0x3fef8b04ab606b7b              # double 0.98572000000000004
	.quad	0x3ff02a0ba1f4b1ee              # double 1.010265
	.quad	0x401fc0257c914b16              # double 7.9376429999999996
	.quad	0x3ffefbeec39799e5              # double 1.936507
	.quad	0x400c595aaf78feef              # double 3.5436299999999998
	.quad	0x4021471ac14c660a              # double 8.6388759999999998
	.quad	0x4005404295a6c5d2              # double 2.656377
	.quad	0x4021591ada76d97b              # double 8.6740329999999997
	.quad	0x4017c82b1f687b14              # double 5.9454770000000003
	.quad	0x3fdab1dd5d3dc8b8              # double 0.41710599999999998
	.quad	0x4011a15e39713ad6              # double 4.4075860000000002
	.quad	0x401e0b73083558a7              # double 7.5111809999999996
	.quad	0x3ffde11233df2a9d              # double 1.8674489999999999
	.quad	0x4010f32c625e99e9              # double 4.2374739999999997
	.quad	0x401e7afd976ff3ae              # double 7.6201080000000001
	.quad	0x3ffacf312b1b36bd              # double 1.675584
	.quad	0x40139043a2163fdd              # double 4.8908829999999996
	.quad	0x40239ce54f7a9197              # double 9.8064370000000007
	.quad	0x400cd2e2fbe33acd              # double 3.6029719999999998
	.quad	0x4004573083558a76              # double 2.542573
	.quad	0x400b28cd63cb8173              # double 3.3949229999999999
	.quad	0x3ffce2bbecaab8a6              # double 1.805355
	.quad	0x401f5f6555c52e73              # double 7.8431600000000001
	.quad	0x4013b8a6dacabc51              # double 4.9303239999999997
	.quad	0x4012472799a1fd15              # double 4.5694869999999996
	.quad	0x400a682be8bc169c              # double 3.3008649999999999
	.quad	0x3ff7f27aa68f4b62              # double 1.496699
	.quad	0x3fc9703f2d3c792d              # double 0.198738
	.quad	0x40006c35ce182667              # double 2.0528369999999998
	.quad	0x401a6a32b12d3416              # double 6.6037090000000003
	.quad	0x4016b6f32bdc26dd              # double 5.6786620000000001
	.quad	0x402121a459103c8e              # double 8.5657069999999997
	.quad	0x400b4185058dde7a              # double 3.4069919999999998
	.quad	0x40212221426fe719              # double 8.5666600000000006
	.quad	0x3fd377f6b1a2a4db              # double 0.304197
	.quad	0x401bdc536d655e29              # double 6.9651620000000003
	.quad	0x401041b5c7cd898b              # double 4.0641699999999998
	.quad	0x3ff7e60f5e41d4b7              # double 1.4936670000000001
	.quad	0x40205ef7c243dccf              # double 8.1854840000000006
	.quad	0x400542ecaeea63b7              # double 2.6576780000000002
	.quad	0x3fd8808c8259e1f4              # double 0.38284600000000002
	.quad	0x40205a1ab4b72c52              # double 8.1759850000000007
	.quad	0x3ff16708ede54b49              # double 1.087655
	.quad	0x401257b7c7820a31              # double 4.5856620000000001
	.quad	0x40035cd5f99c38b0              # double 2.4203299999999999
	.quad	0x4010568405b39e28              # double 4.0844880000000003
	.quad	0x400068f08461f9f0              # double 2.05124
	.quad	0x40185940fecdd0d9              # double 6.0871620000000002
	.quad	0x40216a7e73a365cb              # double 8.7079959999999996
	.quad	0x3ff0a161e4f765fe              # double 1.0394000000000001
	.quad	0x401b218a86d71f36              # double 6.7827549999999999
	.quad	0x4022a210e0221427              # double 9.316535
	.quad	0x4023fd2a62aa1944              # double 9.9944640000000006
	.quad	0x3fff4fede97d06bc              # double 1.957014
	.quad	0x401fa512231832fd              # double 7.9112020000000002
	.quad	0x400c1cf9a06a6e33              # double 3.514148
	.quad	0x3ff0aeb1c432ca58              # double 1.0426500000000001
	.quad	0x4018149f0e4da09d              # double 6.0201380000000002
	.quad	0x4010b3d35eb7457c              # double 4.175611
	.quad	0x4017314a0a0f4d7b              # double 5.7981340000000001
	.quad	0x40062203e63e8dda              # double 2.7666089999999999
	.quad	0x400453953dea465a              # double 2.5408119999999998
	.quad	0x401b9642bf9830e4              # double 6.8967390000000002
	.quad	0x4016be279dd3bafe              # double 5.6856980000000004
	.quad	0x400aba7221858bc6              # double 3.3410380000000002
	.quad	0x3fddfb19e731d2e1              # double 0.46845100000000001
	.quad	0x4008578ab0c88a48              # double 3.042745
	.quad	0x4020edfdac68a937              # double 8.4648260000000004
	.quad	0x3ff63ea0ba1f4b1f              # double 1.39029
	.quad	0x4001c0d0aaa7ded7              # double 2.2191480000000001
	.quad	0x4023e5ef416bdb1a              # double 9.9490909999999992
	.quad	0x4011fcf56eac8605              # double 4.4970299999999996
	.quad	0x4001c73bc903ea70              # double 2.2222819999999999
	.quad	0x4005ca2fcefaa476              # double 2.7237239999999998
	.quad	0x40032d79d0a67621              # double 2.397205
	.quad	0x400c83b688bda943              # double 3.5643129999999998
	.quad	0x40112d3b29a1b9fa              # double 4.2941710000000004
	.quad	0x3fdccc7d1bb4916d              # double 0.44998100000000002
	.quad	0x4018f189bd8383ae              # double 6.2358770000000003
	.quad	0x4020c5c31593e5fb              # double 8.3862539999999992
	.quad	0x402201a005c4651f              # double 9.0031739999999996
	.quad	0x401193d577963993              # double 4.3943690000000002
	.quad	0x4023db6bf444e0db              # double 9.9285580000000006
	.quad	0x3ff991d57ff9b563              # double 1.598104
	.quad	0x4012f4bad7d7c2ca              # double 4.7389939999999999
	.quad	0x40153922d948dc12              # double 5.3057970000000001
	.quad	0x400c639ffd60e94f              # double 3.548645
	.quad	0x402235d206c8711d              # double 9.1051179999999992
	.quad	0x40142f6b1a2a4db1              # double 5.0463069999999997
	.quad	0x401be886594af4f1              # double 6.9770750000000001
	.quad	0x3ffc41450efdc9c5              # double 1.765935
	.quad	0x4018c7ba237cdcca              # double 6.1950459999999996
	.quad	0x401dca2e7f6f4be8              # double 7.4474429999999998
	.quad	0x400888711d798d8b              # double 3.0666220000000002
	.quad	0x401011bd1ed9dfdb              # double 4.0173230000000002
	.quad	0x400ed1550ca1cef2              # double 3.8522129999999999
	.quad	0x400aeddbdb5d8948              # double 3.366142
	.quad	0x4021d4bdba0a5269              # double 8.9155099999999994
	.quad	0x4023135371971c11              # double 9.5377460000000002
	.quad	0x3fc1e6bc621b7e0b              # double 0.13985400000000001
	.quad	0x401b255d1c3ac92a              # double 6.7864880000000003
	.quad	0x3fe66d1e108c3f3e              # double 0.70082
	.quad	0x401741bc55864452              # double 5.8141949999999998
	.quad	0x3ff8c2ff8ec0f883              # double 1.547607
	.quad	0x401a52f2734f82f5              # double 6.5810029999999999
	.quad	0x401eff79c842fa51              # double 7.7494880000000003
	.quad	0x4010b26bb55ac040              # double 4.174239
	.quad	0x3fe2125fe974a340              # double 0.56474299999999999
	.quad	0x401499bb6aa4b988              # double 5.1501289999999997
	.quad	0x3fdd2e3c536d655e              # double 0.45594699999999999
	.quad	0x401840f94c87980f              # double 6.0634509999999997
	.quad	0x401a759b8023a6ce              # double 6.6148509999999998
	.quad	0x400516fda836eb4f              # double 2.6362260000000002
	.quad	0x400b123c42a66dbd              # double 3.3839039999999998
	.quad	0x40058086bdf4c2b5              # double 2.687757
	.quad	0x401af95182a9930c              # double 6.7434750000000001
	.quad	0x401402bc2fc69729              # double 5.0026710000000003
	.quad	0x4020ca13e3e29308              # double 8.3946830000000006
	.quad	0x3fd533a8a3f8982d              # double 0.33127800000000002
	.quad	0x4002af4cf4a558ea              # double 2.3355959999999998
	.quad	0x400b8ad794a6eb92              # double 3.4427940000000001
	.quad	0x400991c8eabffcdb              # double 3.1961840000000001
	.quad	0x400008d002e23290              # double 2.0043030000000002
	.quad	0x4022e9d7ba6698bb              # double 9.4567239999999994
	.quad	0x401c624b33daf8df              # double 7.0959899999999996
	.quad	0x4023e126a65cf67b              # double 9.9397479999999998
	.quad	0x4022c5e8d5410f95              # double 9.3865420000000003
	.quad	0x3fff7f22c01e68a1              # double 1.968539
	.quad	0x3ff4113e81450efe              # double 1.25421
	.quad	0x401bfcaf709b738e              # double 6.9967629999999996
	.quad	0x400086d04e618ce3              # double 2.0658270000000001
	.quad	0x3ff76aec0724b76f              # double 1.4636039999999999
	.quad	0x401ff54823854046              # double 7.9895329999999998
	.quad	0x400a2c2ce4649907              # double 3.2715700000000001
	.quad	0x4013fb02d59d55e7              # double 4.9951280000000002
	.quad	0x4015afaacd9e83e4              # double 5.4215499999999999
	.quad	0x4022044d65e46067              # double 9.0084029999999995
	.quad	0x40237450acc0bdcb              # double 9.7271780000000003
	.quad	0x4021b76ff3ade226              # double 8.858276
	.quad	0x4010d90c4dec1c1d              # double 4.2119609999999996
	.quad	0x40237207f66e86c6              # double 9.7227169999999994
	.quad	0x4003041dd1a21ea3              # double 2.3770099999999998
	.quad	0x40002fa615a8deb1              # double 2.023266
	.quad	0x401af25bfad29995              # double 6.7366789999999996
	.quad	0x4015a40e9d51b4fe              # double 5.4102119999999996
	.quad	0x3fa8ff327aa68f4b              # double 0.048821999999999997
	.quad	0x402033bf93ff25e5              # double 8.1010709999999992
	.quad	0x4017ccdd50a88f00              # double 5.9500630000000001
	.quad	0x4008c159817b95a3              # double 3.0944090000000002
	.quad	0x40030e69f05ea24d              # double 2.3820380000000001
	.quad	0x3ffed56c0d6f544c              # double 1.9271050000000001
	.quad	0x3fc162413db7f173              # double 0.13581099999999999
	.quad	0x3ffe859641f64495              # double 1.9076139999999999
	.quad	0x4014da1c25d07421              # double 5.2129979999999998
	.quad	0x4006539da16616b5              # double 2.7908279999999999
	.quad	0x401aeb1feeb2d0a2              # double 6.7296139999999998
	.quad	0x3fedd959a30984e4              # double 0.932782
	.quad	0x4008aba0e8427419              # double 3.0838030000000001
	.quad	0x401f8bfa0945f9df              # double 7.8866959999999997
	.quad	0x401d37a99fa11a97              # double 7.3043579999999997
	.quad	0x401d93ba3443d46b              # double 7.3942649999999999
	.quad	0x401842630ec31162              # double 6.0648309999999999
	.quad	0x400ec130164840e1              # double 3.8443299999999998
	.quad	0x401f11b866e43aa8              # double 7.7673050000000003
	.quad	0x401e848cb4aec8d6              # double 7.6294430000000002
	.quad	0x401735eb7457c0b1              # double 5.8026559999999998
	.quad	0x4007992641b328b7              # double 2.9497800000000001
	.quad	0x4013e8b5cbff4773              # double 4.9772559999999997
	.quad	0x4013520c8cd63cb8              # double 4.8301259999999999
	.quad	0x400bdf81a5870da6              # double 3.4841340000000001
	.quad	0x40067a4c61d8622c              # double 2.8097159999999999
	.quad	0x40142fc371da37ef              # double 5.0466439999999997
	.quad	0x3fcb4927913e8145              # double 0.21317
	.quad	0x4002772367e414e8              # double 2.308173
	.quad	0x400477082491afc0              # double 2.5581209999999999
	.quad	0x40036779a6b50b0f              # double 2.4255249999999999
	.quad	0x3ff60bfe3b03e20d              # double 1.377928
	.quad	0x401775529fe004b8              # double 5.864573
	.quad	0x4007f2ff4ba51a00              # double 2.9936509999999998
	.quad	0x4021f6bf65dbfceb              # double 8.9819289999999992
	.quad	0x400e2ed6777079e6              # double 3.7728700000000002
	.quad	0x4005aecb74ddf86e              # double 2.7103489999999999
	.quad	0x40162fa8f7db6e50              # double 5.5465429999999998
	.quad	0x3ffb943d03968d76              # double 1.7236910000000001
	.quad	0x4023b5d3dc8b86b1              # double 9.8551319999999993
	.quad	0x4009003686a4ca4f              # double 3.1251039999999999
	.quad	0x400541fa76534374              # double 2.657216
	.quad	0x4004fd8e86402081              # double 2.6238069999999998
	.quad	0x4022a7ae5796bfcb              # double 9.3275020000000008
	.quad	0x400791233df2a9d6              # double 2.9458679999999999
	.quad	0x401b3cf4e874c900              # double 6.8095280000000002
	.quad	0x3fe335a858793dd9              # double 0.60029999999999994
	.quad	0x4021fc3c7d5ed070              # double 8.9926490000000001
	.quad	0x40104074213a0c6b              # double 4.0629429999999997
	.quad	0x402278b545c78a6e              # double 9.2357580000000006
	.quad	0x401200f840181e04              # double 4.500947
	.quad	0x402106050374ff86              # double 8.5117569999999994
	.quad	0x4000aaac1094a2ba              # double 2.0833360000000001
	.quad	0x400d2bae89eba6ad              # double 3.6463290000000002
	.quad	0x4000ea4441787054              # double 2.1143879999999999
	.quad	0x4012f92641b328b7              # double 4.7433100000000001
	.quad	0x3ff0ab09635610ae              # double 1.041757
	.quad	0x4015b01c4fc1df33              # double 5.421983
	.quad	0x3fdd4fb1183b6028              # double 0.45798899999999998
	.quad	0x4021308ce703afb8              # double 8.5948250000000002
	.quad	0x3feb237ef5a964e9              # double 0.84808300000000003
	.quad	0x40218262aa19439e              # double 8.7546590000000002
	.quad	0x401c7efb6dca07f6              # double 7.1240059999999996
	.quad	0x4018ded56b00ffda              # double 6.2176109999999998
	.quad	0x4021516c3fc43b2e              # double 8.6590290000000003
	.quad	0x4022c8c111ada76e              # double 9.3920980000000007
	.quad	0x4011853f7ced9168              # double 4.3801249999999996
	.quad	0x40233696a26e5471              # double 9.6066179999999992
	.quad	0x40059cbb3c81908e              # double 2.7015289999999998
	.quad	0x402383cfb3311a54              # double 9.7574439999999995
	.quad	0x401926916440f239              # double 6.2876640000000004
	.quad	0x4023d5d8b60f1b26              # double 9.9176690000000001
	.quad	0x40206a4c8366516e              # double 8.2076150000000005
	.quad	0x3f81c864883fd502              # double 0.0086829999999999997
	.quad	0x400fb9806f262889              # double 3.9655770000000001
	.quad	0x401d2e34b945308c              # double 7.2951230000000002
	.quad	0x4007a51eb851eb85              # double 2.9556249999999999
	.quad	0x401264c122749f0e              # double 4.5983929999999997
	.quad	0x40146ba27ae9ab2a              # double 5.1051120000000001
	.quad	0x4020f8941c8216c6              # double 8.4855049999999999
	.quad	0x401351d81f106680              # double 4.8299260000000004
	.quad	0x3ff0a6e86c6583e8              # double 1.0407489999999999
	.quad	0x4019324febd09e13              # double 6.2991330000000003
	.quad	0x4021dc9e8815e397              # double 8.9308969999999999
	.quad	0x4013b05dd8f92afa              # double 4.9222330000000003
	.quad	0x40197dc8323358f3              # double 6.3728340000000001
	.quad	0x40137306a2b17050              # double 4.86233
	.quad	0x4021c6e04c059210              # double 8.8884299999999996
	.quad	0x3ff832452c59fb1e              # double 1.512273
	.quad	0x4016efb480a5accd              # double 5.7340869999999997
	.quad	0x401e172bcb5fe543              # double 7.522628
	.quad	0x40225ec8b4395810              # double 9.1851249999999993
	.quad	0x401c306e5cd4ed2d              # double 7.0472960000000002
	.quad	0x3fec74a771c970f8              # double 0.88924000000000003
	.quad	0x400cbd0aed99cbef              # double 3.5923060000000002
	.quad	0x4019566f0cfe1544              # double 6.3344079999999998
	.quad	0x401c91c10d7be985              # double 7.1423379999999996
	.quad	0x401c7841248d7e02              # double 7.1174359999999997
	.quad	0x401235e02ea960b7              # double 4.552613
	.quad	0x3fd26fb8f57f737e              # double 0.28806900000000002
	.quad	0x40180966be7afa72              # double 6.0091809999999999
	.quad	0x4004500e6afcce1c              # double 2.5390899999999998
	.quad	0x400d92d44dca8e2e              # double 3.6966939999999999
	.quad	0x4018d41ede1198af              # double 6.2071490000000002
	.quad	0x40108c471b478423              # double 4.1369899999999999
	.quad	0x400a868b9fdbd2fa              # double 3.315696
	.quad	0x4016f2817763e4ac              # double 5.7368220000000001
	.quad	0x3ff7b3892ee84ad8              # double 1.4813320000000001
	.quad	0x4015d71f36262cba              # double 5.4600799999999996
	.quad	0x401aff52fc2656ac              # double 6.7493400000000001
	.quad	0x40125355cd91eeaa              # double 4.5813819999999996
	.quad	0x40236a1f080303c0              # double 9.7072679999999991
	.quad	0x401b95e679463cfb              # double 6.8963869999999998
	.quad	0x4019780b242070b9              # double 6.3672300000000002
	.quad	0x4018e798958d9b5f              # double 6.2261680000000004
	.quad	0x4002b909f1f14984              # double 2.3403510000000001
	.quad	0x4001e4dca8e2e2b9              # double 2.2367490000000001
	.quad	0x4021f35653c9abb0              # double 8.9752679999999998
	.quad	0x4014582517e77d52              # double 5.0860789999999998
	.quad	0x401c97ef9db22d0e              # double 7.1483749999999997
	.quad	0x4005593469941850              # double 2.6685569999999998
	.quad	0x4010eced0b30b5aa              # double 4.2313729999999996
	.quad	0x3ff2a47a9e2bcf92              # double 1.1651560000000001
	.quad	0x3ff370b6b6e0d23d              # double 1.2150179999999999
	.quad	0x401ffcbab649d389              # double 7.9968060000000003
	.quad	0x4012f49491f2dc2b              # double 4.7388479999999999
	.quad	0x402328b1dd5d3dc9              # double 9.5794820000000004
	.quad	0x402078b780346dc6              # double 8.2357750000000003
	.quad	0x400ccfe8ab4fa483              # double 3.601518
	.quad	0x4023660b0af5fd48              # double 9.6993030000000004
	.quad	0x40164e4b87bdcf03              # double 5.57646
	.quad	0x3ff810c4dec1c1d7              # double 1.504094
	.quad	0x3f69ce075f6fd220              # double 0.00315
	.quad	0x4019a5f91600f345              # double 6.412083
	.quad	0x3ff62055757d5a9f              # double 1.3828940000000001
	.quad	0x3ff30266772d5e07              # double 1.188086
	.quad	0x401d49ab6d00b45b              # double 7.3219430000000001
	.quad	0x40151f71eafee6fb              # double 5.2807079999999997
	.quad	0x4022193a4f8726d0              # double 9.0492729999999995
	.quad	0x402136ca46e08f21              # double 8.6070119999999992
	.quad	0x400ea4c942d490e6              # double 3.8304619999999998
	.quad	0x400304c51116a8b9              # double 2.377329
	.quad	0x40214c15f45e0b4e              # double 8.6486049999999999
	.quad	0x401abd3804534bd7              # double 6.6847839999999996
	.quad	0x3ff08956c0d6f545              # double 1.0335300000000001
	.quad	0x400c37edc7ef177a              # double 3.5273089999999998
	.quad	0x3ff5eaf251c193b4              # double 1.3698600000000001
	.quad	0x3ff6278c0053e2d6              # double 1.384655
	.quad	0x4016f2b734b51373              # double 5.7370270000000003
	.quad	0x4020c2cc2d6a9c56              # double 8.3804639999999999
	.quad	0x40206eef8055fbb5              # double 8.2166709999999998
	.quad	0x401f9bdf8f473040              # double 7.9022199999999998
	.quad	0x401cc29f9ce8d973              # double 7.1900620000000002
	.quad	0x3fe1e3b688bda943              # double 0.55904699999999996
	.quad	0x4019a0a02b841249              # double 6.4068610000000001
	.quad	0x4001efbfc6540cc8              # double 2.2420650000000002
	.quad	0x401466616b54e2b0              # double 5.0999809999999997
	.quad	0x40239b1c64ba94bc              # double 9.8029510000000001
	.quad	0x3ff0b3c81908e582              # double 1.043892
	.quad	0x3fea11ab8e8ea39c              # double 0.81465699999999996
	.quad	0x4004b1a75cd0bb6f              # double 2.5867450000000001
	.quad	0x40194eeed8904f6e              # double 6.327083
	.quad	0x40173c3b4f616723              # double 5.8088199999999999
	.quad	0x400b02856605ee57              # double 3.3762310000000002
	.quad	0x4010e8c75c4a83b2              # double 4.2273230000000002
	.quad	0x4005e60ed80a17b1              # double 2.737333
	.quad	0x40167bc3c5bd0e13              # double 5.6208640000000001
	.quad	0x3fe45ef1fddebd90              # double 0.63658999999999999
	.quad	0x3fdd522ea0fd3b2a              # double 0.45814100000000002
	.quad	0x4020ea00e27e0efa              # double 8.4570380000000007
	.quad	0x3ffc675147f13059              # double 1.7752239999999999
	.quad	0x3ff05707182235b5              # double 1.021247
	.quad	0x3ffc3f06705c896e              # double 1.765387
	.quad	0x401f6ca148ba83f5              # double 7.856084
	.quad	0x4021ab59ddc1e796              # double 8.8346699999999991
	.quad	0x402023dc69f8c21e              # double 8.0700409999999998
	.quad	0x3fde8ec0f88333fd              # double 0.47746300000000003
	.quad	0x3fff4a372606fac6              # double 1.955619
	.quad	0x4013eb09e98dcdb3              # double 4.9795299999999996
	.quad	0x4012b3b42cc2d6aa              # double 4.6754920000000002
	.quad	0x40119ca148ba83f5              # double 4.4029590000000001
	.quad	0x40227c204f2ae07e              # double 9.2424339999999994
	.quad	0x400af052934acaff              # double 3.3673449999999998
	.quad	0x400030856e696a27              # double 2.023692
	.quad	0x4021d7e08aefb2ab              # double 8.9216350000000002
	.quad	0x40220499f1ae2da5              # double 9.0089869999999994
	.quad	0x401de29b280f12c2              # double 7.4712949999999996
	.quad	0x401b4c53f39d1b2e              # double 6.8245389999999997
	.quad	0x4023413d96638434              # double 9.6274230000000002
	.quad	0x401444812be48a59              # double 5.0668990000000003
	.quad	0x4014a1ff7164c72a              # double 5.158201
	.quad	0x402076c6de76427c              # double 8.2319859999999991
	.quad	0x3ff6162ae4b01861              # double 1.380412
	.quad	0x400be855da272863              # double 3.488445
	.quad	0x401b1f9acffa7eb7              # double 6.7808640000000002
	.quad	0x4022493db7f17375              # double 9.1430489999999995
	.quad	0x4017c40746455eaf              # double 5.9414340000000001
	.quad	0x3ff8b71971c10d7c              # double 1.544702
	.quad	0x40210df84cad57bc              # double 8.5272849999999991
	.quad	0x400d8051c9f72f77              # double 3.687656
	.quad	0x4019cc5974e65bea              # double 6.44956
	.quad	0x40076d1f1cfbb949              # double 2.9282819999999998
	.quad	0x401c28d08919ef95              # double 7.0398579999999997
	.quad	0x3fb50f94c87980f5              # double 0.082268999999999995
	.quad	0x3ffee15db3397dd0              # double 1.930021
	.quad	0x40059425f202107b              # double 2.6973379999999998
	.quad	0x3ff1eab042528ae7              # double 1.1197969999999999
	.quad	0x40162884c6a3bde0              # double 5.5395690000000002
	.quad	0x401e001f75104d55              # double 7.5001199999999998
	.quad	0x3ffb6e3e6c4c5975              # double 1.714415
	.quad	0x400ed51a437824d5              # double 3.8540540000000001
	.quad	0x40188bf93ff25e57              # double 6.1366930000000002
	.quad	0x4014a3e468cac4b5              # double 5.1600510000000002
	.quad	0x4014fa3f034b0e1b              # double 5.2443809999999997
	.quad	0x40181495e17e34b9              # double 6.0201029999999998
	.quad	0x3ff292bd3c361134              # double 1.160825
	.quad	0x4006d462c343b70f              # double 2.853704
	.quad	0x3ffbcf345069a4df              # double 1.7380869999999999
	.quad	0x401749949e8815e4              # double 5.8218560000000004
	.quad	0x4018f49f94855da2              # double 6.2388899999999996
	.quad	0x4020d1142b302f73              # double 8.4083570000000005
	.quad	0x4016a9d99029ae4f              # double 5.6658689999999998
	.quad	0x4018bfdb4cc25072              # double 6.18736
	.quad	0x401ad8ea39c51dac              # double 6.7118310000000001
	.quad	0x40000b48d3ae685e              # double 2.0055100000000001
	.quad	0x4011da32b12d3416              # double 4.4630840000000003
	.quad	0x401270d6f544bb1b              # double 4.610195
	.quad	0x400f6770bd017daf              # double 3.9255080000000002
	.quad	0x400e5157abb8800f              # double 3.7897180000000001
	.quad	0x401405e8d5410f95              # double 5.0057710000000002
	.quad	0x3ff36a582dbe7f2c              # double 1.213463
	.quad	0x400e624e59296702              # double 3.7980010000000002
	.quad	0x4015ae631f8a0903              # double 5.4203000000000001
	.quad	0x3fedc96cc5fbf834              # double 0.93083800000000005
	.quad	0x3ff33897204295a7              # double 1.2013160000000001
	.quad	0x4013377d955714ba              # double 4.8041900000000002
	.quad	0x3febdf5ad96a6a01              # double 0.87101499999999998
	.quad	0x40109e3cd9a52264              # double 4.1545290000000001
	.quad	0x3fbb07bfe7e1fc09              # double 0.105587
	.quad	0x402315515054ac2a              # double 9.5416360000000005
	.quad	0x401ee849cb252ce0              # double 7.7268439999999998
	.quad	0x40083260f5e41d4b              # double 3.0245989999999998
	.quad	0x40201339e279dd3c              # double 8.0375510000000005
	.quad	0x401db9efd86a8fc1              # double 7.4315790000000002
	.quad	0x4022b208398a6549              # double 9.3477189999999997
	.quad	0x4021ffad6cb53501              # double 8.9993700000000008
	.quad	0x3ff6c095f2452c5a              # double 1.422018
	.quad	0x401aed0917d6b65b              # double 6.7314800000000004
	.quad	0x401479d9d3458cd2              # double 5.118995
	.quad	0x3feecc22ee41919b              # double 0.96241900000000002
	.quad	0x3faaeb80ecfa69be              # double 0.052578
	.quad	0x40122edc7ef177a7              # double 4.545763
	.quad	0x400e12e8c0485a0c              # double 3.759233
	.quad	0x4023f57eaa2a0a96              # double 9.9794820000000009
	.quad	0x400785c5f7c6759b              # double 2.9403190000000001
	.quad	0x401ba08461f9f01c              # double 6.9067550000000004
	.quad	0x402105729b280f13              # double 8.5106400000000004
	.quad	0x3fda04295a6c5d20              # double 0.40650399999999998
	.quad	0x40041851654d61b3              # double 2.5118740000000002
	.quad	0x3fdc83621fafc8b0              # double 0.445519
	.quad	0x40209753c579f234              # double 8.2955609999999992
	.quad	0x4016770b49e01de2              # double 5.6162539999999996
	.quad	0x40221144aa53fc01              # double 9.0337270000000007
	.quad	0x401267e28240b780              # double 4.6014499999999998
	.quad	0x401ea68f08461f9f              # double 7.6626549999999999
	.quad	0x4011e44806290eed              # double 4.472931
	.quad	0x4019352ef911cf35              # double 6.3019369999999997
	.quad	0x3fd0005814940bbb              # double 0.25002099999999999
	.quad	0x40221ed04e618ce3              # double 9.0601830000000003
	.quad	0x4022827ef9db22d1              # double 9.2548750000000002
	.quad	0x401fb12620253975              # double 7.9229969999999996
	.quad	0x401790ea9e6eeb70              # double 5.8915199999999999
	.quad	0x40053ce63a5c1c61              # double 2.6547360000000002
	.quad	0x401532a0ec743201              # double 5.299442
	.quad	0x4011e9dd3bafd977              # double 4.478383
	.quad	0x3fc00de4c51116a9              # double 0.12542400000000001
	.quad	0x40212bd102bc72e2              # double 8.5855789999999992
	.quad	0x401b88066c2acb86              # double 6.8828370000000003
	.quad	0x3ff4859c8c9320da              # double 1.2826200000000001
	.quad	0x3ff556909aed56b0              # double 1.333634
	.quad	0x40193be4473cd573              # double 6.3084879999999997
	.quad	0x40200de353f7ced9              # double 8.0271249999999998
	.quad	0x401ef89b951c5c57              # double 7.7427809999999999
	.quad	0x40237cbb1af3a14d              # double 9.7436150000000001
	.quad	0x401607fdc5931ca8              # double 5.5078040000000001
	.quad	0x401e59baa1511e00              # double 7.5876260000000002
	.quad	0x400b475ee99a62ed              # double 3.4098489999999999
	.quad	0x401c1c7c0f451761              # double 7.0278169999999998
	.quad	0x4005cf4384ba0e84              # double 2.7262029999999999
	.quad	0x3ff264de3b8a19ca              # double 1.149626
	.quad	0x40083b84db9c7369              # double 3.0290620000000001
	.quad	0x401f13201040bfe4              # double 7.7686770000000003
	.quad	0x401b83059641f645              # double 6.8779510000000004
	.quad	0x4012df6ad70e6f2f              # double 4.7181810000000004
	.quad	0x401fbebbba55d1c4              # double 7.9362630000000003
	.quad	0x3ff2b95703f2d3c8              # double 1.1702490000000001
	.quad	0x40113291fb3fa6df              # double 4.299385
	.quad	0x401f045c358afc48              # double 7.754258
	.quad	0x400c2f7b9e060fe4              # double 3.5231849999999998
	.quad	0x4000d4141e9af5ba              # double 2.1035539999999999
	.quad	0x3fd349ecb31c219f              # double 0.30138700000000002
	.quad	0x40077c4d22c881e4              # double 2.9356939999999998
	.quad	0x3f9a0232096787cf              # double 0.025399000000000001
	.quad	0x4022dbfaf42784a9              # double 9.4296489999999995
	.quad	0x40203857d1782d38              # double 8.1100449999999995
	.quad	0x4015c23033a4723b              # double 5.4396370000000003
	.quad	0x40236a90abb44e51              # double 9.7081350000000004
	.quad	0x3ffb8f7fcfc3f812              # double 1.722534
	.quad	0x3fe0780fdc1615ec              # double 0.514656
	.quad	0x3ff5452b4d8ba40e              # double 1.3293870000000001
	.quad	0x4014109f623076c0              # double 5.0162329999999997
	.quad	0x401e6e1693c03bc5              # double 7.6075080000000001
	.quad	0x401674a01abd1aa8              # double 5.6138919999999999
	.quad	0x3fbc5931ca7d6734              # double 0.110736
	.quad	0x3feba07b352a8438              # double 0.86334
	.quad	0x4005b38de6ddf00b              # double 2.7126730000000001
	.quad	0x400c2432441fea81              # double 3.517674
	.quad	0x401cf893faf42785              # double 7.2427520000000003
	.quad	0x401e44067cf1c326              # double 7.5664309999999997
	.quad	0x40208ce160d887ec              # double 8.275157
	.quad	0x4002dea140570825              # double 2.3587060000000002
	.quad	0x4007d4b838c111ae              # double 2.9788670000000002
	.quad	0x40000b26394face6              # double 2.0054439999999998
	.quad	0x3fffc4d013a92a30              # double 1.9855499999999999
	.quad	0x4010e289dadfb507              # double 4.2212290000000001
	.quad	0x3ff25b691212513b              # double 1.1473169999999999
	.quad	0x3fc864ece9a2c669              # double 0.19058
	.quad	0x4021648e8a71de6a              # double 8.6964000000000006
	.quad	0x401dc4bd33d29564              # double 7.4421280000000003
	.quad	0x40196f88765ba6f0              # double 6.3589190000000002
	.quad	0x3ff0f88fc9363f57              # double 1.060684
	.quad	0x40139483cafb3b75              # double 4.8950339999999999
	.quad	0x3ff57413986338b4              # double 1.3408389999999999
	.quad	0x402074a8e2e2b8c7              # double 8.2278509999999994
	.quad	0x3fe67b480a5accd5              # double 0.70254899999999998
	.quad	0x4023d80d1b71758e              # double 9.9219749999999998
	.quad	0x40184de481f53826              # double 6.0760670000000001
	.quad	0x3fef150331e3a7db              # double 0.97131500000000003
	.quad	0x3fcdf2990f301eac              # double 0.23396600000000001
	.quad	0x3feffa1c25d07421              # double 0.99928099999999997
	.quad	0x40197a6dacabc515              # double 6.3695589999999997
	.quad	0x40019bd7b2031ceb              # double 2.201095
	.quad	0x3ff62a550870110a              # double 1.385335
	.quad	0x3fdea5a038194c01              # double 0.47885899999999998
	.quad	0x401bd69270b06c44              # double 6.959543
	.quad	0x4022cc08fa7a8501              # double 9.3985059999999993
	.quad	0x401ba341e14bdfd2              # double 6.9094309999999997
	.quad	0x400ee10be9424e59              # double 3.8598859999999999
	.quad	0x40125ebc83a96d4c              # double 4.5925159999999998
	.quad	0x40223cb0e1b4bb5e              # double 9.1185369999999999
	.quad	0x40237371758e2196              # double 9.7254749999999994
	.quad	0x400852f33ca31e7e              # double 3.0405030000000002
	.quad	0x4011f1e108c3f3e0              # double 4.4862099999999998
	.quad	0x4015e7c88e79aae7              # double 5.4763510000000002
	.quad	0x401dbad5bee3d5fe              # double 7.4324560000000002
	.quad	0x40225625c810a56a              # double 9.1682570000000005
	.quad	0x402372749f0e4da1              # double 9.7235460000000006
	.quad	0x401549fadafd1138              # double 5.3222459999999998
	.quad	0x4023b25aab474107              # double 9.8483479999999996
	.quad	0x400476ebd4cfd08d              # double 2.5580669999999999
	.quad	0x40161dea033e78e2              # double 5.5292130000000004
	.quad	0x402369832fcac8e7              # double 9.7060790000000007
	.quad	0x3ff3bb71fbc5de9c              # double 1.233263
	.quad	0x40141e978d4fdf3b              # double 5.0298749999999997
	.quad	0x3fd8b1183b60285f              # double 0.38580900000000001
	.quad	0x4015ba4a8c154c98              # double 5.4319249999999997
	.quad	0x4008ce87d2c7b891              # double 3.1008450000000001
	.quad	0x40054c09c3ce208a              # double 2.662128
	.quad	0x3ff1976683c297c0              # double 1.0994630000000001
	.quad	0x40179ec0724b76f7              # double 5.9050310000000001
	.quad	0x40114206c8711d7a              # double 4.3144790000000004
	.quad	0x4016b8fe260b2c84              # double 5.6806570000000001
	.quad	0x40238d7ba6698bb5              # double 9.7763340000000003
	.quad	0x401835ee136e71ce              # double 6.0526660000000003
	.quad	0x3ffa5532617c1bda              # double 1.6457999999999999
	.quad	0x40161fadf2ecf206              # double 5.5309369999999998
	.quad	0x4017cbdb5d894813              # double 5.9490790000000002
	.quad	0x3fe89d3cbc48f10b              # double 0.76919400000000004
	.quad	0x3ff761f9f01b866e              # double 1.4614199999999999
	.quad	0x401d2e175d13d74d              # double 7.2950109999999997
	.quad	0x4008abe61cffeb07              # double 3.0839349999999999
	.quad	0x400dfaef6f8f0414              # double 3.7475269999999998
	.quad	0x4017b8ba83f4eca7              # double 5.9303990000000004
	.quad	0x401c785e80bed741              # double 7.1175480000000002
	.quad	0x40155fcefaa4766c              # double 5.3435629999999996
	.quad	0x40133dd76683c298              # double 4.8103920000000002
	.quad	0x40192ec2ce464990              # double 6.2956649999999996
	.quad	0x40201b743632c1f4              # double 8.0536209999999996
	.quad	0x3ff804e7253da72a              # double 1.5011969999999999
	.quad	0x4007092ddbdb5d89              # double 2.8794819999999999
	.quad	0x4023e44ac6cdaf4b              # double 9.9458830000000002
	.quad	0x4019685be1a82624              # double 6.3519129999999997
	.quad	0x40049f0f5a1016ce              # double 2.5776659999999998
	.quad	0x4022dfe2a3cea6c2              # double 9.4372760000000007
	.quad	0x4021fbf961804d98              # double 8.9921369999999996
	.quad	0x4006d042528ae74f              # double 2.8516889999999999
	.quad	0x3ffd5a37ef5a964f              # double 1.8345260000000001
	.quad	0x40215e008e9b38d6              # double 8.6835979999999999
	.quad	0x401370364388ebcc              # double 4.8595819999999996
	.quad	0x40045db0142f61ed              # double 2.5457459999999998
	.quad	0x4013758c08b75ea6              # double 4.8647919999999996
	.quad	0x4023c2bbecaab8a6              # double 9.8803400000000003
	.quad	0x4020e44bb1af3a15              # double 8.4458900000000003
	.quad	0x401e58cb07d0aeda              # double 7.5867120000000003
	.quad	0x40010e2435696e59              # double 2.1319050000000002
	.quad	0x3ffc93a4f8726d05              # double 1.786046
	.quad	0x3fe123443d46b26c              # double 0.535555
	.quad	0x400ad765fd8adaba              # double 3.355175
	.quad	0x401ab7fb69984a0e              # double 6.6796699999999998
	.quad	0x3fe0e34b945308bc              # double 0.52774600000000005
	.quad	0x40239639f77292c5              # double 9.7934110000000007
	.quad	0x400a1f9e7b80a9df              # double 3.2654390000000002
	.quad	0x3fe1857d1782d384              # double 0.54754499999999995
	.quad	0x401bbe75ff609dd0              # double 6.9359970000000004
	.quad	0x3ff50f1b25f633ce              # double 1.3161879999999999
	.quad	0x401fb29dc725c3df              # double 7.92443
	.quad	0x40201fb3311a543f              # double 8.0619139999999998
	.quad	0x3fb3060bb2bba98f              # double 0.074311000000000002
	.quad	0x40107606fac6045c              # double 4.1152610000000003
	.quad	0x400c7acb4289118c              # double 3.559958
	.quad	0x4017cd29563a9f38              # double 5.9503529999999998
	.quad	0x400b24bb1af3a14d              # double 3.392935
	.quad	0x4019057214f0520d              # double 6.2553179999999999
	.quad	0x3fe540b780346dc6              # double 0.66415000000000002
	.quad	0x4003ad612c6ac216              # double 2.4596580000000001
	.quad	0x400d2950763a1901              # double 3.6451730000000002
	.quad	0x4023243ce2089e34              # double 9.5707769999999996
	.quad	0x3ffddbf26f1dc50d              # double 1.866198
	.quad	0x40019c66d373affb              # double 2.201368
	.quad	0x3fd7306e5cd4ed2d              # double 0.36233100000000001
	.quad	0x40113a176ddaceee              # double 4.3067299999999999
	.quad	0x401ad1094a2b9d3d              # double 6.7041370000000002
	.quad	0x40238abc0e38a7e7              # double 9.7709659999999995
	.quad	0x3fd72189bd8383ae              # double 0.36142200000000002
	.quad	0x4023403b81b64e05              # double 9.6254539999999995
	.quad	0x40130fc829cfdd23              # double 4.7654120000000004
	.quad	0x3fd159c8c9320d99              # double 0.27110499999999998
	.quad	0x4023fdb7cfe58601              # double 9.9955429999999996
	.quad	0x401b330caa326e11              # double 6.7998529999999997
	.quad	0x4003840e1719f7f9              # double 2.4394800000000001
	.quad	0x4021b639713ad5bf              # double 8.8559070000000002
	.quad	0x401f7e071c53f39d              # double 7.8730739999999999
	.quad	0x4022432b990afe61              # double 9.1311920000000004
	.quad	0x402236ad9274e22a              # double 9.1067929999999997
	.quad	0x400ba0663c74fb55              # double 3.4533200000000002
	.quad	0x401f41d5c31593e6              # double 7.814292
	.quad	0x4020363ef284167b              # double 8.1059490000000007
	.quad	0x3fe369a6f826edab              # double 0.60664700000000005
	.quad	0x3ffb12e0e30446b7              # double 1.6921090000000001
	.quad	0x401b76d1e108c3f4              # double 6.8660350000000001
	.quad	0x4020f7f4128bf3bf              # double 8.4842840000000006
	.quad	0x400cf1ba3ca7503c              # double 3.6180310000000002
	.quad	0x40149d6494d50ebb              # double 5.1537040000000003
	.quad	0x40229e0dcb9a9da6              # double 9.3086990000000007
	.quad	0x401907b07075b3e1              # double 6.2575089999999998
	.quad	0x3ffb240fa9c12f0a              # double 1.696304
	.quad	0x40146fb4c3c18b50              # double 5.1090879999999999
	.quad	0x3ffbcd0edc3bd599              # double 1.737563
	.quad	0x40236d41fa765343              # double 9.7133939999999991
	.quad	0x40111d67fd3f5b60              # double 4.2787170000000003
	.quad	0x3ff9229a5ebb773a              # double 1.570948
	.quad	0x40189c96cc5fbf83              # double 6.1529189999999998
	.quad	0x401d0a66dbd72bcb              # double 7.2601579999999997
	.quad	0x40122a6b0d9513f9              # double 4.5414240000000001
	.quad	0x4015d5aee631f8a1              # double 5.4586750000000004
	.quad	0x40033dabe27179c0              # double 2.4051130000000001
	.quad	0x4007b7e0ac7da1ec              # double 2.9647839999999999
	.quad	0x3fcca52ef911cf35              # double 0.22379099999999999
	.quad	0x40142f405f6ba062              # double 5.046144
	.quad	0x3ff49285a921ccd9              # double 1.2857719999999999
	.quad	0x3ff9d00b45ae5ffa              # double 1.6132919999999999
	.quad	0x401fcb024f6598e1              # double 7.9482509999999999
	.quad	0x4015ee73e681a9b9              # double 5.4828640000000002
	.quad	0x3ffe80a17b0f6ad7              # double 1.906404
	.quad	0x4021103148776816              # double 8.5316259999999993
	.quad	0x402093e2b4959e62              # double 8.2888389999999994
	.quad	0x4001591600f34507              # double 2.1684990000000002
	.quad	0x401c870ac3a860dd              # double 7.1318770000000002
	.quad	0x4009393469941850              # double 3.1529319999999998
	.quad	0x40070f6c69b5a640              # double 2.8825310000000002
	.quad	0x4003d53b4b2fa93b              # double 2.479117
	.quad	0x400922c0a4a05dd9              # double 3.141969
	.quad	0x4017c60c7c0f4517              # double 5.9434069999999997
	.quad	0x401f65ae1cde5d18              # double 7.8492969999999999
	.quad	0x401bb588a04d1202              # double 6.9272790000000004
	.quad	0x3ff22af9a8cdea03              # double 1.1354919999999999
	.quad	0x401eeae21101b003              # double 7.7293779999999996
	.quad	0x400bdc11e42e1262              # double 3.482456
	.quad	0x40209fe86833c600              # double 8.3123199999999997
	.quad	0x4018d2189bd8383b              # double 6.2051720000000001
	.quad	0x3fd03868fd199bb2              # double 0.25344299999999997
	.quad	0x401016145953586d              # double 4.0215620000000003
	.quad	0x401a9402d16b97ff              # double 6.6445420000000004
	.quad	0x401798c197e56473              # double 5.8991759999999998
	.quad	0x3fdae7f6f4be835e              # double 0.420408
	.quad	0x3fcc60e51d25aab4              # double 0.22170699999999999
	.quad	0x4014759103c8e25d              # double 5.1148110000000004
	.quad	0x4023e0efbb0e5e68              # double 9.9393290000000007
	.quad	0x400f0b191dde3758              # double 3.8804189999999998
	.quad	0x3fe1556d19dece57              # double 0.54167799999999999
	.quad	0x4021a8fb65668c26              # double 8.8300429999999999
	.quad	0x402150bfc2224edf              # double 8.6577129999999993
	.quad	0x4021c964a59c065b              # double 8.8933459999999993
	.quad	0x3f8c5931ca7d6734              # double 0.013842
	.quad	0x4012254866a11ec9              # double 4.5364089999999999
	.quad	0x402236c0b567557a              # double 9.1069390000000006
	.quad	0x40200426dce39b45              # double 8.0081089999999993
	.quad	0x3fbbc579f2346562              # double 0.108482
	.quad	0x401fbc84b5dcc63f              # double 7.9340999999999999
	.quad	0x401e122c45026890              # double 7.517747
	.quad	0x3ffb4bee807bbb62              # double 1.7060379999999999
	.quad	0x4014c02d16b97fe9              # double 5.1876720000000001
	.quad	0x400af1758e219653              # double 3.3679000000000001
	.quad	0x40193e957470eb25              # double 6.3111170000000003
	.quad	0x3ffd8c6b484d76ab              # double 1.8467819999999999
	.quad	0x401fc2580c308feb              # double 7.9397890000000002
	.quad	0x4020e60eb67c2870              # double 8.4493310000000008
	.quad	0x4023d129cbab649d              # double 9.9085219999999996
	.quad	0x400fe2a0a958537e              # double 3.9856579999999999
	.quad	0x401713fb3743632c              # double 5.7695129999999999
	.quad	0x3f7e0828c36da87a              # double 0.007332
	.quad	0x3ff282c9081c2e34              # double 1.15693
	.quad	0x401b849491f2dc2b              # double 6.8794729999999999
	.quad	0x402367852f7f498c              # double 9.7021879999999995
	.quad	0x40114154434e336a              # double 4.3137980000000002
	.quad	0x401c435dedf1e083              # double 7.0657880000000004
	.quad	0x402336c764adff82              # double 9.6069899999999997
	.quad	0x4023a111f0c34c1b              # double 9.8145900000000008
	.quad	0x4012adc4c9c90c4e              # double 4.6696960000000001
	.quad	0x40182faacd9e83e4              # double 6.0465499999999999
	.quad	0x401d252fc2656abe              # double 7.2863150000000001
	.quad	0x40051610adcd2d45              # double 2.6357740000000001
	.quad	0x3ffb4628027d88c2              # double 1.704628
	.quad	0x400b80945f9df549              # double 3.437783
	.quad	0x3ff1cc7b02d59d56              # double 1.112422
	.quad	0x401ba6a22b3892ef              # double 6.9127280000000004
	.quad	0x400d89c0a8d21bc1              # double 3.6922619999999999
	.quad	0x4012e6d373affb05              # double 4.7254160000000001
	.quad	0x40188ff9f87f023f              # double 6.1406020000000003
	.quad	0x4023cacb85a4f00f              # double 9.8960840000000001
	.quad	0x4022fa305532617c              # double 9.4886499999999998
	.quad	0x401cb9e518f3eccc              # double 7.1815379999999998
	.quad	0x4004db2edfe75bc4              # double 2.6070229999999999
	.quad	0x401cc1ee675147f1              # double 7.1893859999999998
	.quad	0x401caf0b6b6e0d24              # double 7.1709420000000001
	.quad	0x3fe517dae81882ae              # double 0.65916200000000003
	.quad	0x3fee3f1843c3a42f              # double 0.94520199999999999
	.quad	0x3ff4668e3ef28416              # double 1.2750379999999999
	.quad	0x40092a0d349be8ff              # double 3.1455329999999999
	.quad	0x3fe14f97edc7ef17              # double 0.54096599999999995
	.quad	0x3fd53126e978d4fe              # double 0.331125
	.quad	0x40220c60029f16b1              # double 9.0241699999999998
	.quad	0x400bca012599ed7c              # double 3.4736349999999998
	.quad	0x401abb524399b2c4              # double 6.682931
	.quad	0x4003dd30ad46f588              # double 2.4830030000000001
	.quad	0x3fefc059210385c6              # double 0.99222999999999995
	.quad	0x4021097ce52deca2              # double 8.5185309999999994
	.quad	0x401f9254a3c64346              # double 7.8929010000000002
	.quad	0x402196605ee568a5              # double 8.793704
	.quad	0x401e7854046412cf              # double 7.6175079999999999
	.quad	0x400e31b0ccbc05d5              # double 3.7742629999999999
	.quad	0x4018a3b60285ec3e              # double 6.1598740000000003
	.quad	0x3ff2333b96af038e              # double 1.137508
	.quad	0x4023abd273d5bab2              # double 9.8355899999999998
	.quad	0x4013e5ea67e846a6              # double 4.9745270000000001
	.quad	0x401fb126634117f8              # double 7.9229979999999997
	.quad	0x4018a38d60a63305              # double 6.1597189999999999
	.quad	0x400d59780baa582e              # double 3.6686860000000001
	.quad	0x401cd0fc71d60631              # double 7.2040879999999996
	.quad	0x4023ddedf1e0828c              # double 9.9334559999999996
	.quad	0x3faea26e5471715c              # double 0.059832999999999997
	.quad	0x40021c39bcba3012              # double 2.263782
	.quad	0x40197d8b60f1b25f              # double 6.3726019999999997
	.quad	0x4001c036006d0d4a              # double 2.2188530000000002
	.quad	0x4009622e5de15ca7              # double 3.1729400000000001
	.quad	0x3ff74cef672b8844              # double 1.456283
	.quad	0x40118af81626b2f2              # double 4.3857119999999998
	.quad	0x3fed8a04d12018a4              # double 0.92309799999999997
	.quad	0x4011c8e7ddca4b12              # double 4.4461969999999997
	.quad	0x400f5c7ef177a701              # double 3.9201640000000002
	.quad	0x40191b430f49491f              # double 6.2766229999999998
	.quad	0x40211bc4f3ba7760              # double 8.5542370000000005
	.quad	0x3fcb36deb95e5aff              # double 0.212612
	.quad	0x4020a2f1cb89d6ae              # double 8.3182510000000001
	.quad	0x4020c6de54b48d3b              # double 8.3884150000000002
	.quad	0x4000a65c70435efa              # double 2.0812309999999998
	.quad	0x40158739f340d4dc              # double 5.3820569999999996
	.quad	0x3ff3dc63ae2541d9              # double 1.241306
	.quad	0x4021ad6e47dc37a4              # double 8.8387320000000002
	.quad	0x3fd4bd944aa53fc0              # double 0.324071
	.quad	0x401ea23fab10ba62              # double 7.6584459999999996
	.quad	0x3fd822ff08893b7e              # double 0.37713600000000003
	.quad	0x400009a9973d9ec7              # double 2.004718
	.quad	0x3ffef6805a2d7300              # double 1.935181
	.quad	0x401bb358f2e05ccd              # double 6.9251440000000004
	.quad	0x3fe80366516db0de              # double 0.75041500000000005
	.quad	0x401cf98676a7264a              # double 7.2436769999999999
	.quad	0x402266bf444e0daa              # double 9.2006779999999999
	.quad	0x40190b46bacf7447              # double 6.261012
	.quad	0x400a464a9cdc4439              # double 3.284322
	.quad	0x4021cf2990f301eb              # double 8.9046140000000005
	.quad	0x401009c90c4dec1c              # double 4.0095559999999999
	.quad	0x3feab295e9e1b08a              # double 0.83430000000000004
	.quad	0x40227b5bd512ec6c              # double 9.2409350000000003
	.quad	0x40153235f8099179              # double 5.2990339999999998
	.quad	0x40150c6cdaf4adbc              # double 5.2621339999999996
	.quad	0x4018a0e12e83a10a              # double 6.1571090000000002
	.quad	0x4020273493858362              # double 8.0765729999999998
	.quad	0x4012d2c7fbacb429              # double 4.7058410000000004
	.quad	0x40165247021d10b2              # double 5.580349
	.quad	0x40137e13b18dac26              # double 4.8731220000000004
	.quad	0x3fc99a2834d26fa4              # double 0.200017
	.quad	0x401a095e17e34b94              # double 6.5091479999999997
	.quad	0x401a42c2bd7f51f0              # double 6.5651960000000003
	.quad	0x401d87fb267c6b8b              # double 7.3827939999999996
	.quad	0x4000f03e20ccff22              # double 2.1173060000000001
	.quad	0x4020d80fba8826ab              # double 8.4219950000000008
	.quad	0x401c747d3d4280ae              # double 7.1137589999999999
	.quad	0x40148d4ed2cbea4f              # double 5.1379960000000002
	.quad	0x400568bb906466b2              # double 2.676139
	.quad	0x401901a500d5e8d5              # double 6.2516059999999998
	.quad	0x3fee4c87980f55de              # double 0.94684199999999996
	.quad	0x40030915bdd76684              # double 2.3794360000000001
	.quad	0x40170c7d5ed06fef              # double 5.7621969999999996
	.quad	0x4017974ea8da7f3d              # double 5.897761
	.quad	0x40149e785b5b7069              # double 5.1547559999999999
	.quad	0x402386f262888b54              # double 9.7635679999999993
	.quad	0x3ff101eeed8904f7              # double 1.062972
	.quad	0x4022e3b107746888              # double 9.4447100000000006
	.quad	0x401f66583e8576cd              # double 7.849946
	.quad	0x4000e39ffd60e94f              # double 2.111145
	.quad	0x400aa37b4a2339c1              # double 3.329825
	.quad	0x402014dbdf8f4730              # double 8.0407399999999996
	.quad	0x401398a4c1ebc83b              # double 4.8990660000000004
	.quad	0x401cfb28954a7f80              # double 7.2452719999999999
	.quad	0x4020dc299d883ba3              # double 8.4300049999999995
	.quad	0x3ffdb33daf8df7a5              # double 1.85626
	.quad	0x4006721bc126a65d              # double 2.805717
	.quad	0x4020f13682730c67              # double 8.4711189999999998
	.quad	0x4000a2fbe33acd5b              # double 2.0795819999999998
	.quad	0x3fcbf601797cc3a0              # double 0.218445
	.quad	0x402099d3ed527e52              # double 8.3004449999999999
	.quad	0x40200fefbf401c50              # double 8.0311260000000004
	.quad	0x4000170bd017dae8              # double 2.011253
	.quad	0x3fed617e34b94531              # double 0.91815100000000005
	.quad	0x401af45069a4df48              # double 6.738588
	.quad	0x400568ab4fa48302              # double 2.6761080000000002
	.quad	0x400f1acabc515486              # double 3.8880819999999998
	.quad	0x4020a610cf5b1c86              # double 8.3243469999999995
	.quad	0x400b4ee392e1ef74              # double 3.4135200000000001
	.quad	0x3fe6dc3dee781840              # double 0.71438500000000005
	.quad	0x4018c0c62e4d1a65              # double 6.188256
	.quad	0x4005d0d1fa333765              # double 2.726963
	.quad	0x3ff45f3b645a1cac              # double 1.27325
	.quad	0x400f2d3415b1422d              # double 3.8970720000000001
	.quad	0x400562480e8c8abd              # double 2.6729889999999998
	.quad	0x3ffaf7ee4e26d480              # double 1.68553
	.quad	0x3fdd3ed959a30985              # double 0.45696100000000001
	.quad	0x40196e518f3eccc4              # double 6.3577329999999996
	.quad	0x4016144c7b02d59d              # double 5.5198229999999997
	.quad	0x3ff16157689ca18c              # double 1.086265
	.quad	0x4022d0d349be8ff3              # double 9.4078619999999997
	.quad	0x401ef848beb5b2d5              # double 7.7424650000000001
	.quad	0x4014ae21101b0037              # double 5.1700480000000004
	.quad	0x40119c12f09d8c6d              # double 4.4024159999999997
	.quad	0x401f7f0d844d013b              # double 7.8740750000000003
	.quad	0x401c97353b4b2fa9              # double 7.1476639999999998
	.quad	0x3fcd122b7baecd08              # double 0.22711700000000001
	.quad	0x401c714df8b15726              # double 7.1106490000000004
	.quad	0x3fd16c1e364bec68              # double 0.27222400000000002
	.quad	0x401afda7f3cf7015              # double 6.7477109999999998
	.quad	0x3ff6733ebbfd71b0              # double 1.4031359999999999
	.quad	0x40187b8887229e90              # double 6.1206379999999996
	.quad	0x401432f661f18ca0              # double 5.0497680000000003
	.quad	0x3fddb51372a38b8b              # double 0.46417700000000001
	.quad	0x4016e221426fe719              # double 5.7208300000000003
	.quad	0x40178376922d948e              # double 5.8783820000000002
	.quad	0x4018ac9ff92f2b67              # double 6.1685790000000003
	.quad	0x3ff59b1d92b7fe09              # double 1.3503700000000001
	.quad	0x40020a57214f0521              # double 2.2550490000000001
	.quad	0x401713daf8df7a4e              # double 5.7693899999999996
	.quad	0x40213d2e0e30446b              # double 8.6194919999999993
	.quad	0x40006dfefbf401c5              # double 2.053709
	.quad	0x3ffa677a2cecc815              # double 1.650263
	.quad	0x401ce4399b2c40d1              # double 7.2228760000000003
	.quad	0x400f6820e6299525              # double 3.9258440000000001
	.quad	0x401c2dab5c39bcba              # double 7.0445989999999998
	.quad	0x3ffbe37bd05af6c7              # double 1.7430380000000001
	.quad	0x4014b48238540464              # double 5.1762779999999999
	.quad	0x40066b1ab0856e69              # double 2.8022969999999998
	.quad	0x401ae634549b62c7              # double 6.7248089999999996
	.quad	0x3ffab5a5332acfb7              # double 1.6693469999999999
	.quad	0x3ff8368dfbd6a594              # double 1.5133190000000001
	.quad	0x40128e23af31b153              # double 4.638808
	.quad	0x4022224e9c454584              # double 9.0670059999999992
	.quad	0x4008d6102795703f              # double 3.1045229999999999
	.quad	0x4022927ce0fc2eba              # double 9.2861089999999997
	.quad	0x40200833a4723ab0              # double 8.016019
	.quad	0x401c0eebf65dbfcf              # double 7.0145720000000003
	.quad	0x4003aebb341e14be              # double 2.460318
	.quad	0x3fd8bb6ed677707a              # double 0.38644000000000001
	.quad	0x4021b658a32f4491              # double 8.8561449999999997
	.quad	0x4020a3fcc9ea9a3d              # double 8.3202879999999997
	.quad	0x4022f8ebedfa43fe              # double 9.4861749999999994
	.quad	0x4022c141e9af5ba3              # double 9.3774560000000004
	.quad	0x3fd83c46d82ba5a0              # double 0.37867899999999999
	.quad	0x4022a1e16d6dc1a4              # double 9.3161729999999991
	.quad	0x400194b05b7cfe58              # double 2.1976019999999998
	.quad	0x40001d99029ae4f3              # double 2.0144519999999999
	.quad	0x401d5a7eb6bf444e              # double 7.338374
	.quad	0x3ff86b95e5aff2a1              # double 1.5262659999999999
	.quad	0x401ff33d29563a9f              # double 7.9875379999999998
	.quad	0x401f8ddca4b124d1              # double 7.8885370000000003
	.quad	0x40183a4ca4f440af              # double 6.0569329999999999
	.quad	0x3ffde47dc37a3db4              # double 1.8682840000000001
	.quad	0x3fe1669ced0b30b6              # double 0.54377600000000004
	.quad	0x4009725f633ce63a              # double 3.1808459999999998
	.quad	0x3ff0f8b8ae31d713              # double 1.0607230000000001
	.quad	0x4012527e0ef99807              # double 4.580559
	.quad	0x3febf2cf95d4e8fb              # double 0.87339
	.quad	0x4018f1a5ca29845e              # double 6.2359840000000002
	.quad	0x4005ec2d6a9c560c              # double 2.7403209999999998
	.quad	0x4020010fd7e45804              # double 8.0020740000000003
	.quad	0x401078950763a190              # double 4.117756
	.quad	0x4002942e557de0d6              # double 2.3223539999999998
	.quad	0x3ffe7588e368f084              # double 1.9036949999999999
	.quad	0x3ff938fde2ef4e01              # double 1.576414
	.quad	0x40203c2c3c9eecc0              # double 8.1175250000000005
	.quad	0x40210e0ac7da1ec5              # double 8.5274260000000002
	.quad	0x4023482c0a4a05de              # double 9.6409610000000008
	.quad	0x4022722c45026890              # double 9.2229939999999999
	.quad	0x4013284449dbec25              # double 4.7893230000000004
	.quad	0x40217f7403dddb12              # double 8.7489319999999999
	.quad	0x3ff09f70de8f6cf0              # double 1.038926
	.quad	0x400d32de43ed959a              # double 3.6498379999999999
	.quad	0x401e3b0f6ad70e6f              # double 7.5576759999999998
	.quad	0x4010b5ba2be0589b              # double 4.1774680000000002
	.quad	0x402289d51b4fe79f              # double 9.2692040000000002
	.quad	0x3ffed6d4c33b5393              # double 1.927449
	.quad	0x400b5897204295a7              # double 3.4182570000000001
	.quad	0x3ffde88051c9f72f              # double 1.8692629999999999
	.quad	0x4007175968512232              # double 2.8864010000000002
	.quad	0x40059a975afaf859              # double 2.7004839999999999
	.quad	0x4003b4195cc857f3              # double 2.462939
	.quad	0x40085bd273d5bab2              # double 3.044835
	.quad	0x3ff8669270b06c44              # double 1.525042
	.quad	0x401ff01322f27350              # double 7.9844480000000004
	.quad	0x4022f455432873bd              # double 9.4772130000000007
	.quad	0x40038cd29131ec0b              # double 2.4437609999999999
	.quad	0x40195f82b1f687b1              # double 6.3432719999999998
	.quad	0x400c2d983947496b              # double 3.5222630000000001
	.quad	0x4023b8a04d12018a              # double 9.8605979999999995
	.quad	0x3fdbf4c2b51bd61f              # double 0.43681399999999998
	.quad	0x3fca1815a07b352b              # double 0.20386000000000001
	.quad	0x3ffee096787ce960              # double 1.9298310000000001
	.quad	0x4020331e18efbb0e              # double 8.0998389999999993
	.quad	0x400391b1d92b7fe1              # double 2.4461400000000002
	.quad	0x3fe2b84988094e5d              # double 0.58499599999999996
	.quad	0x3ff951697f1f9ad0              # double 1.582376
	.quad	0x401d32888b545c79              # double 7.2993490000000003
	.quad	0x401c7e9424e59296              # double 7.1236119999999996
	.quad	0x401bd3e6c4c5974e              # double 6.9569349999999996
	.quad	0x402067a4a48f96e1              # double 8.2024279999999994
	.quad	0x400bf421c044284e              # double 3.494205
	.quad	0x40149a7546d3f9e8              # double 5.1508380000000002
	.quad	0x4023183ea704bc27              # double 9.5473529999999993
	.quad	0x401ca5c5718eb895              # double 7.161886
	.quad	0x40214b1c219eb639              # double 8.6466989999999999
	.quad	0x401117e56473471f              # double 4.2733359999999996
	.quad	0x4022b0dfdef8487c              # double 9.3454580000000007
	.quad	0x40198da836eb4e98              # double 6.3883369999999999
	.quad	0x4019d6ab9b23dd55              # double 6.4596390000000001
	.quad	0x400c88d002e23290              # double 3.5668030000000002
	.quad	0x401deb0ccbc05d53              # double 7.4795410000000002
	.quad	0x401ec898f605ab3b              # double 7.6958960000000002
	.quad	0x40201699a62ed352              # double 8.0441409999999998
	.quad	0x401401ede1198aec              # double 5.0018840000000004
	.quad	0x3fe887229e90795f              # double 0.76649599999999996
	.quad	0x400e9e4d5d80e497              # double 3.827296
	.quad	0x3ff426c5d206c871              # double 1.259466
	.quad	0x4022fbbecaab8a5d              # double 9.4916900000000001
	.quad	0x4020265e679463d0              # double 8.0749390000000005
	.quad	0x401d474c4cdfaca3              # double 7.3196269999999997
	.quad	0x401c0da87a072d1b              # double 7.0133380000000001
	.quad	0x401d733508f648c7              # double 7.3625069999999999
	.quad	0x40234355043e5322              # double 9.6315080000000002
	.quad	0x4018a0d23d4f15e8              # double 6.1570520000000002
	.quad	0x3fccdbbe0157eed4              # double 0.22545599999999999
	.quad	0x3fdf2003ab862b28              # double 0.48632900000000001
	.quad	0x40123a8b08dd1e54              # double 4.5571710000000003
	.quad	0x3ff1c5953586ca8a              # double 1.110738
	.quad	0x3ff4912dba4d6e48              # double 1.285444
	.quad	0x3ff4a413db7f1737              # double 1.2900579999999999
	.quad	0x4002ffde7210be94              # double 2.3749359999999999
	.quad	0x401e4f4cf4a558ea              # double 7.5774419999999996
	.quad	0x3ff534fd6d7e889c              # double 1.325437
	.quad	0x3ff0f4751ce28ed6              # double 1.059682
	.quad	0x4007cd7a56de3327              # double 2.9753310000000002
	.quad	0x401a44666ec9e237              # double 6.5667970000000002
	.quad	0x4022749bc7714339              # double 9.2277509999999996
	.quad	0x401cb83be6601bca              # double 7.1799160000000004
	.quad	0x400df3ae685db76b              # double 3.7439849999999999
	.quad	0x402038f451761459              # double 8.1112389999999994
	.quad	0x3fef69d51b4fe79f              # double 0.98166900000000001
	.quad	0x40235f3fd933e35c              # double 9.6860339999999993
	.quad	0x401c41f0c34c1a8b              # double 7.0643950000000002
	.quad	0x401275bd0e12e83a              # double 4.6149789999999999
	.quad	0x401c4ab2e1693c04              # double 7.0729480000000002
	.quad	0x401045ce9e5e2479              # double 4.0681710000000004
	.quad	0x4001b0ed3d859c8d              # double 2.2113900000000002
	.quad	0x40016ead0c3d2524              # double 2.1790409999999998
	.quad	0x3fea95ee136e71ce              # double 0.83080200000000004
	.quad	0x4023c2ce67d77fae              # double 9.8804809999999996
	.quad	0x401fb33b96af038e              # double 7.9250319999999999
	.quad	0x4012552e2fbe33ad              # double 4.5831840000000001
	.quad	0x3ff1300ffda4052d              # double 1.0742339999999999
	.quad	0x3ff0b97dd00f776c              # double 1.0452859999999999
	.quad	0x3ff4b1fff79c8430              # double 1.2934570000000001
	.quad	0x40239b3e5753a3ec              # double 9.8032099999999999
	.quad	0x401b93a8e71476b0              # double 6.8941990000000004
	.quad	0x4017238eb0318b93              # double 5.7847239999999998
	.quad	0x4017ddd00f776c48              # double 5.9666139999999999
	.quad	0x402272db8348f53c              # double 9.2243309999999993
	.quad	0x40154acb85a4f00f              # double 5.3230420000000001
	.quad	0x40058308feac42ea              # double 2.6889820000000002
	.quad	0x400ec779207d4e09              # double 3.8473989999999998
	.quad	0x401eda23bff8a8f4              # double 7.7130270000000003
	.quad	0x3ff9773c0c1fc8f3              # double 1.59161
	.quad	0x400732146a1a500d              # double 2.8994529999999998
	.quad	0x4010f5f7403dddb1              # double 4.2402009999999999
	.quad	0x400819d99029ae4f              # double 3.0126219999999999
	.quad	0x3ff70c8ffb8b2639              # double 1.4405669999999999
	.quad	0x4010fd8f92af9a8d              # double 4.2476180000000001
	.quad	0x4022912513b5bf6a              # double 9.2834859999999999
	.quad	0x3ffd2c9e236c15d3              # double 1.823393
	.quad	0x4018bd013a92a305              # double 6.1845749999999997
	.quad	0x401474e69f05ea25              # double 5.1141610000000002
	.quad	0x4023a642e1262025              # double 9.8247289999999996
	.quad	0x4006710e453d20f3              # double 2.8052030000000001
	.quad	0x400cef2e8c0485a1              # double 3.6167880000000001
	.quad	0x401c782385404641              # double 7.1173229999999998
	.quad	0x3fc12deca25529fe              # double 0.134214
	.quad	0x40207e215336deb9              # double 8.2463479999999993
	.quad	0x402128a761027957              # double 8.579402
	.quad	0x4020a7cfc3f811f5              # double 8.3277570000000001
	.quad	0x3feae44b6e935b92              # double 0.840368
	.quad	0x3fe655b89939218a              # double 0.69796400000000003
	.quad	0x401931c7de5082cf              # double 6.2986139999999997
	.quad	0x40227893d9663843              # double 9.2355029999999996
	.quad	0x3ff2bc957cd466f5              # double 1.171041
	.quad	0x3fd63e963dc486ad              # double 0.34756999999999999
	.quad	0x40153344806290ef              # double 5.3000660000000002
	.quad	0x4008dbb62413db7f              # double 3.107281
	.quad	0x40153d4cb9ecf638              # double 5.309863
	.quad	0x4015b41248d7e026              # double 5.4258509999999998
	.quad	0x400abb691212513b              # double 3.3415089999999998
	.quad	0x401e8867ad8e4324              # double 7.6332079999999998
	.quad	0x3fcc51b93037d630              # double 0.221244
	.quad	0x401a5d8c2a454de8              # double 6.5913550000000001
	.quad	0x4010b765ba6efc37              # double 4.1790989999999999
	.quad	0x3fec4b85a4f00ef1              # double 0.88421899999999998
	.quad	0x4012e64c729f59cd              # double 4.724901
	.quad	0x3fffd903a7546d40              # double 1.9904820000000001
	.quad	0x3fe0c67be553ac4f              # double 0.52422899999999995
	.quad	0x4022ac8fd9fd36f8              # double 9.3370350000000001
	.quad	0x401303e8576cce5f              # double 4.7538159999999996
	.quad	0x401be03fb3743633              # double 6.9689930000000002
	.quad	0x3feff9fcb0c026cc              # double 0.99926599999999998
	.quad	0x402358daa0cae643              # double 9.6735430000000004
	.quad	0x401d29f383f0bae9              # double 7.2909680000000003
	.quad	0x400e5c2dadb8348f              # double 3.7950089999999999
	.quad	0x4010f26aec0724b7              # double 4.2367359999999996
	.quad	0x40169bc77143393b              # double 5.6521280000000003
	.quad	0x3ff48edd48451330              # double 1.2848790000000001
	.quad	0x40160563ed0f6274              # double 5.5052640000000004
	.quad	0x4022633a0407cc7d              # double 9.1938019999999998
	.quad	0x40198f36ef8055fc              # double 6.3898580000000003
	.quad	0x4017e1569f490603              # double 5.9700569999999997
	.quad	0x3fe6093532e7b3d9              # double 0.68862400000000001
	.quad	0x4020f09aaa3ad18d              # double 8.4699299999999997
	.quad	0x4023e2f405f6ba06              # double 9.9432679999999998
	.quad	0x4017b7a743a647fe              # double 5.9293490000000002
	.quad	0x3fc1aaac1094a2ba              # double 0.138021
	.quad	0x3fc8e043a2163fdd              # double 0.19434399999999999
	.quad	0x40174e996312f4cf              # double 5.8267569999999997
	.quad	0x401204f5d35653ca              # double 4.5048440000000003
	.quad	0x3ff5a4abe6a337a8              # double 1.352703
	.quad	0x4021826aec0724b7              # double 8.7547219999999992
	.quad	0x4023f32fec56d5d0              # double 9.9749750000000006
	.quad	0x40137d5d80e496ee              # double 4.8724270000000001
	.quad	0x401a39e0e736049f              # double 6.5565220000000002
	.quad	0x4009f0025bfad29a              # double 3.2421920000000002
	.quad	0x4011d9ed3953dea4              # double 4.4628189999999996
	.quad	0x4014da582dbe7f2c              # double 5.2132269999999998
	.quad	0x4020c9116a8b8f15              # double 8.3927110000000003
	.quad	0x4003a5e6ff7dfa01              # double 2.4560070000000001
	.quad	0x4006d1d57ff9b563              # double 2.8524579999999999
	.quad	0x400d6230fcf80dc3              # double 3.6729449999999999
	.quad	0x401fcedb2f661f19              # double 7.9520080000000001
	.quad	0x4020175fd47bedb7              # double 8.0456529999999997
	.quad	0x3fe0eb93ccd0fe8b              # double 0.52875700000000003
	.quad	0x4021ffef177a7009              # double 8.9998710000000006
	.quad	0x401a8916872b020c              # double 6.6338749999999997
	.quad	0x40050858ff759685              # double 2.629076
	.quad	0x3ffe21fe21d96e9c              # double 1.8832990000000001
	.quad	0x400b717acc4ef88c              # double 3.4304100000000002
	.quad	0x40227567dbb16c1e              # double 9.2293079999999996
	.quad	0x400a709f1f14983d              # double 3.3049909999999998
	.quad	0x40214c1fea8112ba              # double 8.6486809999999998
	.quad	0x4018b5f5ad96a6a0              # double 6.1776949999999999
	.quad	0x3ffdcbd1244a6224              # double 1.86226
	.quad	0x401017fe8ee6b830              # double 4.0234319999999997
	.quad	0x4023c342cc2d6a9c              # double 9.8813689999999994
	.quad	0x3fdc3d9ec7000432              # double 0.44126100000000001
	.quad	0x3fcfe047d3d4280b              # double 0.249032
	.quad	0x3ff1248626f60e0f              # double 1.0714170000000001
	.quad	0x4004a27d028a1dfc              # double 2.5793400000000002
	.quad	0x40151ad70e6f2e8c              # double 5.276211
	.quad	0x4006a3d1cc100e6b              # double 2.82999
	.quad	0x401b3d7d7c2ca149              # double 6.8100490000000002
	.quad	0x401b01709310129d              # double 6.7514060000000002
	.quad	0x40065c69728a6117              # double 2.7951229999999998
	.quad	0x401ba21144cbe1eb              # double 6.9082689999999998
	.quad	0x40234ce8fb00bcbe              # double 9.6502149999999993
	.quad	0x3fed98d25edd0529              # double 0.92490499999999998
	.quad	0x4013a6deb95e5aff              # double 4.9129589999999999
	.quad	0x401121cef240fa9c              # double 4.2830159999999999
	.quad	0x402217559b3d07c8              # double 9.0455749999999995
	.quad	0x3fff53b29a1b9f99              # double 1.9579340000000001
	.quad	0x4021f83e20ccff22              # double 8.9848490000000005
	.quad	0x40142e3b03e20cd0              # double 5.045147
	.quad	0x401903293d102bc7              # double 6.2530869999999998
	.quad	0x40216266559f6ec6              # double 8.6921870000000006
	.quad	0x400246d58c8eef1c              # double 2.2845870000000001
	.quad	0x40187e0828c36da8              # double 6.1230779999999996
	.quad	0x3ff61b530ced4e4d              # double 1.3816710000000001
	.quad	0x4018aaa1511dffc5              # double 6.1666309999999998
	.quad	0x3fffdad7518b0d0f              # double 1.990928
	.quad	0x3fe7faebc408d8ed              # double 0.74938000000000005
	.quad	0x3fea1a1b9f98b71c              # double 0.81568700000000005
	.quad	0x4002efd5454152b1              # double 2.3671060000000002
	.quad	0x40225acf0f9d2bf5              # double 9.1773609999999994
	.quad	0x4019de84f09528f2              # double 6.4673040000000004
	.quad	0x401247eae5bc87db              # double 4.5702319999999999
	.quad	0x4014c1ede1198aec              # double 5.1893840000000004
	.quad	0x3ffce856605ee569              # double 1.8067230000000001
	.quad	0x40032bfc2224edf6              # double 2.396477
	.quad	0x3fef6264e48626f6              # double 0.98076099999999999
	.quad	0x40228f603925bb7b              # double 9.2800309999999993
	.quad	0x4016bcddd6e04c06              # double 5.6844400000000004
	.quad	0x400c9cdc443914f5              # double 3.5765920000000002
	.quad	0x3ff7b69014b599aa              # double 1.4820709999999999
	.quad	0x401c9e07e5a78f26              # double 7.1543270000000003
	.quad	0x4012eaf42784a948              # double 4.7294470000000004
	.quad	0x4018785a4f00ef13              # double 6.1175319999999997
	.quad	0x400916f6d7625205              # double 3.1362130000000001
	.quad	0x40098310dbf0563f              # double 3.1889970000000001
	.quad	0x3fe1c039abf33871              # double 0.55471499999999996
	.quad	0x40112297396d0918              # double 4.2837800000000001
	.quad	0x40233afc04c8bc9d              # double 9.6152040000000003
	.quad	0x4020149de0503750              # double 8.0402670000000001
	.quad	0x4021af38e6d15ad1              # double 8.842231
	.quad	0x40173cb923a29c78              # double 5.8093000000000004
	.quad	0x4013b55b4677b396              # double 4.9271060000000002
	.quad	0x40234541fa765343              # double 9.6352689999999991
	.quad	0x4020be338b47c73f              # double 8.3714870000000001
	.quad	0x4004f437c5692b3d              # double 2.6192470000000001
	.quad	0x3fdab4d48882f0e1              # double 0.41728700000000002
	.quad	0x4004be657b84db9c              # double 2.5929669999999998
	.quad	0x401325232d2bb235              # double 4.7862669999999996
	.quad	0x40228fbb0e5e6794              # double 9.2807239999999993
	.quad	0x401bf60ae9680e06              # double 6.9902759999999997
	.quad	0x400509f59ccfaeff              # double 2.6298629999999998
	.quad	0x3ffbe12b5e529bae              # double 1.7424729999999999
	.quad	0x4018563f572de43f              # double 6.0842260000000001
	.quad	0x3fe4b05d95dd4c77              # double 0.64652900000000002
	.quad	0x4014d3137b07075b              # double 5.2061289999999998
	.quad	0x4023067c8d195897              # double 9.5126690000000007
	.quad	0x4015ad16b97fe8ee              # double 5.4190319999999996
	.quad	0x40111a50507a6bd7              # double 4.2756970000000001
	.quad	0x4020229d627bf61b              # double 8.0676070000000006
	.quad	0x40130d3e0bd44999              # double 4.7629320000000002
	.quad	0x401cbcf1800a7c5b              # double 7.1845150000000002
	.quad	0x4022723d70a3d70a              # double 9.2231249999999996
	.quad	0x4005e615a8deb0fb              # double 2.7373460000000001
	.quad	0x40216cf9e3864cb6              # double 8.7128440000000005
	.quad	0x4015ecc100e6afcd              # double 5.4812050000000001
	.quad	0x4021036f19934efd              # double 8.5067070000000005
	.quad	0x401e93f077ccc038              # double 7.6444720000000003
	.quad	0x4021b963779e9d0f              # double 8.8620870000000007
	.quad	0x401d6c985f06f694              # double 7.3560499999999998
	.quad	0x4018619d2391d580              # double 6.095326
	.quad	0x400729845dc83233              # double 2.8952719999999998
	.quad	0x4022905c03361566              # double 9.2819520000000004
	.quad	0x40037d6733ebbfd7              # double 2.436232
	.quad	0x3ff7ef74cd31769b              # double 1.4959610000000001
	.quad	0x40163daceee0f3cb              # double 5.5602299999999998
	.quad	0x4020acece9a2c669              # double 8.337745
	.quad	0x3fca96ededaec4a4              # double 0.207731
	.quad	0x4016e3a3a8e71477              # double 5.7223040000000003
	.quad	0x3ff12af46aa087ca              # double 1.0729869999999999
	.quad	0x4012109e55c0fcb5              # double 4.516229
	.quad	0x40190c7d5ed06fef              # double 6.2621969999999996
	.quad	0x4001655da272862f              # double 2.1744949999999998
	.quad	0x401c6d8cf398e970              # double 7.1069829999999996
	.quad	0x401f878b3700474e              # double 7.8823670000000003
	.quad	0x40235705857afea4              # double 9.6699640000000002
	.quad	0x400681b7e0ac7da2              # double 2.813339
	.quad	0x400b5404ea4a8c15              # double 3.4160249999999999
	.quad	0x401b705143bf7271              # double 6.8596849999999998
	.quad	0x400eb8f7985271bd              # double 3.8403160000000001
	.quad	0x3fce71c53f39d1b3              # double 0.237847
	.quad	0x401de763a1900820              # double 7.4759659999999997
	.quad	0x4005df3493858362              # double 2.7339869999999999
	.quad	0x401de30f49491f2e              # double 7.4717380000000002
	.quad	0x40078dd93c46d82c              # double 2.9442620000000002
	.quad	0x40096b4e11dbca97              # double 3.1773950000000002
	.quad	0x4016da5d28d002e2              # double 5.7132459999999998
	.quad	0x4021a32f87ad080b              # double 8.8187219999999993
	.quad	0x401eae1b8ed1bf7b              # double 7.6700270000000001
	.quad	0x3ff6b2f555821294              # double 1.4186909999999999
	.quad	0x3fe105425f202108              # double 0.53189200000000003
	.quad	0x401758ad25679896              # double 5.8365980000000004
	.quad	0x3ff6c56605ee568a              # double 1.4231929999999999
	.quad	0x3fd70420f6f08cc5              # double 0.35962699999999997
	.quad	0x401b34816f0068dc              # double 6.8012750000000004
	.quad	0x3ffd3355cd91eeaa              # double 1.8250329999999999
	.quad	0x40041b44e50c5eb3              # double 2.513315
	.quad	0x4015be5de15ca6ca              # double 5.435905
	.quad	0x3fcbf1c75818c5ca              # double 0.21831600000000001
	.quad	0x401728b9baa1511e              # double 5.789771
	.quad	0x401044b9cb6848bf              # double 4.0671150000000003
	.quad	0x3ffd9dcc63f14120              # double 1.8510249999999999
	.quad	0x400bf0b5aa715832              # double 3.492534
	.quad	0x40119243dccea290              # double 4.3928370000000001
	.quad	0x40027ec3dab5c39c              # double 2.3118970000000001
	.quad	0x4011f298cc144029              # double 4.4869110000000001
	.quad	0x3fef9cb46bacf744              # double 0.98787899999999995
	.quad	0x4019f59ccfaeff5c              # double 6.4898559999999996
	.quad	0x3ffcbda1ec4e7254              # double 1.796297
	.quad	0x4014b88377186552              # double 5.1801890000000004
	.quad	0x3ffa4d90e23af31b              # double 1.643937
	.quad	0x400d456c508b32cf              # double 3.6588980000000002
	.quad	0x4006c6dacabc5155              # double 2.8470970000000002
	.quad	0x4007afe47991bc56              # double 2.9608850000000002
	.quad	0x400624c8366516db              # double 2.76796
	.quad	0x4003e3547e06961c              # double 2.4860009999999999
	.quad	0x4022b08c1db0142f              # double 9.3448189999999993
	.quad	0x4020635f3d7d3911              # double 8.1940860000000004
	.quad	0x3ffac6833c60029f              # double 1.673465
	.quad	0x3ff002b734b51373              # double 1.0006630000000001
	.quad	0x3ff66e2541d8e864              # double 1.401891
	.quad	0x4023881e8a2ec28b              # double 9.7658579999999997
	.quad	0x40095ab6d00b45ae              # double 3.1692939999999998
	.quad	0x3fe109e774eebf66              # double 0.53245900000000002
	.quad	0x3ffcd4a0a0f4d7ae              # double 1.801911
	.quad	0x401ad63fdd65a145              # double 6.7092280000000004
	.quad	0x400611b584b1ab08              # double 2.7586469999999998
	.quad	0x4001dd640e9d51b5              # double 2.233101
	.quad	0x3f90c49ba5e353f8              # double 0.016375000000000001
	.quad	0x40126cd8127b2cc7              # double 4.606293
	.quad	0x401089bc33f85511              # double 4.1345070000000002
	.quad	0x4015a9780baa582e              # double 5.4154970000000002
	.quad	0x402239f46ed245b3              # double 9.1131930000000008
	.quad	0x4006e545846e8f2a              # double 2.8619490000000001
	.quad	0x40101e835dedf1e1              # double 4.0297980000000004
	.quad	0x4021b8efbb0e5e68              # double 8.8612040000000007
	.quad	0x4019e54c122749f1              # double 6.4739230000000001
	.quad	0x400c58da3c21187e              # double 3.5433849999999998
	.quad	0x4013a1e3eaf6837f              # double 4.9080959999999996
	.quad	0x3fe536d43d03968d              # double 0.66294299999999995
	.quad	0x40197f1bef49cf57              # double 6.3741300000000001
	.quad	0x3fdd2d490e66cb10              # double 0.45588899999999999
	.quad	0x40233c4a4095f245              # double 9.6177539999999996
	.quad	0x40088bd2b6f19935              # double 3.068273
	.quad	0x3fd5e071c53f39d2              # double 0.34182400000000002
	.quad	0x3fe15d0d8cb07d0b              # double 0.54260900000000001
	.quad	0x3fe0ff1737542a24              # double 0.53113900000000003
	.quad	0x4017d00da1a93294              # double 5.9531770000000002
	.quad	0x4017e9a0ad8a1166              # double 5.9781519999999997
	.quad	0x401a02becedd4845              # double 6.5026809999999999
	.quad	0x400593d6c7219221              # double 2.697187
	.quad	0x3fe1d985ad538ac2              # double 0.55780300000000005
	.quad	0x3ff3f2a27f1b6912              # double 1.246737
	.quad	0x3ff75e892253111f              # double 1.46058
	.quad	0x4014ea6b93ccd0ff              # double 5.2289260000000004
	.quad	0x40209c332f01754b              # double 8.305078
	.quad	0x40207008c8259e1f              # double 8.2188169999999996
	.quad	0x40217db0ff10ecb7              # double 8.7454909999999995
	.quad	0x3feacd466f5019f4              # double 0.83755800000000002
	.quad	0x4007ad62bf11f927              # double 2.9596610000000001
	.quad	0x3fdc285ec3dab5c4              # double 0.43996400000000002
	.quad	0x3ff0aa4db163baba              # double 1.0415779999999999
	.quad	0x40122436fc158fb4              # double 4.5353659999999998
	.quad	0x402180a67620ee8d              # double 8.7512699999999999
	.quad	0x3ff84d5f99c38b05              # double 1.5188900000000001
	.quad	0x4015e987c6327ed8              # double 5.4780569999999997
	.quad	0x4021803126e978d5              # double 8.750375
	.quad	0x401dbd912556d19e              # double 7.4351240000000001
	.quad	0x401a61843c3a42f2              # double 6.5952310000000001
	.quad	0x40056fae7924af0c              # double 2.679532
	.quad	0x3fd4c4da9003eea2              # double 0.324515
	.quad	0x402356175d13d74d              # double 9.6681469999999993
	.quad	0x3fc75aaf78feef5f              # double 0.18245500000000001
	.quad	0x401c07ddca4b124d              # double 7.007682
	.quad	0x400c09ea9a3d2d88              # double 3.504842
	.quad	0x4011ee108c3f3e03              # double 4.4824849999999996
	.quad	0x401fbbc48f10a99b              # double 7.9333669999999996
	.quad	0x400f928954a7f801              # double 3.9465509999999999
	.quad	0x3fb351ac9afe1da8              # double 0.075465000000000004
	.quad	0x3fff4cfd08d4bad8              # double 1.956296
	.quad	0x3fff2dce39b456b4              # double 1.9486829999999999
	.quad	0x4018ec0bdcad14a1              # double 6.2305140000000003
	.quad	0x40168f1455219a84              # double 5.6397259999999996
	.quad	0x401cef840181e03f              # double 7.2339019999999996
	.quad	0x4019765dbfceb789              # double 6.3655920000000004
	.quad	0x4018c7bbb62413db              # double 6.1950519999999996
	.quad	0x4023e81d19157abc              # double 9.9533470000000008
	.quad	0x4020864b44a1f080              # double 8.2622929999999997
	.quad	0x4005d39c94f69caa              # double 2.728326
	.quad	0x4021c680389f83be              # double 8.8876969999999993
	.quad	0x3ff61fd47bedb728              # double 1.382771
	.quad	0x4021774d594f26af              # double 8.7330120000000004
	.quad	0x4018eb8023a6ce36              # double 6.2299810000000004
	.quad	0x4021910c4dec1c1d              # double 8.7832969999999992
	.quad	0x401d936262cba733              # double 7.3939300000000001
	.quad	0x40060cbf2b239a39              # double 2.756224
	.quad	0x4019ecd0fe8ab4fa              # double 6.4812659999999997
	.quad	0x401b97924af0bf1a              # double 6.8980189999999997
	.quad	0x4022c832db1e9f27              # double 9.3910129999999991
	.quad	0x3ff1de18efbb0e5e              # double 1.1167229999999999
	.quad	0x3fd8b51372a38b8b              # double 0.38605200000000001
	.quad	0x401a2ac9f2fdb8fe              # double 6.5417860000000001
	.quad	0x401869274e22a2c2              # double 6.1026889999999998
	.quad	0x3fc677036c9c0a8d              # double 0.175507
	.quad	0x40213b12f4cf4a56              # double 8.6153790000000008
	.quad	0x401c08b0d0edc3bd              # double 7.0084869999999997
	.quad	0x4016feaa2a0a9585              # double 5.7486959999999998
	.quad	0x401cfc2224edf612              # double 7.2462239999999998
	.quad	0x400212e8c0485a0c              # double 2.259233
	.quad	0x3ffd1f0d413122b8              # double 1.8200810000000001
	.quad	0x4018882d38476f2a              # double 6.1329849999999997
	.quad	0x4022fbf63c31df76              # double 9.4921129999999998
	.quad	0x4014968db8bac711              # double 5.1470250000000002
	.quad	0x40116892253111f1              # double 4.3521200000000002
	.quad	0x40118eb5f5f0b285              # double 4.3893659999999999
	.quad	0x3fed266559f6ec5b              # double 0.910937
	.quad	0x4021ca875d56f32c              # double 8.8955640000000002
	.quad	0x401be815a07b352b              # double 6.9766450000000004
	.quad	0x4023cfc3f811f4f5              # double 9.9057919999999999
	.quad	0x3fef4e4649906cca              # double 0.97830499999999998
	.quad	0x4015e69d7342edbb              # double 5.4752099999999997
	.quad	0x401a6f00ef1348b2              # double 6.6084019999999999
	.quad	0x4023bc2e557de0d6              # double 9.8675409999999992
	.quad	0x3fe18292817763e5              # double 0.54718900000000004
	.quad	0x400b655fbb517a46              # double 3.424499
	.quad	0x4019473b85e80bed              # double 6.3195629999999996
	.quad	0x401a0438d1d8a548              # double 6.5041229999999999
	.quad	0x400db5a8deb0fadf              # double 3.7137009999999999
	.quad	0x400488405b39e27a              # double 2.5665290000000001
	.quad	0x401833d9231c64bb              # double 6.0506330000000004
	.quad	0x3fd1aa1511dffc54              # double 0.27600599999999997
	.quad	0x400a2409e55c0fcb              # double 3.2675969999999999
	.quad	0x3ffaa7bc3c5bd0e1              # double 1.665951
	.quad	0x400fae6c8f755369              # double 3.9601679999999999
	.quad	0x4022fc9c90c4dec2              # double 9.4933820000000004
	.quad	0x401e869ca9ef5233              # double 7.6314570000000001
	.quad	0x40218b71da37ef5b              # double 8.7723530000000007
	.quad	0x4020a79b994e1a3f              # double 8.3273589999999995
	.quad	0x401db60f5e41d4b7              # double 7.4277930000000003
	.quad	0x4023fd2bd3c36113              # double 9.9944749999999995
	.quad	0x3ff8348b220791c5              # double 1.5128280000000001
	.quad	0x401a0bcf0b6b6e0d              # double 6.5115319999999999
	.quad	0x4004174b40703298              # double 2.511374
	.quad	0x401b142784a9478d              # double 6.7696820000000004
	.quad	0x400b26a098915bdd              # double 3.3938609999999998
	.quad	0x4023970739b024f6              # double 9.7949769999999994
	.quad	0x3fc723571d1d4739              # double 0.18076600000000001
	.quad	0x3ff379f23465625a              # double 1.2172719999999999
	.quad	0x402174c5fbf83383              # double 8.7280730000000002
	.quad	0x40204b211cb039ef              # double 8.1467369999999999
	.quad	0x3ff54ab4fa48301a              # double 1.3307389999999999
	.quad	0x40213475818c5c9a              # double 8.6024589999999996
	.quad	0x4003433d6c721922              # double 2.407832
	.quad	0x3fe0cf54d1e96c40              # double 0.52530900000000003
	.quad	0x4013dee30caa326e              # double 4.9676629999999999
	.quad	0x400b05cf67b1c001              # double 3.377837
	.quad	0x40201d0e56041893              # double 8.0567499999999992
	.quad	0x40111d4067cf1c32              # double 4.2785659999999996
	.quad	0x401954d013a92a30              # double 6.3328249999999997
	.quad	0x4023ed0cc35ce182              # double 9.9629879999999993
	.quad	0x40133e9f2778140e              # double 4.8111540000000002
	.quad	0x40215bb320535c9e              # double 8.6791009999999993
	.quad	0x401cb6d698fe6927              # double 7.178553
	.quad	0x4022c0c1b97353b5              # double 9.3764780000000005
	.quad	0x3ff5789613d31b9b              # double 1.3419399999999999
	.quad	0x401749c12f09d8c7              # double 5.8220260000000001
	.quad	0x3fb35c39bcba3012              # double 0.075625999999999999
	.quad	0x4008f2cb641700ce              # double 3.1185520000000002
	.quad	0x4019697353b4b2fb              # double 6.3529790000000004
	.quad	0x40217f0995aaf790              # double 8.7481200000000001
	.quad	0x3fc9b81b64e05469              # double 0.200931
	.quad	0x401b931fcd24e161              # double 6.8936760000000001
	.quad	0x401a461e0c5a8156              # double 6.5684740000000001
	.quad	0x4021d879e59f2baa              # double 8.9228050000000003
	.quad	0x401b7fcb4f1e4b45              # double 6.8747990000000003
	.quad	0x400e9687b139c94f              # double 3.8235009999999998
	.quad	0x4022a21af7d30ad4              # double 9.3166119999999992
	.quad	0x40063179bfdf7e80              # double 2.7741579999999999
	.quad	0x401cebe9856a37ac              # double 7.2303829999999998
	.quad	0x3fd57b677f6b1a2a              # double 0.33565699999999998
	.quad	0x400da3fcc9ea9a3d              # double 3.7050719999999999
	.quad	0x4023abba3443d46b              # double 9.8354049999999997
	.quad	0x401a9845dc832336              # double 6.6487040000000004
	.quad	0x3fddc6d1e108c3f4              # double 0.46526000000000001
	.quad	0x400b01cc532a4980              # double 3.3758780000000002
	.quad	0x400c5b445ed4a1ad              # double 3.5445639999999998
	.quad	0x400a8b53d640e9d5              # double 3.318031
	.quad	0x400c9170d62bf120              # double 3.5710160000000002
	.quad	0x40239f59253543af              # double 9.8112270000000005
	.quad	0x400223baba7b9171              # double 2.2674460000000001
	.quad	0x4022443a2163fdd6              # double 9.1332559999999994
	.quad	0x4010b6fbd273d5bb              # double 4.1786950000000003
	.quad	0x401502d666a98245              # double 5.2527710000000001
	.quad	0x4020289e12a51e32              # double 8.0793309999999998
	.quad	0x40133ee34fc610f1              # double 4.8114140000000001
	.quad	0x40233f93dd97f62b              # double 9.6241749999999992
	.quad	0x4018417acc4ef88c              # double 6.0639450000000004
	.quad	0x3fe793d31b9b66f9              # double 0.73679499999999998
	.quad	0x4021bf88333fc86d              # double 8.8740860000000001
	.quad	0x3ffbf7d73c925786              # double 1.748008
	.quad	0x402051be2b4959e6              # double 8.1596539999999997
	.quad	0x402156687f455a7d              # double 8.6687659999999997
	.quad	0x40014aed99cbee80              # double 2.1615859999999998
	.quad	0x4012d100e6afcce2              # double 4.7041050000000002
	.quad	0x4020c4caf709b739              # double 8.3843610000000002
	.quad	0x400ffb8f57f737da              # double 3.9978319999999998
	.quad	0x4017f32b990afe61              # double 5.9874710000000002
	.quad	0x3ff2939c94f69caa              # double 1.161038
	.quad	0x4015d2645e4e69f0              # double 5.4554609999999997
	.quad	0x40216fd30ad46f58              # double 8.7184069999999991
	.quad	0x40076710cb295e9e              # double 2.925325
	.quad	0x401a571f36262cba              # double 6.5850799999999996
	.quad	0x3ff49567dbb16c1e              # double 1.286476
	.quad	0x40023d6cf850df16              # double 2.2799930000000002
	.quad	0x401847d06bbdbe3c              # double 6.0701309999999999
	.quad	0x4017cb8aa00192a7              # double 5.9487709999999998
	.quad	0x401e131b9b66f933              # double 7.5186599999999996
	.quad	0x3ff3e2ff08893b7e              # double 1.2429190000000001
	.quad	0x4016f641b328b6d8              # double 5.7404849999999996
	.quad	0x401df8ff75968512              # double 7.4931619999999999
	.quad	0x40221c8216c61523              # double 9.0556800000000006
	.quad	0x4007290de093532e              # double 2.8950459999999998
	.quad	0x401ecc1a47a9e2bd              # double 7.699319
	.quad	0x4021f1e75360d024              # double 8.9724679999999992
	.quad	0x4022f88b9778572a              # double 9.4854400000000005
	.quad	0x3ff07b341e14bdfd              # double 1.030079
	.quad	0x40220f192ea52ef9              # double 9.0294889999999999
	.quad	0x4019f0589acbc8c1              # double 6.4847130000000002
	.quad	0x3ff2be5fb71fbc5e              # double 1.171478
	.quad	0x402316ed8904f6e0              # double 9.5447810000000004
	.quad	0x402091f59ccfaeff              # double 8.2850769999999994
	.quad	0x3ff5c1b003686a4d              # double 1.3597870000000001
	.quad	0x401bb67d77fae361              # double 6.9282130000000004
	.quad	0x4015735daad60200              # double 5.3626620000000003
	.quad	0x4022dec3fc43b2dd              # double 9.4350889999999996
	.quad	0x402282b4959e6256              # double 9.2552839999999996
	.quad	0x3fecb4e54f7a9197              # double 0.89708200000000005
	.quad	0x4020e3a2163fdd66              # double 8.4445960000000007
	.quad	0x40149987c6327ed8              # double 5.1499319999999997
	.quad	0x40220ca4b124d09a              # double 9.0246940000000002
	.quad	0x4000fb55ac03ff69              # double 2.122722
	.quad	0x3fd88387df5cf249              # double 0.38302799999999998
	.quad	0x40237c851a869403              # double 9.7432029999999993
	.quad	0x402107fde7210be9              # double 8.5156089999999995
	.quad	0x401e0c621b7e0ac8              # double 7.5120930000000001
	.quad	0x40137fbd273d5bab              # double 4.8747449999999999
	.quad	0x3fea5ab191dde376              # double 0.82357100000000005
	.quad	0x400ee6a4ca4f440b              # double 3.862619
	.quad	0x4017d8123810e886              # double 5.9610070000000004
	.quad	0x402003a14cec41dd              # double 8.0070899999999998
	.quad	0x400de704bc27631b              # double 3.7378019999999998
	.quad	0x402162ba38759253              # double 8.6928269999999994
	.quad	0x3fe16ad0c3d25248              # double 0.54428900000000002
	.quad	0x40043c2b94d94079              # double 2.5293800000000002
	.quad	0x400b49a6f826edab              # double 3.4109630000000002
	.quad	0x4023cdc22ab25b31              # double 9.9018719999999991
	.quad	0x40125bb8800eae19              # double 4.5895710000000003
	.quad	0x40110fbd6a593a2e              # double 4.265371
	.quad	0x401b47a355043e53              # double 6.8199589999999999
	.quad	0x4014530ad46f587d              # double 5.0810959999999996
	.quad	0x4003fd11be6e6538              # double 2.4985689999999998
	.quad	0x401f0632c1f42bb6              # double 7.7560529999999996
	.quad	0x40119536933a0408              # double 4.3957160000000002
	.quad	0x4019d25a250f8402              # double 6.4554220000000004
	.quad	0x4012b2db1e9f2778              # double 4.6746639999999999
	.quad	0x400ab48dc11e42e1              # double 3.3381609999999999
	.quad	0x401e9dd11be6e654              # double 7.6541180000000004
	.quad	0x40173166e008e9b4              # double 5.7982440000000004
	.quad	0x4019b36694898f60              # double 6.4251959999999997
	.quad	0x401172cdc011d367              # double 4.3621129999999999
	.quad	0x3febd2dcb1465e89              # double 0.86948999999999999
	.quad	0x400fee514c22ee42              # double 3.9913660000000002
	.quad	0x40058d48882f0e0b              # double 2.6939860000000002
	.quad	0x401643d9231c64bb              # double 5.5662580000000004
	.quad	0x400936512a94ff00              # double 3.1515219999999999
	.quad	0x40200cf5b1c86488              # double 8.0253119999999996
	.quad	0x3ffea33ce63a5c1c              # double 1.9148529999999999
	.quad	0x4020801d19157abc              # double 8.2502220000000008
	.quad	0x402180807357e671              # double 8.7509800000000002
	.quad	0x3fa19b6f5caf2d80              # double 0.034389000000000003
	.quad	0x4016a38151a43782              # double 5.6596729999999997
	.quad	0x401a555e28aa4335              # double 6.583367
	.quad	0x3fe7572de43ed95a              # double 0.72939200000000004
	.quad	0x400dd16ebd4cfd09              # double 3.7272620000000001
	.quad	0x40209a0f0a5efe93              # double 8.3008959999999998
	.quad	0x40231048387df5cf              # double 9.5318009999999997
	.quad	0x4012b6527a20578e              # double 4.6780489999999997
	.quad	0x3fed2615a8deb0fb              # double 0.91089900000000001
	.quad	0x401eaabda0dfdef8              # double 7.6667389999999997
	.quad	0x3ff7bbf727136a40              # double 1.48339
	.quad	0x3ff8734b51372a39              # double 1.5281480000000001
	.quad	0x400e4fa1e3eaf683              # double 3.7888829999999998
	.quad	0x4017c7bedb7281fe              # double 5.9450640000000003
	.quad	0x40189d5e8d5410f9              # double 6.1536809999999997
	.quad	0x3feb6a209aaa3ad2              # double 0.85670500000000005
	.quad	0x4015605ee568a505              # double 5.344112
	.quad	0x40213b5d249e44fa              # double 8.615945
	.quad	0x3ffed703f2d3c793              # double 1.927494
	.quad	0x401b8438088509c0              # double 6.8791200000000003
	.quad	0x4010f03bc4d22c88              # double 4.2346029999999999
	.quad	0x401a549e8815e397              # double 6.5826359999999999
	.quad	0x3ff8ae9680e06530              # double 1.542624
	.quad	0x3fde6ec5b078d930              # double 0.47551100000000002
	.quad	0x4015b668c2613900              # double 5.428134
	.quad	0x4010047ae147ae14              # double 4.0043749999999996
	.quad	0x4020f87a28bb0a2d              # double 8.4853070000000006
	.quad	0x3ff10638c9752978              # double 1.064019
	.quad	0x401fbbb40b34e768              # double 7.9333039999999996
	.quad	0x401a5638433d6c72              # double 6.5841989999999999
	.quad	0x4013eaec4a4095f2              # double 4.9794169999999998
	.quad	0x40112ce95faa8a83              # double 4.2938590000000003
	.quad	0x4013ed86ec17ebaf              # double 4.9819599999999999
	.quad	0x401434577d955715              # double 5.0511150000000002
	.quad	0x40060ffc9795b35b              # double 2.757806
	.quad	0x4013954477ff151e              # double 4.8957689999999996
	.quad	0x4014b99c7bcc2939              # double 5.1812610000000001
	.quad	0x4022419dece57109              # double 9.1281580000000009
	.quad	0x4022d2e946801712              # double 9.4119360000000007
	.quad	0x3fb7415b1422ccb4              # double 0.090841000000000005
	.quad	0x400352b3cc4ac6ce              # double 2.4153820000000001
	.quad	0x4010206961c36977              # double 4.0316520000000002
	.quad	0x401e32deca2552a0              # double 7.5496780000000001
	.quad	0x4021ca9ed3953dea              # double 8.8957429999999995
	.quad	0x40164a604e1e7104              # double 5.5726329999999997
	.quad	0x40044d22c881e471              # double 2.5376639999999999
	.quad	0x3fd3c9795b35b0bc              # double 0.309172
	.quad	0x401dcc02f2f98740              # double 7.44923
	.quad	0x4010d872f33ca31e              # double 4.2113759999999996
	.quad	0x40105ddbdb5d8948              # double 4.0916589999999999
	.quad	0x3f97d84901d19158              # double 0.023286000000000001
	.quad	0x40211fd805e5f30e              # double 8.5621949999999991
	.quad	0x4022ecaa7589efd8              # double 9.4622379999999992
	.quad	0x3ff438cda6e75ff6              # double 1.263868
	.quad	0x40218c8323358f2e              # double 8.774438
	.quad	0x401a10870110a138              # double 6.51614
	.quad	0x3ff6e31d712a0ec7              # double 1.4304479999999999
	.quad	0x401ddc4cdfaca362              # double 7.4651370000000004
	.quad	0x400189fbe76c8b44              # double 2.1923750000000002
	.quad	0x3ff3006290eed02d              # double 1.187594
	.quad	0x3fc518548a9bcfd5              # double 0.16480500000000001
	.quad	0x401d325e99e94ab2              # double 7.2991890000000001
	.quad	0x40170310dbf0563f              # double 5.7529940000000002
	.quad	0x3ff6781733226c3c              # double 1.4043190000000001
	.quad	0x400bce2435696e59              # double 3.4756550000000002
	.quad	0x40148edbb59ddc1e              # double 5.1395099999999996
	.quad	0x4016c74213a0c6b5              # double 5.6945880000000004
	.quad	0x401174c8366516db              # double 4.364045
	.quad	0x40181a71151611ba              # double 6.0258219999999998
	.quad	0x3fff1a9b8cb8e087              # double 1.9439960000000001
	.quad	0x40192e874c8ffb8b              # double 6.2954379999999999
	.quad	0x4005222ff08893b8              # double 2.6416930000000001
	.quad	0x401a26b54e2b063e              # double 6.537801
	.quad	0x400df9018e757929              # double 3.7465850000000001
	.quad	0x400b9f3e0370cdc8              # double 3.4527549999999998
	.quad	0x4021a69fcf3dc055              # double 8.8254380000000001
	.quad	0x400290b7c3504c49              # double 2.3206630000000001
	.quad	0x40203ba57eaa2a0b              # double 8.1164970000000007
	.quad	0x400faeeae9ee45c3              # double 3.9604089999999998
	.quad	0x4002cb990afe60c4              # double 2.3494130000000002
	.quad	0x40222d43aeab7996              # double 9.0884070000000001
	.quad	0x401fbe4a7b4e54f8              # double 7.9358310000000003
	.quad	0x4013d8b3700474da              # double 4.9616220000000002
	.quad	0x402026f954eb13e0              # double 8.0761210000000005
	.quad	0x401d405ff1d81f10              # double 7.3128659999999996
	.quad	0x400e53250b52439a              # double 3.7905980000000001
	.quad	0x401a7559b3d07c85              # double 6.6146000000000003
	.quad	0x3fe086705c896dd2              # double 0.51641099999999995
	.quad	0x40201409e55c0fcb              # double 8.0391379999999995
	.quad	0x3fd69945b6c3760c              # double 0.353105
	.quad	0x401fa2517e77d524              # double 7.9085140000000003
	.quad	0x40222e049ecb31c2              # double 9.0898789999999998
	.quad	0x3ffbec9c0a8d21bc              # double 1.745266
	.quad	0x3ffee06b7aa25d8d              # double 1.9297899999999999
	.quad	0x401a61a048e043a2              # double 6.5953379999999999
	.quad	0x4013af8487b99d45              # double 4.9214039999999999
	.quad	0x402227c3b4f61672              # double 9.0776649999999997
	.quad	0x400d39f3c70c996b              # double 3.6532969999999998
	.quad	0x3ffd46748e475600              # double 1.829701
	.quad	0x40108394317acc4f              # double 4.128495
	.quad	0x3ff0584b1ab0856e              # double 1.0215559999999999
	.quad	0x400c7687b139c94f              # double 3.5578759999999998
	.quad	0x402165ea24cc6823              # double 8.699052
	.quad	0x4003f75fb2edfe76              # double 2.4957880000000001
	.quad	0x4023d74f50a02b84              # double 9.9205269999999998
	.quad	0x3fe2dcf893faf428              # double 0.58947400000000005
	.quad	0x3fe43a68b19a415f              # double 0.63212999999999997
	.quad	0x401f0bdad7518b0d              # double 7.7615769999999999
	.quad	0x40180e1869835159              # double 6.0137650000000002
	.quad	0x4017f46801711948              # double 5.9886780000000002
	.quad	0x4020c80e6afcce1c              # double 8.3907349999999993
	.quad	0x40107d7e02645e4e              # double 4.1225509999999996
	.quad	0x40145fbfc6540cc8              # double 5.0935050000000004
	.quad	0x402105d35653c9ac              # double 8.5113780000000005
	.quad	0x400b90dae3e6c4c6              # double 3.4457300000000002
	.quad	0x4020c07e5a78f25a              # double 8.3759639999999997
	.quad	0x4017adaa0cae642c              # double 5.919594
	.quad	0x401eda515ce9e5e2              # double 7.7132009999999998
	.quad	0x3ff90853c148344c              # double 1.564533
	.quad	0x401155c31593e5fb              # double 4.3337519999999996
	.quad	0x401c76b6e0d23d4f              # double 7.1159319999999999
	.quad	0x3fffeedbb59ddc1e              # double 1.9958149999999999
	.quad	0x401c877d0f1f57b4              # double 7.1323129999999999
	.quad	0x40209151a437824d              # double 8.2838259999999995
	.quad	0x40145e7296f6512b              # double 5.0922340000000004
	.quad	0x40007a871a3b14a9              # double 2.059828
	.quad	0x4022a6d916872b02              # double 9.3258749999999999
	.quad	0x4018ce6c4c5974e6              # double 6.2015849999999997
	.quad	0x400c7f141205bc02              # double 3.5620500000000002
	.quad	0x4011b6e04c059210              # double 4.4285899999999998
	.quad	0x400241919ac79703              # double 2.282016
	.quad	0x3feb0ae31d712a0f              # double 0.84507900000000002
	.quad	0x4010900e27e0ef9a              # double 4.1406790000000004
	.quad	0x400df5a20ddc6195              # double 3.7449379999999999
	.quad	0x4014d0de4c51116b              # double 5.2039730000000004
	.quad	0x401305336deb95e6              # double 4.7550790000000003
	.quad	0x40218902795703f3              # double 8.7675970000000003
	.quad	0x400ddccd96206855              # double 3.7328139999999999
	.quad	0x401ee1b93037d630              # double 7.7204329999999999
	.quad	0x4023f84e831ad213              # double 9.9849739999999993
	.quad	0x4022f7c7820a30db              # double 9.4839439999999992
	.quad	0x401b5c426351def0              # double 6.8400970000000001
	.quad	0x40195fad6cb53501              # double 6.3434350000000004
	.quad	0x4018e4b2745bf26f              # double 6.2233369999999999
	.quad	0x4019726b2f23033a              # double 6.3617369999999998
	.quad	0x40220c649906cca3              # double 9.0242050000000003
	.quad	0x401140a84be40421              # double 4.313142
	.quad	0x401cf80dc33721d5              # double 7.2422399999999998
	.quad	0x400852af1455219b              # double 3.0403730000000002
	.quad	0x4011fb6aa4b9884c              # double 4.4955239999999996
	.quad	0x40010cebbba55d1c              # double 2.1313089999999999
	.quad	0x40224d028a1dfb94              # double 9.1504100000000008
	.quad	0x401fdad8a116659d              # double 7.9637169999999999
	.quad	0x4020f653c9abb01d              # double 8.4811080000000008
	.quad	0x402036d3f9e7b80b              # double 8.1070860000000007
	.quad	0x40101e4f3343fa2b              # double 4.0295990000000002
	.quad	0x4019da3fcc9ea9a4              # double 6.4631340000000002
	.quad	0x4011a88b545c78a7              # double 4.4145940000000001
	.quad	0x3fd34d2f5dbb9cfa              # double 0.30158600000000002
	.quad	0x402032ed56b00ffe              # double 8.0994670000000006
	.quad	0x40206c7b45f17bd9              # double 8.2118780000000005
	.quad	0x40146be121ee6751              # double 5.1053509999999998
	.quad	0x3fe2d206c8711d7a              # double 0.58813800000000005
	.quad	0x401621eb851eb852              # double 5.5331250000000001
	.quad	0x3ff4230121682f94              # double 1.2585459999999999
	.quad	0x4001876a7264a16a              # double 2.1911209999999999
	.quad	0x3ff9161a1db877ab              # double 1.567896
	.quad	0x401b4d1e96c3fc44              # double 6.8253120000000003
	.quad	0x400db245b291b824              # double 3.7120470000000001
	.quad	0x4018a9a88ace24bc              # double 6.1656820000000003
	.quad	0x400fc0b2c83ec893              # double 3.9690910000000001
	.quad	0x401f710c2c5e2cdc              # double 7.860398
	.quad	0x40158fba450acc0c              # double 5.3903590000000001
	.quad	0x40118a7829068987              # double 4.385224
	.quad	0x4013802602c9081c              # double 4.8751449999999998
	.quad	0x3fd25fbb517a4617              # double 0.28709299999999999
	.quad	0x401b3882adc4c9c9              # double 6.805186
	.quad	0x3fec7f6b1a2a4db1              # double 0.89055399999999996
	.quad	0x400f2eac8605681f              # double 3.8977900000000001
	.quad	0x4001d1737542a23c              # double 2.227271
	.quad	0x40117b430f49491f              # double 4.3703729999999998
	.quad	0x4018f8bc9cd3e0bd              # double 6.2429069999999998
	.quad	0x40033953586ca8a0              # double 2.4029910000000001
	.quad	0x4016e10c6f7a0b5f              # double 5.7197740000000001
	.quad	0x3ffe6d080b673c4f              # double 1.9016189999999999
	.quad	0x40234b63d3e4ef03              # double 9.6472460000000008
	.quad	0x401c356798958d9b              # double 7.0521529999999997
	.quad	0x40127fadafd11383              # double 4.6246859999999996
	.quad	0x401ffbc2b94d9408              # double 7.9958600000000004
	.quad	0x400f88344c37e6f7              # double 3.941506
	.quad	0x4019040aaeafab54              # double 6.2539470000000001
	.quad	0x3fffd21144cbe1eb              # double 1.9887859999999999
	.quad	0x4011a4e054690de1              # double 4.4110120000000004
	.quad	0x4001feec39799e52              # double 2.2494740000000002
	.quad	0x401ecf97edc7ef17              # double 7.7027279999999995
	.quad	0x4013d27bf61aa3f0              # double 4.9555509999999998
	.quad	0x4014f5cd91eeaa6d              # double 5.2400419999999999
	.quad	0x4014382f944241c4              # double 5.0548690000000001
	.quad	0x4021e0bef8ceb357              # double 8.9389570000000003
	.quad	0x40118fa7221858bc              # double 4.3902859999999997
	.quad	0x3fc229f9ce8d972d              # double 0.141906
	.quad	0x4023a138de6ddf01              # double 9.8148870000000005
	.quad	0x400d33343fa2ad3f              # double 3.6500020000000002
	.quad	0x40171a9478c868ba              # double 5.775957
	.quad	0x4004b0dbf0563ed1              # double 2.586357
	.quad	0x401b0ee17a02fb5d              # double 6.764532
	.quad	0x40225ba647fdc593              # double 9.1790029999999998
	.quad	0x3ffb5a8cdea033e8              # double 1.7096070000000001
	.quad	0x40083c0767d34df0              # double 3.0293109999999999
	.quad	0x3fd8d8cf398e9707              # double 0.38823299999999999
	.quad	0x3ffe17c0b1359792              # double 1.8807990000000001
	.quad	0x401368f9b13165d4              # double 4.8525150000000004
	.quad	0x4008d3d20f2becee              # double 3.1034280000000001
	.quad	0x400973dc054ef45a              # double 3.1815720000000001
	.quad	0x40144c64fdb09a67              # double 5.0746039999999999
	.quad	0x4019583d790752db              # double 6.3361720000000004
	.quad	0x4014d51a86940358              # double 5.2081090000000003
	.quad	0x3ff3ffe7e1fc08fa              # double 1.2499769999999999
	.quad	0x401b167e414e7ee9              # double 6.7719659999999999
	.quad	0x401348cc144028e5              # double 4.821091
	.quad	0x401fe0aa64c2f838              # double 7.9694000000000002
	.quad	0x4014288094e5d5b2              # double 5.0395529999999997
	.quad	0x3ff639e060fe4799              # double 1.38913
	.quad	0x401fb35d67ba237d              # double 7.9251610000000001
	.quad	0x400882a1b5c7cd8a              # double 3.0637850000000002
	.quad	0x4021ea0ec7432010              # double 8.9571439999999996
	.quad	0x401c673abc947065              # double 7.1008100000000001
	.quad	0x400fa4bd33d29564              # double 3.9554390000000001
	.quad	0x4022adbe3c105187              # double 9.3393420000000003
	.quad	0x40224b86f47b677f              # double 9.1475139999999993
	.quad	0x401c209917939a7c              # double 7.0318339999999999
	.quad	0x4010646a1a500d5f              # double 4.0980610000000004
	.quad	0x401c4bdf090f733b              # double 7.0740930000000004
	.quad	0x3ff9cb6f9fcb0c02              # double 1.6121669999999999
	.quad	0x4017a31550ca1cef              # double 5.9092609999999999
	.quad	0x400becf850df15a5              # double 3.4907080000000001
	.quad	0x4020279bfdf7e804              # double 8.0773620000000008
	.quad	0x401a4c382a12f901              # double 6.574433
	.quad	0x4015a1de26916441              # double 5.408074
	.quad	0x40113384fd2a62aa              # double 4.3003119999999999
	.quad	0x4020f1fd7a13c255              # double 8.4726370000000006
	.quad	0x4012f9efd86a8fc1              # double 4.7440790000000002
	.quad	0x401a085271bcdbbe              # double 6.508127
	.quad	0x400bbcd78c436fc1              # double 3.4672079999999998
	.quad	0x401dd40bbb1f255f              # double 7.4570759999999998
	.quad	0x4012bddb1209edc0              # double 4.6854060000000004
	.quad	0x3ff4e58e64b23140              # double 1.306044
	.quad	0x4021fcc92146a1a5              # double 8.993722
	.quad	0x4022db8130164841              # double 9.4287200000000002
	.quad	0x4003ec57e23f24d9              # double 2.490402
	.quad	0x3ff985479d4d8341              # double 1.5950390000000001
	.quad	0x401c44da4ce8101f              # double 7.0672389999999998
	.quad	0x40101f672b884407              # double 4.0306670000000002
	.quad	0x3fe271c3265add9c              # double 0.57638699999999998
	.quad	0x401fff9591cd1c7e              # double 7.9995940000000001
	.quad	0x401e806c8711d799              # double 7.6254140000000001
	.quad	0x4016af4b61fe21d9              # double 5.6711859999999996
	.quad	0x400a1a89b951c5c5              # double 3.2629579999999998
	.quad	0x3fcc26b2f23033a4              # double 0.21993099999999999
	.quad	0x40125dd2f1a9fbe7              # double 4.5916249999999996
	.quad	0x401946cb10342aaa              # double 6.319134
	.quad	0x40073c6f36ef8056              # double 2.904509
	.quad	0x4017432d2bb23572              # double 5.8156020000000002
	.quad	0x401395a7d24180d4              # double 4.8961480000000002
	.quad	0x401e412ec6bce853              # double 7.5636549999999997
	.quad	0x400c4dddf43c7d5f              # double 3.5380210000000001
	.quad	0x4001204af922962d              # double 2.140768
	.quad	0x3ffb01248d7e0264              # double 1.6877789999999999
	.quad	0x4021516030c23fab              # double 8.6589369999999999
	.quad	0x4010447a17f4128c              # double 4.066872
	.quad	0x4002e2120e1f7d74              # double 2.3603860000000001
	.quad	0x4008840fa9c12f0a              # double 3.0644830000000001
	.quad	0x4019cee84ad794a7              # double 6.4520580000000001
	.quad	0x4005ae054690de09              # double 2.7099709999999999
	.quad	0x3ff7d287c200c0f0              # double 1.488899
	.quad	0x4018122ee41919ac              # double 6.0177569999999996
	.quad	0x40208594af4f0d84              # double 8.2608999999999994
	.quad	0x3fe48870110a137f              # double 0.64165499999999998
	.quad	0x4012046f9b994e1a              # double 4.5043319999999998
	.quad	0x4021c783dff3f0fe              # double 8.889678
	.quad	0x4023c791819d2392              # double 9.8897820000000003
	.quad	0x4012a393ab430f49              # double 4.6597429999999997
	.quad	0x4015e435ac8a3726              # double 5.472861
	.quad	0x4022d94e1a3f4667              # double 9.4244240000000001
	.quad	0x4018a8583621fafd              # double 6.1643990000000004
	.quad	0x4010703a7546d3fa              # double 4.1095980000000001
	.quad	0x401ff21bc126a65d              # double 7.986434
	.quad	0x4010654ac29bf163              # double 4.0989180000000003
	.quad	0x3fedbfa4c61d8623              # double 0.92964400000000003
	.quad	0x400634f2f123c42a              # double 2.7758539999999998
	.quad	0x400ea3358f2e05cd              # double 3.8296920000000001
	.quad	0x3fdd56c0d6f544bb              # double 0.45841999999999999
	.quad	0x400200e5a35d67ba              # double 2.2504379999999999
	.quad	0x3ffff709b738e6d1              # double 1.9978119999999999
	.quad	0x4004effda4052d66              # double 2.6171829999999998
	.quad	0x4002cfc69728a611              # double 2.3514529999999998
	.quad	0x4019e00eae18ac9f              # double 6.4688059999999998
	.quad	0x40150644523f67f5              # double 5.2561200000000001
	.quad	0x3ff6803361565c2d              # double 1.406299
	.quad	0x40221092ccf6be38              # double 9.0323700000000002
	.quad	0x3fd01a47a9e2bcf9              # double 0.25160399999999999
	.quad	0x40152b66f9335d25              # double 5.2923850000000003
	.quad	0x40204c6cfc829cfe              # double 8.1492690000000003
	.quad	0x40006bb23571d1d4              # double 2.0525859999999998
	.quad	0x401ee9c3ce2089e3              # double 7.7282859999999997
	.quad	0x401daf562174c4ce              # double 7.421227
	.quad	0x4007d158b827fa1a              # double 2.97722
	.quad	0x40150e646f156191              # double 5.2640549999999999
	.quad	0x4021ba0ad8a11666              # double 8.8633640000000007
	.quad	0x4013e357a355043e              # double 4.9720139999999997
	.quad	0x3ffb3f8b9baa1512              # double 1.703014
	.quad	0x4014d73e24febd0a              # double 5.2101980000000001
	.quad	0x400c3d06bbdbe3c1              # double 3.529798
	.quad	0x4023c817763e4abe              # double 9.8908039999999993
	.quad	0x4008ee525892684d              # double 3.116368
	.quad	0x401f57470eb24a6b              # double 7.8352320000000004
	.quad	0x401e0b40b34e7686              # double 7.5109890000000004
	.quad	0x3ff49c0229a5ebb7              # double 1.2880879999999999
	.quad	0x401dc36c15d2d01c              # double 7.440842
	.quad	0x401e891105e1c151              # double 7.6338540000000003
	.quad	0x4022659f0a1be2b5              # double 9.1984790000000007
	.quad	0x40057d46b26bf877              # double 2.6861700000000002
	.quad	0x3fd168405b39e27a              # double 0.27198800000000001
	.quad	0x3fc4a4fca42aed14              # double 0.16128500000000001
	.quad	0x3feaded8d36b4c7f              # double 0.83970299999999998
	.quad	0x401688f605ab3aac              # double 5.6337510000000002
	.quad	0x40138104d551d68c              # double 4.8759949999999996
	.quad	0x3fe51f2dc2b0ea18              # double 0.66005599999999998
	.quad	0x3f561e4f765fd8ae              # double 0.0013500000000000001
	.quad	0x3ff2d056c508b32d              # double 1.175864
	.quad	0x40224396073de1e3              # double 9.1320040000000002
	.quad	0x4019b89bd8383ada              # double 6.4302820000000001
	.quad	0x4016e5cfede97d07              # double 5.7244260000000002
	.quad	0x4011e31098d477bc              # double 4.471743
	.quad	0x3fd502d9cf13ceea              # double 0.32829900000000001
	.quad	0x4011d25cc426351e              # double 4.4554320000000001
	.quad	0x4020bfde7210be94              # double 8.3747439999999997
	.quad	0x4014421fafc8b008              # double 5.0645740000000004
	.quad	0x400a3a95421c0443              # double 3.2786050000000002
	.quad	0x401b5d30f062d40b              # double 6.8410070000000003
	.quad	0x3fe9947496aad1d0              # double 0.79937199999999997
	.quad	0x401c12f2734f82f5              # double 7.0185029999999999
	.quad	0x3fffe5e892253112              # double 1.99363
	.quad	0x4021694e5d5b24ea              # double 8.7056760000000004
	.quad	0x4017ff6f08cc575c              # double 5.999447
	.quad	0x401547fa5d28d003              # double 5.3202910000000001
	.quad	0x4014c49906cca2db              # double 5.1919899999999997
	.quad	0x400eb04c48adeebb              # double 3.8360829999999999
	.quad	0x400e3675ddd2ae8e              # double 3.7765919999999999
	.quad	0x4011bda33bd9cae2              # double 4.4351929999999999
	.quad	0x3fff3882278d0cc3              # double 1.9512959999999999
	.quad	0x401d1d052934acb0              # double 7.27834
	.quad	0x401e56a0dbad3a60              # double 7.5845979999999997
	.quad	0x401a7f09528f190d              # double 6.6240589999999999
	.quad	0x40087dad7518b0d1              # double 3.061366
	.quad	0x401510268900c522              # double 5.2657720000000001
	.quad	0x401b306c00da1a93              # double 6.7972869999999999
	.quad	0x401778b827fa1a0d              # double 5.8678900000000001
	.quad	0x400979bdc69f8c22              # double 3.1844440000000001
	.quad	0x401f7f893faf4278              # double 7.8745469999999997
	.quad	0x4011a63779e9d0ea              # double 4.4123210000000004
	.quad	0x3ff2fcd035371972              # double 1.1867220000000001
	.quad	0x4023397c3d68405b              # double 9.6122759999999995
	.quad	0x401cdf34d6a161e5              # double 7.217975
	.quad	0x40101f6e82949a56              # double 4.0306949999999997
	.quad	0x4005f0c88a47ecff              # double 2.7425700000000002
	.quad	0x400101b75a74c09c              # double 2.1258379999999999
	.quad	0x400fb146e4c0df59              # double 3.9615610000000001
	.quad	0x3fe9bc0a06e9ff0d              # double 0.80420400000000003
	.quad	0x400d08259e1f3a58              # double 3.628978
	.quad	0x3ff4bb7521144cbe              # double 1.295766
	.quad	0x4004c9856a37ac3f              # double 2.5983990000000001
	.quad	0x401174c7b02d59d5              # double 4.3640429999999997
	.quad	0x4004a0d23d4f15e8              # double 2.5785260000000001
	.quad	0x401028676a7264a1              # double 4.0394569999999996
	.quad	0x4018e1cd1c7de508              # double 6.2205089999999998
	.quad	0x4021f1ae924f227d              # double 8.972035
	.quad	0x40116f3a14cec41e              # double 4.3586200000000002
	.quad	0x4020b090b417ca21              # double 8.3448539999999998
	.quad	0x3fbdcc857f3061c8              # double 0.11640200000000001
	.quad	0x4010d6bc1eff9f88              # double 4.2097020000000001
	.quad	0x4009ee8c0485a0be              # double 3.2414779999999999
	.quad	0x400721e3eaf6837f              # double 2.8915479999999998
	.quad	0x40175a9d1f601798              # double 5.8384900000000002
	.quad	0x40193a2595bbbe88              # double 6.3067840000000004
	.quad	0x401dc4306e5cd4ed              # double 7.4415909999999998
	.quad	0x40172ec679cc74b8              # double 5.7956789999999998
	.quad	0x4020645fbf83382e              # double 8.1960429999999995
	.quad	0x3fd350763a190082              # double 0.301786
	.quad	0x3ff909246bf01323              # double 1.564732
	.quad	0x400e9e586013660e              # double 3.8273169999999999
	.quad	0x402319df761cbccf              # double 9.5505329999999997
	.quad	0x400e1f644955b467              # double 3.7653279999999998
	.quad	0x400f9e9856a37ac4              # double 3.952439
	.quad	0x400dc2a90cd423d9              # double 3.7200489999999999
	.quad	0x3ffddf212d773190              # double 1.8669750000000001
	.quad	0x40239ba7daa4fca4              # double 9.8040149999999997
	.quad	0x400d7dc8b86b15f9              # double 3.6864180000000002
	.quad	0x4022d4c70867ad8e              # double 9.4155809999999995
	.quad	0x401570f4d7add15f              # double 5.360309
	.quad	0x4014c350d2806af4              # double 5.1907379999999996
	.quad	0x401d8d7f0ed3d85a              # double 7.3881800000000002
	.quad	0x4011ddf9378ee286              # double 4.4667709999999996
	.quad	0x40159ead0c3d2524              # double 5.4049569999999996
	.quad	0x4005d0025bfad29a              # double 2.7265670000000002
	.quad	0x400f86916440f239              # double 3.9407070000000002
	.quad	0x400847685985ad54              # double 3.0348670000000002
	.quad	0x400cd16872b020c5              # double 3.6022500000000002
	.quad	0x4006db8d3f1843c4              # double 2.8572030000000002
	.quad	0x4023d3ac929aa1d7              # double 9.9134259999999994
	.quad	0x4019a9129888f862              # double 6.4151100000000003
	.quad	0x400f4514c22ee419              # double 3.908731
	.quad	0x400190a350d2806b              # double 2.195624
	.quad	0x401b344c37e6f71a              # double 6.8010719999999996
	.quad	0x401bea6c5d206c87              # double 6.9789289999999999
	.quad	0x4011b9c9d5a187a5              # double 4.4314340000000003
	.quad	0x40237d7b41bfbdf1              # double 9.7450810000000007
	.quad	0x4018c744f5d35654              # double 6.1945990000000002
	.quad	0x3fee69ea14057082              # double 0.95042899999999997
	.quad	0x401fae91c8eabffd              # double 7.9204780000000001
	.quad	0x40115bf0563ed0f6              # double 4.3397839999999999
	.quad	0x4000b863beec397a              # double 2.0900340000000002
	.quad	0x401fb6f75d9a0f0a              # double 7.9286779999999997
	.quad	0x4017c52b4d8ba40e              # double 5.9425480000000004
	.quad	0x4007e880d801b435              # double 2.9885269999999999
	.quad	0x4011ce807bbb6241              # double 4.4516619999999998
	.quad	0x4004e7ba237cdcca              # double 2.6131479999999998
	.quad	0x40166c83a96d4c34              # double 5.6059710000000003
	.quad	0x4020ee53c9abb01d              # double 8.4654830000000008
	.quad	0x4022d21fafc8b008              # double 9.4103980000000007
	.quad	0x40045443d46b26c0              # double 2.5411450000000002
	.quad	0x4022c9dba908a266              # double 9.3942540000000001
	.quad	0x3fd26e978d4fdf3b              # double 0.28799999999999998
	.quad	0x4013fc209246bf01              # double 4.9962179999999998
	.quad	0x4011d5ab7dc7abfc              # double 4.4586620000000003
	.quad	0x4023c6bf8769ec2d              # double 9.8881800000000001
	.quad	0x401d0538ef34d6a1              # double 7.2550999999999997
	.quad	0x40078e95faa8a82a              # double 2.9446219999999999
	.quad	0x40175f294dd72368              # double 5.8429310000000001
	.quad	0x40162dae81882adc              # double 5.5446109999999997
	.quad	0x401593d07c84b5dd              # double 5.3943500000000002
	.quad	0x4023e76fb09203a3              # double 9.9520239999999997
	.quad	0x4017b301eabbcb1d              # double 5.9248120000000002
	.quad	0x3ff62baf102363b2              # double 1.3856649999999999
	.quad	0x4021be8900c521de              # double 8.8721390000000007
	.quad	0x401520b955f7835a              # double 5.2819570000000002
	.quad	0x3fd2b53d640e9d52              # double 0.29231200000000002
	.quad	0x4021acee5abc0e39              # double 8.8377560000000006
	.quad	0x40153d7774aba387              # double 5.3100259999999997
	.quad	0x4021355043e5321e              # double 8.6041279999999993
	.quad	0x4001fce18266772d              # double 2.2484769999999998
	.quad	0x4023f77d955714ba              # double 9.9833800000000004
	.quad	0x401cb5e529bae46d              # double 7.177632
	.quad	0x400af2f901083dbc              # double 3.3686389999999999
	.quad	0x4014b99c7bcc2939              # double 5.1812610000000001
	.quad	0x40102f416bdb1a6d              # double 4.0461479999999996
	.quad	0x3ff182070b8cfbfc              # double 1.0942449999999999
	.quad	0x401ee40fecdd0d8d              # double 7.7227170000000002
	.quad	0x400765c78a6dacac              # double 2.9246970000000001
	.quad	0x402217fcfc3f811f              # double 9.0468519999999994
	.quad	0x400538af3e468cac              # double 2.6526779999999999
	.quad	0x3ff123b8a19c9d5a              # double 1.071221
	.quad	0x4011849e8815e397              # double 4.3795109999999999
	.quad	0x401d240b34e7685a              # double 7.2851990000000004
	.quad	0x3fef6d5ae1cde5d2              # double 0.98209900000000006
	.quad	0x4003e3ce2089e343              # double 2.4862329999999999
	.quad	0x401cf856e696a26e              # double 7.2425189999999997
	.quad	0x400843f0bae89eba              # double 3.0331739999999998
	.quad	0x40204f8cc9a77e5f              # double 8.1553710000000006
	.quad	0x4020a4384ba0e842              # double 8.3207419999999992
	.quad	0x40007f162ae4b018              # double 2.0620539999999998
	.quad	0x4011e1ac57e23f25              # double 4.4703840000000001
	.quad	0x40069e2c55c96031              # double 2.8272330000000001
	.quad	0x40008a51e321a2e8              # double 2.067539
	.quad	0x4003b15ad106ee31              # double 2.4615990000000001
	.quad	0x401669df1172ef0b              # double 5.6033900000000001
	.quad	0x3ff23fd933e35c5b              # double 1.1405879999999999
	.quad	0x401dc13ad5bee3d6              # double 7.438701
	.quad	0x40238bb42cc2d6aa              # double 9.7728590000000004
	.quad	0x400e1814940bbb1f              # double 3.7617579999999999
	.quad	0x40198d18d25edd05              # double 6.3877899999999999
	.quad	0x400a8e63a5c1c609              # double 3.3195260000000002
	.quad	0x3ff9885a0be51090              # double 1.5957889999999999
	.quad	0x4004b17939a7c17b              # double 2.5866570000000002
	.quad	0x401bc58d5842b735              # double 6.9429220000000003
	.quad	0x3fdaebd09e12a51e              # double 0.42064299999999999
	.quad	0x4016421f6cacd185              # double 5.5645730000000002
	.quad	0x40211d0ca1cef241              # double 8.556737
	.quad	0x401099999999999a              # double 4.1500000000000004
	.quad	0x401e864a9cdc4439              # double 7.6311439999999999
	.quad	0x3fd80a88effe2a3d              # double 0.375643
	.quad	0x401a8f82b1f687b1              # double 6.6401469999999998
	.quad	0x401bc215336deb96              # double 6.9395340000000001
	.quad	0x40217cc2f837b4a2              # double 8.7436749999999996
	.quad	0x3fe676a2f48c2e77              # double 0.70198199999999999
	.quad	0x402399c065b63d3e              # double 9.8002959999999995
	.quad	0x401fe01d9f4d37c1              # double 7.9688629999999998
	.quad	0x3ff2655e28aa4335              # double 1.149748
	.quad	0x401b08f4b61fe21e              # double 6.7587460000000004
	.quad	0x4013f915379fa97e              # double 4.9932449999999999
	.quad	0x401a61105e1c1509              # double 6.5947889999999996
	.quad	0x4016bc8472c0e7bc              # double 5.6840989999999998
	.quad	0x400b764f11b60ae9              # double 3.4327679999999998
	.quad	0x400bea936c58eeaf              # double 3.4895390000000002
	.quad	0x4020b9f4b1ee2435              # double 8.3631949999999993
	.quad	0x402051f7b5aea316              # double 8.1600929999999998
	.quad	0x4020e73eab367a10              # double 8.4516500000000008
	.quad	0x400726f049a9973e              # double 2.8940130000000002
	.quad	0x4015afff36ac6477              # double 5.4218719999999996
	.quad	0x402378c804102ff9              # double 9.7359010000000001
	.quad	0x4022caaa3ad18d26              # double 9.3958300000000001
	.quad	0x4023972b020c49ba              # double 9.7952499999999993
	.quad	0x402156106ab14ec2              # double 8.668094
	.quad	0x401fed4845132f88              # double 7.9817210000000002
	.quad	0x3fca2f0e0a84be40              # double 0.20456099999999999
	.quad	0x40184443d46b26c0              # double 6.0666650000000004
	.quad	0x4018a605681ecd4b              # double 6.1621300000000003
	.quad	0x40072b9991361dc9              # double 2.8962889999999999
	.quad	0x40235d5dc4007571              # double 9.6823560000000004
	.quad	0x4009bb1183b60286              # double 3.216342
	.quad	0x40230005814940bc              # double 9.5000420000000005
	.quad	0x3ffcb01322f27350              # double 1.7929870000000001
	.quad	0x401c419a847b2464              # double 7.0640660000000004
	.quad	0x3fff224dd2f1a9fc              # double 1.945875
	.quad	0x402104b76f6d7625              # double 8.5092119999999998
	.quad	0x4017e830a0b1bbcf              # double 5.9767479999999997
	.quad	0x401bd22d948dc11e              # double 6.9552519999999998
	.quad	0x4022fde54b48d3ae              # double 9.4958899999999993
	.quad	0x400dd9d0203e63e9              # double 3.7313540000000001
	.quad	0x40166d3ed527e521              # double 5.6066849999999997
	.quad	0x401f1e82517e77d5              # double 7.7797939999999998
	.quad	0x4006741f212d7732              # double 2.8067000000000002
	.quad	0x40149a72a7bd48cb              # double 5.1508279999999997
	.quad	0x4020ef9c62a1b5c8              # double 8.4679900000000004
	.quad	0x3ff7d0f1f57b41c0              # double 1.4885120000000001
	.quad	0x3ff574f93bc0a06f              # double 1.3410580000000001
	.quad	0x3feade9e1b089a02              # double 0.83967499999999995
	.quad	0x401ca7cc39ffd60f              # double 7.1638650000000004
	.quad	0x402227df3b645a1d              # double 9.0778750000000006
	.quad	0x401d2083dbc23316              # double 7.2817530000000001
	.quad	0x3fe72009f623076c              # double 0.722661
	.quad	0x400f6408d8ec95c0              # double 3.923845
	.quad	0x400f5042528ae74f              # double 3.9141889999999999
	.quad	0x40013912556d19df              # double 2.1528670000000001
	.quad	0x400ba8027d88c1db              # double 3.457036
	.quad	0x401758ad25679896              # double 5.8365980000000004
	.quad	0x401f41c8a7a41e58              # double 7.8142420000000001
	.quad	0x3fdaebd09e12a51e              # double 0.42064299999999999
	.quad	0x40172b224aada33c              # double 5.7921230000000001
	.quad	0x401f42bb6672fba0              # double 7.8151679999999999
	.quad	0x4014109e98dcdb38              # double 5.0162300000000002
	.quad	0x3ff77d6c72192210              # double 1.468121
	.quad	0x401c5e90ff972474              # double 7.0923499999999997
	.quad	0x401e42e275ab7dc8              # double 7.5653170000000002
	.quad	0x401e4c68a936c58f              # double 7.574618
	.quad	0x3feaa71151611ba4              # double 0.83289400000000002
	.quad	0x3fcca6c1a048e044              # double 0.22383900000000001
	.quad	0x401bf9a5a89b951c              # double 6.9937959999999997
	.quad	0x4023bf487fcb923a              # double 9.8735999999999997
	.quad	0x4019783b60285ec4              # double 6.3674140000000001
	.quad	0x40102b4fa48301a8              # double 4.0422960000000003
	.quad	0x401e04c51116a8b9              # double 7.504658
	.quad	0x402268769ec2ce46              # double 9.2040299999999995
	.quad	0x401d333bd9cae211              # double 7.300033
	.quad	0x3ff900d3cff64cf9              # double 1.562702
	.quad	0x4013ae55c0fcb4f2              # double 4.9202490000000001
	.quad	0x401a773bc903ea70              # double 6.6164389999999997
	.quad	0x401d904a2fcefaa4              # double 7.3909079999999996
	.quad	0x400895f99c38b04b              # double 3.0732300000000001
	.quad	0x3ff95c65c70435f0              # double 1.5850580000000001
	.quad	0x4018013165d39970              # double 6.0011650000000003
	.quad	0x401fe24bba12b5e5              # double 7.9709919999999998
	.quad	0x3fec560e94ee392e              # double 0.88550499999999999
	.quad	0x4022086f69446738              # double 9.0164749999999998
	.quad	0x40127cd5f99c38b0              # double 4.6219099999999997
	.quad	0x4002ae429e0a41a2              # double 2.3350879999999998
	.quad	0x40171e0828c36da8              # double 5.7793279999999996
	.quad	0x40184d234eb9a177              # double 6.0753300000000001
	.quad	0x3fee13c89f40a287              # double 0.93991499999999994
	.quad	0x4008ec18f81e8a2f              # double 3.1152820000000001
	.quad	0x4019a0e5a35d67ba              # double 6.4071259999999999
	.quad	0x3fdf6d1e108c3f3e              # double 0.491035
	.quad	0x4020cf64adff822c              # double 8.4050650000000004
	.quad	0x4023113e5fb71fbc              # double 9.5336789999999993
	.quad	0x402033135979181a              # double 8.0997570000000003
	.quad	0x3fce1094a2b9d3cc              # double 0.23488100000000001
	.quad	0x400b9f42352eb5f6              # double 3.452763
	.quad	0x402314942d490e67              # double 9.5401930000000003
	.quad	0x40187437c5692b3d              # double 6.1134940000000002
	.quad	0x400a7304039abf34              # double 3.3061600000000002
	.quad	0x3ff72f0281ba7fc3              # double 1.448977
	.quad	0x4021e8b780346dc6              # double 8.9545250000000003
	.quad	0x40135c78e9f6a93f              # double 4.8403049999999999
	.quad	0x4021836d22424a27              # double 8.7566919999999993
	.quad	0x4004339da16616b5              # double 2.5252029999999999
	.quad	0x4004a897635e742a              # double 2.5823200000000002
	.quad	0x3ff33d30ad46f588              # double 1.202439
	.quad	0x401ca3b320535c9e              # double 7.1598629999999996
	.quad	0x400c31209edbf8ba              # double 3.5239880000000001
	.quad	0x4018386d2ed783e0              # double 6.055104
	.quad	0x40138d63886594af              # double 4.8880749999999997
	.quad	0x400d1e7d9988d2a2              # double 3.639888
	.quad	0x401fe6a2f48c2e77              # double 7.9752309999999999
	.quad	0x400e20807357e671              # double 3.7658700000000001
	.quad	0x3ff38a6a875d56f3              # double 1.221293
	.quad	0x40123ff2a172abef              # double 4.562449
	.quad	0x400615cbff47735c              # double 2.760643
	.quad	0x4001fdfdac68a937              # double 2.2490190000000001
	.quad	0x40051a41e57d9dbb              # double 2.6378210000000002
	.quad	0x4023ba6e115592da              # double 9.8641210000000008
	.quad	0x400d6b5568e820e6              # double 3.6774089999999999
	.quad	0x402394e67d77fae3              # double 9.7908209999999993
	.quad	0x40164b9c7368ad69              # double 5.5738390000000004
	.quad	0x40087a048e043a21              # double 3.0595789999999998
	.quad	0x3ff83691212513b6              # double 1.5133220000000001
	.quad	0x4019979c842fa509              # double 6.3980579999999998
	.quad	0x4018e84449dbec25              # double 6.2268230000000004
	.quad	0x4021989c7ffde721              # double 8.7980689999999999
	.quad	0x40157cbab649d389              # double 5.3718060000000003
	.quad	0x401f6c981beb1811              # double 7.8560489999999996
	.quad	0x3fd8b780346dc5d6              # double 0.38619999999999999
	.quad	0x3fd3bf2b239a38fc              # double 0.30854300000000001
	.quad	0x401005b63d3e4ef0              # double 4.0055779999999999
	.quad	0x401682799a1fd157              # double 5.6274170000000003
	.quad	0x4010e82278d0cc36              # double 4.2266940000000002
	.quad	0x4001fcd2d44dca8e              # double 2.2484489999999999
	.quad	0x401dd0d4562e09ff              # double 7.4539350000000004
	.quad	0x4013002ff8ec0f88              # double 4.7501829999999998
	.quad	0x402220b45ae5ffa4              # double 9.0638760000000004
	.quad	0x40069b5a63f9a49c              # double 2.8258559999999999
	.quad	0x401b72db1e9f2778              # double 6.8621639999999999
	.quad	0x40076113404ea4a9              # double 2.9224000000000001
	.quad	0x4001cfffbce4217d              # double 2.2265619999999999
	.quad	0x401c2324851a8694              # double 7.034319
	.quad	0x3fce73471f79420b              # double 0.23789299999999999
	.quad	0x401574f3343fa2ad              # double 5.3642089999999998
	.quad	0x4023f48605681ecd              # double 9.9775849999999994
	.quad	0x4000a93e1c9b413a              # double 2.0826380000000002
	.quad	0x401684aa53fc0097              # double 5.629556
	.quad	0x40231a5f202107b8              # double 9.5515070000000008
	.quad	0x401c0db09a671ef3              # double 7.013369
	.quad	0x40203886bdf4c2b5              # double 8.1104029999999998
	.quad	0x40234788333fc86d              # double 9.6397110000000001
	.quad	0x401d8f0dc768dfbd              # double 7.3897009999999996
	.quad	0x3ffedae0c1765775              # double 1.928437
	.quad	0x402041e85fd04a30              # double 8.1287260000000003
	.quad	0x401c40ce0b912dba              # double 7.0632859999999997
	.quad	0x4014a59bc33f8551              # double 5.161727
	.quad	0x401046a0dbad3a60              # double 4.0689729999999997
	.quad	0x400c8004b7f5a533              # double 3.5625089999999999
	.quad	0x4009cdab191dde37              # double 3.2254239999999998
	.quad	0x3ff1840181e03f70              # double 1.0947279999999999
	.quad	0x3ff83e8ccdd93c47              # double 1.515271
	.quad	0x4021ef4491298890              # double 8.9673200000000008
	.quad	0x400165e679463cfb              # double 2.1747559999999999
	.quad	0x3ff0c178705425f2              # double 1.047234
	.quad	0x4015df0025bfad2a              # double 5.4677740000000004
	.quad	0x3fe1963dc486ad2e              # double 0.54959000000000002
	.quad	0x40142050374ff866              # double 5.0315560000000001
	.quad	0x401f8b4bf8fcd680              # double 7.8860320000000002
	.quad	0x40187b8b69552e30              # double 6.1206490000000002
	.quad	0x3ff9415a07b352a8              # double 1.5784549999999999
	.quad	0x3ffbfdd872f33ca3              # double 1.749474
	.quad	0x40151bca1059ea57              # double 5.2771379999999999
	.quad	0x401ed25bfad29995              # double 7.7054289999999996
	.quad	0x4014b89f40a2877f              # double 5.1802950000000001
	.quad	0x40172299d883ba34              # double 5.7837899999999998
	.quad	0x40194d6fda836eb5              # double 6.3256220000000001
	.quad	0x401eb302b40f66a5              # double 7.6748149999999997
	.quad	0x3ffada272862f599              # double 1.6782600000000001
	.quad	0x40223fe4ffc9795b              # double 9.1247939999999996
	.quad	0x3ff2711409a24031              # double 1.1526069999999999
	.quad	0x4023df0da5daf07c              # double 9.935651
	.quad	0x3ffdf064ece9a2c6              # double 1.8711899999999999
	.quad	0x40015f644955b467              # double 2.1715779999999998
	.quad	0x40195e492bc2fc69              # double 6.3420759999999996
	.quad	0x401cc4406c00da1b              # double 7.1916520000000004
	.quad	0x4020b190afe60c39              # double 8.3468070000000001
	.quad	0x400577121ab4b72c              # double 2.6831399999999999
	.quad	0x400e790d173fb7a6              # double 3.809107
	.quad	0x3fea7023e9ea1405              # double 0.82618899999999995
	.quad	0x40051f3f0fe047d4              # double 2.6402570000000001
	.quad	0x401be26e978d4fdf              # double 6.9711249999999998
	.quad	0x4018b20bc382a130              # double 6.1738730000000004
	.quad	0x401da5ac8a372607              # double 7.411791
	.quad	0x400bfe2ac3222920              # double 3.4991050000000001
	.quad	0x401353fbbd7b2032              # double 4.8320150000000002
	.quad	0x400d5484d76ab580              # double 3.6662689999999998
	.quad	0x4022a9c4545846e9              # double 9.3315760000000001
	.quad	0x40089653c9abb01d              # double 3.0734020000000002
	.quad	0x40139009b30728e9              # double 4.8906619999999998
	.quad	0x4012d16b54e2b064              # double 4.7045110000000001
	.quad	0x4022cf5360d02470              # double 9.4049329999999998
	.quad	0x40085ca8e2e2b8c7              # double 3.0452439999999998
	.quad	0x400e10d5a5b9628d              # double 3.7582200000000001
	.quad	0x400196f75d9a0f0a              # double 2.1987139999999998
	.quad	0x401d7365cb35f3d8              # double 7.3626930000000001
	.quad	0x400c125b317efe0d              # double 3.5089630000000001
	.quad	0x401b0ca31e7d9989              # double 6.7623410000000002
	.quad	0x40116486ad2dcb14              # double 4.3481699999999996
	.quad	0x3ff0f1327243137b              # double 1.058886
	.quad	0x4017ccf0739b024f              # double 5.9501359999999996
	.quad	0x401291d68c692f6f              # double 4.6424200000000004
	.quad	0x402092393ee5eedd              # double 8.2855930000000004
	.quad	0x4022458b3f63c31e              # double 9.1358280000000001
	.quad	0x4022fa0a52695960              # double 9.4883600000000001
	.quad	0x40076548a9bcfd4c              # double 2.924455
	.quad	0x3ff447ca643cc07b              # double 1.2675270000000001
	.quad	0x3ff3927c39368273              # double 1.223263
	.quad	0x40220e55193708ab              # double 9.0279930000000004
	.quad	0x40153dbdf8f47304              # double 5.310295
	.quad	0x4013b1615ebfa8f8              # double 4.9232230000000001
	.quad	0x3ff844534bd76ee7              # double 1.5166809999999999
	.quad	0x3ff55435ac8a3726              # double 1.333059
	.quad	0x4015ddf58c08b75f              # double 5.4667570000000003
	.quad	0x3fe2c9c0a8d21bc1              # double 0.58712799999999998
	.quad	0x400797f1f9acffa8              # double 2.949192
	.quad	0x4003d14ca925fe97              # double 2.4771969999999999
	.quad	0x401d6c4007570c56              # double 7.3557129999999997
	.quad	0x4001a994185058de              # double 2.207802
	.quad	0x4019bef3d3a1d324              # double 6.436477
	.quad	0x40207835158b8280              # double 8.2347800000000007
	.quad	0x401fcfcc9ea9a3d3              # double 7.9529290000000001
	.quad	0x401f682cb20fb225              # double 7.8517330000000002
	.quad	0x4023c9f2990f301f              # double 9.8944290000000005
	.quad	0x4020c0c75c4a83b2              # double 8.3765210000000003
	.quad	0x4021efec3547e069              # double 8.9685989999999993
	.quad	0x4004b294573a7979              # double 2.5871970000000002
	.quad	0x4003f5ad96a6a012              # double 2.4949599999999998
	.quad	0x40112d82ba5a0382              # double 4.2944440000000004
	.quad	0x40007250b524399b              # double 2.0558179999999999
	.quad	0x4002169552e2fbe3              # double 2.2610269999999999
	.quad	0x3fe3cbd1244a6224              # double 0.61863000000000001
	.quad	0x401d3b4245f5ad97              # double 7.3078700000000003
	.quad	0x3ffa4792d1287c20              # double 1.642474
	.quad	0x3fee30318b934699              # double 0.94338299999999997
	.quad	0x40132d76683c297c              # double 4.794397
	.quad	0x3ffedeadd590c0ad              # double 1.929365
	.quad	0x3ff77a3ec02f2f98              # double 1.4673449999999999
	.quad	0x40194896dd26b724              # double 6.3208880000000001
	.quad	0x4010c512ec6bce85              # double 4.1924549999999998
	.quad	0x401482a6b0d95140              # double 5.1275890000000004
	.quad	0x4022c7f36ac64778              # double 9.3905290000000007
	.quad	0x3ff867a20578e5c5              # double 1.525301
	.quad	0x4022fc2e33eff195              # double 9.4925399999999999
	.quad	0x40226acb641700ce              # double 9.2085830000000009
	.quad	0x401ccbf6e3f78bbd              # double 7.1991839999999998
	.quad	0x3fe717aeddce7cd0              # double 0.72164099999999998
	.quad	0x3fec48711d798d8b              # double 0.88384300000000005
	.quad	0x40045e0ded288ce7              # double 2.545925
	.quad	0x40161367e414e7ef              # double 5.5189510000000004
	.quad	0x3ff4151d68c692f7              # double 1.255155
	.quad	0x3f884fd2a62aa194              # double 0.011871
	.quad	0x40209ed6dc1a47aa              # double 8.3102330000000002
	.quad	0x402140285ec3dab6              # double 8.6253080000000004
	.quad	0x4003feeaa6d26741              # double 2.4994710000000002
	.quad	0x4017dfc5cdd50a89              # double 5.9685280000000001
	.quad	0x40118a4b9884c6a4              # double 4.3850540000000002
	.quad	0x4022307a28bb0a2d              # double 9.0946820000000006
	.quad	0x40202de9185cee18              # double 8.0896690000000007
	.quad	0x40226232b12d3416              # double 9.1917930000000005
	.quad	0x4009e5d7cb2d905c              # double 3.237228
	.quad	0x400e4008e9b38d61              # double 3.7812670000000002
	.quad	0x400e04b5dcc63f14              # double 3.7523
	.quad	0x400d12684cf0739b              # double 3.633988
	.quad	0x401a2d12018a43bb              # double 6.5440139999999998
	.quad	0x4016532db1e9f278              # double 5.5812290000000004
	.quad	0x4022d6603d577964              # double 9.4187030000000007
	.quad	0x400d0f52b90a7829              # double 3.632482
	.quad	0x40181854046412cf              # double 6.0237579999999999
	.quad	0x4011e617c1bda512              # double 4.4747000000000003
	.quad	0x401d07c8d195896a              # double 7.2576020000000003
	.quad	0x40136154003254e7              # double 4.8450470000000001
	.quad	0x402256492bc2fc69              # double 9.1685269999999992
	.quad	0x4010a885d31337eb              # double 4.1645729999999999
	.quad	0x3ffffa109d0635a4              # double 1.998551
	.quad	0x401fdedfa43fe5c9              # double 7.9676499999999998
	.quad	0x4010f220791c4b90              # double 4.2364519999999999
	.quad	0x4015a1ee675147f1              # double 5.4081359999999998
	.quad	0x4014f564302b40f6              # double 5.2396399999999996
	.quad	0x401be720c8cd63cc              # double 6.9757110000000004
	.quad	0x40213c77bbf93ff2              # double 8.6181009999999993
	.quad	0x3feaf82d38476f2a              # double 0.84279499999999996
	.quad	0x401b22e275ab7dc8              # double 6.7840670000000003
	.quad	0x40125d8c6d612c6b              # double 4.5913560000000002
	.quad	0x40160dd8b60f1b26              # double 5.513522
	.quad	0x4015e33f42352eb6              # double 5.471921
	.quad	0x4014f15d701d9f4d              # double 5.2357079999999998
	.quad	0x40105873ffac1d2a              # double 4.0863800000000001
	.quad	0x3ffa2cbc8c0ce91d              # double 1.6359220000000001
	.quad	0x4015cf3b213e3e29              # double 5.4523739999999998
	.quad	0x401c95857afea3df              # double 7.1460169999999996
	.quad	0x3fff8101f31f46ed              # double 1.968996
	.quad	0x401c8f88fc9363f5              # double 7.1401709999999996
	.quad	0x4014fac322291fb4              # double 5.244885
	.quad	0x40119323358f2e06              # double 4.3936890000000002
	.quad	0x4002663065f9591d              # double 2.2998970000000001
	.quad	0x3ff478bef8ceb357              # double 1.279479
	.quad	0x4022f1c3ac929aa2              # double 9.4721960000000003
	.quad	0x401b09d7ba6698bb              # double 6.7596119999999997
	.quad	0x400298e7ddca4b12              # double 2.3246609999999999
	.quad	0x401de563a9f383f1              # double 7.4740130000000002
	.quad	0x40041e41d4b6a61a              # double 2.5147740000000001
	.quad	0x400092ee84ad794a              # double 2.0717439999999998
	.quad	0x40200ba27ae9ab2a              # double 8.0227240000000002
	.quad	0x400a8a2ca9ac3654              # double 3.3174679999999999
	.quad	0x401cd1049235f80a              # double 7.2041190000000004
	.quad	0x4013175c4a83b1d1              # double 4.7728130000000002
	.quad	0x40205aacfb762d84              # double 8.1771010000000003
	.quad	0x40223e325d4a5df2              # double 9.1214779999999998
	.quad	0x3ff0d4a0a0f4d7ae              # double 1.051911
	.quad	0x401b5baa582dbe7f              # double 6.8395169999999998
	.quad	0x4020b45d206c8712              # double 8.3522730000000003
	.quad	0x400bb8b502ababeb              # double 3.4651890000000001
	.quad	0x402170b8cfbfc654              # double 8.7201599999999999
	.quad	0x3fdc090b417ca212              # double 0.438052
	.quad	0x4022f3a9b0681238              # double 9.4759039999999998
	.quad	0x4022c4a2b9d3cbc5              # double 9.3840540000000007
	.quad	0x401deaa3ad18d25f              # double 7.4791400000000001
	.quad	0x3f9d90e23af31b15              # double 0.028872999999999999
	.quad	0x40203fa02fb5d031              # double 8.124269
	.quad	0x400bbd5f56a7ac82              # double 3.4674670000000001
	.quad	0x4012a3b7952d234f              # double 4.6598800000000002
	.quad	0x4022af852f7f498c              # double 9.3428129999999995
	.quad	0x400edadf2ecf2064              # double 3.8568709999999999
	.quad	0x401a6f9fcb0c026d              # double 6.6090080000000002
	.quad	0x4021c86af46aa088              # double 8.8914410000000004
	.quad	0x3fd8b7b6bb129025              # double 0.38621299999999997
	.quad	0x40214fbfa4c61d86              # double 8.6557589999999998
	.quad	0x400984406c00da1b              # double 3.1895760000000002
	.quad	0x40238933a0407cc8              # double 9.7679720000000003
	.quad	0x40231c7820a30db7              # double 9.5556040000000006
	.quad	0x4017dd102bc72e27              # double 5.9658819999999997
	.quad	0x401a08611fd5885d              # double 6.5081829999999998
	.quad	0x401b98b2e9ccb7d4              # double 6.8991199999999999
	.quad	0x3ffd46887a8d64d8              # double 1.82972
	.quad	0x400047724fa8b4c0              # double 2.0348860000000002
	.quad	0x4014a355475a31a5              # double 5.1595050000000002
	.quad	0x401b1f038e29f9cf              # double 6.7802870000000004
	.quad	0x40210c6ae3a3a8e7              # double 8.5242529999999999
	.quad	0x402377cbd556084a              # double 9.7339769999999994
	.quad	0x4007096c3fc43b2e              # double 2.8796010000000001
	.quad	0x3fe8282c6ef3d3a2              # double 0.75490400000000002
	.quad	0x401736eeb702602d              # double 5.8036450000000004
	.quad	0x401713d31b9b66f9              # double 5.7693599999999998
	.quad	0x4012542f61ed5ae2              # double 4.5822120000000002
	.quad	0x40178e92d55a3a08              # double 5.8892319999999998
	.quad	0x4015e77b7c7820a3              # double 5.476057
	.quad	0x4007357689ca18bd              # double 2.9011049999999998
	.quad	0x400dc805a2d72ffd              # double 3.7226669999999999
	.quad	0x40158d51f81a5871              # double 5.3880080000000001
	.quad	0x400974f50a02b841              # double 3.1821079999999999
	.quad	0x401dd7d0f1f57b42              # double 7.4607580000000002
	.quad	0x4009e518f3eccc47              # double 3.2368640000000002
	.quad	0x40175ee41919ac79              # double 5.8426669999999996
	.quad	0x402019a2834d26fa              # double 8.0500679999999996
	.quad	0x4021e65aff2a172b              # double 8.9499130000000005
	.quad	0x3ff4342dcf4623d1              # double 1.2627390000000001
	.quad	0x400e7b9d7fd82774              # double 3.8103590000000001
	.quad	0x3ff3ad6238da3c21              # double 1.22983
	.quad	0x40068fd5454152b1              # double 2.8202310000000002
	.quad	0x401b67eb6bf444e1              # double 6.8514840000000001
	.quad	0x40139884c6a3bde0              # double 4.8989440000000002
	.quad	0x4022bc37a3db3bfb              # double 9.3676119999999993
	.quad	0x401a23fe5c91d14e              # double 6.5351499999999998
	.quad	0x400630cd423d9232              # double 2.7738290000000001
	.quad	0x400c3bd0e12e83a1              # double 3.529207
	.quad	0x3fb761672324c836              # double 0.091329999999999994
	.quad	0x3ff8ded9dfdac68b              # double 1.5544070000000001
	.quad	0x402344538ef34d6a              # double 9.6334499999999998
	.quad	0x401580aed99cbee8              # double 5.375667
	.quad	0x3fd5c9eecbfb15b5              # double 0.34044999999999997
	.quad	0x400215885d31337f              # double 2.2605140000000001
	.quad	0x40122bd987c6327f              # double 4.5428220000000001
	.quad	0x40171f7596851223              # double 5.7807219999999999
	.quad	0x4018c8611fd5885d              # double 6.1956829999999998
	.quad	0x40128652bd3c3611              # double 4.6311749999999998
	.quad	0x3feef8ab0c88a47f              # double 0.96785500000000002
	.quad	0x40227c894c447c31              # double 9.2432350000000003
	.quad	0x3feee924f227d029              # double 0.96596000000000004
	.quad	0x3ffb367b1c0010c7              # double 1.700801
	.quad	0x40212d0ebaade658              # double 8.5880030000000005
	.quad	0x4015a7f90d9d7775              # double 5.4140360000000003
	.quad	0x4002f3a14cec41dd              # double 2.36896
	.quad	0x401bc292c4934268              # double 6.9400130000000004
	.quad	0x4005423c85c24c40              # double 2.6573419999999999
	.quad	0x4013feac8605681f              # double 4.9987050000000002
	.quad	0x40071b5a63f9a49c              # double 2.8883559999999999
	.quad	0x3fc29b280f12c27a              # double 0.14535999999999999
	.quad	0x40210f6965f5275f              # double 8.5301010000000002
	.quad	0x3fd281d7dbf487fd              # double 0.28917500000000002
	.quad	0x4015cbc4d22c881e              # double 5.4489929999999998
	.quad	0x401e39378ee28672              # double 7.5558759999999996
	.quad	0x3ffea28fe260b2c8              # double 1.9146879999999999
	.quad	0x40209065f9591cd2              # double 8.2820280000000004
	.quad	0x40226c4dec1c1d6d              # double 9.2115320000000001
	.quad	0x3ff9fdda8bd230ba              # double 1.624476
	.quad	0x4013471bcdbbe015              # double 4.8194419999999996
	.quad	0x3fe605814940bbb2              # double 0.68817200000000001
	.quad	0x3fc6a9717df19d67              # double 0.17704600000000001
	.quad	0x402286bedfa43fe6              # double 9.2631750000000004
	.quad	0x4010bb4cc2507208              # double 4.1829099999999997
	.quad	0x400f681669ced0b3              # double 3.925824
	.quad	0x3feb7d24180d3cff              # double 0.85902599999999996
	.quad	0x40010779207d4e09              # double 2.1286489999999998
	.quad	0x4017fdeefe4ffc98              # double 5.9979820000000004
	.quad	0x401475985ad538ac              # double 5.1148389999999999
	.quad	0x3fc1c7abfb9bed31              # double 0.138906
	.quad	0x4017e2cd7cf5f4e4              # double 5.9714869999999998
	.quad	0x401644538ef34d6a              # double 5.5667249999999999
	.quad	0x40020892253111f1              # double 2.2541850000000001
	.quad	0x402200997c3d6840              # double 9.0011709999999993
	.quad	0x4014d3b6cbd987c6              # double 5.2067519999999998
	.quad	0x401ac2253111f0c3              # double 6.6895949999999997
	.quad	0x40209067cf1c3266              # double 8.2820420000000006
	.quad	0x400a08f4b61fe21e              # double 3.2543730000000002
	.quad	0x401e0c3a860dcb9b              # double 7.5119420000000003
	.quad	0x3fdda08398a65493              # double 0.462922
	.quad	0x401125cd4ed2cbea              # double 4.2869159999999997
	.quad	0x40190dc4c9c90c4e              # double 6.2634460000000001
	.quad	0x4021fad6ece13f4b              # double 8.9899210000000007
	.quad	0x4017275536933a04              # double 5.788411
	.quad	0x4015e9cbab649d39              # double 5.4783160000000004
	.quad	0x40116b167ec7863c              # double 4.3545780000000001
	.quad	0x401d697cc39ffd61              # double 7.3530150000000001
	.quad	0x401c39a6b50b0f28              # double 7.0563000000000002
	.quad	0x3fe626117720c8cd              # double 0.69214699999999996
	.quad	0x3fae69f05ea24cc7              # double 0.059402000000000003
	.quad	0x400ebb60ae9680e0              # double 3.8414929999999998
	.quad	0x400dd02602c9081c              # double 3.7266349999999999
	.quad	0x401dcaaf35e310dc              # double 7.4479340000000001
	.quad	0x40183a9b06812381              # double 6.0572319999999999
	.quad	0x4021cf20ea5b530d              # double 8.9045480000000001
	.quad	0x3ffb9aec4a4095f2              # double 1.7253229999999999
	.quad	0x4011c496ededaec5              # double 4.4419820000000003
	.quad	0x40030e8fb00bcbe6              # double 2.3821099999999999
	.quad	0x4014ce5753a3ec03              # double 5.201505
	.quad	0x40135d074213a0c7              # double 4.8408480000000003
	.quad	0x3ff91ba6efc371da              # double 1.569251
	.quad	0x4012c756861e9292              # double 4.6946659999999998
	.quad	0x40187004fb1183b6              # double 6.109394
	.quad	0x4012476ea4228998              # double 4.5697580000000002
	.quad	0x4003a9acbc8c0ce9              # double 2.457849
	.quad	0x3ff0ba81dc0db270              # double 1.045534
	.quad	0x400381b3aeee9574              # double 2.4383309999999998
	.quad	0x40176a09cc319c5a              # double 5.8535529999999998
	.quad	0x3ffd11dbca9691a7              # double 1.8168599999999999
	.quad	0x40225904b3c3e74b              # double 9.173864
	.quad	0x401896e75ff609dd              # double 6.147367
	.quad	0x4010aeecbfb15b57              # double 4.1708249999999998
	.quad	0x401be2074ea8da7f              # double 6.9707309999999998
	.quad	0x4020ece13f4a98ab              # double 8.4626560000000008
	.quad	0x3fdfa48b65237048              # double 0.49441800000000002
	.quad	0x3ff970d844d013a9              # double 1.59005
	.quad	0x402002e4f3343fa3              # double 8.0056530000000006
	.quad	0x400a91b823c85c25              # double 3.3211520000000001
	.quad	0x4008bbba55d1c3ad              # double 3.0916640000000002
	.quad	0x40214ab92c061848              # double 8.6459440000000001
	.quad	0x400e17eb28d8665e              # double 3.761679
	.quad	0x3fb826351deefe50              # double 0.094333
	.quad	0x4019607c84b5dcc6              # double 6.3442249999999998
	.quad	0x401c79f94855da27              # double 7.1191149999999999
	.quad	0x3fd31ac57e23f24e              # double 0.29850900000000002
	.quad	0x401f8ad9ad85dfa8              # double 7.8855959999999996
	.quad	0x401e3e8576cce5f7              # double 7.5610559999999997
	.quad	0x40102df3b645a1cb              # double 4.0448750000000002
	.quad	0x40166d095af294dd              # double 5.6064809999999996
	.quad	0x4022b008c8259e1f              # double 9.3438169999999996
	.quad	0x40229caf922962d0              # double 9.3060270000000003
	.quad	0x4014aab862b27cbf              # double 5.1667189999999996
	.quad	0x40192b97fe8ee6b8              # double 6.2925719999999998
	.quad	0x4019cc2e33eff195              # double 6.449395
	.quad	0x40014a4ca4f440af              # double 2.161279
	.quad	0x401c2144cbe1eb42              # double 7.032489
	.quad	0x40196078d92fb19e              # double 6.3442109999999996
	.quad	0x401d0a704bc27632              # double 7.2601940000000003
	.quad	0x401f73afb7e90ff9              # double 7.8629749999999996
	.quad	0x402234a72ead9275              # double 9.1028380000000002
	.quad	0x401dc81dc0db2703              # double 7.4454260000000003
	.quad	0x40090f850df15a4b              # double 3.1325780000000001
	.quad	0x401baaf922962cfe              # double 6.9169660000000004
	.quad	0x401bdd47bedb7282              # double 6.966094
	.quad	0x400dadf58c08b75f              # double 3.7099410000000002
	.quad	0x40176784662bae04              # double 5.8510910000000003
	.quad	0x4002cf8594291750              # double 2.3513289999999998
	.quad	0x400a6e6d15ad106f              # double 3.3039190000000001
	.quad	0x4003b3bfb58d1527              # double 2.4627680000000001
	.quad	0x40207c317acc4ef9              # double 8.2425650000000008
	.quad	0x401778d045fe1112              # double 5.8679819999999996
	.quad	0x3fdd348b220791c5              # double 0.45633200000000002
	.quad	0x3ffcc5d206c8711d              # double 1.7982959999999999
	.quad	0x4013579fa97e132b              # double 4.8355699999999997
	.quad	0x4021480a393ee5ef              # double 8.6407030000000002
	.quad	0x4000f5781c714fce              # double 2.1198579999999998
	.quad	0x401856fc158fb43e              # double 6.0849460000000004
	.quad	0x401e1bc5154866a1              # double 7.5271189999999998
	.quad	0x402209f10667f90e              # double 9.0194170000000007
	.quad	0x402236c54bcf0b6b              # double 9.1069739999999992
	.quad	0x4016f0dd3fe1975f              # double 5.7352189999999998
	.quad	0x3fe6f8e1932d6ece              # double 0.71788099999999999
	.quad	0x402009a32b12d341              # double 8.0188229999999994
	.quad	0x400a925e56cd6c2f              # double 3.321469
	.quad	0x3fd0eb8d82342246              # double 0.26437699999999997
	.quad	0x4020d9fff79c8430              # double 8.4257810000000006
	.quad	0x3ffde3f034b0e1b5              # double 1.8681490000000001
	.quad	0x3fddf1800a7c5ac4              # double 0.46786499999999998
	.quad	0x402282f709b738e7              # double 9.2557910000000003
	.quad	0x400aae6fb4c3c18b              # double 3.3351739999999999
	.quad	0x40165814d72799a2              # double 5.586017
	.quad	0x400567c286f8ad25              # double 2.6756639999999998
	.quad	0x4020c69b9d7fd827              # double 8.3879059999999992
	.quad	0x4022c08b545c78a7              # double 9.3760630000000002
	.quad	0x400387e6b3fe9fae              # double 2.4413580000000001
	.quad	0x40210b3d6c721922              # double 8.5219529999999999
	.quad	0x4022e06705c896dd              # double 9.4382859999999997
	.quad	0x401e133fc86cebbc              # double 7.5187980000000003
	.quad	0x40199869c66d373b              # double 6.398841
	.quad	0x3fe703382e44b6e9              # double 0.71914299999999998
	.quad	0x401d6e7d1351159c              # double 7.3578989999999997
	.quad	0x401c0a7ce0fc2eba              # double 7.0102419999999999
	.quad	0x401960d66f0cfe15              # double 6.3445679999999998
	.quad	0x40027a847b24638d              # double 2.3098230000000002
	.quad	0x400a0aea747d805e              # double 3.2553299999999998
	.quad	0x3ffc60620ab71327              # double 1.773531
	.quad	0x40178846e8f29d41              # double 5.8830830000000001
	.quad	0x4016d0ecb74ddf87              # double 5.7040280000000001
	.quad	0x4005677bbf93ff26              # double 2.675529
	.quad	0x402112176ddaceee              # double 8.5353349999999999
	.quad	0x4020d8214ad362e9              # double 8.422129
	.quad	0x401da4934267839d              # double 7.4107180000000001
	.quad	0x401e3dc0980b2420              # double 7.5603049999999996
	.quad	0x40239cd338f79852              # double 9.8062989999999992
	.quad	0x3fd33e1c9b413986              # double 0.30066599999999999
	.quad	0x4021197891e21533              # double 8.5497479999999992
	.quad	0x4016c82491afc04d              # double 5.6954520000000004
	.quad	0x4014428e0c9d9d34              # double 5.0649949999999997
	.quad	0x4003ae4d1a650614              # double 2.460108
	.quad	0x402200bae89eba6b              # double 9.0014260000000004
	.quad	0x401cf45846e8f29d              # double 7.2386179999999998
	.quad	0x4000daf07bfe7e20              # double 2.1069040000000001
	.quad	0x3ff748f10a99b6f6              # double 1.455308
	.quad	0x4010ca3c21187e7c              # double 4.197495
	.quad	0x401ad0129cbab64a              # double 6.7031960000000002
	.quad	0x4010bc8754f3775c              # double 4.1841100000000004
	.quad	0x3ff2b8a3b57c4e2f              # double 1.170078
	.quad	0x4018e66c6de76428              # double 6.2250230000000002
	.quad	0x4023689d8c6d612c              # double 9.7043269999999992
	.quad	0x4023c0e53eb399f6              # double 9.8767490000000002
	.quad	0x400e518fc504816f              # double 3.789825
	.quad	0x401561083dbc2331              # double 5.3447579999999997
	.quad	0x4003aa960b6f9fcb              # double 2.458294
	.quad	0x3ff4fc4e2f37fbf0              # double 1.311598
	.quad	0x3ffd27c7c5260f5e              # double 1.8222119999999999
	.quad	0x400c051761459536              # double 3.5024860000000002
	.quad	0x401a01672324c836              # double 6.5013699999999996
	.quad	0x402023ee807bbb62              # double 8.0701789999999995
	.quad	0x3ffefa22706d5065              # double 1.9360679999999999
	.quad	0x3fd98f4b61fe21d9              # double 0.39937099999999998
	.quad	0x40220d1d4738a3b5              # double 9.0256139999999991
	.quad	0x3fdfb2d4d4024b34              # double 0.49529000000000001
	.quad	0x401503fdd65a1449              # double 5.2538980000000004
	.quad	0x401c5a0c282c6ef4              # double 7.0879370000000002
	.quad	0x401b751f3e89a88b              # double 6.8643770000000002
	.quad	0x401b084f09528f19              # double 6.758114
	.quad	0x3ff36cb74ddf86e4              # double 1.2140420000000001
	.quad	0x4021b7d60e94ee39              # double 8.8590549999999997
	.quad	0x3feada9003eea20a              # double 0.83918000000000004
	.quad	0x4018c091c087442c              # double 6.1880559999999996
	.quad	0x3ffe81a5870da5db              # double 1.906652
	.quad	0x40154fb8b26394fb              # double 5.3278530000000002
	.quad	0x401e9cee17a02fb6              # double 7.6532520000000002
	.quad	0x4021171bcdbbe015              # double 8.5451339999999991
	.quad	0x400693a92a305532              # double 2.8220999999999998
	.quad	0x4012607c4199fe43              # double 4.5942239999999996
	.quad	0x401abca46e08f217              # double 6.684221
	.quad	0x401e06cebbba55d2              # double 7.5066480000000002
	.quad	0x401a21754b05b7d0              # double 6.5326740000000001
	.quad	0x40115336deb95e5b              # double 4.331264
	.quad	0x400036494d50ebab              # double 2.0265070000000001
	.quad	0x4023a2e514c22ee4              # double 9.8181539999999998
	.quad	0x3ff6cacf312b1b37              # double 1.4245140000000001
	.quad	0x400b251761459536              # double 3.3931110000000002
	.quad	0x3fef9d388a8b08dd              # double 0.98794199999999998
	.quad	0x40159fddebd9018e              # double 5.4061199999999996
	.quad	0x400afc6a7ef9db23              # double 3.3732500000000001
	.quad	0x40229922fad6cb53              # double 9.2990949999999994
	.quad	0x4007b883ba3443d4              # double 2.9650949999999998
	.quad	0x401690e3cd9a5226              # double 5.6414939999999998
	.quad	0x3ffbb21a2e7f6f4c              # double 1.730982
	.quad	0x4015280064a9cdc4              # double 5.2890639999999998
	.quad	0x4009b3300de4c511              # double 3.212494
	.quad	0x3ff2f8b3700474da              # double 1.185718
	.quad	0x40151d0aaa7ded6c              # double 5.2783610000000003
	.quad	0x4022a49fd7a13c25              # double 9.3215319999999995
	.quad	0x4007d3111f0c34c2              # double 2.9780600000000002
	.quad	0x400b34a44c7b02d6              # double 3.4007040000000002
	.quad	0x3fc0e043a2163fdd              # double 0.13184399999999999
	.quad	0x40222ff9b56323bc              # double 9.0937020000000003
	.quad	0x3ff6588f42fe8251              # double 1.3966209999999999
	.quad	0x3feac8d1958969a1              # double 0.83701400000000004
	.quad	0x40221275cd0bb6ed              # double 9.0360549999999993
	.quad	0x40035379fa97e133              # double 2.4157600000000001
	.quad	0x40058d55a3a08399              # double 2.6940110000000002
	.quad	0x3fdec3b0c4588a05              # double 0.48069400000000001
	.quad	0x4021d14e1a3f4667              # double 8.9087990000000001
	.quad	0x40235b0446b69db6              # double 9.6777669999999993
	.quad	0x401c08ec0f883340              # double 7.0087130000000002
	.quad	0x401580cae642bf98              # double 5.3757739999999998
	.quad	0x402116a550870111              # double 8.5442300000000007
	.quad	0x3ff35ebd9018e758              # double 1.2106300000000001
	.quad	0x401f696af038e2a0              # double 7.8529470000000003
	.quad	0x4021e0b545c78a6e              # double 8.9388830000000006
	.quad	0x4022ccbdba0a5269              # double 9.3998849999999993
	.quad	0x4004981626b2f230              # double 2.5742609999999999
	.quad	0x401d4e56473471f8              # double 7.3265010000000003
	.quad	0x40163c9bc7714339              # double 5.5591879999999998
	.quad	0x4006d6909aed56b0              # double 2.854768
	.quad	0x40135dd2ae8e1d65              # double 4.8416240000000004
	.quad	0x401209af1886df83              # double 4.5094570000000003
	.quad	0x4011b33daf8df7a5              # double 4.4250400000000001
	.quad	0x4010bc36113404ea              # double 4.1837999999999997
	.quad	0x402093c536d655e3              # double 8.2886140000000008
	.quad	0x401b9377e1b8ed1c              # double 6.894012
	.quad	0x400fdbe09bb6aa4c              # double 3.9823620000000002
	.quad	0x3fd64ad362e90364              # double 0.34831699999999999
	.quad	0x3ff3b40a2877ee4e              # double 1.231455
	.quad	0x401e7d7ecbb7f9d7              # double 7.622554
	.quad	0x4021cf95703f2d3c              # double 8.9054369999999991
	.quad	0x401e6e22e5de15ca              # double 7.6075549999999996
	.quad	0x4023d7e32a0663c7              # double 9.9216549999999994
	.quad	0x3ffc3586ca89fc6e              # double 1.7630680000000001
	.quad	0x3fed7f7dfa00e27e              # double 0.92181299999999999
	.quad	0x4012854da4ce8102              # double 4.630179
	.quad	0x402256273929ed39              # double 9.1682679999999994
	.quad	0x401d43c968943e10              # double 7.316198
	.quad	0x4019bb6177ea1c69              # double 6.4329890000000001
	.quad	0x4022f8b06c43f5f9              # double 9.4857209999999998
	.quad	0x40217ca6223e186a              # double 8.7434550000000009
	.quad	0x4009ab2af145521a              # double 3.2085780000000002
	.quad	0x40220eb69db65ecc              # double 9.0287369999999995
	.quad	0x3fd87ff151e75361              # double 0.38280900000000001
	.quad	0x401c8795703f2d3c              # double 7.1324059999999996
	.quad	0x400398dc55000c95              # double 2.4496389999999999
	.quad	0x4007eb5aa715831f              # double 2.989919
	.quad	0x4022431a2a4db164              # double 9.1310590000000004
	.quad	0x3ff8fb267c6b8b69              # double 1.5613159999999999
	.quad	0x40204f19d66adb40              # double 8.1544939999999997
	.quad	0x3ff4f96d0917d6b6              # double 1.3108949999999999
	.quad	0x4019463992c8c500              # double 6.3185789999999997
	.quad	0x4019fea9e6eeb702              # double 6.4986949999999997
	.quad	0x3ff8a9df97aaac11              # double 1.5414730000000001
	.quad	0x401b9ea077036c9c              # double 6.904909
	.quad	0x4014c56d19dece57              # double 5.1927989999999999
	.quad	0x4015727b6fe2e6eb              # double 5.3617990000000004
	.quad	0x4022a65bc87db2b3              # double 9.3249189999999995
	.quad	0x40020df58c08b75f              # double 2.2568160000000002
	.quad	0x3ffb5ab29e4d5d81              # double 1.709643
	.quad	0x4003df4eca686e7e              # double 2.4840369999999998
	.quad	0x4023c4c575c0767d              # double 9.8843189999999996
	.quad	0x4001f13ad5bee3d6              # double 2.242788
	.quad	0x40220cbc27631b58              # double 9.0248729999999995
	.quad	0x40196425aee631f9              # double 6.3478000000000003
	.quad	0x3fd73a3ec02f2f98              # double 0.36292999999999997
	.quad	0x3ff205681ecd4aa1              # double 1.12632
	.quad	0x401221704ff4341a              # double 4.5326550000000001
	.quad	0x4013a399b2c40d0b              # double 4.9097660000000003
	.quad	0x400588049667b5f2              # double 2.6914150000000001
	.quad	0x3ff0b9cc74b838c1              # double 1.045361
	.quad	0x3ff93dac68a936c6              # double 1.5775570000000001
	.quad	0x4022e4cbc05d52c1              # double 9.4468669999999992
	.quad	0x4005f18eb8950764              # double 2.7429480000000002
	.quad	0x402304cccccccccd              # double 9.5093750000000003
	.quad	0x401995cfaacd9e84              # double 6.3963000000000001
	.quad	0x40230330d738609a              # double 9.5062320000000007
	.quad	0x3fd6dedaec4a4096              # double 0.357352
	.quad	0x401dd41355475a32              # double 7.4571050000000003
	.quad	0x40228e1a8262456f              # double 9.2775459999999992
	.quad	0x400752a27f1b6912              # double 2.915349
	.quad	0x3ff0a55f3519bd40              # double 1.0403739999999999
	.quad	0x40113c5a8155d5f5              # double 4.3089389999999996
	.quad	0x402222f922962cfe              # double 9.0683070000000008
	.quad	0x4023c448f10a99b7              # double 9.883369
	.quad	0x4020a33764f11b61              # double 8.3187820000000006
	.quad	0x4022fd3d4280ae10              # double 9.4946079999999995
	.quad	0x4021a3b384fd2a63              # double 8.8197290000000006
	.quad	0x4022a4c1a8ac5c14              # double 9.32179
	.quad	0x4008b41f212d7732              # double 3.0879500000000002
	.quad	0x40148981beb18117              # double 5.1342840000000001
	.quad	0x401a7b955f78359c              # double 6.6206870000000002
	.quad	0x3ff0fc5260f5e41d              # double 1.0616019999999999
	.quad	0x4016a1876188b114              # double 5.657743
	.quad	0x3fa66b3fe9fadafd              # double 0.043787
	.quad	0x400f6d79d0a67621              # double 3.928455
	.quad	0x4002c157eed45e92              # double 2.3444060000000002
	.quad	0x401276a2b1704ff4              # double 4.6158549999999998
	.quad	0x402179ceaf251c19              # double 8.7379049999999996
	.quad	0x40200c0e1719f7f9              # double 8.0235450000000004
	.quad	0x4009b5c853c14834              # double 3.2137609999999999
	.quad	0x401e33e9ea140571              # double 7.5506970000000004
	.quad	0x3ff2ac7000431bdf              # double 1.1670990000000001
	.quad	0x401826f19934efcc              # double 6.0380310000000001
	.quad	0x4019006f262888b5              # double 6.2504239999999998
	.quad	0x3ff90ee5abc0e38a              # double 1.5661369999999999
	.quad	0x400de06530058149              # double 3.7345679999999999
	.quad	0x401a2fd47bedb728              # double 6.5467089999999999
	.quad	0x3fc0515ce9e5e248              # double 0.12748300000000001
	.quad	0x4011a474538ef34d              # double 4.4105999999999996
	.quad	0x402246b09635610b              # double 9.1380660000000002
	.quad	0x40174b1e5c0b9991              # double 5.8233579999999998
	.quad	0x40091f9ce8d972cd              # double 3.1404359999999998
	.quad	0x400f530ad46f587d              # double 3.9155479999999998
	.quad	0x40019291b823c85c              # double 2.1965669999999999
	.quad	0x3fee087ca643cc08              # double 0.93853600000000003
	.quad	0x4022929cdc443915              # double 9.2863530000000001
	.quad	0x40226b76b3bb83cf              # double 9.2098899999999997
	.quad	0x4002273365881a15              # double 2.2691409999999999
	.quad	0x401f1351deefe500              # double 7.7688670000000002
	.quad	0x400b1add9c27e953              # double 3.388118
	.quad	0x401439a847b24639              # double 5.0563060000000002
	.quad	0x40215f70bd017daf              # double 8.6864070000000009
	.quad	0x3fffece032db1e9f              # double 1.995331
	.quad	0x402006ffc115df65              # double 8.0136699999999994
	.quad	0x3ffb61ee675147f1              # double 1.711409
	.quad	0x401a396b76709fa5              # double 6.5560739999999997
	.quad	0x40225ef30a4e379b              # double 9.1854479999999992
	.quad	0x40193b12d3415b14              # double 6.3076889999999999
	.quad	0x4021716e37154003              # double 8.7215439999999997
	.quad	0x3fe07566490a350d              # double 0.51433099999999998
	.quad	0x4016688ca3e7d135              # double 5.6020989999999999
	.quad	0x401e521101b00368              # double 7.5801429999999996
	.quad	0x402168589acbc8c1              # double 8.7038010000000003
	.quad	0x40183b078d92fb1a              # double 6.0576460000000001
	.quad	0x3ff6ef99806f2629              # double 1.4334960000000001
	.quad	0x4022b746887a8d65              # double 9.3579600000000002
	.quad	0x4023f8eeae9ee45c              # double 9.9861959999999996
	.quad	0x400df07b784662bb              # double 3.7424230000000001
	.quad	0x402208d759ee88df              # double 9.0172679999999996
	.quad	0x400432999567dbb1              # double 2.5247069999999998
	.quad	0x4009a54046412cf1              # double 3.205689
	.quad	0x40042b5b2d4d4025              # double 2.5211700000000001
	.quad	0x401c9e4f765fd8ae              # double 7.1546000000000003
	.quad	0x40123dd616f86a0a              # double 4.5603870000000004
	.quad	0x4018d96787ce95fb              # double 6.2123090000000003
	.quad	0x401151c64ba94bbe              # double 4.3298579999999998
	.quad	0x3ff35fadafd11383              # double 1.2108589999999999
	.quad	0x401c88ede54b48d4              # double 7.1337200000000003
	.quad	0x4020e0b6b6e0d23d              # double 8.4388939999999995
	.quad	0x4008f966be7afa72              # double 3.1217779999999999
	.quad	0x3ffa37143393ab43              # double 1.638447
	.quad	0x4021af991bc55864              # double 8.8429649999999995
	.quad	0x401513fcc9ea9a3d              # double 5.2695189999999998
	.quad	0x401d2741d084e832              # double 7.2883370000000003
	.quad	0x40170c8f75536934              # double 5.7622660000000003
	.quad	0x4006ea52ef911cf3              # double 2.8644159999999999
	.quad	0x400667f8012dfd69              # double 2.8007659999999999
	.quad	0x401fc6fb09203a32              # double 7.9443169999999999
	.quad	0x40095fbce4217d28              # double 3.1717469999999999
	.quad	0x401387625204af92              # double 4.8822109999999999
	.quad	0x4023ac1615ebfa8f              # double 9.8361059999999991
	.quad	0x402051a9b8cb8e08              # double 8.1594979999999993
	.quad	0x3fdbebfec13b9f12              # double 0.43627899999999997
	.quad	0x401027763e4abe6a              # double 4.0385369999999998
	.quad	0x40109f6d330941c8              # double 4.1556899999999999
	.quad	0x401f26b228dc981c              # double 7.7877890000000001
	.quad	0x40207536f7e3d1cc              # double 8.2289349999999998
	.quad	0x3fb34cb1897a67a5              # double 0.075388999999999998
	.quad	0x401bec64ba94bbe4              # double 6.9808529999999998
	.quad	0x4021a218372e6a77              # double 8.8165910000000007
	.quad	0x4000063beec3979a              # double 2.003044
	.quad	0x4021d76b3bb83cf3              # double 8.9207400000000003
	.quad	0x4002ef345069a4df              # double 2.3667989999999999
	.quad	0x4018f17268d32831              # double 6.2357880000000003
	.quad	0x40205c3b4f616723              # double 8.1801399999999997
	.quad	0x400328fc9363f573              # double 2.3950130000000001
	.quad	0x3fcc20d5629d840a              # double 0.219752
	.quad	0x40207af251c193b4              # double 8.2401300000000006
	.quad	0x401409c99285a922              # double 5.0095580000000002
	.quad	0x4021ec6c97d8cf3a              # double 8.9617660000000008
	.quad	0x3fe8a0e8427418d7              # double 0.76964200000000005
	.quad	0x3ff32719b4dcebff              # double 1.1970460000000001
	.quad	0x4021559d3458cd21              # double 8.6672150000000006
	.quad	0x4011e55c96030c24              # double 4.473986
	.quad	0x4012440d0aaa7ded              # double 4.5664559999999996
	.quad	0x4010355eaee2003b              # double 4.0521190000000002
	.quad	0x400c17d6b65a9a80              # double 3.5116399999999999
	.quad	0x40027eacc92146a2              # double 2.3118530000000002
	.quad	0x400ff98958d9b5e9              # double 3.9968439999999998
	.quad	0x401c42f449129889              # double 7.065385
	.quad	0x402114e8533b1077              # double 8.5408349999999995
	.quad	0x3ffbb5ad96a6a012              # double 1.7318549999999999
	.quad	0x4018599a62ed3522              # double 6.0875029999999999
	.quad	0x401c3b3d4ae429e1              # double 7.0578510000000003
	.quad	0x3ff66977c88e79ab              # double 1.400749
	.quad	0x400371569f490603              # double 2.4303409999999999
	.quad	0x40020d6ece13f4aa              # double 2.2565590000000002
	.quad	0x400f1e4d5d80e497              # double 3.889796
	.quad	0x3fcf5f3519bd4068              # double 0.24509300000000001
	.quad	0x3ffa7f22c01e68a1              # double 1.656039
	.quad	0x4000b0d38cda6e76              # double 2.086341
	.quad	0x3fe4e1eb4202d9cf              # double 0.65257799999999999
	.quad	0x3fd542a66dbd72bd              # double 0.33219300000000002
	.quad	0x40151a35935fc3b5              # double 5.275595
	.quad	0x4015a5054ac29bf1              # double 5.4111529999999997
	.quad	0x3fd874080f98fa37              # double 0.38208199999999998
	.quad	0x3ff375a31a4bdba1              # double 1.2162200000000001
	.quad	0x4012876a2f48c2e7              # double 4.6322409999999996
	.quad	0x401dc3a92a305532              # double 7.4410749999999997
	.quad	0x40192966be7afa72              # double 6.2904309999999999
	.quad	0x401b63cc07aaef2c              # double 6.8474579999999996
	.quad	0x40193396073de1e3              # double 6.3003770000000001
	.quad	0x3fd1acb85a4f00ef              # double 0.276167
	.quad	0x40194caf2d7f950c              # double 6.3248870000000004
	.quad	0x3fff1606317268d3              # double 1.942877
	.quad	0x3ff8b5aa715831f0              # double 1.5443519999999999
	.quad	0x4021e347a5b0ff11              # double 8.9439060000000001
	.quad	0x400489185cee17a0              # double 2.5669409999999999
	.quad	0x40225d99ed7c6fbd              # double 9.1828149999999997
	.quad	0x402143fe9fadafd1              # double 8.6328019999999999
	.quad	0x4010346cfc829cfe              # double 4.0511970000000002
	.quad	0x40167b45f17bd8be              # double 5.6203839999999996
	.quad	0x3fe3c26dce39b457              # double 0.61748400000000003
	.quad	0x3ff86b5b2d4d4025              # double 1.5262100000000001
	.quad	0x402328d1d8a54824              # double 9.5797260000000008
	.quad	0x3fa182a9930be0df              # double 0.034200000000000001
	.quad	0x401d100de4c51117              # double 7.2656780000000003
	.quad	0x402093c9cd3e0bd4              # double 8.2886489999999994
	.quad	0x402256d00b45ae60              # double 9.169556
	.quad	0x4004365a9a804966              # double 2.5265399999999998
	.quad	0x402044b9884c6a3c              # double 8.1342280000000002
	.quad	0x3ff45eb52c9d16fd              # double 1.2731220000000001
	.quad	0x400f4dcb9a9da598              # double 3.9129860000000001
	.quad	0x3ff91769a91105e2              # double 1.5682160000000001
	.quad	0x4011c476f2a5a46a              # double 4.4418600000000001
	.quad	0x400116db940fecdd              # double 2.136161
	.quad	0x401c8dd095af294e              # double 7.1384910000000001
	.quad	0x4015ba76965f5276              # double 5.4320930000000001
	.quad	0x40163b107746887b              # double 5.5576800000000004
	.quad	0x40097359791819d2              # double 3.1813229999999999
	.quad	0x402250b545c78a6e              # double 9.1576330000000006
	.quad	0x40224ecb31c219eb              # double 9.1538939999999993
	.quad	0x4018e34efcbd5561              # double 6.2219810000000004
	.quad	0x4002d2b55ef1fddf              # double 2.3528850000000001
	.quad	0x40114e0e736049ed              # double 4.3262270000000003
	.quad	0x401782c83ec892ab              # double 5.8777169999999996
	.quad	0x401e19a8cdea033e              # double 7.5250579999999996
	.quad	0x4018119c5a3e39f7              # double 6.0171979999999996
	.quad	0x4019d94467381d7e              # double 6.4621750000000002
	.quad	0x4013935c182ecaef              # double 4.8939060000000003
	.quad	0x4021100218def417              # double 8.5312660000000005
	.quad	0x3ff4e28fe260b2c8              # double 1.3053129999999999
	.quad	0x3ff135a5332acfb7              # double 1.0755969999999999
	.quad	0x3ff26809d495182b              # double 1.1504000000000001
	.quad	0x3ff09b0bbf50e347              # double 1.0378529999999999
	.quad	0x4010793d102bc72e              # double 4.1183969999999999
	.quad	0x4011101f75104d55              # double 4.2657449999999999
	.quad	0x4001156b87379746              # double 2.135459
	.quad	0x3ff3eda33bd9cae2              # double 1.245517
	.quad	0x4012bbe553ac4f7f              # double 4.6834920000000002
	.quad	0x3ff1c5436b8f9b13              # double 1.11066
	.quad	0x401dca93f290abb4              # double 7.4478299999999997
	.quad	0x40112620685553ef              # double 4.2872329999999996
	.quad	0x40034a7503b81b65              # double 2.4113560000000001
	.quad	0x4008f6b00ffda405              # double 3.1204529999999999
	.quad	0x4021c1e518f3eccc              # double 8.8787009999999995
	.quad	0x40101d3aa369fcf4              # double 4.0285440000000001
	.quad	0x3fb8bae89eba6aca              # double 0.096601999999999993
	.quad	0x401e0e3e6c4c5975              # double 7.51391
	.quad	0x4020c5388a8b08dd              # double 8.3851969999999998
	.quad	0x3ffba429e0a41a26              # double 1.727579
	.quad	0x4023f45baf533f42              # double 9.9772619999999996
	.quad	0x4013e36deb95e5b0              # double 4.972099
	.quad	0x4009a82b62845996              # double 3.2071139999999998
	.quad	0x40230ca2fcefaa47              # double 9.5246809999999992
	.quad	0x4021b20902de00d2              # double 8.8477250000000005
	.quad	0x4012cf6ba0620ab7              # double 4.7025589999999999
	.quad	0x3ffcdf633ce63a5c              # double 1.804538
	.quad	0x4022ec6d612c6ac2              # double 9.4617719999999998
	.quad	0x4020a85985ad538b              # double 8.3288080000000004
	.quad	0x40151bbb62413db8              # double 5.2770820000000001
	.quad	0x402005ae81882adc              # double 8.0110969999999994
	.quad	0x401c8939a7c17a89              # double 7.1340089999999998
	.quad	0x3fee014f8b588e37              # double 0.93766000000000005
	.quad	0x3ff95492ff4ba51a              # double 1.583148
	.quad	0x4020958ab92c0618              # double 8.2920739999999995
	.quad	0x4012904039abf338              # double 4.6408699999999996
	.quad	0x40232bb16c1e364c              # double 9.5853380000000001
	.quad	0x401e6438088509c0              # double 7.5978700000000003
	.quad	0x40100f3f0fe047d4              # double 4.0148890000000002
	.quad	0x4022c92a30553261              # double 9.3928999999999991
	.quad	0x4019a8bc169c23b8              # double 6.4147800000000004
	.quad	0x401064e7685985ad              # double 4.0985389999999997
	.quad	0x40232a26c3b927d4              # double 9.5823269999999993
	.quad	0x400fe0c0f01fb82c              # double 3.9847429999999999
	.quad	0x40232a9068986fce              # double 9.5831330000000001
	.quad	0x401115097c80841f              # double 4.2705440000000001
	.quad	0x401ee67839cd8128              # double 7.7250680000000002
	.quad	0x4014f5e95b78cc9a              # double 5.2401479999999996
	.quad	0x401bb82ba5a03819              # double 6.9298539999999997
	.quad	0x400e50ab2e1693c0              # double 3.7893889999999999
	.quad	0x400814f483cafb3b              # double 3.0102319999999998
	.quad	0x40234d23d4f15e7d              # double 9.6506640000000007
	.quad	0x401247457c0b1359              # double 4.5696009999999996
	.quad	0x4020c0f7db6e5040              # double 8.3768910000000005
	.quad	0x3fe1e88bda9435ad              # double 0.55963700000000005
	.quad	0x4023b7ad4b2745bf              # double 9.8587439999999997
	.quad	0x401eb69fcf3dc055              # double 7.678344
	.quad	0x4003197a24894c44              # double 2.3874399999999998
	.quad	0x3fbac5b078d92fb2              # double 0.10457900000000001
	.quad	0x40134c0c62e4d1a6              # double 4.8242659999999997
	.quad	0x3ff9a36199780baa              # double 1.6023879999999999
	.quad	0x400bdf59ccfaeff6              # double 3.4840580000000001
	.quad	0x401457be9856a37b              # double 5.0856880000000002
	.quad	0x4020c5842b734b51              # double 8.3857739999999996
	.quad	0x4006558537e2c55d              # double 2.7917580000000002
	.quad	0x4022307ce95faa8b              # double 9.0947030000000009
	.quad	0x401e12ea52ef911d              # double 7.518472
	.quad	0x4017c2c01e68a0d3              # double 5.9401859999999997
	.quad	0x401b6c73eee52589              # double 6.8559109999999999
	.quad	0x40213e4d3bf2f556              # double 8.6216830000000009
	.quad	0x400c2806290eed03              # double 3.5195430000000001
	.quad	0x402065bbe0157eed              # double 8.1986989999999995
	.quad	0x4019cc17a89331a1              # double 6.4493090000000004
	.quad	0x402291719f7f8ca8              # double 9.2840699999999998
	.quad	0x3fcda6266fd651b1              # double 0.23163300000000001
	.quad	0x401b3f605ab3aabd              # double 6.8118910000000001
	.quad	0x4019ab96f21f6cad              # double 6.4175680000000002
	.quad	0x401da4102ff8ec10              # double 7.4102180000000004
	.quad	0x3ff788626f60e0eb              # double 1.4707969999999999
	.quad	0x4023e142f61ed5ae              # double 9.9399639999999998
	.quad	0x401d1ffe2a3cea6c              # double 7.2812429999999999
	.quad	0x3ffe481a1554fbdb              # double 1.892603
	.quad	0x40097fe111276fb1              # double 3.1874410000000002
	.quad	0x400c9932d6ece13f              # double 3.5748039999999999
	.quad	0x401483c254a3c643              # double 5.1286709999999998
	.quad	0x4022fecdf266ba49              # double 9.4976649999999995
	.quad	0x3ff626f60e0eb67c              # double 1.384512
	.quad	0x402063b4d48882f1              # double 8.1947390000000002
	.quad	0x401a3db61bb05faf              # double 6.5602650000000002
	.quad	0x4023498efbb0e5e6              # double 9.6436689999999992
	.quad	0x3ff74999dcb5781c              # double 1.4554689999999999
	.quad	0x402357beb9e492bc              # double 9.6713769999999996
	.quad	0x401c8a350d2806af              # double 7.1349679999999998
	.quad	0x4014f04c48adeebb              # double 5.2346659999999998
	.quad	0x3ff060945f9df549              # double 1.023579
	.quad	0x3ff1376b3bb83cf3              # double 1.07603
	.quad	0x401dea2f48c2e771              # double 7.4786960000000002
	.quad	0x3ffc296d4c33b539              # double 1.760114
	.quad	0x4022f5796bfca85d              # double 9.4794420000000005
	.quad	0x401f0a743a647fdc              # double 7.7602089999999997
	.quad	0x400351b79d909f1f              # double 2.414901
	.quad	0x3fd4af3a14cec41e              # double 0.32319500000000001
	.quad	0x40079cc4ac6cdaf5              # double 2.9515470000000001
	.quad	0x401953fe1975f2cb              # double 6.3320239999999997
	.quad	0x400865e892253112              # double 3.04976
	.quad	0x401934245f5ad96a              # double 6.3009199999999996
	.quad	0x3ff8f67f4dbdf8f4              # double 1.5601799999999999
	.quad	0x40224e1c36976bc2              # double 9.1525590000000001
	.quad	0x40234b864cb5bb38              # double 9.6475089999999994
	.quad	0x3fa6145953586ca9              # double 0.043124000000000003
	.quad	0x40201a04f2ae07e6              # double 8.0508190000000006
	.quad	0x3ff99afc04c8bc9d              # double 1.600338
	.quad	0x400420902de00d1b              # double 2.5158999999999998
	.quad	0x3fff37b17ce52ded              # double 1.9510970000000001
	.quad	0x400f68a71de69ad4              # double 3.9260999999999999
	.quad	0x3ff0b0fadf2ecf20              # double 1.0432079999999999
	.quad	0x4015dc02f2f98740              # double 5.464855
	.quad	0x4012fedaec4a4096              # double 4.748882
	.quad	0x400f376c4827b6fe              # double 3.9020619999999999
	.quad	0x3f8ecedd48451330              # double 0.015043000000000001
	.quad	0x4003ee5471715c64              # double 2.4913720000000001
	.quad	0x402136709fa54c55              # double 8.6063279999999995
	.quad	0x402321d81f106680              # double 9.5661020000000008
	.quad	0x401c235b91f70de9              # double 7.034529
	.quad	0x402323e425aee632              # double 9.5701000000000001
	.quad	0x402388a116659d13              # double 9.7668540000000003
	.quad	0x401fd1c4b90214ad              # double 7.9548519999999998
	.quad	0x3ffa3cff64cf8d71              # double 1.6398919999999999
	.quad	0x3ff599d55e6bc622              # double 1.3500570000000001
	.quad	0x3fd380baa582dbe8              # double 0.304732
	.quad	0x401bce5ffa3b9ae1              # double 6.9515380000000002
	.quad	0x3fef98676a7264a1              # double 0.98735399999999995
	.quad	0x4017fa5ca29845dd              # double 5.9944940000000004
	.quad	0x3fe3ed8f4f93bc0a              # double 0.622749
	.quad	0x4009c6822ff08894              # double 3.2219280000000001
	.quad	0x3fc019220ff54089              # double 0.12576699999999999
	.quad	0x40217c8c5004fb12              # double 8.7432580000000009
	.quad	0x40129b2dd377e1b9              # double 4.6515420000000001
	.quad	0x4022e3c07ee0b0af              # double 9.4448279999999993
	.quad	0x3feb62c77574f723              # double 0.85580800000000001
	.quad	0x3fd5e07e5a78f25a              # double 0.34182699999999999
	.quad	0x4011e8123810e886              # double 4.4766320000000004
	.quad	0x400c488ace24bba1              # double 3.5354209999999999
	.quad	0x400bf99ed7c6fbd2              # double 3.4968849999999998
	.quad	0x40136aafff36ac64              # double 4.8541869999999996
	.quad	0x3fe00240b780346e              # double 0.50027500000000003
	.quad	0x400e7e71044f1a1a              # double 3.8117390000000002
	.quad	0x3fe9f865d7cb2d90              # double 0.81157199999999996
	.quad	0x401cec3ac929aa1d              # double 7.2306929999999996
	.quad	0x4009073ffac1d29e              # double 3.1285400000000001
	.quad	0x400344ae85b9e8c4              # double 2.4085359999999998
	.quad	0x401a427bf61aa3f0              # double 6.5649259999999998
	.quad	0x400cd5f2452c59fb              # double 3.6044659999999999
	.quad	0x401a2fcfc3f811f5              # double 6.546691
	.quad	0x40057f266ba493c9              # double 2.6870850000000002
	.quad	0x3fed15ca6ca03c4b              # double 0.90891
	.quad	0x3fd6ef240fa9c12f              # double 0.358346
	.quad	0x3ffa3649d388a8b1              # double 1.6382540000000001
	.quad	0x40035e5c91d14e3c              # double 2.4210750000000001
	.quad	0x3ff2173a797891e2              # double 1.130671
	.quad	0x40228b89b52007dd              # double 9.2725349999999995
	.quad	0x401e76188b11409a              # double 7.6153279999999999
	.quad	0x3fd26933a0407cc8              # double 0.28767100000000001
	.quad	0x401cd459103c8e26              # double 7.2073710000000002
	.quad	0x40031ce3150dae3e              # double 2.3891049999999998
	.quad	0x3fd681ba7fc32ebe              # double 0.35166799999999998
	.quad	0x400ae61e92923e5c              # double 3.3623630000000002
	.quad	0x40202c065b63d3e5              # double 8.0859860000000001
	.quad	0x401d4a73b42cc2d7              # double 7.3227070000000003
	.quad	0x401dba415f45e0b5              # double 7.4318900000000001
	.quad	0x400d2adb402d16b9              # double 3.6459259999999998
	.quad	0x401fc3db7f173754              # double 7.9412669999999999
	.quad	0x4014491d57ff9b56              # double 5.0714009999999998
	.quad	0x4019990de093532e              # double 6.3994669999999996
	.quad	0x401e7ca70d1fa333              # double 7.6217309999999995
	.quad	0x4020fdefa615a8df              # double 8.4959690000000005
	.quad	0x401ed15b573eab36              # double 7.7044499999999996
	.quad	0x402252bc72e275ab              # double 9.1615939999999991
	.quad	0x4000b9cd8127b2cc              # double 2.0907239999999998
	.quad	0x400041cd5f99c38b              # double 2.03213
	.quad	0x40160a2df9378ee3              # double 5.5099410000000004
	.quad	0x40157e429e0a41a2              # double 5.3733009999999997
	.quad	0x4006e75a31a4bdba              # double 2.862965
	.quad	0x40150c75c4a83b1d              # double 5.262168
	.quad	0x4015559b8023a6ce              # double 5.3336009999999998
	.quad	0x4023d5b1e9f27781              # double 9.9173729999999995
	.quad	0x3ff614b167ec7864              # double 1.3800520000000001
	.quad	0x4016ab1960fa15db              # double 5.6670889999999998
	.quad	0x3fda99fe43675ddd              # double 0.41564899999999999
	.quad	0x3fea4d7060bb2bbb              # double 0.82195300000000004
	.quad	0x400b8221c8a7a41e              # double 3.4385409999999998
	.quad	0x401a38123810e886              # double 6.5547570000000004
	.quad	0x402157503b81b64e              # double 8.670534
	.quad	0x3ff2a76640a6b93d              # double 1.165869
	.quad	0x400fc82f944241c4              # double 3.972747
	.quad	0x40197ea5f84cad58              # double 6.3736800000000002
	.quad	0x4021efc0096feb4a              # double 8.9682619999999993
	.quad	0x4021ba7136a400fc              # double 8.8641450000000006
	.quad	0x401aa1f57b41bfbe              # double 6.6581630000000001
	.quad	0x400233d188f42fe8              # double 2.2753019999999999
	.quad	0x400c5e6cd29131ec              # double 3.546106
	.quad	0x40108e3bcd35a858              # double 4.1388999999999996
	.quad	0x4011b7b56b873797              # double 4.4294029999999998
	.quad	0x40213583a53b8e4c              # double 8.6045200000000008
	.quad	0x40188a7c17a89332              # double 6.1352390000000003
	.quad	0x4020e4404a72ead9              # double 8.4458029999999997
	.quad	0x4022bf25a250f840              # double 9.3733339999999998
	.quad	0x400e980dc33721d5              # double 3.8242449999999999
	.quad	0x401ce31df761cbcd              # double 7.221794
	.quad	0x4022a427418d690a              # double 9.3206120000000006
	.quad	0x4000710129cbab65              # double 2.0551780000000002
	.quad	0x4012f949a5657fb7              # double 4.7434450000000004
	.quad	0x400353ef6b5d462c              # double 2.4159839999999999
	.quad	0x3fe983b3e9a6f827              # double 0.79732700000000001
	.quad	0x3fc5df548ecd8dbd              # double 0.170878
	.quad	0x4021df9f44d44567              # double 8.9367619999999999
	.quad	0x4010008c3f3e0371              # double 4.0005350000000002
	.quad	0x3fe77166e008e9b4              # double 0.73259300000000005
	.quad	0x401a518e325d4a5e              # double 6.579644
	.quad	0x4020cd182a9930be              # double 8.4005749999999999
	.quad	0x400c96f9fcb0c027              # double 3.5737190000000001
	.quad	0x40013c1b54195cc8              # double 2.1543489999999998
	.quad	0x40139a2120e1f7d7              # double 4.9005169999999998
	.quad	0x4018691819d2391d              # double 6.1026309999999997
	.quad	0x3fe2563886594af5              # double 0.57302500000000001
	.quad	0x3fc485caafbc1ace              # double 0.160333
	.quad	0x401caf5caf2d7f95              # double 7.171252
	.quad	0x402293ab862b27cc              # double 9.288418
	.quad	0x3ff125f633ce63a6              # double 1.0717680000000001
	.quad	0x4021f18611fd5886              # double 8.9717260000000003
	.quad	0x4021430fadf2ecf2              # double 8.630979
	.quad	0x3ff97e1eb4202d9d              # double 1.593291
	.quad	0x4011e2352eb5f5f1              # double 4.4709060000000003
	.quad	0x3ff2fb7b6bb12902              # double 1.1863969999999999
	.quad	0x40160c5a8155d5f5              # double 5.5120639999999996
	.quad	0x4015c4606748e475              # double 5.4417739999999997
	.quad	0x3ff836f7e3d1cc10              # double 1.51342
	.quad	0x4001bd86a8fc0d2c              # double 2.2175419999999999
	.quad	0x4023ec219eb6390d              # double 9.9611940000000007
	.quad	0x401d422424a276b8              # double 7.314591
	.quad	0x400bc8af81626b2f              # double 3.4729909999999999
	.quad	0x40000a37ac3eb7cc              # double 2.0049890000000001
	.quad	0x4020372fba01eeee              # double 8.1077860000000008
	.quad	0x40123c80c73abc94              # double 4.5590849999999996
	.quad	0x3fe41101b003686a              # double 0.62707599999999997
	.quad	0x4010c2452c59fb1e              # double 4.1897169999999999
	.quad	0x40014cb1897a67a5              # double 2.1624479999999999
	.quad	0x4023b93ccd0fe8ab              # double 9.8617919999999994
	.quad	0x3ff8644d87724fa9              # double 1.5244880000000001
	.quad	0x401fcf90539fba45              # double 7.9526989999999999
	.quad	0x4017ea47a9e2bcf9              # double 5.9787889999999999
	.quad	0x401fb88094e5d5b2              # double 7.9301779999999997
	.quad	0x401b990ea9e6eeb7              # double 6.89947
	.quad	0x40090b063e07a28c              # double 3.1303830000000001
	.quad	0x40216e00f345069a              # double 8.7148509999999995
	.quad	0x3fee9d7dbf487fcc              # double 0.95672500000000004
	.quad	0x4010ccdc443914f5              # double 4.2000590000000004
	.quad	0x40152787ce95faa9              # double 5.2886040000000003
	.quad	0x3fface4755ffe6d6              # double 1.6753610000000001
	.quad	0x4004107852f7f499              # double 2.5080420000000001
	.quad	0x40126b267c6b8b69              # double 4.6046389999999997
	.quad	0x3ff7c288051c9f73              # double 1.484993
	.quad	0x401de9bed30f062d              # double 7.4782669999999998
	.quad	0x40220318116ebd4d              # double 9.006043
	.quad	0x4021e114f483cafb              # double 8.9396129999999996
	.quad	0x400299b6f5caf2d8              # double 2.325056
	.quad	0x3ff4fb41bfbdf091              # double 1.311342
	.quad	0x400aa4adff822bbf              # double 3.3304100000000001
	.quad	0x400bf888f861a60d              # double 3.4963549999999999
	.quad	0x401e663cb8173322              # double 7.5998409999999996
	.quad	0x401f9aeff5c6c11a              # double 7.9013059999999999
	.quad	0x4001d944aa53fc01              # double 2.2310880000000002
	.quad	0x401de08f21709310              # double 7.4692959999999999
	.quad	0x3ffb69f9ce8d972d              # double 1.713373
	.quad	0x401469dcf893faf4              # double 5.1033819999999999
	.quad	0x4021bc894c447c31              # double 8.8682350000000003
	.quad	0x401735f30e7ff584              # double 5.8026850000000003
	.quad	0x40233a339c0ebee0              # double 9.6136750000000006
	.quad	0x401a7964e8b7e4de              # double 6.6185489999999998
	.quad	0x3fe6474107314ca9              # double 0.69619799999999998
	.quad	0x400304c7b02d59d5              # double 2.3773339999999998
	.quad	0x400691e18efbb0e6              # double 2.821231
	.quad	0x40215dc5931ca7d6              # double 8.6831479999999992
	.quad	0x4022ec7f34938583              # double 9.4619079999999993
	.quad	0x401b6b242070b8d0              # double 6.8546300000000002
	.quad	0x4013c064ece9a2c6              # double 4.9378849999999996
	.quad	0x401531fe21d96e9c              # double 5.2988210000000002
	.quad	0x4004f74e65bea0ba              # double 2.6207549999999999
	.quad	0x4005b5c9e6687f45              # double 2.7137639999999998
	.quad	0x4013b67b1c0010c7              # double 4.928204
	.quad	0x400babd2b6f19935              # double 3.458898
	.quad	0x4011762dc6e2a800              # double 4.3654089999999997
	.quad	0x4011fab29e4d5d81              # double 4.4948220000000001
	.quad	0x3fc057a786c22681              # double 0.12767500000000001
	.quad	0x3ffeea91d9b1b79e              # double 1.9322680000000001
	.quad	0x4007f0e560418937              # double 2.9926249999999999
	.quad	0x3fc068ec52a411c3              # double 0.12820200000000001
	.quad	0x3ff76eb1c432ca58              # double 1.4645250000000001
	.quad	0x40071ec3dab5c39c              # double 2.8900220000000001
	.quad	0x4004bab430f49492              # double 2.591164
	.quad	0x402314d4024b33db              # double 9.54068
	.quad	0x4009281c2e33eff2              # double 3.1445850000000002
	.quad	0x4015f5000c9539b9              # double 5.4892580000000004
	.quad	0x4021cd73215fcc18              # double 8.9012689999999992
	.quad	0x4007c64b2314013f              # double 2.9718230000000001
	.quad	0x402257f08893b7d8              # double 9.1717569999999994
	.quad	0x40190b688bda9436              # double 6.2611410000000003
	.quad	0x401a81da37ef5a96              # double 6.6268089999999997
	.quad	0x4016a6feb4a6655a              # double 5.663081
	.quad	0x401524f3775b8130              # double 5.2860849999999999
	.quad	0x400f0ec0724b76f7              # double 3.8822030000000001
	.quad	0x40216aa087ca643d              # double 8.7082560000000004
	.quad	0x3fe76ba0620ab713              # double 0.73188799999999998
	.quad	0x401ba56d5cfaacda              # double 6.9115500000000001
	.quad	0x401c27b84db9c737              # double 7.0387890000000004
	.quad	0x40046ef1348b2208              # double 2.5541710000000002
	.quad	0x400930385c67dfe3              # double 3.1485449999999999
	.quad	0x3ff3da73b42cc2d7              # double 1.2408330000000001
	.quad	0x401fa9717df19d67              # double 7.9154720000000002
	.quad	0x40226d3e2d6238da              # double 9.2133649999999996
	.quad	0x400dc5cc426351df              # double 3.721581
	.quad	0x4022c749f0e4da0a              # double 9.3892360000000004
	.quad	0x4023f07485e3da30              # double 9.9696390000000008
	.quad	0x40102c1871e6cd29              # double 4.0430619999999999
	.quad	0x4004298bb4d48883              # double 2.520286
	.quad	0x401986d00b45ae60              # double 6.381653
	.quad	0x40218992428d434a              # double 8.768694
	.quad	0x4004031832fcac8e              # double 2.5015109999999998
	.quad	0x4003158f2e05ccc9              # double 2.3855270000000002
	.quad	0x3fe5d199bb2788db              # double 0.681836
	.quad	0x4022a1884c6a3bde              # double 9.315493
	.quad	0x401aac3fc43b2dd3              # double 6.6682119999999996
	.quad	0x401140114d2f5dbc              # double 4.3125660000000003
	.quad	0x402141b8cb8e086c              # double 8.6283630000000002
	.quad	0x40231e837f7be122              # double 9.5595970000000001
	.quad	0x4003ff9b56323bbc              # double 2.4998079999999998
	.quad	0x402174c92146a1a5              # double 8.728097
	.quad	0x3ff1710cb295e9e2              # double 1.0901000000000001
	.quad	0x4020a91a543f1c76              # double 8.3302790000000009
	.quad	0x401bec51116a8b8f              # double 6.9807779999999999
	.quad	0x40227360ae9680e0              # double 9.2253469999999993
	.quad	0x4004b51efb6dca08              # double 2.588438
	.quad	0x40020584f4c6e6da              # double 2.2526950000000001
	.quad	0x40017c78a6dacabc              # double 2.1857769999999999
	.quad	0x401a1d09e12a51e3              # double 6.5283579999999999
	.quad	0x4021a83a96d4c33b              # double 8.8285719999999994
	.quad	0x4021f1ad6451b930              # double 8.9720259999999996
	.quad	0x402282e1693c03bc              # double 9.2556259999999994
	.quad	0x4017c265f0f5a101              # double 5.9398419999999996
	.quad	0x400db7db6e503fb3              # double 3.7147739999999998
	.quad	0x3fe301c92ddbdb5e              # double 0.59396800000000005
	.quad	0x40085fbf83382e45              # double 3.0467520000000001
	.quad	0x402229d0e991ff71              # double 9.0816719999999993
	.quad	0x40078a1f080303c0              # double 2.9424419999999998
	.quad	0x400694e11dbca969              # double 2.822695
	.quad	0x3fd79d66adb402d1              # double 0.36898199999999998
	.quad	0x400e9905c0336156              # double 3.8247179999999998
	.quad	0x4005be69270b06c4              # double 2.7179739999999999
	.quad	0x40145d46b26bf877              # double 5.0910900000000003
	.quad	0x400a16b5d462c344              # double 3.2610890000000001
	.quad	0x401a18b8f14db595              # double 6.5241429999999996
	.quad	0x401ea7a8d64d7f0f              # double 7.6637300000000001
	.quad	0x401d76c8b4395810              # double 7.3659999999999997
	.quad	0x401d3abffcdab192              # double 7.3073730000000001
	.quad	0x3ff56602c9081c2e              # double 1.337405
	.quad	0x4016bdba0a526959              # double 5.6852799999999997
	.quad	0x4022e5e8b3b32053              # double 9.4490409999999993
	.quad	0x3ffcba4a48f96e16              # double 1.7954810000000001
	.quad	0x401508df7a4e7ab7              # double 5.2586649999999997
	.quad	0x3fd781a9b8cb8e08              # double 0.36728899999999998
	.quad	0x40238d32c625e99f              # double 9.7757780000000007
	.quad	0x40236dc22ab25b31              # double 9.7143719999999991
	.quad	0x4007f737542a23c0              # double 2.995711
	.quad	0x4015179a6b50b0f2              # double 5.2730499999999996
	.quad	0x4014f86f8ad2567a              # double 5.2426130000000004
	.quad	0x401413c03bc4d22d              # double 5.0192880000000004
	.quad	0x4019314fce746cb9              # double 6.2981559999999996
	.quad	0x3feb228fe260b2c8              # double 0.84796899999999997
	.quad	0x3fd27625204af923              # double 0.28846100000000002
	.quad	0x40140f7164c729f6              # double 5.0150810000000003
	.quad	0x400612452c59fb1e              # double 2.758921
	.quad	0x4010fd698fe69271              # double 4.2474730000000003
	.quad	0x400b1e121ee67514              # double 3.3896829999999998
	.quad	0x401a1626f60e0eb6              # double 6.5216329999999996
	.quad	0x400a579207d4e098              # double 3.2927590000000002
	.quad	0x40165d0d49949e88              # double 5.5908709999999999
	.quad	0x3ffd3966be7afa72              # double 1.826514
	.quad	0x40119d987c6327ee              # double 4.4039020000000004
	.quad	0x40232499d0203e64              # double 9.5714860000000001
	.quad	0x3ff4b1a6d698fe69              # double 1.293372
	.quad	0x3ff5252263d816ad              # double 1.321566
	.quad	0x4013d646f1561911              # double 4.9592549999999997
	.quad	0x401115ee568a5050              # double 4.2714169999999996
	.quad	0x401e6a42f1ed17c6              # double 7.6037710000000001
	.quad	0x40131cdea033e78e              # double 4.7781929999999999
	.quad	0x4011d0a84be40421              # double 4.453767
	.quad	0x40035e06961c3697              # double 2.4209109999999998
	.quad	0x4009e2424a276b7f              # double 3.2354780000000001
	.quad	0x4016f14b599aa609              # double 5.7356389999999999
	.quad	0x3fff038e29f9ce8e              # double 1.9383680000000001
	.quad	0x3fd114e3bcd35a86              # double 0.26690000000000003
	.quad	0x40212d6fb8f57f73              # double 8.5887429999999991
	.quad	0x401a1918e325d4a6              # double 6.5245090000000001
	.quad	0x400ea2174c4cdfad              # double 3.8291460000000002
	.quad	0x401c9ebf65dbfceb              # double 7.1550269999999996
	.quad	0x40209d33b1077469              # double 8.3070350000000008
	.quad	0x401c339a7c17a893              # double 7.0503939999999998
	.quad	0x401c34199fe43676              # double 7.0508790000000001
	.quad	0x4001d884406c00da              # double 2.230721
	.quad	0x401c93ad5bee3d60              # double 7.1442160000000001
	.quad	0x402139351159c497              # double 8.6117329999999992
	.quad	0x3ff4eb474107314d              # double 1.3074410000000001
	.quad	0x40144d8ba40d90e2              # double 5.0757279999999998
	.quad	0x3ff9fe4de3b8a19d              # double 1.6245860000000001
	.quad	0x4021989aaa3ad18d              # double 8.7980549999999997
	.quad	0x4010ba8e71476af9              # double 4.1821840000000003
	.quad	0x40200be4ef0281ba              # double 8.0232309999999991
	.quad	0x4019c6d549731099              # double 6.4441730000000002
	.quad	0x401f5fd71b046845              # double 7.8435940000000003
	.quad	0x4002eb630a91537a              # double 2.364935
	.quad	0x40220eb8305d95dd              # double 9.0287489999999995
	.quad	0x4002bd4b6a619daa              # double 2.3424290000000001
	.quad	0x4020745f394b7b29              # double 8.2272890000000007
	.quad	0x4019a54ac29bf163              # double 6.4114180000000003
	.quad	0x3fe76364bec679cc              # double 0.73088299999999995
	.quad	0x4011e753e707e176              # double 4.4759060000000002
	.quad	0x401fa7f1737542a2              # double 7.9140069999999998
	.quad	0x402023341e14bdfd              # double 8.0687569999999997
	.quad	0x4007e25529fe004b              # double 2.9855139999999998
	.quad	0x3ffd506dd69d3027              # double 1.832136
	.quad	0x3fcd0a6fc58ab92c              # double 0.226881
	.quad	0x401850f2becedd48              # double 6.0790509999999998
	.quad	0x3fe0bdd54da4ce81              # double 0.523173
	.quad	0x401ba2d16b97fe8f              # double 6.9090020000000001
	.quad	0x3fbc754f3775b813              # double 0.111165
	.quad	0x40120d666a98244f              # double 4.5130860000000004
	.quad	0x40174f383f0bae8a              # double 5.8273630000000001
	.quad	0x40207423b7952d23              # double 8.2268349999999995
	.quad	0x40227452a411c2a0              # double 9.2271929999999998
	.quad	0x4015738f7985271c              # double 5.3628520000000002
	.quad	0x401497de5082cf53              # double 5.1483090000000002
	.quad	0x3fe87a67a52ac754              # double 0.76494200000000001
	.quad	0x401bd8c2e770bd01              # double 6.9616809999999996
	.quad	0x40166dca4b124d0a              # double 5.6072170000000003
	.quad	0x4020fcc7b02d59d5              # double 8.4937109999999993
	.quad	0x4020ff16b11c6d1e              # double 8.4982199999999999
	.quad	0x3ff350b417ca2121              # double 1.207203
	.quad	0x4020a352007dd441              # double 8.3189849999999996
	.quad	0x401bda8a82a5614e              # double 6.963419
	.quad	0x4020f1dabe27179c              # double 8.472372
	.quad	0x40133ece9a2c6690              # double 4.8113349999999997
	.quad	0x4002b65ba6efc372              # double 2.3390420000000001
	.quad	0x401d5d3f5b5fa227              # double 7.341062
	.quad	0x401f0fff36ac6477              # double 7.7656219999999995
	.quad	0x400a37c45cbbc2b9              # double 3.2772299999999999
	.quad	0x4023fcb09e98dcdb              # double 9.9935349999999996
	.quad	0x3fed6f60e0eb67c3              # double 0.91984600000000005
	.quad	0x401f088a8b08dd1e              # double 7.7583409999999997
	.quad	0x401ff7b677f6b1a3              # double 7.9919070000000003
	.quad	0x3fe81de050374ff8              # double 0.75364699999999996
	.quad	0x400cb12556d19ded              # double 3.586497
	.quad	0x3ff3d5c74751ce29              # double 1.239692
	.quad	0x4020bfc914b167ec              # double 8.3745809999999991
	.quad	0x4000f13165d39970              # double 2.1177700000000002
	.quad	0x400cfc6540cc78ea              # double 3.62324
	.quad	0x4017b9a671ef30a5              # double 5.9312990000000001
	.quad	0x400729e774eebf66              # double 2.8954610000000001
	.quad	0x400c58055fbb517a              # double 3.5429789999999999
	.quad	0x3ff4798a65492ff5              # double 1.2796730000000001
	.quad	0x401f5f7c6759ab6d              # double 7.8432479999999999
	.quad	0x4015a64883fd5022              # double 5.4123859999999997
	.quad	0x401c955193708aad              # double 7.1458190000000004
	.quad	0x40110e468cac4b4d              # double 4.263941
	.quad	0x400830563ed0f627              # double 3.0236019999999999
	.quad	0x400b295bff04577e              # double 3.3951950000000002
	.quad	0x4016c5653c9abb02              # double 5.6927690000000002
	.quad	0x4008306fef7c243e              # double 3.0236510000000001
	.quad	0x4021e869835158b8              # double 8.9539299999999997
	.quad	0x401db3e9ea140571              # double 7.4256970000000004
	.quad	0x402131afe1da7b0b              # double 8.5970449999999996
	.quad	0x3ff63f78bbd38045              # double 1.390496
	.quad	0x3fff748b220791c5              # double 1.9659530000000001
	.quad	0x400371f8a0902de0              # double 2.43065
	.quad	0x3ff0c04ea4a8c155              # double 1.04695
	.quad	0x3ff367c6b8b69553              # double 1.212836
	.quad	0x40095a365cb35f3d              # double 3.1690489999999998
	.quad	0x400c42602c9081c3              # double 3.53241
	.quad	0x40038a92e62131a9              # double 2.442663
	.quad	0x400f4f34d6a161e5              # double 3.913675
	.quad	0x400d4dec1c1d6cf8              # double 3.6630479999999999
	.quad	0x4020862d40aaeafb              # double 8.2620640000000005
	.quad	0x400b4194c0160525              # double 3.407022
	.quad	0x401f72014727dcbe              # double 7.8613330000000001
	.quad	0x4013bb4a66559f6f              # double 4.9329010000000002
	.quad	0x400a00e30446b69e              # double 3.2504330000000001
	.quad	0x40060d0bb6ed6777              # double 2.75637
	.quad	0x401e02a4db163bac              # double 7.5025820000000003
	.quad	0x4021d82730c67169              # double 8.922174
	.quad	0x4020eab21815a07b              # double 8.4583899999999996
	.quad	0x401a9a8262456f76              # double 6.6508880000000001
	.quad	0x401207e34b945309              # double 4.5077030000000002
	.quad	0x40154308787485e4              # double 5.3154620000000001
	.quad	0x40146495182a9931              # double 5.0982250000000002
	.quad	0x4015c616b54e2b06              # double 5.4434459999999998
	.quad	0x4017875a31a4bdba              # double 5.88218
	.quad	0x40218ba57eaa2a0b              # double 8.7727470000000007
	.quad	0x3ff6109070fbeb9e              # double 1.3790439999999999
	.quad	0x402350ffb8b26395              # double 9.658201
	.quad	0x400ac7bdcf0307f2              # double 3.3475299999999999
	.quad	0x40126e7e62dc6e2b              # double 4.6079040000000004
	.quad	0x400f6c5436b8f9b1              # double 3.9278949999999999
	.quad	0x401863b18dac258d              # double 6.0973569999999997
	.quad	0x401d4d3f1843c3a4              # double 7.3254359999999998
	.quad	0x401fb5cb35f3d7d4              # double 7.9275330000000004
	.quad	0x400691dbca9691a7              # double 2.8212199999999998
	.quad	0x40183704ff43419e              # double 6.0537299999999998
	.quad	0x3fb6159817b95a29              # double 0.086266999999999996
	.quad	0x4011e137f38c5437              # double 4.4699400000000002
	.quad	0x3ff676427c7c5261              # double 1.403872
	.quad	0x40140a4766c6de76              # double 5.0100379999999998
	.quad	0x400633532e7b3d8e              # double 2.775061
	.quad	0x40158864020817fc              # double 5.3831939999999996
	.quad	0x40239a7eb6bf444e              # double 9.8017479999999999
	.quad	0x402343b4b2fa93af              # double 9.6322379999999991
	.quad	0x40142692b3cc4ac7              # double 5.0376690000000002
	.quad	0x4020af86c226809d              # double 8.3428249999999995
	.quad	0x40227f18eb895076              # double 9.2482369999999996
	.quad	0x4012b0d4562e09ff              # double 4.6726850000000004
	.quad	0x3fe2b38f7985271c              # double 0.58441900000000002
	.quad	0x3fe804ec636b0963              # double 0.75060099999999996
	.quad	0x401612480e8c8abd              # double 5.5178529999999997
	.quad	0x3ff18201cd5f99c4              # double 1.0942400000000001
	.quad	0x4020fd2220bc382a              # double 8.4944009999999999
	.quad	0x3fe3748c2e770bd0              # double 0.60797699999999999
	.quad	0x40050a04d12018a4              # double 2.6298919999999999
	.quad	0x40158f9a49c2c1b1              # double 5.3902369999999999
	.quad	0x3ffb635c5b4aa971              # double 1.7117579999999999
	.quad	0x4011908d08919ef9              # double 4.3911629999999997
	.quad	0x3fe349d7ba6698bb              # double 0.60276399999999997
	.quad	0x3ff6dd7342edbb5a              # double 1.429065
	.quad	0x400617036c9c0a8d              # double 2.7612369999999999
	.quad	0x40137c560c7c0f45              # double 4.8714219999999999
	.quad	0x3ff6f7681669ced1              # double 1.4354020000000001
	.quad	0x3f8faf42784a9479              # double 0.015471
	.quad	0x4008f3b9f127f5e8              # double 3.1190069999999999
	.quad	0x402036fcdee34fc6              # double 8.1073979999999999
	.quad	0x400d0d5bee3d5fdd              # double 3.6315230000000001
	.quad	0x4006a5f6ba0620ab              # double 2.8310369999999998
	.quad	0x401b83c31df761cc              # double 6.8786740000000002
	.quad	0x40053795b35b0bbf              # double 2.6521409999999999
	.quad	0x3ffe678f25a250f8              # double 1.9002829999999999
	.quad	0x3ff9dedbf8b9baa1              # double 1.6169089999999999
	.quad	0x3ffbfeee52589268              # double 1.7497389999999999
	.quad	0x400b1074a771c971              # double 3.383035
	.quad	0x4022d717bd8be729              # double 9.4201029999999992
	.quad	0x400b4c1ca3a4b557              # double 3.4121640000000002
	.quad	0x4021fe15ca6ca03c              # double 8.9962599999999994
	.quad	0x4021caa92e62131b              # double 8.8958220000000008
	.quad	0x3ffa3d3804534bd7              # double 1.6399459999999999
	.quad	0x40035f86e3b46fdf              # double 2.4216440000000001
	.quad	0x4018fa350d2806af              # double 6.2443429999999998
	.quad	0x4006dfa871a3b14b              # double 2.8592080000000002
	.quad	0x400bb84ba0e84274              # double 3.464988
	.quad	0x4011a2fdb8fde2ef              # double 4.4091709999999997
	.quad	0x4002d87bdcf0307f              # double 2.3557049999999999
	.quad	0x401f692ee84ad795              # double 7.8527180000000003
	.quad	0x3ff855d2d01c0ca6              # double 1.520953
	.quad	0x40108a7e30878748              # double 4.1352469999999997
	.quad	0x401cc2784a9478c8              # double 7.1899119999999996
	.quad	0x3fe294af4f0d844d              # double 0.58065
	.quad	0x400e8bfdb4cc2507              # double 3.8183549999999999
	.quad	0x4003bbbf0dc768e0              # double 2.4666730000000001
	.quad	0x401231d4f9c1f85d              # double 4.5486639999999996
	.quad	0x401854f722e1ac58              # double 6.0829740000000001
	.quad	0x4005ffc08fa7a850              # double 2.749879
	.quad	0x40222ab1704ff434              # double 9.0833849999999998
	.quad	0x40120cb6c7a7c9de              # double 4.512416
	.quad	0x4021dcc2f837b4a2              # double 8.9311749999999996
	.quad	0x40210bec02f2f987              # double 8.5232849999999996
	.quad	0x3ff8c3009b30728f              # double 1.5476080000000001
	.quad	0x4019db10fd7e4580              # double 6.4639319999999998
	.quad	0x3ff7d6c82e019b0b              # double 1.4899370000000001
	.quad	0x3fc34b48d3ae685e              # double 0.15073500000000001
	.quad	0x3ffdca6223e18698              # double 1.86191
	.quad	0x4010aeed8904f6e0              # double 4.1708280000000002
	.quad	0x40235ba2595bbbe8              # double 9.6789729999999991
	.quad	0x401b2d8d79d0a676              # double 6.7944849999999999
	.quad	0x401a29f23465625a              # double 6.5409629999999996
	.quad	0x40170906466b1e5c              # double 5.758813
	.quad	0x401d69531550ca1d              # double 7.3528560000000001
	.quad	0x3ff01e892253111f              # double 1.007455
	.quad	0x3ff9c9d16fc9bc77              # double 1.611772
	.quad	0x401cedf86e3b46fe              # double 7.2323930000000001
	.quad	0x401e5f8ec0f88334              # double 7.593318
	.quad	0x3ff8403eea209aaa              # double 1.5156849999999999
	.quad	0x401243fe1975f2cb              # double 4.5663989999999997
	.quad	0x401305831f03d146              # double 4.7553830000000001
	.quad	0x3fc152f3c2dadb83              # double 0.13534399999999999
	.quad	0x40100336deb95e5b              # double 4.003139
	.quad	0x3ff93518b0d0edc4              # double 1.5754630000000001
	.quad	0x4014206e9ff0cbb0              # double 5.0316720000000004
	.quad	0x4004427b6fe2e6eb              # double 2.5324620000000002
	.quad	0x3ffafbfb58d1526e              # double 1.6865190000000001
	.quad	0x401cfeafee6fb4c4              # double 7.2487180000000002
	.quad	0x4010de4cd7492791              # double 4.2170899999999998
	.quad	0x40052772d5e071c5              # double 2.6442619999999999
	.quad	0x401aa5a708ede54b              # double 6.6617699999999997
	.quad	0x4002eb2e9ccb7d41              # double 2.3648349999999998
	.quad	0x4010862a5e785b5b              # double 4.1310209999999996
	.quad	0x3ff860d349be8ff3              # double 1.523639
	.quad	0x401a9b62413db7f1              # double 6.6517419999999996
	.quad	0x3ff23faa044ae85c              # double 1.1405430000000001
	.quad	0x400933edd8b60f1b              # double 3.1503559999999999
	.quad	0x3fe5451dabe2717a              # double 0.66468700000000003
	.quad	0x3fd308398a654930              # double 0.297377
	.quad	0x401e5898b2e9ccb8              # double 7.5865200000000001
	.quad	0x4003a1d5c31593e6              # double 2.454021
	.quad	0x3ff3de74299d883c              # double 1.2418100000000001
	.quad	0x40154deb0fadf2ed              # double 5.326092
	.quad	0x3ff698aa8650e779              # double 1.412272
	.quad	0x40101122fad6cb53              # double 4.0167349999999997
	.quad	0x401a2068123810e9              # double 6.5316470000000004
	.quad	0x40189efd438d1d8a              # double 6.1552629999999997
	.quad	0x3fe0e0bf5d78811b              # double 0.52743499999999999
	.quad	0x4023bc83a96d4c34              # double 9.8681920000000005
	.quad	0x401c28348f53c57a              # double 7.039263
	.quad	0x4010355f3519bd40              # double 4.0521209999999996
	.quad	0x40117e071c53f39d              # double 4.3730739999999999
	.quad	0x4015ffbacb428912              # double 5.4997360000000004
	.quad	0x40165e7967caea74              # double 5.5922599999999996
	.quad	0x3f856e264e48626f              # double 0.010463999999999999
	.quad	0x40226231e7d9988d              # double 9.1917869999999997
	.quad	0x3ff4419c17225b75              # double 1.2660180000000001
	.quad	0x40203023e9ea1405              # double 8.0940239999999992
	.quad	0x40137564b662fdfc              # double 4.8646419999999999
	.quad	0x40202f4b838c111b              # double 8.0923730000000003
	.quad	0x40096807357e670e              # double 3.1757949999999999
	.quad	0x401d15d67ba237ce              # double 7.2713260000000002
	.quad	0x40226048beb5b2d5              # double 9.1880550000000003
	.quad	0x3fff79a38fbca106              # double 1.9671970000000001
	.quad	0x3ff0165f5275ee9a              # double 1.0054620000000001
	.quad	0x400f14690de09353              # double 3.8849659999999999
	.quad	0x401bba237cdcca71              # double 6.9317760000000002
	.quad	0x401846d549731099              # double 6.0691730000000002
	.quad	0x400b02ba5a038195              # double 3.3763320000000001
	.quad	0x4022506b59146e4c              # double 9.1570689999999999
	.quad	0x4017f9de05037500              # double 5.9940110000000004
	.quad	0x4009efd438d1d8a5              # double 3.2421039999999999
	.quad	0x4010b87db2b34613              # double 4.180167
	.quad	0x4013c77963992c8c              # double 4.9447989999999997
	.quad	0x4020da5fa658c4bd              # double 8.4265109999999996
	.quad	0x400fe5de15ca6ca0              # double 3.9872399999999999
	.quad	0x402245e6362a5e78              # double 9.1365219999999993
	.quad	0x3ff18d62bf11f927              # double 1.097018
	.quad	0x3ff27c3504c48adf              # double 1.155324
	.quad	0x4022c692d55a3a08              # double 9.3878389999999996
	.quad	0x40161580c308feac              # double 5.5209989999999998
	.quad	0x401b61fc4c165908              # double 6.8456890000000001
	.quad	0x40212208e1501190              # double 8.5664739999999995
	.quad	0x4014af0a1be2b496              # double 5.1709370000000003
	.quad	0x40207ff822bbecab              # double 8.2499400000000005
	.quad	0x40231a5a469d7343              # double 9.5514700000000001
	.quad	0x3fcf0667f90d9d77              # double 0.24238299999999999
	.quad	0x400c7e84f09528f2              # double 3.5617770000000002
	.quad	0x3fc881d3aa369fcf              # double 0.19146199999999999
	.quad	0x401653e186983516              # double 5.5819150000000004
	.quad	0x40093dbec2480e8d              # double 3.1551490000000002
	.quad	0x401fdc322291fb40              # double 7.9650350000000003
	.quad	0x3fbe6a65cf67b1c0              # double 0.118811
	.quad	0x401c97dc7abfb9bf              # double 7.1483020000000002
	.quad	0x4021fa0e6299524c              # double 8.988391
	.quad	0x4021cfe111276fb1              # double 8.9060140000000008
	.quad	0x3fc38811b1d92b80              # double 0.15259
	.quad	0x3ff53fad2999567e              # double 1.3280460000000001
	.quad	0x4001e1ed5ae1cde6              # double 2.2353160000000001
	.quad	0x40034ee631f8a090              # double 2.4135249999999999
	.quad	0x401cb9463cfb3312              # double 7.1809320000000003
	.quad	0x3ffe74f82f512663              # double 1.9035569999999999
	.quad	0x401bdae924f227d0              # double 6.9637799999999999
	.quad	0x401251355475a31a              # double 4.5793049999999997
	.quad	0x40239322d0e56042              # double 9.7873750000000008
	.quad	0x401f090e23af31b1              # double 7.7588429999999997
	.quad	0x40003f9f87f023ea              # double 2.031066
	.quad	0x3ffa3562174c4ce0              # double 1.6380330000000001
	.quad	0x3ff043be6601bc99              # double 1.0165390000000001
	.quad	0x3fe8e770bd017daf              # double 0.77825200000000005
	.quad	0x4023b6b0b7c3504c              # double 9.8568169999999994
	.quad	0x40106c729f59ccfb              # double 4.1059060000000001
	.quad	0x400ceab9b23dd54e              # double 3.6146120000000002
	.quad	0x401d05e9e1b089a0              # double 7.2557749999999999
	.quad	0x3fce5bfad2999568              # double 0.237182
	.quad	0x40202355cd91eeaa              # double 8.0690139999999992
	.quad	0x3ff411c2a0232096              # double 1.2543359999999999
	.quad	0x3ffac852b4d8ba41              # double 1.673907
	.quad	0x400df4e22a2c2374              # double 3.7445719999999998
	.quad	0x402272e09fe86834              # double 9.2243700000000004
	.quad	0x40223ccb3a2595bc              # double 9.1187380000000004
	.quad	0x4023f1b0ee49f517              # double 9.9720530000000007
	.quad	0x3fd46523704790b8              # double 0.31867299999999998
	.quad	0x3ff7619c9d5a187a              # double 1.4613309999999999
	.quad	0x3ff636a1e81cb46c              # double 1.3883380000000001
	.quad	0x401959a671ef30a5              # double 6.3375490000000001
	.quad	0x3ff06685db76b3bc              # double 1.0250300000000001
	.quad	0x4014971715c63ae2              # double 5.1475489999999997
	.quad	0x4023623ec02f2f98              # double 9.6918849999999991
	.quad	0x40238491d14e3bcd              # double 9.7589249999999996
	.quad	0x4013b8c0485a0be5              # double 4.9304209999999999
	.quad	0x4008bc94f69ca9ef              # double 3.0920809999999999
	.quad	0x40118163779e9d0f              # double 4.3763560000000004
	.quad	0x40148f0c77dd872f              # double 5.1396959999999998
	.quad	0x4010fb51372a38b9              # double 4.2454270000000003
	.quad	0x401474a8c154c986              # double 5.1139250000000001
	.quad	0x4013b583a53b8e4c              # double 4.9272600000000004
	.quad	0x400eae6e221c8a7a              # double 3.8351709999999999
	.quad	0x4021f18e97071822              # double 8.9717909999999996
	.quad	0x40218d41b75a74c1              # double 8.7758920000000007
	.quad	0x401ec64f97edc7ef              # double 7.6936629999999999
	.quad	0x4007827e0ef99807              # double 2.938717
	.quad	0x4011dceaaf35e311              # double 4.4657390000000001
	.quad	0x4014abcd78c436fc              # double 5.1677759999999999
	.quad	0x401b80b135979182              # double 6.8756760000000003
	.quad	0x3fec7ebd09e12a52              # double 0.89047100000000001
	.quad	0x400e558c08b75ea6              # double 3.7917709999999998
	.quad	0x4022601cd5f99c39              # double 9.1877200000000006
	.quad	0x40207e9f2778140e              # double 8.2473080000000003
	.quad	0x4012cb1b79d909f2              # double 4.6983470000000001
	.quad	0x401d1379314445aa              # double 7.2690169999999998
	.quad	0x40200b60f1b25f63              # double 8.0222239999999996
	.quad	0x4013e83558a76102              # double 4.9767659999999996
	.quad	0x4015645f9df548ed              # double 5.3480210000000001
	.quad	0x400babf1a5ca2984              # double 3.4589569999999998
	.quad	0x3fe578b7a1c25d07              # double 0.67098599999999997
	.quad	0x400db4f3fd933e36              # double 3.7133560000000001
	.quad	0x40208b35d249e450              # double 8.2718950000000007
	.quad	0x4021201adea89763              # double 8.5627049999999993
	.quad	0x40151c0e7bc3c5bd              # double 5.277399
	.quad	0x4009ddcb5781c715              # double 3.233298
	.quad	0x4016c43d89ce4a7b              # double 5.6916409999999997
	.quad	0x3ffea746cb966be8              # double 1.9158390000000001
	.quad	0x400907d6733ebbfd              # double 3.1288269999999998
	.quad	0x401e71d25aab4741              # double 7.6111539999999999
	.quad	0x401dad35eb7457c1              # double 7.4191510000000003
	.quad	0x3ffc9056c508b32d              # double 1.785239
	.quad	0x4004f29d40ee06d9              # double 2.6184639999999999
	.quad	0x401c772862f5989e              # double 7.1163650000000001
	.quad	0x400a2e41919ac797              # double 3.272586
	.quad	0x401f4aade657b84e              # double 7.8229290000000002
	.quad	0x4021302730c67169              # double 8.594049
	.quad	0x4020eb471f79420b              # double 8.4595269999999996
	.quad	0x3ff782938de6ddf0              # double 1.469379
	.quad	0x4023bee3d5fdcdf7              # double 9.8728320000000007
	.quad	0x401cc49be8ff327b              # double 7.1920010000000003
	.quad	0x3fe06aab042528ae              # double 0.51302099999999995
	.quad	0x3ff592bc2fc69729              # double 1.3483240000000001
	.quad	0x4023f3c42a66dbd7              # double 9.9761059999999996
	.quad	0x3ff03be9856a37ac              # double 1.0146269999999999
	.quad	0x3fb62ca9ac365450              # double 0.086619000000000001
	.quad	0x401ed05c03361566              # double 7.7034760000000002
	.quad	0x401fec200c0f01fc              # double 7.9805910000000004
	.quad	0x401cd24cc6822ff1              # double 7.2053710000000004
	.quad	0x401ab07aaef2c732              # double 6.6723429999999997
	.quad	0x3fe732fec56d5cfb              # double 0.72497500000000004
	.quad	0x400bb98bf7f06706              # double 3.4655990000000001
	.quad	0x401b5781c714fce7              # double 6.8354559999999998
	.quad	0x40175f1ae2da554c              # double 5.8428760000000004
	.quad	0x3fed2cbea4ebdd33              # double 0.91171199999999997
	.quad	0x3ffd3cddd6e04c06              # double 1.8273600000000001
	.quad	0x401239f44d445671              # double 4.5565959999999999
	.quad	0x40226ffcfc3f811f              # double 9.2187269999999994
	.quad	0x402250d66f0cfe15              # double 9.1578859999999995
	.quad	0x40100e685db76b3c              # double 4.0140700000000002
	.quad	0x4005c9563a9f383f              # double 2.723309
	.quad	0x3ff76c710cb295ea              # double 1.463975
	.quad	0x4022cbbe22e5de16              # double 9.3979350000000004
	.quad	0x401fdc46d82ba5a0              # double 7.9651139999999998
	.quad	0x400fc0b673c4f3ba              # double 3.9690979999999998
	.quad	0x4020deffc115df65              # double 8.4355449999999994
	.quad	0x4015f36501e2584f              # double 5.4876899999999997
	.quad	0x401f72502eec7c95              # double 7.8616339999999995
	.quad	0x3ff8ac2374794ea0              # double 1.5420259999999999
	.quad	0x3fec7cd24e160d88              # double 0.89023699999999995
	.quad	0x4014bd40aaeafab5              # double 5.1848169999999998
	.quad	0x3fdc386d2ed783e0              # double 0.440944
	.quad	0x401503f1c75818c6              # double 5.2538520000000002
	.quad	0x4022043265add9c2              # double 9.0081969999999991
	.quad	0x4020f0620ab71327              # double 8.4694979999999997
	.quad	0x400672ec6bce8534              # double 2.8061150000000001
	.quad	0x401a64528283d35f              # double 6.5979710000000003
	.quad	0x40149862b27cbf6e              # double 5.1488139999999998
	.quad	0x4022ba4f227d028a              # double 9.3638849999999998
	.quad	0x40114a66559f6ec6              # double 4.3226560000000003
	.quad	0x4018b4f3fd933e36              # double 6.1767120000000002
	.quad	0x3ff4f3c2dadb8349              # double 1.309512
	.quad	0x4019a07d0aed99cc              # double 6.4067270000000001
	.quad	0x4017ace789e774ef              # double 5.9188520000000002
	.quad	0x401d348beb5b2d4d              # double 7.3013149999999998
	.quad	0x3ff8ae3c536d655e              # double 1.542538
	.quad	0x4012eea67e846a5d              # double 4.7330569999999996
	.quad	0x4011830f49491f2e              # double 4.3779880000000002
	.quad	0x3ff0415f45e0b4e1              # double 1.01596
	.quad	0x40054f6be37de93a              # double 2.66378
	.quad	0x401b935696e58a33              # double 6.893885
	.quad	0x40056aa9717df19d              # double 2.6770809999999998
	.quad	0x402384638c975297              # double 9.7585719999999991
	.quad	0x401f13a0c6b484d7              # double 7.7691679999999996
	.quad	0x4014f1682f944242              # double 5.2357490000000002
	.quad	0x400d064dc22ab25b              # double 3.6280779999999999
	.quad	0x4023769680e06530              # double 9.731617
	.quad	0x3ff5c30c23fab10c              # double 1.3601190000000001
	.quad	0x4023ee379b77c02b              # double 9.965268
	.quad	0x4021cfeb28d8665e              # double 8.906091
	.quad	0x40085901083dbc23              # double 3.0434589999999999
	.quad	0x401c1c5de9c0229a              # double 7.0277019999999997
	.quad	0x3fc2c6ac215b9a5b              # double 0.14668800000000001
	.quad	0x400998b827fa1a0d              # double 3.19957
	.quad	0x400d369a91105e1c              # double 3.651662
	.quad	0x402104c447c30d30              # double 8.5093099999999993
	.quad	0x401e0c6b05319829              # double 7.5121270000000004
	.quad	0x401460dbad3a604e              # double 5.0945879999999999
	.quad	0x3ffdc00218def417              # double 1.8593770000000001
	.quad	0x4009cdbe7f2c300a              # double 3.2254610000000001
	.quad	0x401e2614163779ea              # double 7.5371860000000002
	.quad	0x401d8f2c7325918a              # double 7.389818
	.quad	0x4008dfa871a3b14b              # double 3.1092080000000002
	.quad	0x401a1a53b8e4b87c              # double 6.5257100000000001
	.quad	0x4011009fe86833c6              # double 4.25061
	.quad	0x4013a924af0bf1a6              # double 4.9151790000000002
	.quad	0x3fd4ef8055fbb518              # double 0.32711800000000002
	.quad	0x3ffb750d2806af47              # double 1.7160770000000001
	.quad	0x4020a7460242d05f              # double 8.3267059999999997
	.quad	0x40192294141e9af6              # double 6.2837680000000002
	.quad	0x400098583621fafd              # double 2.0743870000000002
	.quad	0x400a76dc1a47a9e3              # double 3.3080370000000001
	.quad	0x402032d948dc11e4              # double 8.0993139999999996
	.quad	0x4017c39de481f538              # double 5.9410319999999999
	.quad	0x4013bd0678c0053e              # double 4.9345949999999998
	.quad	0x401a7fbdad7518b1              # double 6.6247470000000002
	.quad	0x401922a3486f049b              # double 6.2838260000000004
	.quad	0x400754445aa2e3c5              # double 2.9161459999999999
	.quad	0x3ff6cc9539b88872              # double 1.424947
	.quad	0x401496dacabc5155              # double 5.1473190000000004
	.quad	0x3fdd4395810624dd              # double 0.45724999999999999
	.quad	0x401bacd8127b2cc7              # double 6.918793
	.quad	0x401014bba12b5e53              # double 4.0202470000000003
	.quad	0x4010367381d7dbf5              # double 4.0531750000000004
	.quad	0x3fe2cc5974e65bea              # double 0.58744499999999999
	.quad	0x3fa4bba12b5e529c              # double 0.040494000000000002
	.quad	0x4018f78ee2867275              # double 6.2417559999999996
	.quad	0x4017f87980f55de6              # double 5.9926510000000004
	.quad	0x4020c3a109d0635a              # double 8.3820879999999995
	.quad	0x400c25143bf72713              # double 3.5181049999999998
	.quad	0x401012220bc382a1              # double 4.0177079999999998
	.quad	0x3ffb8ae642bf9831              # double 1.721411
	.quad	0x401a788d2a1f8e3b              # double 6.6177260000000002
	.quad	0x400ef5a20ddc6195              # double 3.8699379999999999
	.quad	0x4005246173b85e81              # double 2.6427640000000001
	.quad	0x401ed57ff9b56324              # double 7.7084960000000002
	.quad	0x3fe0859ea57214f0              # double 0.51631099999999996
	.quad	0x4022d8a265f0f5a1              # double 9.4231139999999999
	.quad	0x401cb0e3cd9a5226              # double 7.1727439999999998
	.quad	0x4010e60c7c0f4517              # double 4.2246569999999997
	.quad	0x401a99b52007dd44              # double 6.6501049999999999
	.quad	0x3fe0cf15e7c8d196              # double 0.52527900000000005
	.quad	0x400b1dcdf6987831              # double 3.3895529999999998
	.quad	0x40228afcefaa4767              # double 9.2714610000000004
	.quad	0x40011e01de269164              # double 2.1396519999999999
	.quad	0x400bdf9f87f023ea              # double 3.484191
	.quad	0x401e28622c450269              # double 7.5394370000000003
	.quad	0x3fdf258d5842b735              # double 0.48666700000000002
	.quad	0x3fc42bc72e275ab8              # double 0.157586
	.quad	0x3fe55b50f40e5a36              # double 0.66739700000000002
	.quad	0x40227fb6bb129025              # double 9.2494409999999991
	.quad	0x4023db599aa60914              # double 9.9284180000000006
	.quad	0x40171160d887ebf2              # double 5.7669709999999998
	.quad	0x402034894c447c31              # double 8.1026100000000003
	.quad	0x4012c3bc903ea705              # double 4.6911490000000002
	.quad	0x3fb2d6777079e59f              # double 0.073584999999999998
	.quad	0x400252263d816acf              # double 2.2901120000000001
	.quad	0x40118ed45e9185cf              # double 4.3894820000000001
	.quad	0x401d5945b6c3760c              # double 7.33718
	.quad	0x40225bcf4e874c90              # double 9.179316
	.quad	0x40213ae3608d0892              # double 8.6150160000000007
	.quad	0x3fcbeec39799e519              # double 0.218224
	.quad	0x4000ef87ad080b67              # double 2.1169579999999999
	.quad	0x40236d73215fcc18              # double 9.7137689999999992
	.quad	0x3f9fa22706d50657              # double 0.030891999999999999
	.quad	0x40231749adc8fb87              # double 9.545484
	.quad	0x4023a0c7c0f45176              # double 9.8140239999999999
	.quad	0x3ffbd80f55de58e6              # double 1.7402489999999999
	.quad	0x402229a1554fbdad              # double 9.0813089999999992
	.quad	0x40139da3c21187e8              # double 4.9039450000000002
	.quad	0x4019f59a73b42cc3              # double 6.4898470000000001
	.quad	0x400877ff9b56323c              # double 3.0585930000000001
	.quad	0x401fb18e757928e1              # double 7.9233950000000001
	.quad	0x401ec31e7d9988d3              # double 7.6905460000000003
	.quad	0x3fe36f5232d2bb23              # double 0.60733899999999996
	.quad	0x40232be7296f6513              # double 9.5857480000000006
	.quad	0x40153540cc78e9f7              # double 5.3020050000000003
	.quad	0x3fe7500713f077cd              # double 0.72851900000000003
	.quad	0x400128943e100608              # double 2.1448140000000002
	.quad	0x400128f648c7192f              # double 2.1450010000000002
	.quad	0x401b76f9b994e1a4              # double 6.866187
	.quad	0x3fedf1d606317269              # double 0.93577100000000002
	.quad	0x4013d7fa5d28d003              # double 4.9609160000000001
	.quad	0x4011129f16b11c6d              # double 4.2681849999999999
	.quad	0x401175568e820e63              # double 4.3645880000000004
	.quad	0x40135fbdf090f734              # double 4.8434980000000003
	.quad	0x4015f83621fafc8b              # double 5.492394
	.quad	0x4021627674d16335              # double 8.6923100000000009
	.quad	0x3fddcc20d5629d84              # double 0.465584
	.quad	0x401a35918a009f62              # double 6.5523129999999998
	.quad	0x402314dfaca36199              # double 9.5407689999999991
	.quad	0x402126defc7a3982              # double 8.57592
	.quad	0x400751758e219653              # double 2.9147750000000001
	.quad	0x4022e9c79b34a44c              # double 9.4566009999999991
	.quad	0x40020925fe974a34              # double 2.254467
	.quad	0x3f4c58255b035bd5              # double 8.6499999999999999E-4
	.quad	0x400fd7b17ce52ded              # double 3.9803190000000002
	.quad	0x400348882f0e0a85              # double 2.4104160000000001
	.quad	0x4017103c4b09e98e              # double 5.7658550000000002
	.quad	0x400e09856a37ac3f              # double 3.7546490000000001
	.quad	0x401fc64e48626f61              # double 7.9436580000000001
	.quad	0x40184751ce28ed5f              # double 6.0696479999999999
	.quad	0x400e9c42a66dbd73              # double 3.8262990000000001
	.quad	0x3fd6d41743e963dc              # double 0.35669499999999998
	.quad	0x401d84e3ffef3908              # double 7.3797759999999996
	.quad	0x40233022ff08893b              # double 9.5940169999999991
	.quad	0x40061f8b1572580c              # double 2.7654019999999999
	.quad	0x3ff3bac0c62e4d1a              # double 1.2330939999999999
	.quad	0x4021ea76fb09203a              # double 8.9579389999999997
	.quad	0x4020b21cac083127              # double 8.3478750000000002
	.quad	0x4012439c0ebedfa4              # double 4.5660249999999998
	.quad	0x400e048c7192ea53              # double 3.752221
	.quad	0x40022c0f88333fc8              # double 2.2715139999999998
	.quad	0x40211f47f993d534              # double 8.5610959999999991
	.quad	0x4010b7e670e2c12b              # double 4.1795900000000001
	.quad	0x4012ccbc48f10a9a              # double 4.6999370000000003
	.quad	0x401496640a6b93cd              # double 5.1468660000000002
	.quad	0x3fded634549b62c7              # double 0.48182399999999997
	.quad	0x400c55ac03ff6901              # double 3.5418319999999999
	.quad	0x4004dc7abfb9bed3              # double 2.607656
	.quad	0x4003dfbdf090f734              # double 2.4842490000000002
	.quad	0x4000292ab68cef67              # double 2.0201009999999999
	.quad	0x401d1f71eafee6fb              # double 7.2807079999999997
	.quad	0x402234cd0fe8ab50              # double 9.1031270000000006
	.quad	0x3fef2e62131a8ef7              # double 0.97441199999999994
	.quad	0x4017600e6afcce1c              # double 5.8438049999999997
	.quad	0x3ff9380a9de8b3b3              # double 1.576182
	.quad	0x4021e820a30db6a2              # double 8.9533740000000002
	.quad	0x4017f0cf9e3864cb              # double 5.9851669999999997
	.quad	0x402331856a37ac3f              # double 9.5967210000000005
	.quad	0x401506909aed56b0              # double 5.2564109999999999
	.quad	0x4021d74d9c6b0532              # double 8.9205140000000007
	.quad	0x40208a27ae9ab29e              # double 8.2698339999999995
	.quad	0x40108c27a63736ce              # double 4.13687
	.quad	0x4021a10ff9724745              # double 8.8145749999999996
	.quad	0x4011d3543aeab799              # double 4.4563759999999997
	.quad	0x4004aaef2c732592              # double 2.5834640000000002
	.quad	0x401c060285ec3dab              # double 7.0058689999999997
	.quad	0x40213086df82b1f7              # double 8.5947790000000008
	.quad	0x3feac27418d6909b              # double 0.83623700000000001
	.quad	0x40234f221858bc5a              # double 9.6545570000000005
	.quad	0x4018e5bee3d5fdce              # double 6.224361
	.quad	0x401fc7ad4b2745bf              # double 7.9449969999999998
	.quad	0x401c5b6bf444e0db              # double 7.0892790000000003
	.quad	0x401fa1511dffc548              # double 7.9075360000000003
	.quad	0x3fd0fbefd00713f0              # double 0.26537699999999997
	.quad	0x3fce460ed80a17b1              # double 0.236513
	.quad	0x401827f4128bf3bf              # double 6.0390170000000003
	.quad	0x3fa12dfd694ccab4              # double 0.033554
	.quad	0x401dde9ee45c358b              # double 7.467403
	.quad	0x3fa98f1d3ed527e5              # double 0.049919999999999999
	.quad	0x400294289118c198              # double 2.322343
	.quad	0x401b979ee02a77a3              # double 6.8980670000000002
	.quad	0x3feff09fa54c5543              # double 0.99812299999999998
	.quad	0x400f8c7f34938583              # double 3.9436019999999998
	.quad	0x402371b62c77574f              # double 9.7220929999999992
	.quad	0x4023cf70de8f6cf0              # double 9.9051580000000001
	.quad	0x401abb302f72b453              # double 6.6828010000000004
	.quad	0x401071c64ba94bbe              # double 4.1111079999999998
	.quad	0x4022c03558a76102              # double 9.3754069999999992
	.quad	0x402073a6266fd652              # double 8.2258770000000005
	.quad	0x401116b228dc981c              # double 4.2721640000000001
	.quad	0x40205bedb7281fda              # double 8.1795480000000005
	.quad	0x3fe24bbe4473cd57              # double 0.57174599999999998
	.quad	0x401115d63886594b              # double 4.271325
	.quad	0x4018935935fc3b4f              # double 6.1438949999999997
	.quad	0x4007294d50ebaade              # double 2.8951669999999998
	.quad	0x401be15a8deb0fae              # double 6.970072
	.quad	0x401807ba237cdcca              # double 6.0075459999999996
	.quad	0x401f543a2163fdd6              # double 7.8322529999999997
	.quad	0x4023174f50a02b84              # double 9.5455269999999998
	.quad	0x4012013f4a98aa86              # double 4.5012179999999997
	.quad	0x3fd85b899392189c              # double 0.38058700000000001
	.quad	0x401b82fd75e2046c              # double 6.8779199999999996
	.quad	0x3fb83d68405b39e2              # double 0.094686999999999993
	.quad	0x401a14567125dd09              # double 6.5198609999999997
	.quad	0x40141005814940bc              # double 5.0156460000000003
	.quad	0x4021f1d0203e63e9              # double 8.9722910000000002
	.quad	0x4013361d8622c450              # double 4.8028469999999999
	.quad	0x400b2fc0d2c386d3              # double 3.398317
	.quad	0x40120bce8533b107              # double 4.5115299999999996
	.quad	0x400a02352eb5f5f1              # double 3.2510780000000001
	.quad	0x401e04b1ab0856e7              # double 7.5045840000000003
	.quad	0x40011851654d61b3              # double 2.1368740000000002
	.quad	0x4011f7f88765ba6f              # double 4.492159
	.quad	0x3fe72836eb4e9814              # double 0.72365900000000005
	.quad	0x40206d80a17b0f6b              # double 8.2138720000000003
	.quad	0x40112b70ef566491              # double 4.2924230000000003
	.quad	0x3fcffe974a3400b9              # double 0.24995700000000001
	.quad	0x40035d5c31593e60              # double 2.4205860000000001
	.quad	0x401442003ab862b2              # double 5.0644539999999996
	.quad	0x40114917507e9d95              # double 4.3213780000000002
	.quad	0x40128b0813cab820              # double 4.6357730000000004
	.quad	0x4023c3b55ac03ff7              # double 9.8822430000000007
	.quad	0x400a8b5f5f0b2852              # double 3.3180529999999999
	.quad	0x401efae2541d8e86              # double 7.7450039999999998
	.quad	0x400e7a5119ce075f              # double 3.8097249999999998
	.quad	0x4013f80c308feac4              # double 4.9922339999999998
	.quad	0x4023e7fb267c6b8b              # double 9.9530879999999992
	.quad	0x400a2d7060bb2bbb              # double 3.2721870000000002
	.quad	0x400073c68661ae71              # double 2.0565310000000001
	.quad	0x3fc39ee88df3732a              # double 0.15328700000000001
	.quad	0x402027fb8b263950              # double 8.0780910000000006
	.quad	0x4021fe481f53825e              # double 8.9966439999999999
	.quad	0x400403254e6e221d              # double 2.5015360000000002
	.quad	0x400e01be2b4959e6              # double 3.7508509999999999
	.quad	0x3feec4fa05143bf7              # double 0.96154499999999998
	.quad	0x40064303c07ee0b1              # double 2.7827220000000001
	.quad	0x3fe9b45ae5ffa3ba              # double 0.80326600000000004
	.quad	0x401da11276fb0920              # double 7.4072969999999998
	.quad	0x401ec19c5a3e39f7              # double 7.6890729999999996
	.quad	0x400e053543aeab7a              # double 3.7525430000000002
	.quad	0x400a8588a04d1202              # double 3.3152020000000002
	.quad	0x4002cfc7a398201d              # double 2.3514550000000001
	.quad	0x3fe5a5b9628cbd12              # double 0.67647999999999997
	.quad	0x4005efd6d7e889c2              # double 2.7421090000000001
	.quad	0x4002fa5657fb6998              # double 2.3722349999999999
	.quad	0x401dfe978d4fdf3b              # double 7.4986249999999996
	.quad	0x3faa2e5de15ca6ca              # double 0.051135
	.quad	0x3ff552fc2656abde              # double 1.3327599999999999
	.quad	0x4021b25482385404              # double 8.8483009999999993
	.quad	0x401fc90214ad362f              # double 7.9462970000000003
	.quad	0x401604ca4f440af2              # double 5.5046780000000002
	.quad	0x400e9415b1422ccb              # double 3.8223069999999999
	.quad	0x3ff53220791c4b90              # double 1.324738
	.quad	0x401b88c1db0142f6              # double 6.8835519999999999
	.quad	0x401efdd0d8cb07d1              # double 7.7478670000000002
	.quad	0x4000948ecd8dbced              # double 2.0725380000000002
	.quad	0x400987e175d13d75              # double 3.1913480000000001
	.quad	0x401b0270f3882279              # double 6.7523840000000002
	.quad	0x400f5f4c2b51bd62              # double 3.921532
	.quad	0x3ffd7911cf355cd9              # double 1.842058
	.quad	0x402375bf6a0dbad4              # double 9.7299760000000006
	.quad	0x3ffe8cb4aec8d5c7              # double 1.9093519999999999
	.quad	0x400b2d81adea8976              # double 3.3972199999999999
	.quad	0x3ff4266234a87e39              # double 1.259371
	.quad	0x3fda13ad5bee3d60              # double 0.40745100000000001
	.quad	0x400f3a16e7a311e8              # double 3.9033639999999998
	.quad	0x3fb056a37ac3eb7d              # double 0.063822000000000004
	.quad	0x3ff9adc2b0ea1837              # double 1.604922
	.quad	0x4023dd3f39d1b2e6              # double 9.9321230000000007
	.quad	0x4014c42f61ed5ae2              # double 5.1915870000000002
	.quad	0x40214b762d83c6c9              # double 8.6473859999999991
	.quad	0x4021a6e286727568              # double 8.8259469999999993
	.quad	0x400e5dc26dce39b4              # double 3.7957809999999998
	.quad	0x4011d35eb7457c0b              # double 4.4564159999999999
	.quad	0x3fe70088509bf9c6              # double 0.71881499999999998
	.quad	0x4011eef49cf56ead              # double 4.4833550000000004
	.quad	0x3fec340b34e7685a              # double 0.88135300000000005
	.quad	0x400bd8fe69270b07              # double 3.4809540000000001
	.quad	0x402049ca18bd6627              # double 8.1441199999999991
	.quad	0x401b1d0a24463066              # double 6.778359
	.quad	0x401676634117f844              # double 5.6156129999999997
	.quad	0x3fd19ee02a77a2cf              # double 0.27532200000000001
	.quad	0x4020758d1526d8b2              # double 8.2295920000000002
	.quad	0x3ffc36b4c7f34938              # double 1.7633559999999999
	.quad	0x401767fc32ebe597              # double 5.8515480000000002
	.quad	0x4004791cd1c7de51              # double 2.5591370000000002
	.quad	0x3ff6734a44c7b02d              # double 1.4031469999999999
	.quad	0x401c91d96e9bbf0e              # double 7.1424310000000002
	.quad	0x401510e8427418d7              # double 5.2665110000000004
	.quad	0x400fa68986fcdee3              # double 3.9563169999999999
	.quad	0x3ff6c8fb86f47b67              # double 1.4240679999999999
	.quad	0x401f663886594af5              # double 7.849825
	.quad	0x4011dcde19fc2a88              # double 4.4656909999999996
	.quad	0x40006105186db50f              # double 2.0473729999999999
	.quad	0x401f2bb16c1e364c              # double 7.7926690000000001
	.quad	0x4014dc764adff823              # double 5.2152950000000002
	.quad	0x4021579f666234a8              # double 8.6711379999999991
	.quad	0x4003a1cac083126f              # double 2.4540000000000002
	.quad	0x400af3647baa9b4a              # double 3.3688440000000002
	.quad	0x40115634dad31fcd              # double 4.3341859999999999
	.quad	0x4014d85ad538ac19              # double 5.211284
	.quad	0x402015daf07bfe7e              # double 8.0426859999999998
	.quad	0x401d98b8f14db595              # double 7.3991429999999996
	.quad	0x3fe45f090f733a8a              # double 0.63660099999999997
	.quad	0x3ff90887229e9079              # double 1.5645819999999999
	.quad	0x3fe9163baba7b917              # double 0.78396399999999999
	.quad	0x401231f57b41bfbe              # double 4.5487880000000001
	.quad	0x400a27bc3c5bd0e1              # double 3.2694019999999999
	.quad	0x401c8677f6b1a2a5              # double 7.1313170000000001
	.quad	0x3ff999f1ae2da555              # double 1.6000840000000001
	.quad	0x401d3dcd7060bb2c              # double 7.3103540000000002
	.quad	0x4015fbf9830e3cda              # double 5.4960690000000003
	.quad	0x3fe3070fbeb9e493              # double 0.59461200000000003
	.quad	0x4021926e978d4fdf              # double 8.7859999999999996
	.quad	0x3fe11e731d2e0e30              # double 0.53496699999999997
	.quad	0x401db76bc1eff9f8              # double 7.4291219999999995
	.quad	0x4012df01fb82c2bd              # double 4.7177809999999996
	.quad	0x4015057689ca18bd              # double 5.2553349999999996
	.quad	0x3ffc074fb549f948              # double 1.7517849999999999
	.quad	0x4006cc0e7bc3c5bd              # double 2.849637
	.quad	0x3fe80c1b97353b4b              # double 0.75147799999999998
	.quad	0x400f2396073de1e3              # double 3.8923760000000001
	.quad	0x401e2fa2ad3e920c              # double 7.546519
	.quad	0x401003df2a9d627c              # double 4.003781
	.quad	0x4000823f24d8fd5d              # double 2.0635970000000001
	.quad	0x3fdbcc1ca3a4b557              # double 0.43433300000000002
	.quad	0x401fd255f3519bd4              # double 7.955406
	.quad	0x3fe510fb65668c26              # double 0.65832299999999999
	.quad	0x4011b37542a23c00              # double 4.4252520000000004
	.quad	0x40129a775fb2edfe              # double 4.6508459999999996
	.quad	0x401e6393250b5244              # double 7.5972410000000004
	.quad	0x4021267ad8e43244              # double 8.5751559999999997
	.quad	0x40193e4b44a1f080              # double 6.3108339999999998
	.quad	0x401eeb068123810f              # double 7.7295170000000004
	.quad	0x4002c72a7bd48cb5              # double 2.3472490000000001
	.quad	0x400efeb5b2d4d402              # double 3.8743699999999999
	.quad	0x400789a5ebb7739f              # double 2.9422109999999999
	.quad	0x401b0f0307f23cc9              # double 6.7646600000000001
	.quad	0x401efcddd6e04c06              # double 7.7469400000000004
	.quad	0x3fc5d4e8fb00bcbe              # double 0.17055999999999999
	.quad	0x3f66fa82e87d2c7c              # double 0.0028050000000000002
	.quad	0x40107620ab713272              # double 4.1153589999999998
	.quad	0x4022b7633ce63a5c              # double 9.3581789999999998
	.quad	0x3ffcda30984e3fff              # double 1.803269
	.quad	0x3ff63c0d2c386d2f              # double 1.389661
	.quad	0x4012bdae81882adc              # double 4.6852359999999997
	.quad	0x401c65c0b9991362              # double 7.0993680000000001
	.quad	0x401936bf8769ec2d              # double 6.3034650000000001
	.quad	0x3ff69b22d0e56042              # double 1.4128750000000001
	.quad	0x4022fed2cbea4ebe              # double 9.4977020000000003
	.quad	0x4022297204295a6c              # double 9.0809479999999993
	.quad	0x400bbaa79bbadc0a              # double 3.4661400000000002
	.quad	0x400f7ac1d29dc726              # double 3.9349400000000001
	.quad	0x3ffe6a454de7ea60              # double 1.9009450000000001
	.quad	0x401817414a4d2b2c              # double 6.02271
	.quad	0x401c7696a26e5471              # double 7.1158089999999996
	.quad	0x3fff6d70e6f2e8c0              # double 1.9642189999999999
	.quad	0x4010f2a023209678              # double 4.2369389999999996
	.quad	0x4018105186db50f4              # double 6.015936
	.quad	0x3fe2501c0ca600b0              # double 0.57227899999999998
	.quad	0x3fe8a6f3f52fc265              # double 0.77037999999999995
	.quad	0x401b382730c67169              # double 6.804837
	.quad	0x3ffd2e81882adc4d              # double 1.8238540000000001
	.quad	0x40143a4ce8101f32              # double 5.056934
	.quad	0x3ff579fec99f1ae3              # double 1.342284
	.quad	0x40224711d798d8a9              # double 9.1388079999999992
	.quad	0x40122e09fe86833c              # double 4.5449599999999997
	.quad	0x401b55fa658c4bd3              # double 6.8339629999999998
	.quad	0x4006bccb7d41743f              # double 2.8421850000000002
	.quad	0x401eb70e6f2e8c05              # double 7.6787660000000004
	.quad	0x3fc621426fe718a8              # double 0.17288999999999999
	.quad	0x4013de6106ab14ec              # double 4.9671669999999999
	.quad	0x3ffa43e74afd5454              # double 1.641578
	.quad	0x4011f76188b1140a              # double 4.4915830000000003
	.quad	0x4023b2372606fac6              # double 9.848077
	.quad	0x4023f056605ee569              # double 9.9694090000000006
	.quad	0x4020a2f4cf4a558f              # double 8.3182740000000006
	.quad	0x4005c7d566cf41f2              # double 2.722575
	.quad	0x400c831ad2135dab              # double 3.5640160000000001
	.quad	0x4019aa0b1bbcf4e8              # double 6.4160579999999996
	.quad	0x40021f30e7ff583a              # double 2.2652299999999999
	.quad	0x401fd6073de1e2df              # double 7.9590120000000004
	.quad	0x3fe6b96af038e2a0              # double 0.71013400000000004
	.quad	0x4006028b2a6b0d95              # double 2.751242
	.quad	0x401b08c692f6e829              # double 6.7585699999999997
	.quad	0x3fa872f33ca31e7e              # double 0.047752000000000003
	.quad	0x3fff6d9ad85dfa87              # double 1.964259
	.quad	0x400018c7e28240b8              # double 2.0121000000000002
	.quad	0x4021979aae6c8f75              # double 8.7961019999999994
	.quad	0x40227d0a88effe2a              # double 9.2442209999999996
	.quad	0x4016233daf8df7a5              # double 5.5344150000000001
	.quad	0x40213ec093d96638              # double 8.6225629999999995
	.quad	0x4005834b0e1b4bb6              # double 2.6891080000000001
	.quad	0x400f4ca70d1fa333              # double 3.9124279999999998
	.quad	0x40227890d5a5b963              # double 9.2354800000000008
	.quad	0x402373906466b1e6              # double 9.7257110000000004
	.quad	0x4019cc764adff823              # double 6.4496700000000002
	.quad	0x40000e297396d091              # double 2.0069149999999998
	.quad	0x40089b0142f61ed6              # double 3.0756860000000001
	.quad	0x401f860b6f9fcb0c              # double 7.880903
	.quad	0x401627fae3608d09              # double 5.5390430000000004
	.quad	0x400ca5d3996fa82f              # double 3.5809700000000002
	.quad	0x400f63486f049a99              # double 3.9234779999999998
	.quad	0x4022751aa821f299              # double 9.2287189999999999
	.quad	0x402156a0fd3b29a2              # double 8.6691970000000005
	.quad	0x400d8a0e410b630b              # double 3.6924100000000002
	.quad	0x4011befaa4766c6e              # double 4.4365030000000001
	.quad	0x4013e57bc7f77af6              # double 4.9741049999999998
	.quad	0x4011d85fd04a2fcf              # double 4.461303
	.quad	0x4023df342edbb59e              # double 9.9359450000000002
	.quad	0x40165740c4156e26              # double 5.5852079999999997
	.quad	0x402381b52007dd44              # double 9.7533349999999998
	.quad	0x401c35ce18266773              # double 7.0525440000000001
	.quad	0x40187b2c40d0aaa8              # double 6.1202860000000001
	.quad	0x4004c71bcdbbe015              # double 2.5972209999999998
	.quad	0x3fec7a97e132b55f              # double 0.88996500000000001
	.quad	0x3ff6921ff2e48e8a              # double 1.4106749999999999
	.quad	0x4023728b6d86ec18              # double 9.7237200000000001
	.quad	0x401caee3d5fdcdf7              # double 7.1707910000000004
	.quad	0x4001b262888b545c              # double 2.2121019999999998
	.quad	0x40006d57bc7f77af              # double 2.0533899999999998
	.quad	0x40171ec4a4095f24              # double 5.7800469999999997
	.quad	0x40142232096787cf              # double 5.0333940000000004
	.quad	0x40191dcbddb9841b              # double 6.2790980000000003
	.quad	0x4006ec996b7670a0              # double 2.8655270000000002
	.quad	0x3ff5e7f3cf70153c              # double 1.369129
	.quad	0x40198c19c17225b7              # double 6.3868169999999997
	.quad	0x400e3e60c38f3669              # double 3.7804579999999999
	.quad	0x400e0e0653005815              # double 3.7568480000000002
	.quad	0x4005e96bb98c7e28              # double 2.7389749999999999
	.quad	0x401bb2a66dbd72bd              # double 6.9244630000000003
	.quad	0x400e97aa25d8d79d              # double 3.824055
	.quad	0x400ec9adc8fb86f4              # double 3.8484759999999998
	.quad	0x3fea3bd1676640a7              # double 0.81980200000000003
	.quad	0x40203010c6f7a0b6              # double 8.0938780000000001
	.quad	0x4017bafdda8bd231              # double 5.9326090000000002
	.quad	0x401a270e2c12ad82              # double 6.5381400000000003
	.quad	0x4022816c82e019b1              # double 9.2527810000000006
	.quad	0x400dc3bc4d22c882              # double 3.720574
	.quad	0x4020ffd5ed06fef8              # double 8.4996790000000004
	.quad	0x40225216616b54e3              # double 9.1603270000000006
	.quad	0x401f42be8bc169c2              # double 7.8151799999999998
	.quad	0x3ffabd29563a9f38              # double 1.6711819999999999
	.quad	0x4019af31f46ed246              # double 6.4210890000000003
	.quad	0x400cef3a14cec41e              # double 3.6168100000000001
	.quad	0x401db4484944ed70              # double 7.4260570000000001
	.quad	0x4023313ad5bee3d6              # double 9.596152
	.quad	0x4023c5916872b021              # double 9.8858750000000004
	.quad	0x40022ffda4052d66              # double 2.2734329999999998
	.quad	0x402059f623076c05              # double 8.1757059999999999
	.quad	0x3fa3d512ec6bce85              # double 0.038734999999999999
	.quad	0x3ffc70d0edc3bd5a              # double 1.7775430000000001
	.quad	0x3ffdb0cfe154434e              # double 1.855667
	.quad	0x3ff34d035371971c              # double 1.206302
	.quad	0x401ae2674080f990              # double 6.7210970000000003
	.quad	0x400a63dee78183f9              # double 3.2987649999999999
	.quad	0x3ff1202645e4e69f              # double 1.070349
	.quad	0x400d2a3f034b0e1b              # double 3.6456279999999999
	.quad	0x4009731db445ed4a              # double 3.181209
	.quad	0x4021da7c9de05037              # double 8.9267319999999994
	.quad	0x40221e1cffeb074a              # double 9.0588149999999992
	.quad	0x3feaa686e7e62dc7              # double 0.83282800000000001
	.quad	0x4010ed434a01abd2              # double 4.2317020000000003
	.quad	0x3fed2cb4289118c2              # double 0.91170700000000005
	.quad	0x4001ccec41dd1a22              # double 2.22506
	.quad	0x401ebc0f01fb82c3              # double 7.6836510000000002
	.quad	0x3f972c5197a24895              # double 0.022630000000000001
	.quad	0x4001e96a26e54717              # double 2.238972
	.quad	0x401d0ce63a5c1c61              # double 7.2625970000000004
	.quad	0x40217f609dcf8940              # double 8.7487840000000006
	.quad	0x40036cf0739b024f              # double 2.4281929999999998
	.quad	0x400dbbe553ac4f7f              # double 3.7167460000000001
	.quad	0x400b4a2120e1f7d7              # double 3.4111959999999999
	.quad	0x4011e586013660e5              # double 4.4741439999999999
	.quad	0x4006f27ed84d338f              # double 2.8684059999999998
	.quad	0x401130ed80a17b0f              # double 4.2977809999999996
	.quad	0x4004f63f572de43f              # double 2.6202380000000001
	.quad	0x400b6958537e2c56              # double 3.4264380000000001
	.quad	0x401d2c11e42e1262              # double 7.293037
	.quad	0x40068c8259e1f3a5              # double 2.8186079999999998
	.quad	0x40239a50d6b228dd              # double 9.8013980000000007
	.quad	0x40168af31b152f3c              # double 5.6356929999999998
	.quad	0x401ba766c6de7642              # double 6.9134779999999996
	.quad	0x4016098c3b0c4589              # double 5.5093240000000003
	.quad	0x4022e6bce8533b10              # double 9.4506599999999992
	.quad	0x4010f9806f262889              # double 4.2436540000000003
	.quad	0x401746b484d76ab6              # double 5.8190480000000004
	.quad	0x3fe32276b7ed41b7              # double 0.59795699999999996
	.quad	0x4018097edc7ef178              # double 6.0092730000000003
	.quad	0x4010d65300581494              # double 4.209301
	.quad	0x400a00bdcad14a0a              # double 3.250362
	.quad	0x3fff83cf2cf95d4f              # double 1.9696800000000001
	.quad	0x4022604ee7c49fd8              # double 9.1881020000000006
	.quad	0x401ed3810e8858ff              # double 7.7065469999999995
	.quad	0x40206fbe76c8b439              # double 8.2182499999999994
	.quad	0x400c038da3c21188              # double 3.501735
	.quad	0x3ff1f6f3b213e3e3              # double 1.1227910000000001
	.quad	0x4012d266772d5e07              # double 4.7054689999999999
	.quad	0x40049ecaab8a5ce6              # double 2.5775350000000001
	.quad	0x400cdf56a7ac81d4              # double 3.6090520000000001
	.quad	0x40159300de4c5111              # double 5.3935579999999996
	.quad	0x401526a400fba882              # double 5.2877349999999996
	.quad	0x401a7985f06f6944              # double 6.6186749999999996
	.quad	0x3fd761cbccf28c7a              # double 0.365344
	.quad	0x40084d330941c821              # double 3.0376949999999998
	.quad	0x40183fdda8bd230c              # double 6.0623690000000003
	.quad	0x402143530ced4e4d              # double 8.6314930000000007
	.quad	0x4009b74aba387592              # double 3.2144979999999999
	.quad	0x400e09478c868ba0              # double 3.7545310000000001
	.quad	0x401020329802c0a5              # double 4.0314430000000003
	.quad	0x400c82e6ea854478              # double 3.563917
	.quad	0x401164acaff6d331              # double 4.3483150000000004
	.quad	0x40149969a0ad8a11              # double 5.1498169999999996
	.quad	0x400c86b2f23033a4              # double 3.5657709999999998
	.quad	0x3fe3b43fa2ad3e92              # double 0.61575299999999999
	.quad	0x401ed0c1b97353b5              # double 7.7038640000000002
	.quad	0x4002485b18548a9c              # double 2.2853300000000001
	.quad	0x3ffde57a786c2268              # double 1.868525
	.quad	0x400789ccb7d41744              # double 2.942285
	.quad	0x4022714ccab3edd9              # double 9.2212890000000005
	.quad	0x400a608bfc2224ee              # double 3.297142
	.quad	0x3ff2b2a497fa5d29              # double 1.168614
	.quad	0x401c466d7a56de33              # double 7.0687769999999999
	.quad	0x401059c1f85d744f              # double 4.0876539999999997
	.quad	0x401ce6b4c7f34938              # double 7.2252989999999997
	.quad	0x4020f7c6da448494              # double 8.4839389999999994
	.quad	0x401f259c065b63d4              # double 7.7867280000000001
	.quad	0x3fe0715e7c8d1959              # double 0.51383900000000005
	.quad	0x3fe029b280f12c28              # double 0.50509000000000004
	.quad	0x40126332f01754b0              # double 4.5968739999999997
	.quad	0x4011f0d51f81a587              # double 4.485188
	.quad	0x401bb26b2f23033a              # double 6.9242369999999998
	.quad	0x3ff071f57b41bfbe              # double 1.027822
	.quad	0x401c40d66f0cfe15              # double 7.0633179999999998
	.quad	0x3f994ee392e1ef74              # double 0.024715000000000001
	.quad	0x40215b4a44c7b02d              # double 8.6783009999999994
	.quad	0x4012d01cd5f99c39              # double 4.7032350000000003
	.quad	0x401ad73bc903ea70              # double 6.7101889999999997
	.quad	0x3ff16c2acb85a4f0              # double 1.088908
	.quad	0x3fe79d798d8a979e              # double 0.73797299999999999
	.quad	0x4019e8abd5dc4007              # double 6.4772179999999997
	.quad	0x4017a364388ebcc7              # double 5.9095620000000002
	.quad	0x401309b66f9335d2              # double 4.7594849999999997
	.quad	0x400401db0142f61f              # double 2.5009060000000001
	.quad	0x4020449ff92f2b67              # double 8.1340330000000005
	.quad	0x401ebc892ab68cef              # double 7.6841169999999996
	.quad	0x3fe693f6c2699c7c              # double 0.70556200000000002
	.quad	0x4002e26cc1ca3a4b              # double 2.3605589999999999
	.quad	0x400e1142b302f72b              # double 3.7584279999999999
	.quad	0x3fbfecedd4845133              # double 0.124709
	.quad	0x4007fb36bd2b6f1a              # double 2.9976630000000002
	.quad	0x401f45b20b8066c3              # double 7.8180620000000002
	.quad	0x4022c8ecd8dbcec8              # double 9.3924319999999994
	.quad	0x401752b27cbf6e3f              # double 5.8307589999999996
	.quad	0x3ffcb2c83ec892ab              # double 1.7936479999999999
	.quad	0x40126e846a5d6bec              # double 4.6079270000000001
	.quad	0x3ff3283f4eca686e              # double 1.1973259999999999
	.quad	0x3fddc857f3061c7a              # double 0.46535300000000002
	.quad	0x4009dc3ac929aa1d              # double 3.2325339999999998
	.quad	0x4019cf33ca31e7da              # double 6.4523460000000004
	.quad	0x3ff8dd41fa765343              # double 1.5540179999999999
	.quad	0x3ff7ec8605681ecd              # double 1.4952449999999999
	.quad	0x401adaaa7ded6ba9              # double 6.7135410000000002
	.quad	0x40071c60029f16b1              # double 2.888855
	.quad	0x400f3bb8c32a8c9c              # double 3.9041610000000002
	.quad	0x40212b31e3a7daa5              # double 8.584365
	.quad	0x400bbe7c8d195897              # double 3.4680110000000002
	.quad	0x40219deaee2003ac              # double 8.8084330000000008
	.quad	0x40218273929ed395              # double 8.7547879999999996
	.quad	0x40205e748e475600              # double 8.1844830000000002
	.quad	0x3fef9578a2a90cd4              # double 0.98699599999999998
	.quad	0x3ff0504b3c3e74b0              # double 1.019603
	.quad	0x4012ff190d173fb8              # double 4.7491190000000003
	.quad	0x401af1116a8b8f15              # double 6.7354180000000001
	.quad	0x3ff36dfe75bc44bf              # double 1.2143539999999999
	.quad	0x400c65fbf83382e4              # double 3.5497969999999999
	.quad	0x400d71a8ef77f280              # double 3.680498
	.quad	0x400be2595bbbe879              # double 3.485522
	.quad	0x4014a136e71cda2b              # double 5.1574359999999997
	.quad	0x401b3cd035371972              # double 6.8093880000000002
	.quad	0x40081e1b8ed1bf7b              # double 3.0147010000000001
	.quad	0x401638f1d3ed527e              # double 5.5556099999999997
	.quad	0x3fa20d9945b6c376              # double 0.03526
	.quad	0x40205df1172ef0ae              # double 8.1834799999999994
	.quad	0x3fc355714b9cb685              # double 0.15104500000000001
	.quad	0x401e904c05921038              # double 7.6409149999999996
	.quad	0x4020dad256798959              # double 8.4273860000000003
	.quad	0x40184c964628027e              # double 6.0747920000000004
	.quad	0x3fd5caab8a5ce5b4              # double 0.34049499999999999
	.quad	0x4020edaff2a172ac              # double 8.4642330000000001
	.quad	0x3fff07d0aed99cbf              # double 1.939408
	.quad	0x3f91d084e831ad21              # double 0.017396999999999999
	.quad	0x401415aea3161a1e              # double 5.0211740000000002
	.quad	0x40163c8d9f9053a0              # double 5.5591340000000002
	.quad	0x3ff2d0247021d10b              # double 1.175816
	.quad	0x4010bd6d7e889c1b              # double 4.1849879999999997
	.quad	0x400d4b2a6b0d9514              # double 3.661702
	.quad	0x400167685985ad54              # double 2.1754920000000002
	.quad	0x3ff50cb5bb384fd3              # double 1.3156030000000001
	.quad	0x3fee72495e17e34c              # double 0.95145100000000004
	.quad	0x4005928d8665e02f              # double 2.6965590000000002
	.quad	0x401ef6b00ffda405              # double 7.7409059999999998
	.quad	0x400fad46f587d6f9              # double 3.9596079999999998
	.quad	0x40168fb3743632c2              # double 5.640333
	.quad	0x4014271bcdbbe015              # double 5.0381919999999996
	.quad	0x40036b4528283d36              # double 2.427378
	.quad	0x401e235b91f70de9              # double 7.534529
	.quad	0x401686f60e0eb67c              # double 5.6317979999999999
	.quad	0x4000002ea960b6fa              # double 2.000089
	.quad	0x4020a3ea4ebdd335              # double 8.3201470000000004
	.quad	0x401e1ef6f8f04146              # double 7.5302389999999999
	.quad	0x401bc9dcf893faf4              # double 6.9471319999999999
	.quad	0x400a67b02d59d55e              # double 3.3006289999999998
	.quad	0x400f1bb2fec56d5d              # double 3.888525
	.quad	0x400aaac8605681ed              # double 3.3333900000000001
	.quad	0x401cc32c1f42bb66              # double 7.1905979999999996
	.quad	0x3ffd05b6c3760bf6              # double 1.813895
	.quad	0x400526202539756d              # double 2.6436160000000002
	.quad	0x4001288ace24bba1              # double 2.1447959999999999
	.quad	0x4020939c51dabe27              # double 8.2883019999999998
	.quad	0x40173c6d1e108c3f              # double 5.8090099999999998
	.quad	0x3fdd125dd095af29              # double 0.45424599999999998
	.quad	0x401ae0817fc7607c              # double 6.7192439999999998
	.quad	0x401474af4f0d844d              # double 5.11395
	.quad	0x401fb673c4f3ba77              # double 7.9281759999999996
	.quad	0x401866ce789e774f              # double 6.1003970000000001
	.quad	0x3fc22af5771001d6              # double 0.14193600000000001
	.quad	0x3ff976e2eb1c432d              # double 1.5915250000000001
	.quad	0x401a5c4da9003eea              # double 6.5901399999999999
	.quad	0x40232b4549b62c77              # double 9.5845129999999993
	.quad	0x3ff3799d451fc4c1              # double 1.2171909999999999
	.quad	0x40218d0d49949e88              # double 8.7754919999999998
	.quad	0x4008d84988094e5d              # double 3.1056089999999998
	.quad	0x3fedd938151a4378              # double 0.93276599999999998
	.quad	0x3fcd0f301eabbcb2              # double 0.22702600000000001
	.quad	0x401b88826aa8eb46              # double 6.8833099999999998
	.quad	0x3ff6826dce39b457              # double 1.4068430000000001
	.quad	0x3ffc74623d0bfa09              # double 1.7784139999999999
	.quad	0x3ff2fbd7b2031ceb              # double 1.186485
	.quad	0x402392116659d12d              # double 9.7852890000000005
	.quad	0x3fe4a5d8d79d0a67              # double 0.64524499999999996
	.quad	0x4023fb68cef672b9              # double 9.9910340000000008
	.quad	0x4021cf5a964e8b7e              # double 8.9049879999999995
	.quad	0x400153ca31e7d999              # double 2.1659130000000002
	.quad	0x40074afee6fb4c3c              # double 2.911619
	.quad	0x4014f7e62dc6e2a8              # double 5.242089
	.quad	0x4009a829cfdd2285              # double 3.2071109999999998
	.quad	0x40216077ee4e26d5              # double 8.6884150000000009
	.quad	0x40199eaab042528b              # double 6.4049480000000001
	.quad	0x401dfaabcd78c437              # double 7.494796
	.quad	0x40227d5f56a7ac82              # double 9.2448680000000003
	.quad	0x4021c16f4384ba0f              # double 8.8778020000000009
	.quad	0x4023722f05a708ee              # double 9.7230150000000002
	.quad	0x400ef0afe60c38f3              # double 3.8675229999999998
	.quad	0x3fc4806290eed02d              # double 0.160168
	.quad	0x401cd142f61ed5ae              # double 7.2043569999999999
	.quad	0x40202fde7210be94              # double 8.0934939999999997
	.quad	0x4014c9b068123811              # double 5.1969620000000001
	.quad	0x4018e3fc86cebbba              # double 6.2226429999999997
	.quad	0x400829d8c6d612c7              # double 3.0204330000000001
	.quad	0x40185afaf8594291              # double 6.0888479999999996
	.quad	0x4005d11622813448              # double 2.727093
	.quad	0x400c9a57eaa2a0a9              # double 3.5753629999999998
	.quad	0x3fe2d7d9dba908a2              # double 0.58884899999999996
	.quad	0x40150fa54c554328              # double 5.2652789999999996
	.quad	0x401eefe32a0663c7              # double 7.7342649999999997
	.quad	0x401389635610adcd              # double 4.8841679999999998
	.quad	0x3fe459d12cadddf4              # double 0.63596399999999997
	.quad	0x4000c5e1c15097c8              # double 2.096622
	.quad	0x401d5fd113836a83              # double 7.3435709999999998
	.quad	0x401c25b0ff10ecb7              # double 7.0368079999999997
	.quad	0x401da908a265f0f6              # double 7.4150720000000003
	.quad	0x401260cfe154434e              # double 4.5945429999999998
	.quad	0x4021ced4801f7510              # double 8.9039649999999995
	.quad	0x401715dd8f92af9b              # double 5.7713530000000004
	.quad	0x40100a6aca793576              # double 4.010173
	.quad	0x4022dbdc69f8c21e              # double 9.4294159999999998
	.quad	0x4016a24cc6822ff1              # double 5.6584960000000004
	.quad	0x4022e22d948dc11e              # double 9.4417539999999995
	.quad	0x401fa46e8f29d40f              # double 7.9105780000000001
	.quad	0x3ff33cbd556084a5              # double 1.202329
	.quad	0x401a6b6d43d03969              # double 6.6049090000000001
	.quad	0x40231d5eaee2003b              # double 9.5573630000000004
	.quad	0x3fe9f8d0cc35ce18              # double 0.81162299999999998
	.quad	0x401eb86cebbba55d              # double 7.6801029999999999
	.quad	0x4020424b76f6d762              # double 8.1294819999999994
	.quad	0x4013829cfdd22856              # double 4.8775519999999997
	.quad	0x40235b08feac42ea              # double 9.6778030000000008
	.quad	0x4013538eb0318b93              # double 4.8315989999999998
	.quad	0x3fed2c01e68a0d35              # double 0.91162200000000004
	.quad	0x40158aab042528ae              # double 5.3854179999999996
	.quad	0x402309db0142f61f              # double 9.5192490000000002
	.quad	0x4009b91f70de8f6d              # double 3.215392
	.quad	0x401e1d1b71758e22              # double 7.5284250000000004
	.quad	0x400274a0a0f4d7ae              # double 2.3069470000000001
	.quad	0x3fd4edaec4a4095f              # double 0.32700699999999999
	.quad	0x3fe9876e1deacc92              # double 0.79778199999999999
	.quad	0x40202d106ee30caa              # double 8.0880159999999997
	.quad	0x400fdf138bcdfefc              # double 3.983924
	.quad	0x400d30d9d7774aba              # double 3.6488529999999999
	.quad	0x400ea4a4095f2453              # double 3.8303910000000001
	.quad	0x4020451f3e89a88b              # double 8.1350040000000003
	.quad	0x3feb8b524399b2c4              # double 0.86075699999999999
	.quad	0x400df0abb44e50c6              # double 3.742515
	.quad	0x40208c7f562174c5              # double 8.2744090000000003
	.quad	0x3fe1384ba0e84274              # double 0.53812199999999999
	.quad	0x40140ee78183f91e              # double 5.0145549999999997
	.quad	0x3ff1cb34e7685986              # double 1.1121110000000001
	.quad	0x4021c009d495182b              # double 8.8750750000000007
	.quad	0x40210cbb7f9d6f11              # double 8.5248679999999997
	.quad	0x3ffb437e6f71a7e3              # double 1.703978
	.quad	0x4021d113e81450f0              # double 8.9083550000000002
	.quad	0x3f6f7f8ca8198f1d              # double 0.0038449999999999999
	.quad	0x4023011105e1c151              # double 9.5020830000000007
	.quad	0x400fa70c996b7671              # double 3.9565670000000002
	.quad	0x401367cb2d905c03              # double 4.8513609999999998
	.quad	0x402111728a611772              # double 8.5340769999999999
	.quad	0x3ff94e3369b9d7fe              # double 1.5815920000000001
	.quad	0x401f2efed634549b              # double 7.7958939999999996
	.quad	0x4002de07a28bb0a3              # double 2.3584130000000001
	.quad	0x3fd9547e06961c37              # double 0.39578200000000002
	.quad	0x4008f9791819d239              # double 3.1218129999999999
	.quad	0x3ff158cb07d0aeda              # double 1.0841780000000001
	.quad	0x4019154003254e6e              # double 6.2707519999999999
	.quad	0x400f584a0e410b63              # double 3.91811
	.quad	0x4002ba0cf1800a7c              # double 2.3408449999999998
	.quad	0x4005f6e58a32f449              # double 2.745555
	.quad	0x4023e1ff0cbaf966              # double 9.9413990000000005
	.quad	0x401e5ecd4aa10e02              # double 7.5925799999999999
	.quad	0x3ffc142e557de0d6              # double 1.7549269999999999
	.quad	0x4019bd2a1f8e3ac1              # double 6.4347310000000002
	.quad	0x4022cf7121ab4b73              # double 9.4051600000000004
	.quad	0x40175cc78e9f6a94              # double 5.840605
	.quad	0x400f78b9baa1511e              # double 3.933948
	.quad	0x4010afd3b29a1ba0              # double 4.1717060000000004
	.quad	0x401be56798958d9b              # double 6.9740279999999997
	.quad	0x400a068fd199bb28              # double 3.2532040000000002
	.quad	0x402279e01de26916              # double 9.2380379999999995
	.quad	0x4019d893faf42785              # double 6.4615020000000003
	.quad	0x4017e97c3d68405b              # double 5.9780129999999998
	.quad	0x401d379f23465626              # double 7.3043180000000003
	.quad	0x3fed730a4e379b78              # double 0.92029300000000003
	.quad	0x4023681cd5f99c39              # double 9.7033450000000005
	.quad	0x3ff769d8409e55c1              # double 1.463341
	.quad	0x400ee10f94c87981              # double 3.859893
	.quad	0x3ff86a9b499d0204              # double 1.526027
	.quad	0x40226844284dfce3              # double 9.2036449999999999
	.quad	0x401059b17481b21c              # double 4.0875909999999998
	.quad	0x40212055b8993922              # double 8.5631540000000008
	.quad	0x400f8eb3dd11be6e              # double 3.9446789999999998
	.quad	0x401d04330d73860a              # double 7.2541010000000004
	.quad	0x3fdd9b30728e92d5              # double 0.46259699999999998
	.quad	0x40226d454152b0a7              # double 9.213419
	.quad	0x4001e141e9af5ba3              # double 2.2349890000000001
	.quad	0x4020d4dd72367e41              # double 8.4157519999999995
	.quad	0x401465daad601ffb              # double 5.0994669999999998
	.quad	0x4008f0763a190082              # double 3.117413
	.quad	0x4022578d0cc35ce2              # double 9.1709980000000009
	.quad	0x4020c230b9dc2f40              # double 8.3792779999999993
	.quad	0x3ff982a454de7ea6              # double 1.594395
	.quad	0x40099215fcc1871e              # double 3.1963309999999998
	.quad	0x402053e0ded288ce              # double 8.1638249999999992
	.quad	0x401c57cf1c3265ae              # double 7.0857510000000001
	.quad	0x3ff1262456f75d9a              # double 1.071812
	.quad	0x3fdd822bbecaab8a              # double 0.46106999999999998
	.quad	0x3ffc807570c564f9              # double 1.7813619999999999
	.quad	0x3fd81d4738a3b57c              # double 0.37678699999999998
	.quad	0x401c9c32ebe596c8              # double 7.1525379999999998
	.quad	0x4002f1e3a7daa4fd              # double 2.3681100000000002
	.quad	0x40238271151611ba              # double 9.7547689999999996
	.quad	0x3ffd31c0010c6f7a              # double 1.824646
	.quad	0x40207f8594291750              # double 8.2490659999999991
	.quad	0x401bdaeab7995ee1              # double 6.9637859999999998
	.quad	0x401627bf61aa3f03              # double 5.5388159999999997
	.quad	0x401a3bc126a65cf6              # double 6.5583539999999996
	.quad	0x3fe6e46cfc829cfe              # double 0.71538400000000002
	.quad	0x4010e1359791819d              # double 4.2199309999999999
	.quad	0x3fe91b866e43aa7a              # double 0.78461000000000003
	.quad	0x402180a697aeddce              # double 8.7512709999999991
	.quad	0x401a6d0f1f57b41c              # double 6.606503
	.quad	0x40133a4f8726d04e              # double 4.8069439999999997
	.quad	0x402091b77c02afde              # double 8.2846030000000006
	.quad	0x3fe086c226809d49              # double 0.51644999999999996
	.quad	0x4023e82f0e0a84be              # double 9.9534839999999996
	.quad	0x3fd19a2403148777              # double 0.27503300000000003
	.quad	0x402252ab25b317f0              # double 9.1614620000000002
	.quad	0x4021102ababead4f              # double 8.5315759999999994
	.quad	0x401048eb463497b7              # double 4.0712099999999998
	.quad	0x401ceda9003eea21              # double 7.2320900000000004
	.quad	0x400fbad4b2745bf2              # double 3.9662259999999998
	.quad	0x4010b6f049a9973e              # double 4.1786510000000003
	.quad	0x4013bd3804534bd7              # double 4.9347839999999996
	.quad	0x4016b0902de00d1b              # double 5.6724249999999996
	.quad	0x4010e5625a682b63              # double 4.2240080000000004
	.quad	0x400d7bd1676640a7              # double 3.6854580000000001
	.quad	0x3ff952e83a109d06              # double 1.582741
	.quad	0x4013e1733226c3b9              # double 4.9701659999999999
	.quad	0x4020eccbc05d52c1              # double 8.4624919999999992
	.quad	0x4019ec7000431bdf              # double 6.4808960000000004
	.quad	0x402285e7210be942              # double 9.2615289999999995
	.quad	0x401a522749f0e4da              # double 6.580228
	.quad	0x4000b950b955f783              # double 2.0904859999999998
	.quad	0x401a652ce032db1f              # double 6.5988040000000003
	.quad	0x4013f75ee99a62ed              # double 4.9915729999999998
	.quad	0x400e86583e8576cd              # double 3.815598
	.quad	0x401b732b990afe61              # double 6.8624710000000002
	.quad	0x4012ab9fdbd2fa0d              # double 4.6676019999999996
	.quad	0x3fed1db445ed4a1b              # double 0.90987600000000002
	.quad	0x3ffc60168b5cbff4              # double 1.7734589999999999
	.quad	0x4007c102795703f3              # double 2.9692430000000001
	.quad	0x40188c5d63886595              # double 6.1370750000000003
	.quad	0x4018decc814d727a              # double 6.2175770000000004
	.quad	0x4021232617c1bda5              # double 8.5686499999999999
	.quad	0x4010c806f262888b              # double 4.1953389999999997
	.quad	0x4015726351deefe5              # double 5.361707
	.quad	0x3ff58cc6822ff089              # double 1.3468690000000001
	.quad	0x400489e1b089a027              # double 2.5673249999999999
	.quad	0x40051197a24894c4              # double 2.6335899999999999
	.quad	0x3ff28fbdad7518b1              # double 1.160093
	.quad	0x401b73b213e3e293              # double 6.862984
	.quad	0x401628b2a6b0d951              # double 5.5397439999999998
	.quad	0x3feb95c63ae2541e              # double 0.86203300000000005
	.quad	0x4021fb6848beb5b3              # double 8.9910300000000003
	.quad	0x40076e5f7403dddb              # double 2.928893
	.quad	0x3fef5cc63f141206              # double 0.98007500000000003
	.quad	0x3fe544b90214ad36              # double 0.66463899999999998
	.quad	0x3fe6000431bde82d              # double 0.68750199999999995
	.quad	0x400b784901d19158              # double 3.4337330000000001
	.quad	0x400e73cff64cf8d7              # double 3.806549
	.quad	0x4014a5629d8409e5              # double 5.1615089999999997
	.quad	0x4023fc5846e8f29d              # double 9.9928609999999995
	.quad	0x400c4fe718a86d72              # double 3.539015
	.quad	0x4020098eb8950764              # double 8.0186670000000006
	.quad	0x4023b9d3458cd20b              # double 9.86294
	.quad	0x3f9db7ae5796bfcb              # double 0.029021000000000002
	.quad	0x40166f2b239a38fc              # double 5.6085630000000002
	.quad	0x401582e44b6e935c              # double 5.3778240000000004
	.quad	0x4020749c6f36ef80              # double 8.2277559999999994
	.quad	0x4021481083dbc233              # double 8.6407509999999998
	.quad	0x40239c0485a0be51              # double 9.8047219999999999
	.quad	0x3ffed2e946801712              # double 1.9264920000000001
	.quad	0x401d3a647fdc5932              # double 7.3070240000000002
	.quad	0x401a6e0a41a261bf              # double 6.6074609999999998
	.quad	0x401ff8826aa8eb46              # double 7.9926849999999998
	.quad	0x4020586a8fc0d2c4              # double 8.1726880000000008
	.quad	0x3fcd6b3354c12275              # double 0.22983400000000001
	.quad	0x4012b20b3d4ae42a              # double 4.6738710000000001
	.quad	0x3fd4d199bb2788db              # double 0.325293
	.quad	0x3ff0159c49774257              # double 1.0052760000000001
	.quad	0x3fed60807357e671              # double 0.91803000000000001
	.quad	0x4014f65a9a804966              # double 5.2405799999999996
	.quad	0x3ffa5bd72bcb5fe5              # double 1.6474219999999999
	.quad	0x4020ce7f2c3009b3              # double 8.403314
	.quad	0x401195e0f7fcfc40              # double 4.3963660000000004
	.quad	0x400e1aa369fcf3dc              # double 3.763007
	.quad	0x401aee29b6b2af14              # double 6.7325809999999997
	.quad	0x4023e8a372606fac              # double 9.9543719999999993
	.quad	0x4015927ed84d338f              # double 5.3930619999999996
	.quad	0x400b3a7f3cf70154              # double 3.4035630000000001
	.quad	0x4014398cc144028e              # double 5.0562009999999997
	.quad	0x4023b308dd1e53a8              # double 9.8496769999999997
	.quad	0x3fc3041461b6d43d              # double 0.148562
	.quad	0x40226af2d7f950b9              # double 9.2088839999999994
	.quad	0x3fe411a543f1c758              # double 0.62715399999999999
	.quad	0x401e19817b95a294              # double 7.5249079999999999
	.quad	0x401633a4723aafff              # double 5.5504319999999998
	.quad	0x401af10e02214270              # double 6.7354050000000001
	.quad	0x401cc72324c83665              # double 7.1944699999999999
	.quad	0x40116a3c64345cff              # double 4.3537460000000001
	.quad	0x4016badb402d16b9              # double 5.6824769999999996
	.quad	0x4004aef2c7325919              # double 2.5854240000000002
	.quad	0x4021142a02320968              # double 8.5393830000000008
	.quad	0x402031dffc5479d5              # double 8.0974120000000003
	.quad	0x4020d24a05dd8f93              # double 8.4107210000000006
	.quad	0x40196a871a3b14a9              # double 6.354031
	.quad	0x402244a5375c8da0              # double 9.1340730000000008
	.quad	0x3fffb04039abf338              # double 1.9805299999999999
	.quad	0x40216ca3400b88ca              # double 8.7121829999999996
	.quad	0x4022318e757928e1              # double 9.0967900000000004
	.quad	0x4020d7746887a8d6              # double 8.4208099999999995
	.quad	0x40237743c7d5ed07              # double 9.732939
	.quad	0x4008b5fb2edfe75c              # double 3.0888580000000001
	.quad	0x3fe40c32a8c9b845              # double 0.62648899999999996
	.quad	0x401920b3d4ae429e              # double 6.281936
	.quad	0x402307b99d451fc5              # double 9.5150880000000004
	.quad	0x4013cefed634549b              # double 4.9521439999999997
	.quad	0x4004bcbccf28c79f              # double 2.5921569999999998
	.quad	0x402075bbe0157eed              # double 8.2299489999999995
	.quad	0x40223061847f5621              # double 9.0944939999999992
	.quad	0x4007412cf0f9d2bf              # double 2.9068239999999999
	.quad	0x401fd7460242d05f              # double 7.9602279999999998
	.quad	0x3fb3c105186db50f              # double 0.077163999999999996
	.quad	0x401babb40b34e768              # double 6.9176789999999997
	.quad	0x4016b56d19dece57              # double 5.6771739999999999
	.quad	0x4010206573215fcc              # double 4.0316369999999999
	.quad	0x4000b3743632c1f4              # double 2.0876239999999999
	.quad	0x3ff1ff69014b599b              # double 1.1248560000000001
	.quad	0x4005e3f705857aff              # double 2.7363110000000002
	.quad	0x4023d433b5393251              # double 9.9144570000000005
	.quad	0x4013717acc4ef88c              # double 4.8608200000000004
	.quad	0x400f49ed3953dea4              # double 3.9110969999999998
	.quad	0x4021d4b9a9da597d              # double 8.9154789999999995
	.quad	0x40117c322291fb40              # double 4.3712850000000003
	.quad	0x401648f96e158751              # double 5.5712640000000002
	.quad	0x4021de4302b40f67              # double 8.9341050000000006
	.quad	0x40211a37ac3eb7cc              # double 8.5512060000000005
	.quad	0x3fdac3bd5992428d              # double 0.41819699999999999
	.quad	0x4011091d14e3bcd3              # double 4.2588999999999997
	.quad	0x40120dddf43c7d5f              # double 4.5135420000000002
	.quad	0x3ff895b6805a2d73              # double 1.536551
	.quad	0x4015af23cc8de2ac              # double 5.4210349999999998
	.quad	0x4021d723cc8de2ac              # double 8.9201949999999996
	.quad	0x401469a49c2c1b11              # double 5.103167
	.quad	0x400392fcac8e68e4              # double 2.446771
	.quad	0x40218f37da61e0c6              # double 8.7797230000000006
	.quad	0x401cc6f049a9973e              # double 7.1942760000000003
	.quad	0x401cec209246bf01              # double 7.2305929999999998
	.quad	0x40222dc486ad2dcb              # double 9.0893899999999999
	.quad	0x40225a0f4d7add16              # double 9.1758980000000001
	.quad	0x3fdfe0f7fcfc3f81              # double 0.49810599999999999
	.quad	0x4023a310fd7e4580              # double 9.8184889999999996
	.quad	0x3ff0b8854cdb7ae5              # double 1.0450489999999999
	.quad	0x401bc94855da2728              # double 6.9465649999999997
	.quad	0x40207c467381d7dc              # double 8.242725
	.quad	0x401a29bb6aa4b988              # double 6.5407539999999997
	.quad	0x4012703686a4ca4f              # double 4.6095829999999998
	.quad	0x3ffe1286b59146e5              # double 1.8795230000000001
	.quad	0x400e43400b88ca3e              # double 3.7828369999999998
	.quad	0x401ce0b84988094e              # double 7.2194529999999997
	.quad	0x4001971eafee6fb5              # double 2.1987890000000001
	.quad	0x40203b0ad46f587d              # double 8.1153169999999992
	.quad	0x401b7a459103c8e2              # double 6.8694059999999997
	.quad	0x3fd7080b673c4f3c              # double 0.35986600000000002
	.quad	0x40133cb4aec8d5c7              # double 4.8092829999999998
	.quad	0x4020d14898f605ab              # double 8.4087569999999996
	.quad	0x401e3a9bcfd4bf0a              # double 7.5572350000000004
	.quad	0x4017f3fcc9ea9a3d              # double 5.9882689999999998
	.quad	0x4022ba30fcf80dc3              # double 9.3636549999999996
	.quad	0x402087505d0fa58f              # double 8.2642849999999992
	.quad	0x3fff97bfa4c61d86              # double 1.974548
	.quad	0x401fc1b2e59af9ec              # double 7.9391590000000001
	.quad	0x4015fe73a365cb36              # double 5.498488
	.quad	0x40202be74afd5454              # double 8.0857489999999998
	.quad	0x4021dfdc1615ebfb              # double 8.9372260000000008
	.quad	0x401e72e019b0ab2e              # double 7.6121829999999999
	.quad	0x401ed0685553ef6b              # double 7.7035229999999996
	.quad	0x401c3acabc515486              # double 7.0574139999999996
	.quad	0x401270880d801b43              # double 4.6098939999999997
	.quad	0x402014302b40f66a              # double 8.0394299999999994
	.quad	0x400f026dce39b457              # double 3.8761860000000001
	.quad	0x4007c9e6687f455a              # double 2.9735839999999998
	.quad	0x4014623ee1bd1eda              # double 5.0959430000000001
	.quad	0x401876e008e9b38d              # double 6.1160889999999997
	.quad	0x40227bedfa43fe5d              # double 9.2420500000000008
	.quad	0x4008adad31fcd24e              # double 3.084803
	.quad	0x40185a6c5d206c87              # double 6.0883039999999999
	.quad	0x402114dcca70d1fa              # double 8.5407469999999996
	.quad	0x4005568b5cbff477              # double 2.6672579999999999
	.quad	0x400e80ee8d10f51b              # double 3.8129550000000001
	.quad	0x40035cbee807bbb6              # double 2.4202859999999999
	.quad	0x4009c108c3f3e037              # double 3.219255
	.quad	0x40045525cc426352              # double 2.5415760000000001
	.quad	0x4011b37bd05af6c7              # double 4.4252770000000003
	.quad	0x4017f9da16616b55              # double 5.9939960000000001
	.quad	0x3fd64cf4a558ea7d              # double 0.34844700000000001
	.quad	0x40216f5b5fa22707              # double 8.7174940000000003
	.quad	0x4011d1d606317269              # double 4.4549180000000002
	.quad	0x401e271e6cd29132              # double 7.5382020000000001
	.quad	0x401c97cfa26a22b4              # double 7.1482530000000004
	.quad	0x40221713059641f6              # double 9.0450669999999995
	.quad	0x401e28c1db0142f6              # double 7.5398019999999999
	.quad	0x40224bdee78183f9              # double 9.1481849999999998
	.quad	0x4020c7383f0bae8a              # double 8.3891010000000001
	.quad	0x400b831055b89939              # double 3.4389959999999999
	.quad	0x40036d01c0ca600b              # double 2.428226
	.quad	0x40218be878fabda1              # double 8.7732580000000002
	.quad	0x4021fa93293d102c              # double 8.9894040000000003
	.quad	0x40136c81d3aa36a0              # double 4.8559640000000002
	.quad	0x402121233df2a9d6              # double 8.5647219999999997
	.quad	0x402245f202107b78              # double 9.1366119999999995
	.quad	0x3fd0bea4ebdd334c              # double 0.26163599999999998
	.quad	0x402237bdf090f734              # double 9.1088710000000006
	.quad	0x401de0f238972043              # double 7.4696740000000004
	.quad	0x400521f644955b46              # double 2.6415829999999998
	.quad	0x4021230f062d40ab              # double 8.5684740000000001
	.quad	0x4016d2d3415b1423              # double 5.7058840000000002
	.quad	0x4016f3461309c800              # double 5.7375720000000001
	.quad	0x4002011233df2a9d              # double 2.2505229999999998
	.quad	0x4017a96b76709fa5              # double 5.9154489999999997
	.quad	0x401e2ce5b4245f5b              # double 7.5438450000000001
	.quad	0x401493f1c75818c6              # double 5.1444770000000002
	.quad	0x40060672756861e9              # double 2.7531479999999999
	.quad	0x4021852a411c2a02              # double 8.7600879999999997
	.quad	0x3fee11e646f15619              # double 0.93968499999999999
	.quad	0x4022799bb2788db0              # double 9.2375159999999994
	.quad	0x3ffbc52502eec7c9              # double 1.7356309999999999
	.quad	0x4021e875b3e1437c              # double 8.9540229999999993
	.quad	0x4019d18548a9bcfd              # double 6.4546099999999997
	.quad	0x401209d12cadddf4              # double 4.5095869999999998
	.quad	0x4014655970b49e02              # double 5.0989740000000001
	.quad	0x401f3012dfd694cd              # double 7.7969470000000002
	.quad	0x401638ff75968512              # double 5.5556619999999999
	.quad	0x3febd651b0ccbc06              # double 0.86991200000000002
	.quad	0x3ffec9aed56b0100              # double 1.924239
	.quad	0x3ff0fd3a1d323fee              # double 1.061823
	.quad	0x400ea44c7b02d59d              # double 3.8302239999999999
	.quad	0x3fb4c532a497fa5d              # double 0.081133999999999998
	.quad	0x40149239a38fbca1              # double 5.142798
	.quad	0x3fe7e47991bc5586              # double 0.74663999999999997
	.quad	0x3ffa2ce9a2c66905              # double 1.6359649999999999
	.quad	0x40230b3b96af038e              # double 9.5219389999999997
	.quad	0x400d3c30d306a2b1              # double 3.6543899999999998
	.quad	0x40222e5d7cb2d906              # double 9.0905570000000004
	.quad	0x401445c810a569b1              # double 5.0681459999999996
	.quad	0x3ff7394face67d78              # double 1.451492
	.quad	0x400dbe15ca6ca03c              # double 3.7178149999999999
	.quad	0x401830678c0053e3              # double 6.0472700000000001
	.quad	0x4023a224aada33be              # double 9.8166860000000007
	.quad	0x3fe7e5e02ea960b7              # double 0.746811
	.quad	0x4007bdad7518b0d1              # double 2.967616
	.quad	0x401800fdc1615ec0              # double 6.0009680000000003
	.quad	0x3ff382fe82517e78              # double 1.219481
	.quad	0x40099b5f1bef49cf              # double 3.2008649999999998
	.quad	0x401d5a0232096788              # double 7.3378990000000001
	.quad	0x402205e9185cee18              # double 9.0115440000000007
	.quad	0x3fdf9884c6a3bde0              # double 0.49368400000000001
	.quad	0x400c9a919695d91b              # double 3.5754730000000001
	.quad	0x4018f0b6b6e0d23d              # double 6.2350719999999997
	.quad	0x4018057a786c2268              # double 6.00535
	.quad	0x4023826394face68              # double 9.7546660000000002
	.quad	0x401989118c197e56              # double 6.3838559999999998
	.quad	0x4022321a0cf1800a              # double 9.0978549999999991
	.quad	0x400cb562174c4ce0              # double 3.5885660000000001
	.quad	0x4013f6c2efd438d2              # double 4.9909780000000001
	.quad	0x4020390e66cb1034              # double 8.1114379999999997
	.quad	0x4014d220ff540896              # double 5.2052040000000002
	.quad	0x40194933e35c5b4b              # double 6.3214870000000003
	.quad	0x400bf7da61e0c5a8              # double 3.496022
	.quad	0x4015ae0ded288ce7              # double 5.419975
	.quad	0x3fd55989df1172ef              # double 0.33359
	.quad	0x4021f0d32830a0b2              # double 8.9703610000000005
	.quad	0x40097a0cf1800a7c              # double 3.1845949999999998
	.quad	0x401240296b3354c1              # double 4.5626579999999999
	.quad	0x4023c7f950b955f8              # double 9.8905740000000008
	.quad	0x3fe8e28cbd1244a6              # double 0.77765499999999999
	.quad	0x401a716d6dc1a47b              # double 6.6107690000000003
	.quad	0x4014861f18c9fb61              # double 5.1309779999999998
	.quad	0x400215dc4007570c              # double 2.2606739999999999
	.quad	0x401c8605ab3aabcd              # double 7.1308809999999996
	.quad	0x4010a4ae85b9e8c4              # double 4.1608219999999996
	.quad	0x4020cfe931876189              # double 8.4060760000000005
	.quad	0x3fd06f4be835dedf              # double 0.25679299999999999
	.quad	0x3fe78198f1d3ed52              # double 0.73456999999999995
	.quad	0x3ffd5752977c88e8              # double 1.8338190000000001
	.quad	0x4017970da5daf07c              # double 5.897513
	.quad	0x3ff5469a4df47f99              # double 1.3297369999999999
	.quad	0x401fa6d2674080fa              # double 7.9129120000000004
	.quad	0x3fcec39ffd60e94f              # double 0.240345
	.quad	0x3ffe2c00da1a9329              # double 1.8857429999999999
	.quad	0x400fd6909aed56b0              # double 3.979768
	.quad	0x4013f34a01abd1ab              # double 4.9875870000000004
	.quad	0x401301bd1ed9dfdb              # double 4.7516980000000002
	.quad	0x40217f8d0cc35ce2              # double 8.7491230000000009
	.quad	0x4012bc842fa50939              # double 4.6840979999999997
	.quad	0x40125da9435ac8a3              # double 4.5914659999999996
	.quad	0x4011ba81dc0db270              # double 4.4321359999999999
	.quad	0x4000ea04d12018a4              # double 2.1142669999999999
	.quad	0x40194dfce3150dae              # double 6.3261599999999998
	.quad	0x40214352a8438088              # double 8.6314899999999994
	.quad	0x3ff93bdad7518b0d              # double 1.577113
	.quad	0x40146339c0ebedfa              # double 5.0968999999999998
	.quad	0x4013f94467381d7e              # double 4.9934250000000002
	.quad	0x3fe816cc5fbf8338              # double 0.75278299999999998
	.quad	0x401d2d20296b3355              # double 7.2940680000000002
	.quad	0x401404e0114d2f5e              # double 5.0047610000000002
	.quad	0x3ffdc8f75536933a              # double 1.861564
	.quad	0x3ff6acbd987c6328              # double 1.417173
	.quad	0x3ff161deacc92147              # double 1.0863940000000001
	.quad	0x4019a30caa326e11              # double 6.4092279999999997
	.quad	0x4014d004fb1183b6              # double 5.203144
	.quad	0x400504f00ef1348b              # double 2.6274109999999999
	.quad	0x401c2c83a96d4c34              # double 7.0434710000000003
	.quad	0x401f206e19b90eaa              # double 7.7816700000000001
	.quad	0x401183005814940c              # double 4.3779310000000002
	.quad	0x401701fe21d96e9c              # double 5.7519460000000002
	.quad	0x3fe046cdaf4adbc6              # double 0.50864299999999996
	.quad	0x40224d8644523f68              # double 9.151415
	.quad	0x401b94998d045fe1              # double 6.8951169999999999
	.quad	0x4007140bbb1f255f              # double 2.8847879999999999
	.quad	0x401e918c5c9a34ca              # double 7.642137
	.quad	0x401122b063e07a29              # double 4.2838760000000002
	.quad	0x40236daee631f8a1              # double 9.7142250000000007
	.quad	0x400940fe47991bc5              # double 3.1567349999999998
	.quad	0x4013b296b3354c12              # double 4.9244029999999999
	.quad	0x4023bd7d3910c2c6              # double 9.8700960000000002
	.quad	0x4020ca1f6cacd185              # double 8.3947710000000004
	.quad	0x4021fde2478854ce              # double 8.9958670000000005
	.quad	0x4013a96af038e2a0              # double 4.9154470000000003
	.quad	0x401795a14488c60d              # double 5.8961230000000002
	.quad	0x3ff0201de2691644              # double 1.007841
	.quad	0x402268ee49f51698              # double 9.204943
	.quad	0x4006dbe4cd749279              # double 2.85737
	.quad	0x40237a4a276b7ed4              # double 9.7388469999999998
	.quad	0x40009e00d1b71759              # double 2.0771500000000001
	.quad	0x40008ba301216830              # double 2.0681820000000002
	.quad	0x3fc0f7c243dccea3              # double 0.13256100000000001
	.quad	0x401d428b2a6b0d95              # double 7.3149839999999999
	.quad	0x401e4b4f61672325              # double 7.5735450000000001
	.quad	0x401fa52c59fb1e19              # double 7.911302
	.quad	0x4009fcb5fe542e55              # double 3.2483939999999998
	.quad	0x40109a2120e1f7d7              # double 4.1505169999999998
	.quad	0x4012e053e2d6238e              # double 4.7190700000000003
	.quad	0x401123da72a7bd49              # double 4.2850130000000002
	.quad	0x401ca601ffb480a6              # double 7.1621170000000003
	.quad	0x4020629e0a41a262              # double 8.1926120000000004
	.quad	0x4007d450efdc9c4e              # double 2.9786700000000002
	.quad	0x4003b002e2328f9f              # double 2.4609429999999999
	.quad	0x400839e060fe4799              # double 3.02826
	.quad	0x4023d8b30b5aa716              # double 9.9232410000000009
	.quad	0x3fe94bb5e0f7fcfc              # double 0.79049199999999997
	.quad	0x401c801c92ddbdb6              # double 7.1251090000000001
	.quad	0x401a6940789613d3              # double 6.6027849999999999
	.quad	0x40217dad31fcd24e              # double 8.7454619999999998
	.quad	0x401237ecfe9b7bf2              # double 4.5546150000000001
	.quad	0x3ffbf76d5497310a              # double 1.7479070000000001
	.quad	0x402339118c197e56              # double 9.6114619999999995
	.quad	0x402243037d63022e              # double 9.1308860000000003
	.quad	0x400e0816f0068db9              # double 3.7539500000000001
	.quad	0x4022c4aa9717df1a              # double 9.3841140000000003
	.quad	0x3fdb39bcba301217              # double 0.42539900000000003
	.quad	0x401f323f24d8fd5d              # double 7.7990690000000002
	.quad	0x40160ba9d1f60179              # double 5.5113899999999996
	.quad	0x40155ccf6be37de9              # double 5.3406349999999998
	.quad	0x40127b0b39192642              # double 4.6201600000000003
	.quad	0x401ebe83a109d063              # double 7.6860489999999996
	.quad	0x40149d6ba8c58687              # double 5.1537309999999996
	.quad	0x40225efc9bc77143              # double 9.1855209999999996
	.quad	0x4022b46173b85e81              # double 9.3523060000000005
	.quad	0x4019bc13fd0d0679              # double 6.4336700000000002
	.quad	0x4020f20232096788              # double 8.4726730000000003
	.quad	0x40162ee6fb4c3c19              # double 5.5458030000000003
	.quad	0x3fde9085f4a1272d              # double 0.47757100000000002
	.quad	0x400ddfc9ff92f2b6              # double 3.7342719999999998
	.quad	0x40164279dd3bafd9              # double 5.5649179999999996
	.quad	0x3fd7f8012dfd694d              # double 0.37451200000000001
	.quad	0x3f8e5aff2a172abf              # double 0.014822
	.quad	0x40065c4d22c881e4              # double 2.7950689999999998
	.quad	0x4023a95336deb95e              # double 9.8307129999999994
	.quad	0x4023e4bb3c81908e              # double 9.9467409999999994
	.quad	0x401448d32830a0b2              # double 5.0711180000000002
	.quad	0x4014b5ec3dab5c3a              # double 5.1776590000000002
	.quad	0x401b093d5347a5b1              # double 6.759023
	.quad	0x40201af661f18ca0              # double 8.0526610000000005
	.quad	0x40193c5b4aa9717e              # double 6.308942
	.quad	0x40117550ca1cef24              # double 4.3645659999999999
	.quad	0x400da5814940bbb2              # double 3.705813
	.quad	0x40119fe60c38f367              # double 4.4061510000000004
	.quad	0x3ffe70329802c0a5              # double 1.9023920000000001
	.quad	0x401500b242070b8d              # double 5.25068
	.quad	0x40171935fc3b4f61              # double 5.7746199999999996
	.quad	0x4015ef8487b99d45              # double 5.4839039999999999
	.quad	0x4022b36a619da9ca              # double 9.3504210000000008
	.quad	0x4021b1567dbb16c2              # double 8.8463630000000002
	.quad	0x401424dc22ab25b3              # double 5.0359959999999999
	.quad	0x4008c36c9c0a8d22              # double 3.0954220000000001
	.quad	0x400affce3150dae4              # double 3.374905
	.quad	0x4020debdf4c2b51c              # double 8.4350430000000002
	.quad	0x3f7a7c17a89331a1              # double 0.0064660000000000004
	.quad	0x40134313be22e5de              # double 4.8155049999999999
	.quad	0x3ffe41d3aa369fcf              # double 1.8910709999999999
	.quad	0x4004061309c7ffde              # double 2.5029659999999998
	.quad	0x402207d6d7e889c2              # double 9.0153110000000005
	.quad	0x3ff48b0292817764              # double 1.283938
	.quad	0x40213bfbbd7b2032              # double 8.6171550000000003
	.quad	0x401ceb24e9c45458              # double 7.2296329999999998
	.quad	0x3ff2babda0dfdef8              # double 1.1705909999999999
	.quad	0x401426c3fc43b2dd              # double 5.0378569999999998
	.quad	0x4012d471f79420b4              # double 4.7074660000000002
	.quad	0x4010709203a322af              # double 4.1099319999999997
	.quad	0x4014ffbf83382e45              # double 5.2497540000000003
	.quad	0x4019e6f0068db8bb              # double 6.4755250000000002
	.quad	0x3ff1a2fcefaa4767              # double 1.102292
	.quad	0x401b30fd7e45803d              # double 6.7978420000000002
	.quad	0x401212ba9d1f6018              # double 4.5182900000000004
	.quad	0x40086332f01754b0              # double 3.0484369999999998
	.quad	0x401b201e68a0d34a              # double 6.7813660000000002
	.quad	0x401226366d7a56de              # double 4.5373169999999998
	.quad	0x3ffed24af0bf1a5d              # double 1.9263410000000001
	.quad	0x3fc16323bbc6eb0b              # double 0.13583799999999999
	.quad	0x4014ec8d5c74751d              # double 5.2310080000000001
	.quad	0x4023123ee1bd1eda              # double 9.5356360000000002
	.quad	0x4004d4e9813879c4              # double 2.603961
	.quad	0x4011c5cbbc2b94d9              # double 4.4431599999999998
	.quad	0x40167eee52589268              # double 5.6239559999999997
	.quad	0x4023379d2bf55150              # double 9.6086209999999994
	.quad	0x3fbfc05d52c16df4              # double 0.124029
	.quad	0x400d384901d19158              # double 3.6524830000000001
	.quad	0x40211160d887ebf2              # double 8.5339419999999997
	.quad	0x4018efe60c38f367              # double 6.2342760000000004
	.quad	0x4023047e8ccdd93c              # double 9.5087779999999995
	.quad	0x402186dc7ef177a7              # double 8.763401
	.quad	0x4022aeb7ed41b75a              # double 9.3412469999999992
	.quad	0x3fd13519bd4067cf              # double 0.26886599999999999
	.quad	0x401450ee8d10f51b              # double 5.0790350000000002
	.quad	0x3f9e358298cc1440              # double 0.029500999999999999
	.quad	0x4016e84449dbec25              # double 5.7268230000000004
	.quad	0x4022c15703f2d3c8              # double 9.3776170000000007
	.quad	0x3ff22994185058de              # double 1.135151
	.quad	0x3fd73f39d1b2e59b              # double 0.363234
	.quad	0x401c75daad601ffb              # double 7.1150919999999998
	.quad	0x402243daf8df7a4e              # double 9.1325299999999991
	.quad	0x4006b8655193708b              # double 2.8400370000000001
	.quad	0x40238bbd1676640a              # double 9.7729269999999993
	.quad	0x3fe5b2702a3486f0              # double 0.67803199999999997
	.quad	0x401a37d523b3636f              # double 6.5545239999999998
	.quad	0x4022a8b827fa1a0d              # double 9.3295300000000001
	.quad	0x401510fba8826aa9              # double 5.2665850000000001
	.quad	0x400ded0b30b5aa71              # double 3.7407439999999998
	.quad	0x4009d65b20b8066c              # double 3.2296659999999999
	.quad	0x3f940895d0b73d19              # double 0.019564000000000002
	.quad	0x40205206c8711d7a              # double 8.1602080000000008
	.quad	0x3fca2fb5d031055c              # double 0.20458100000000001
	.quad	0x400dbb38d60a6330              # double 3.7164169999999999
	.quad	0x3fee37cdcca70d20              # double 0.94431200000000004
	.quad	0x4023daae6c8f7553              # double 9.9271119999999992
	.quad	0x400449070fbeb9e5              # double 2.5356580000000002
	.quad	0x402361d5e4a38327              # double 9.6910849999999992
	.quad	0x40232dbf8b9baa15              # double 9.5893519999999998
	.quad	0x3fefe4f11b60ae97              # double 0.99669700000000005
	.quad	0x3ffd32dc2b0ea183              # double 1.8249169999999999
	.quad	0x4009ab7fe08aefb3              # double 3.2087400000000001
	.quad	0x3fbe4d3bf2f55582              # double 0.118366
	.quad	0x402202a715831f04              # double 9.0051810000000003
	.quad	0x3ff2b05532617c1c              # double 1.16805
	.quad	0x4008b2abef06b378              # double 3.0872419999999998
	.quad	0x40095972474538ef              # double 3.1686749999999999
	.quad	0x4020378a0902de01              # double 8.1084750000000003
	.quad	0x3ffc461e92923e5c              # double 1.7671190000000001
	.quad	0x401a27329c347e8d              # double 6.5382790000000002
	.quad	0x400b8b524399b2c4              # double 3.443028
	.quad	0x3ffd906466b1e5c1              # double 1.8477520000000001
	.quad	0x4009d79f23465626              # double 3.2302840000000002
	.quad	0x401fcc88e79aae6d              # double 7.9497410000000003
	.quad	0x40214e9b9d7fd827              # double 8.6535309999999992
	.quad	0x3fd11e53a81dc0db              # double 0.26747599999999999
	.quad	0x40076169c23b7953              # double 2.9225650000000001
	.quad	0x3ffadb6a1e81cb47              # double 1.6785680000000001
	.quad	0x4010bab3aabcd78c              # double 4.1823259999999998
	.quad	0x4000ebf9830e3cda              # double 2.1152220000000002
	.quad	0x400b0f1561911490              # double 3.3823650000000001
	.quad	0x400b36e58a32f449              # double 3.401805
	.quad	0x4021c3a493c89f41              # double 8.8821150000000006
	.quad	0x401b2f2e05ccc89b              # double 6.7960739999999999
	.quad	0x40108bfec13b9f12              # double 4.1367139999999996
	.quad	0x401cabf401c4fc1e              # double 7.167923
	.quad	0x401e6039ef0f16f4              # double 7.5939709999999998
	.quad	0x4016d7168f8e7ddd              # double 5.7100470000000003
	.quad	0x4000b1465e892253              # double 2.08656
	.quad	0x4010107af20ea5b5              # double 4.0160939999999998
	.quad	0x3ff67a9c12f09d8c              # double 1.4049339999999999
	.quad	0x4015857d5a9eb207              # double 5.3803609999999997
	.quad	0x401077357e670e2c              # double 4.1164149999999999
	.quad	0x3fe6c9cf56eac860              # double 0.71213499999999996
	.quad	0x4023213532e7b3d9              # double 9.5648590000000002
	.quad	0x401c0f93ff25e56d              # double 7.0152130000000001
	.quad	0x3fdbfbfc6540cc79              # double 0.437255
	.quad	0x4002070435efa616              # double 2.2534260000000002
	.quad	0x3ff0a37bd05af6c7              # double 1.0399130000000001
	.quad	0x402023ef49cf56eb              # double 8.0701850000000004
	.quad	0x4009e9b52007dd44              # double 3.239115
	.quad	0x401f138327674d16              # double 7.7690549999999998
	.quad	0x4016d99ed7c6fbd2              # double 5.7125199999999996
	.quad	0x401de79463cfb331              # double 7.4761519999999999
	.quad	0x3ffedeafee6fb4c4              # double 1.9293670000000001
	.quad	0x3ffb5d6e04c05921              # double 1.71031
	.quad	0x3fe3e05ccc89b0ee              # double 0.62113799999999997
	.quad	0x3ff778f47304039b              # double 1.4670300000000001
	.quad	0x3ff6f5818c5c9a35              # double 1.434938
	.quad	0x4022624acf312b1b              # double 9.1919769999999996
	.quad	0x4022c307af20ea5b              # double 9.3809179999999994
	.quad	0x3f7adc8fb86f47b6              # double 0.0065579999999999996
	.quad	0x40096d6ba8c58687              # double 3.1784279999999998
	.quad	0x40220f4927913e81              # double 9.0298549999999995
	.quad	0x3fe8648626f60e0f              # double 0.76227100000000003
	.quad	0x3ffd5f1e0828c36e              # double 1.8357220000000001
	.quad	0x401fca0c282c6ef4              # double 7.9473120000000002
	.quad	0x3fdac7abfb9bed31              # double 0.418437
	.quad	0x401a0ba6efc371da              # double 6.5113789999999998
	.quad	0x40177964a59c065b              # double 5.8685479999999997
	.quad	0x401ce8b3700474da              # double 7.2272470000000002
	.quad	0x4001d5ddd2ae8e1d              # double 2.2294269999999998
	.quad	0x3ff68a4b124d099e              # double 1.408763
	.quad	0x401c39103c8e25c8              # double 7.0557259999999999
	.quad	0x401a2a116659d12d              # double 6.5410820000000003
	.quad	0x3ffc420b3d4ae42a              # double 1.766124
	.quad	0x40098ba0620ab713              # double 3.1931769999999999
	.quad	0x402248ef99806f26              # double 9.1424529999999997
	.quad	0x400adc7b45f17bd9              # double 3.3576570000000001
	.quad	0x4012d5a9eb2074eb              # double 4.7086560000000004
	.quad	0x3fe4b5fc3b4f6167              # double 0.64721499999999998
	.quad	0x4017c48cf7caa759              # double 5.9419440000000003
	.quad	0x4005bc2d2780778a              # double 2.7168830000000002
	.quad	0x4000fb6e503fb374              # double 2.1227689999999999
	.quad	0x3ff8b8df7a4e7ab7              # double 1.5451349999999999
	.quad	0x40163d98bf7f0670              # double 5.5601529999999997
	.quad	0x40093f8f4730403a              # double 3.1560350000000001
	.quad	0x400e6b65a9a80496              # double 3.8024399999999998
	.quad	0x3fc4b14ec204f2ae              # double 0.161661
	.quad	0x4006c68b9fdbd2fa              # double 2.846946
	.quad	0x401b6fd5cb790fb6              # double 6.8592139999999997
	.quad	0x40195fdcdf698783              # double 6.3436159999999999
	.quad	0x3fe32c94b380cb6c              # double 0.59919199999999995
	.quad	0x40231eb2074ea8da              # double 9.5599519999999991
	.quad	0x40204e605ee568a5              # double 8.153079
	.quad	0x401025bb384fd2a6              # double 4.0368469999999999
	.quad	0x4020b8196d8f4f94              # double 8.3595690000000004
	.quad	0x3ff1cbb906466b1e              # double 1.1122369999999999
	.quad	0x4013daad601ffb48              # double 4.963552
	.quad	0x40136f3fd933e35c              # double 4.8586419999999997
	.quad	0x401835a964e8b7e5              # double 6.0524040000000001
	.quad	0x4007c7e06961c369              # double 2.9725959999999998
	.quad	0x3f8409a240314877              # double 0.0097839999999999992
	.quad	0x401a221ab4b72c52              # double 6.5333050000000004
	.quad	0x4014a84fd2a62aa2              # double 5.1643670000000004
	.quad	0x4009fa89331a08c0              # double 3.2473320000000001
	.quad	0x4002f4299d883ba3              # double 2.3692199999999999
	.quad	0x402091fd9ba1b196              # double 8.2851379999999999
	.quad	0x401ec0bf5d78811b              # double 7.6882299999999998
	.quad	0x4022b142917507ea              # double 9.3462110000000003
	.quad	0x402258789613d31c              # double 9.1727950000000006
	.quad	0x401ff85553ef6b5d              # double 7.9925129999999997
	.quad	0x40015c6c97d8cf3a              # double 2.1701290000000002
	.quad	0x4021a8f5e41d4b6a              # double 8.8300009999999993
	.quad	0x4022ca76d97b30f9              # double 9.3954380000000004
	.quad	0x40024053e2d6238e              # double 2.2814100000000002
	.quad	0x40205b0df9bdc6a0              # double 8.1778410000000008
	.quad	0x400e9ba450acc0be              # double 3.8259970000000001
	.quad	0x3ff3d363f572de44              # double 1.239109
	.quad	0x3ff8dc8a7a41e57e              # double 1.5538430000000001
	.quad	0x4022428e92d55a3a              # double 9.1299939999999999
	.quad	0x401c30a0f4d7add1              # double 7.0474889999999997
	.quad	0x401a10f840181e04              # double 6.516572
	.quad	0x4009e0685553ef6b              # double 3.2345739999999998
	.quad	0x4020a0fe47991bc5              # double 8.3144399999999994
	.quad	0x4023077b395c4220              # double 9.5146119999999996
	.quad	0x4013330c67168f8e              # double 4.7998519999999996
	.quad	0x401f7b90a7829069              # double 7.8706690000000003
	.quad	0x401d6102bc72e276              # double 7.3447370000000003
	.quad	0x4000681a9b8cb8e1              # double 2.0508320000000002
	.quad	0x3ff12117720c8cd6              # double 1.0705789999999999
	.quad	0x402294ead0c3d252              # double 9.2908539999999995
	.quad	0x400c4dadb8348f54              # double 3.5379290000000001
	.quad	0x40191cd9e83e425b              # double 6.2781750000000001
	.quad	0x4021fa9003eea20a              # double 8.9893800000000006
	.quad	0x3fe98485a0be5109              # double 0.797427
	.quad	0x401f397d49d7ba67              # double 7.8061420000000003
	.quad	0x400bafd438d1d8a5              # double 3.4608539999999999
	.quad	0x40122e15ca6ca03c              # double 4.5450049999999997
	.quad	0x401dcf18c9fb6135              # double 7.4522430000000002
	.quad	0x4008290795f676ea              # double 3.0200339999999999
	.quad	0x40229585be1a8262              # double 9.2920359999999995
	.quad	0x40028ec4a4095f24              # double 2.3197109999999999
	.quad	0x400f0426351deefe              # double 3.8770259999999999
	.quad	0x3fb2e957470eb24a              # double 0.073872999999999994
	.quad	0x3fa616b54e2b063e              # double 0.043142
	.quad	0x4023e1df548ecd8e              # double 9.9411570000000005
	.quad	0x3fba5d074213a0c7              # double 0.102982
	.quad	0x4023ccaec8d5c747              # double 9.8997709999999994
	.quad	0x3fd7787ce95faa8b              # double 0.36672900000000003
	.quad	0x401f0c493426783a              # double 7.7619980000000002
	.quad	0x4020206df82b1f68              # double 8.0633389999999991
	.quad	0x40147ee06d938152              # double 5.1239030000000003
	.quad	0x3fe9c3009b30728f              # double 0.80505400000000005
	.quad	0x40123c918e325d4a              # double 4.5591489999999997
	.quad	0x40112723ee1bd1ee              # double 4.2882230000000003
	.quad	0x3f556191148fd9fd              # double 0.001305
	.quad	0x4013a400b88ca3e8              # double 4.9101590000000002
	.quad	0x401785a426bb55ac              # double 5.880509
	.quad	0x401d8443914f483d              # double 7.3791640000000003
	.quad	0x401a8c679cc74b84              # double 6.6371140000000004
	.quad	0x400c7ec02f2f9874              # double 3.56189
	.quad	0x3ff7d479d4d83409              # double 1.489374
	.quad	0x401ed9f1f14983d8              # double 7.7128370000000003
	.quad	0x400b1c9c4da9003f              # double 3.38897
	.quad	0x3fe0e00f345069a5              # double 0.52735100000000001
	.quad	0x40163b762d83c6c9              # double 5.5580679999999996
	.quad	0x400375b1422ccb3a              # double 2.4324669999999999
	.quad	0x400513df2a9d627c              # double 2.634703
	.quad	0x401ee4751ce28ed6              # double 7.7231030000000001
	.quad	0x401f3ac6cdaf4adc              # double 7.8073990000000002
	.quad	0x4017010adcd2d44e              # double 5.7510180000000002
	.quad	0x400a467dfe32a066              # double 3.2844199999999999
	.quad	0x4011a1e364bec67a              # double 4.4080940000000002
	.quad	0x4023410e8858ff76              # double 9.6270640000000007
	.quad	0x40187858793dd97f              # double 6.1175249999999997
	.quad	0x4003cc4ac6cdaf4b              # double 2.4747520000000001
	.quad	0x40197947064ece9a              # double 6.3684349999999998
	.quad	0x4019dd83c6c97d8d              # double 6.466323
	.quad	0x401549c065b63d3e              # double 5.3220229999999997
	.quad	0x3ffe052d666a9824              # double 1.8762639999999999
	.quad	0x40187ce3de6149c7              # double 6.121963
	.quad	0x402220dd82fd75e2              # double 9.06419
	.quad	0x401a308461f9f01c              # double 6.5473800000000004
	.quad	0x401947e80389f83c              # double 6.3202210000000001
	.quad	0x4021b95e17e34b94              # double 8.8620459999999994
	.quad	0x400f0e5d5b24e9c4              # double 3.8820139999999999
	.quad	0x3ff53a5093964a5a              # double 1.3267370000000001
	.quad	0x400177a13c254a3c              # double 2.1834129999999998
	.quad	0x4018a64b2314013f              # double 6.1623960000000002
	.quad	0x4012d435696e58a3              # double 4.7072349999999998
	.quad	0x40021897a67a52ac              # double 2.2620079999999998
	.quad	0x401ef5d462c343b7              # double 7.7400679999999999
	.quad	0x402121e7967caea7              # double 8.5662199999999995
	.quad	0x4020bf572de43ed9              # double 8.3737119999999994
	.quad	0x4010b1ba3ca7503c              # double 4.1735620000000004
	.quad	0x400c8734f82f5126              # double 3.5660189999999998
	.quad	0x40200b97dd00f777              # double 8.0226430000000004
	.quad	0x401f39d99029ae4f              # double 7.8064939999999998
	.quad	0x401eb89656eefa1e              # double 7.6802609999999998
	.quad	0x3fcb1098d477bbf9              # double 0.21144399999999999
	.quad	0x400466efc371da38              # double 2.550262
	.quad	0x40186db8fde2ef4e              # double 6.107151
	.quad	0x3ff145e6ff7dfa01              # double 1.079566
	.quad	0x4020cc558644523f              # double 8.3990899999999993
	.quad	0x401c60a7c5ac471b              # double 7.0943899999999998
	.quad	0x3ffb1c66d373affb              # double 1.694434
	.quad	0x3fe7e0ac7da1ec4e              # double 0.74617599999999995
	.quad	0x401e0adb8348f53c              # double 7.5106029999999997
	.quad	0x400e2800eae18aca              # double 3.769533
	.quad	0x3fdeae2541d8e864              # double 0.479379
	.quad	0x4010978811b1d92b              # double 4.1479799999999996
	.quad	0x4006c621b7e0ac7e              # double 2.8467440000000002
	.quad	0x401622ea52ef911d              # double 5.534097
	.quad	0x401889103c8e25c8              # double 6.1338509999999999
	.quad	0x400376049ecb31c2              # double 2.432626
	.quad	0x40027a8ef77f27fe              # double 2.3098429999999999
	.quad	0x4020e78140dd3fe2              # double 8.4521580000000007
	.quad	0x4013bf0a5efe9318              # double 4.9365629999999996
	.quad	0x4023dd9f2ba9d1f6              # double 9.932855
	.quad	0x4006ab96f21f6cad              # double 2.8337840000000001
	.quad	0x400684e054690de1              # double 2.8148810000000002
	.quad	0x400fe316a055757d              # double 3.9858829999999998
	.quad	0x4004a7f1b6912125              # double 2.582004
	.quad	0x400c18dde7a743a6              # double 3.5121419999999999
	.quad	0x400234dc65c70436              # double 2.275811
	.quad	0x4003c63065f9591d              # double 2.4717720000000001
	.quad	0x401bf8954a7f8013              # double 6.9927570000000001
	.quad	0x4022d23c85c24c40              # double 9.4106179999999994
	.quad	0x40041369b9d7fd82              # double 2.5094789999999998
	.quad	0x400d1d388a8b08dd              # double 3.6392679999999999
	.quad	0x3fd830e3cd9a5226              # double 0.37798399999999999
	.quad	0x401e0569b17481b2              # double 7.5052859999999999
	.quad	0x40141a2339c0ebee              # double 5.025525
	.quad	0x4022e984e3ffef39              # double 9.4560919999999999
	.quad	0x4022b6ed245b291c              # double 9.3572780000000008
	.quad	0x401f5b456b441bb9              # double 7.8391320000000002
	.quad	0x4013e8e71476af8a              # double 4.9774440000000002
	.quad	0x4021eead9274e22a              # double 8.9661679999999997
	.quad	0x4023b1feeb2d0a24              # double 9.8476479999999995
	.quad	0x401d96dbd72bcb60              # double 7.3973230000000001
	.quad	0x400f599780baa583              # double 3.9187460000000001
	.quad	0x401db12ba16e7a31              # double 7.4230179999999999
	.quad	0x401d44fc1df3300e              # double 7.3173680000000001
	.quad	0x401ac78ab0c88a48              # double 6.6948650000000001
	.quad	0x401689f623076c05              # double 5.634728
	.quad	0x4023fb7ac3eb7cbb              # double 9.9911709999999995
	.quad	0x3ff1f91e646f1562              # double 1.1233200000000001
	.quad	0x3fc785942917507f              # double 0.18376400000000001
	.quad	0x4018ca908a265f0f              # double 6.1978169999999997
	.quad	0x4012b1ecd4aa10e0              # double 4.6737549999999999
	.quad	0x4010db8ed1bf7ad5              # double 4.2144120000000003
	.quad	0x40077dba908a265f              # double 2.936391
	.quad	0x3ffdd5ae1cde5d18              # double 1.864668
	.quad	0x3fe1c7b02d59d55e              # double 0.55562599999999995
	.quad	0x401820225742dcf4              # double 6.0313809999999997
	.quad	0x4012bb4bf8fcd680              # double 4.6829070000000002
	.quad	0x3ff1ef60e0eb67c3              # double 1.1209420000000001
	.quad	0x401099aa60913a50              # double 4.1500640000000004
	.quad	0x4023ef78feef5ec8              # double 9.9677199999999999
	.quad	0x3fed8685553ef6b6              # double 0.92267100000000002
	.quad	0x3fd7d6dc1a47a9e3              # double 0.37248900000000001
	.quad	0x3fd9023a6ce3582a              # double 0.39076100000000002
	.quad	0x401ceb69db65ecc4              # double 7.2298960000000001
	.quad	0x401fd7f349385836              # double 7.9608889999999999
	.quad	0x4011bc0379314446              # double 4.4336070000000003
	.quad	0x40174aff2a172abf              # double 5.8232390000000001
	.quad	0x400e23d9a95421c0              # double 3.7675049999999999
	.quad	0x401a22ba16e7a312              # double 6.5339130000000001
	.quad	0x401351290257c915              # double 4.8292580000000003
	.quad	0x4004148c2e770bd0              # double 2.510033
	.quad	0x4014fb45f17bd8be              # double 5.2453839999999996
	.quad	0x401a92dcf4623d0c              # double 6.643421
	.quad	0x401ca20c49ba5e35              # double 7.1582499999999998
	.quad	0x40094aaf35e310dc              # double 3.161467
	.quad	0x400af8da3c21187e              # double 3.3715099999999998
	.quad	0x4003b7292c493426              # double 2.4644339999999998
	.quad	0x4019153c579f2346              # double 6.2707379999999997
	.quad	0x4021d1e7967caea7              # double 8.9099699999999995
	.quad	0x4008e11dbca9691a              # double 3.1099199999999998
	.quad	0x4018b9817b95a294              # double 6.1811579999999999
	.quad	0x4016a37f7be121ee              # double 5.6596659999999996
	.quad	0x401d77d1bb4916ca              # double 7.3670109999999998
	.quad	0x4003e6748e475600              # double 2.487527
	.quad	0x400ee39f77292c49              # double 3.8611439999999999
	.quad	0x4010efae7924af0c              # double 4.234064
	.quad	0x401cf57928e0c9da              # double 7.2397200000000002
	.quad	0x3fd4059641f64495              # double 0.31284099999999998
	.quad	0x3ffbd959a30984e4              # double 1.740564
	.quad	0x4020574a1272c94b              # double 8.1704869999999996
	.quad	0x3fdb5842b734b513              # double 0.42726199999999998
	.quad	0x401dc1a9b8cb8e08              # double 7.4391239999999996
	.quad	0x3fda50374ff865d8              # double 0.41114600000000001
	.quad	0x40233d1ac9afe1da              # double 9.6193449999999991
	.quad	0x401997dbf487fcb9              # double 6.3982999999999999
	.quad	0x4010fe779207d4e1              # double 4.2485030000000004
	.quad	0x3fe810ff97247454              # double 0.75207500000000004
	.quad	0x3ffdf07af20ea5b5              # double 1.871211
	.quad	0x401b415097c80842              # double 6.8137840000000001
	.quad	0x3ff799ddc1e7967d              # double 1.4750650000000001
	.quad	0x40110f9c1f85d745              # double 4.265244
	.quad	0x400b745846e8f29d              # double 3.4318089999999999
	.quad	0x401d337c99ae924f              # double 7.3002799999999999
	.quad	0x40096e48e8a71de7              # double 3.1788500000000002
	.quad	0x40205eaf251c193b              # double 8.1849299999999996
	.quad	0x40201eeb702602c9              # double 8.0603899999999999
	.quad	0x401fbbcfd4bf0996              # double 7.9334100000000003
	.quad	0x401c19b8887229e9              # double 7.025118
	.quad	0x401e029bf162ae4b              # double 7.5025479999999999
	.quad	0x401bb3482be8bc17              # double 6.9250800000000003
	.quad	0x400f8f6d7625204b              # double 3.945033
	.quad	0x400d2642bf9830e4              # double 3.6436820000000001
	.quad	0x4016b9b951c5c572              # double 5.6813710000000004
	.quad	0x4000d1b8aa00192a              # double 2.1024029999999998
	.quad	0x40163052934acaff              # double 5.5471899999999996
	.quad	0x400f5ccea28fe261              # double 3.9203160000000001
	.quad	0x40182164c729f59d              # double 6.0326110000000002
	.quad	0x40115507a6bd6e8b              # double 4.333037
	.quad	0x402144c63f141206              # double 8.6343250000000005
	.quad	0x4004cf5e41d4b6a6              # double 2.601254
	.quad	0x401955d3dc8b86b1              # double 6.3338159999999997
	.quad	0x40061d50a88effe3              # double 2.7643140000000002
	.quad	0x40222dc74751ce29              # double 9.0894110000000001
	.quad	0x40235b2a0663c750              # double 9.6780550000000005
	.quad	0x4007e950fc71d606              # double 2.9889239999999999
	.quad	0x4015a2d38476f2a6              # double 5.4090100000000003
	.quad	0x3fef57c6fbd273d6              # double 0.97946500000000003
	.quad	0x3fec7fa7eb6bf445              # double 0.89058300000000001
	.quad	0x401e087229e90796              # double 7.508248
	.quad	0x401b5004b7f5a533              # double 6.8281429999999999
	.quad	0x400f6d740c4156e2              # double 3.9284439999999998
	.quad	0x4018d1dde37585be              # double 6.2049479999999999
	.quad	0x3fba2120e1f7d73d              # double 0.10206800000000001
	.quad	0x40127acc4ef88b97              # double 4.6199199999999996
	.quad	0x401c7da4484944ed              # double 7.1226969999999996
	.quad	0x401e661672324c83              # double 7.5996949999999996
	.quad	0x401e4441bb8c32a9              # double 7.5666570000000001
	.quad	0x401bab54195cc858              # double 6.917313
	.quad	0x40210ea593a2df93              # double 8.5286069999999992
	.quad	0x402209b3f20a73f7              # double 9.0189509999999995
	.quad	0x401cf98f1d3ed528              # double 7.2437100000000001
	.quad	0x4019d55c52e72da1              # double 6.4583599999999999
	.quad	0x401e0cf63800218e              # double 7.5126580000000001
	.quad	0x4008df9e7b80a9df              # double 3.1091890000000002
	.quad	0x40183a2d72ffd1dd              # double 6.0568140000000001
	.quad	0x400c5ada76d97b31              # double 3.544362
	.quad	0x3ff025881a1554fc              # double 1.009163
	.quad	0x4014dad46f587d70              # double 5.2137010000000004
	.quad	0x400789a5657fb69a              # double 2.9422100000000002
	.quad	0x4018e935fc3b4f61              # double 6.2277449999999996
	.quad	0x400d75ab7dc7abfc              # double 3.6824560000000002
	.quad	0x40207f2f9873ffac              # double 8.2484099999999997
	.quad	0x40094181e03f7058              # double 3.1569859999999998
	.quad	0x4015f3c70c996b76              # double 5.4880639999999996
	.quad	0x401d7d2e94680171              # double 7.3722479999999999
	.quad	0x4020790342aa9f7b              # double 8.2363529999999994
	.quad	0x4017dbe22e5de15d              # double 5.9647300000000003
	.quad	0x4021fd186db50f41              # double 8.9943270000000002
	.quad	0x4021b248f96e1587              # double 8.8482129999999994
	.quad	0x402287ee9142b303              # double 9.2654920000000001
	.quad	0x3ffd7644955b4678              # double 1.8413740000000001
	.quad	0x4008f5bd512ec6bd              # double 3.11999
	.quad	0x40224aad81adea89              # double 9.1458549999999992
	.quad	0x40212a645e4e69f0              # double 8.5827969999999993
	.quad	0x4004a3a6ce358299              # double 2.5799080000000001
	.quad	0x4002925b317efe0d              # double 2.3214630000000001
	.quad	0x4021ec75818c5c9a              # double 8.9618339999999996
	.quad	0x401b025f202107b8              # double 6.7523160000000004
	.quad	0x401436ab5807fed2              # double 5.053388
	.quad	0x40092d15ad106ee3              # double 3.147014
	.quad	0x4012fd19157abb88              # double 4.747166
	.quad	0x4016578811b1d92b              # double 5.5854799999999996
	.quad	0x40125c7b890d5a5c              # double 4.5903150000000004
	.quad	0x4020cb1dd5d3dc8c              # double 8.3967120000000008
	.quad	0x402274f95d4e8fb0              # double 9.2284649999999999
	.quad	0x400e96709fa54c55              # double 3.8234569999999999
	.quad	0x400079c065b63d3e              # double 2.0594489999999999
	.quad	0x4020972990f301eb              # double 8.2952390000000005
	.quad	0x3ffb56ba8c586877              # double 1.708674
	.quad	0x402206fd00713f07              # double 9.0136489999999991
	.quad	0x3fb80aaeafab53d6              # double 0.093912999999999996
	.quad	0x3fea76c69b5a63fa              # double 0.82699900000000004
	.quad	0x4020f1a8cdea033e              # double 8.4719909999999992
	.quad	0x401ec149c6f36ef8              # double 7.6887579999999999
	.quad	0x3fce146a1a500d5f              # double 0.23499800000000001
	.quad	0x3ff30317acc4ef89              # double 1.1882550000000001
	.quad	0x401947f6f4be835e              # double 6.3202780000000001
	.quad	0x40144fd2a62aa194              # double 5.0779519999999998
	.quad	0x401c014df8b15726              # double 7.0012740000000004
	.quad	0x4003c28dc981beb2              # double 2.4699970000000002
	.quad	0x3ff8e8eb463497b7              # double 1.5568649999999999
	.quad	0x402114efaa4766c7              # double 8.5408910000000002
	.quad	0x40201394d9407896              # double 8.0382449999999999
	.quad	0x3ff4c70867ad8e43              # double 1.298592
	.quad	0x401ca66f5019f3c7              # double 7.162534
	.quad	0x3fff60fb224aada3              # double 1.9611769999999999
	.quad	0x401c26ca89fc6da4              # double 7.0378819999999997
	.quad	0x400895a858793dd9              # double 3.0730749999999998
	.quad	0x4018bee06d938152              # double 6.1864030000000003
	.quad	0x4004ad57bc7f77af              # double 2.5846399999999998
	.quad	0x400865eedcc20d56              # double 3.0497719999999999
	.quad	0x401650bf1a5ca298              # double 5.5788539999999998
	.quad	0x401d0c422036006d              # double 7.261971
	.quad	0x4013e190d173fb7a              # double 4.9702789999999997
	.quad	0x3ff0f0e3cd9a5226              # double 1.0588109999999999
	.quad	0x401eba5870da5daf              # double 7.6819779999999999
	.quad	0x4020e17edc7ef178              # double 8.4404210000000006
	.quad	0x4015213c68661ae7              # double 5.282457
	.quad	0x3fd8f19d66adb403              # double 0.38974700000000001
	.quad	0x4011e949a5657fb7              # double 4.4778200000000004
	.quad	0x40100872f33ca31e              # double 4.0082509999999996
	.quad	0x402255caf2d7f951              # double 9.1675640000000005
	.quad	0x401f5ac471b47842              # double 7.8386399999999998
	.quad	0x40211a3aafff36ac              # double 8.5512289999999993
	.quad	0x4012bd79d0a67621              # double 4.6850350000000001
	.quad	0x4022d51d4738a3b5              # double 9.4162389999999991
	.quad	0x40135fdda8bd230c              # double 4.8436190000000003
	.quad	0x40211c5fbf83382e              # double 8.5554179999999995
	.quad	0x3ffe8116ebd4cfd1              # double 1.9065160000000001
	.quad	0x4006649f51697f20              # double 2.7991320000000002
	.quad	0x401bf76cce5f7404              # double 6.9916260000000001
	.quad	0x3fb7f13059641f64              # double 0.093523999999999996
	.quad	0x401812b734b51373              # double 6.0182770000000003
	.quad	0x40239ec2699c7bcc              # double 9.8100769999999997
	.quad	0x400ffa658c4bd33d              # double 3.9972639999999999
	.quad	0x4023b0c88a47ecff              # double 9.8452800000000006
	.quad	0x400049728a611772              # double 2.035863
	.quad	0x4015265b20b8066c              # double 5.2874569999999999
	.quad	0x40214c58eeae9ee4              # double 8.6491159999999993
	.quad	0x401b60247021d10b              # double 6.8438889999999999
	.quad	0x40013565c2d27807              # double 2.1510729999999998
	.quad	0x3feb609bb6aa4b99              # double 0.85554300000000005
	.quad	0x40031d36f7e3d1cc              # double 2.389265
	.quad	0x4023e161e4f765fe              # double 9.9402000000000008
	.quad	0x4020008d6d3b6cbe              # double 8.0010790000000007
	.quad	0x4010dcf28c79f666              # double 4.2157689999999999
	.quad	0x401c726202539757              # double 7.1117020000000002
	.quad	0x40141b40f66a5508              # double 5.0266149999999996
	.quad	0x3ff5d16659d12cae              # double 1.363623
	.quad	0x3fff9bcba3012168              # double 1.975536
	.quad	0x3fcf6a26e5471716              # double 0.24542700000000001
	.quad	0x401ae07d912556d2              # double 6.7192290000000003
	.quad	0x401ed66c2acb85a5              # double 7.709397
	.quad	0x3ff8851654d61b2a              # double 1.532492
	.quad	0x401f6c957cd466f5              # double 7.8560389999999999
	.quad	0x40135227d028a1e0              # double 4.8302300000000002
	.quad	0x401089d16fc9bc77              # double 4.1345879999999999
	.quad	0x4016f9a027525461              # double 5.7437750000000003
	.quad	0x40175b7bf1e8e608              # double 5.83934
	.quad	0x4022578982cb20fb              # double 9.1709709999999998
	.quad	0x4011a6f0cfe15443              # double 4.4130279999999997
	.quad	0x4021db1fab96f21f              # double 8.9279759999999992
	.quad	0x401c0d4c33b53932              # double 7.0129859999999997
	.quad	0x4005742352eb5f5f              # double 2.681708
	.quad	0x4020290b1feeb2d1              # double 8.0801630000000007
	.quad	0x400cd94c01605250              # double 3.6061019999999999
	.quad	0x3fa5464dc22ab25b              # double 0.041551999999999999
	.quad	0x3ff4b9a8cdea033e              # double 1.2953269999999999
	.quad	0x4022c40a4a05dd90              # double 9.3828910000000008
	.quad	0x401f45eaee2003ac              # double 7.8182790000000004
	.quad	0x400be36fc158fb44              # double 3.4860530000000001
	.quad	0x4017875a31a4bdba              # double 5.88218
	.quad	0x4021900ef1348b22              # double 8.7813639999999999
	.quad	0x4014527b2cc70868              # double 5.0805480000000003
	.quad	0x3fe7c473cd573648              # double 0.74273100000000003
	.quad	0x3fde6aa8eb463498              # double 0.47526000000000002
	.quad	0x4023fcf93bc0a06f              # double 9.9940890000000006
	.quad	0x3fdc73365881a155              # double 0.44453199999999998
	.quad	0x401f5323358f2e06              # double 7.8311890000000002
	.quad	0x400e16494d50ebab              # double 3.7608820000000001
	.quad	0x4021dbd2fa0d77b8              # double 8.9293440000000004
	.quad	0x40082c100e6afcce              # double 3.021515
	.quad	0x400736c69b5a63fa              # double 2.9017460000000002
	.quad	0x4021ab3f85510d39              # double 8.8344690000000003
	.quad	0x4020168bc169c23b              # double 8.0440349999999992
	.quad	0x3fe21d084e831ad2              # double 0.56604399999999999
	.quad	0x4023972d7f950b95              # double 9.7952689999999993
	.quad	0x3fecfcd035371972              # double 0.90586100000000003
	.quad	0x40059878b7a1c25d              # double 2.699449
	.quad	0x40232bdd11be6e65              # double 9.5856709999999996
	.quad	0x3ffb8113c68661ae              # double 1.7190129999999999
	.quad	0x3ffc3fa4c61d8623              # double 1.7655380000000001
	.quad	0x401805f2cb641701              # double 6.0058090000000002
	.quad	0x4011294d94078961              # double 4.2903349999999998
	.quad	0x401f70121682f944              # double 7.8594439999999999
	.quad	0x40179c165907d912              # double 5.9024289999999997
	.quad	0x40216abd5dc40075              # double 8.7084759999999992
	.quad	0x401d0db877ab3248              # double 7.2633989999999997
	.quad	0x40232ffda4052d66              # double 9.5937319999999993
	.quad	0x4021ef470eb24a6b              # double 8.9673390000000008
	.quad	0x3fedcfa69be09bb7              # double 0.93159800000000004
	.quad	0x3fd22a7bd48cb4af              # double 0.28384300000000001
	.quad	0x4010cf0068db8bac              # double 4.2021499999999996
	.quad	0x3fcc79fec99f1ae3              # double 0.222473
	.quad	0x3fb2f8055fbb517a              # double 0.074096999999999996
	.quad	0x401e804534bd76ee              # double 7.6252639999999996
	.quad	0x401cb58bc59b8024              # double 7.1772910000000003
	.quad	0x401ae0d66f0cfe15              # double 6.7195679999999998
	.quad	0x40106e3150dae3e7              # double 4.1076100000000002
	.quad	0x4009ab55ef1fddec              # double 3.2086600000000001
	.quad	0x4021a9b068123811              # double 8.8314240000000002
	.quad	0x401e3577d955714c              # double 7.5522150000000003
	.quad	0x401db9806f262889              # double 7.4311540000000003
	.quad	0x3fe5edca07f66e87              # double 0.68527700000000003
	.quad	0x4018830e7ff583a5              # double 6.1279849999999998
	.quad	0x4020743d89ce4a7b              # double 8.2270319999999995
	.quad	0x402060344c37e6f7              # double 8.1878989999999998
	.quad	0x401bdd94d0dcfcc6              # double 6.9663880000000002
	.quad	0x401467b4a2339c0f              # double 5.1012750000000002
	.quad	0x4015802645e4e69f              # double 5.375146
	.quad	0x401385fcc1871e6d              # double 4.8808470000000002
	.quad	0x4007d402d16b97ff              # double 2.9785210000000002
	.quad	0x4018d474107314cb              # double 6.2074740000000004
	.quad	0x3fb41d3aa369fcf4              # double 0.078571000000000002
	.quad	0x401dc3d3e4ef0282              # double 7.4412380000000002
	.quad	0x4013d6ce789e774f              # double 4.9597720000000001
	.quad	0x400b68661ae70c13              # double 3.4259759999999999
	.quad	0x40113bc0a06e9ff1              # double 4.3083520000000002
	.quad	0x40239def8487b99d              # double 9.8084679999999995
	.quad	0x4001a8c0ce91c8eb              # double 2.2073990000000001
	.quad	0x3fbc7c6327ed84d3              # double 0.111273
	.quad	0x40220cb70ac3a861              # double 9.0248340000000002
	.quad	0x40204fa43fe5c91d              # double 8.1555499999999999
	.quad	0x4022726dce39b457              # double 9.2234940000000005
	.quad	0x4022d01344806291              # double 9.4063970000000001
	.quad	0x402045dc6195464e              # double 8.1364470000000004
	.quad	0x4020e465c70435f0              # double 8.4460890000000006
	.quad	0x401b467a52ac753e              # double 6.8188259999999996
	.quad	0x4010c147f1305964              # double 4.1887509999999999
	.quad	0x40161ae70c1333b9              # double 5.5262719999999996
	.quad	0x40106e978d4fdf3b              # double 4.1079999999999997
	.quad	0x40212b1c219eb639              # double 8.5841989999999999
	.quad	0x4010f612839042d9              # double 4.2403050000000002
	.quad	0x401f05568e820e63              # double 7.7552130000000003
	.quad	0x4017c5db3397dd01              # double 5.943219
	.quad	0x40226e72756861e9              # double 9.2157169999999997
	.quad	0x4011f94ee392e1ef              # double 4.4934649999999996
	.quad	0x4004a4b7f5a5332b              # double 2.5804290000000001
	.quad	0x401907e846a5d6bf              # double 6.2577220000000002
	.quad	0x4021efd5cb790fb6              # double 8.9684279999999994
	.quad	0x4010238a3b57c4e3              # double 4.034707
	.quad	0x40016631f8a0902e              # double 2.1749000000000001
	.quad	0x400be093532e7b3e              # double 3.4846560000000002
	.quad	0x400cfed6ba8c5868              # double 3.6244329999999998
	.quad	0x402390789613d31c              # double 9.7821700000000006
	.quad	0x40226b6e0d23d4f1              # double 9.2098239999999993
	.quad	0x400619f4d37c1377              # double 2.7626740000000001
	.quad	0x401cd75cd0bb6ed6              # double 7.2103149999999996
	.quad	0x4009ab606b7aa25e              # double 3.2086800000000002
	.quad	0x4012271a3b14a904              # double 4.5381859999999996
	.quad	0x3fe9fa05143bf727              # double 0.81176999999999999
	.quad	0x401877ab324851a8              # double 6.1168639999999996
	.quad	0x40118dd22856605f              # double 4.3884970000000001
	.quad	0x400e205ff1d81f10              # double 3.7658079999999998
	.quad	0x4011809784ec636b              # double 4.375578
	.quad	0x3ff0ae5abc0e38a8              # double 1.042567
	.quad	0x40056e032db1e9f2              # double 2.6787169999999998
	.quad	0x4016fa12b5e529bb              # double 5.7442120000000001
	.quad	0x3fe48d2e514c22ee              # double 0.64223399999999997
	.quad	0x4019dae1049235f8              # double 6.463749
	.quad	0x40160f1b25f633ce              # double 5.5147519999999997
	.quad	0x401305a6c5d206c8              # double 4.7555189999999996
	.quad	0x400bc333764f11b6              # double 3.470313
	.quad	0x401668ac9f2fdb90              # double 5.6022210000000001
	.quad	0x3ffdb521144cbe1f              # double 1.8567210000000001
	.quad	0x3fbd65d3996fa82f              # double 0.11483500000000001
	.quad	0x4015fe9d0e991ff7              # double 5.4986459999999999
	.quad	0x400d7daf8df7a4e8              # double 3.6863700000000001
	.quad	0x401684f69ca9ef52              # double 5.6298469999999998
	.quad	0x3ff0d1b3f20a73f7              # double 1.0511969999999999
	.quad	0x3fe9f0b06c43f5f9              # double 0.81063099999999999
	.quad	0x3fe8ce69f05ea24d              # double 0.77519700000000002
	.quad	0x400c5e6bc621b7e1              # double 3.5461040000000001
	.quad	0x40170fd5454152b1              # double 5.7654620000000003
	.quad	0x3fe9bee3d5fdcdf7              # double 0.80455200000000004
	.quad	0x4019fe89a88ace25              # double 6.4985720000000002
	.quad	0x401532e72da122fb              # double 5.2997100000000001
	.quad	0x4022467cd0353719              # double 9.1376709999999992
	.quad	0x401501bda5119ce0              # double 5.2516999999999996
	.quad	0x4022ac8c0ce91c8f              # double 9.3370060000000006
	.quad	0x40087d5c31593e60              # double 3.0612110000000001
	.quad	0x401f866a11ec918e              # double 7.8812639999999998
	.quad	0x400b46d58c8eef1c              # double 3.4095870000000001
	.quad	0x3fff8148344c37e7              # double 1.969063
	.quad	0x400d2e557de0d66f              # double 3.647624
	.quad	0x40221b4330d73861              # double 9.0532470000000007
	.quad	0x400413458cd20afa              # double 2.5094099999999999
	.quad	0x4023c0e1f7d73c92              # double 9.8767239999999994
	.quad	0x40120219220ff541              # double 4.5020490000000004
	.quad	0x3ff11857afea3df7              # double 1.068443
	.quad	0x4011f736049ecb32              # double 4.4914170000000002
	.quad	0x3ff3dfc7e6b3fea0              # double 1.2421340000000001
	.quad	0x3ff8d6df3f961805              # double 1.552459
	.quad	0x3fca672fba01eeee              # double 0.20627400000000001
	.quad	0x40188a6c5d206c87              # double 6.1351789999999999
	.quad	0x4013a7d912556d1a              # double 4.9139140000000001
	.quad	0x4010f86876e1dead              # double 4.2425860000000002
	.quad	0x4004eb95e5aff2a1              # double 2.6150319999999998
	.quad	0x401e529563a9f384              # double 7.5806480000000001
	.quad	0x401be1526d8b1dd6              # double 6.9700410000000002
	.quad	0x401ca6ce789e774f              # double 7.1628970000000001
	.quad	0x3fba56eefa1e3eaf              # double 0.10288899999999999
	.quad	0x4017d8c32a8c9b84              # double 5.9616819999999997
	.quad	0x3ff2465f9591cd1c              # double 1.1421809999999999
	.quad	0x4019abc2fc69728a              # double 6.4177359999999997
	.quad	0x4012799d883ba344              # double 4.6187649999999998
	.quad	0x4003f7a0b5ed8d37              # double 2.4959120000000001
	.quad	0x40203d32e7b3d8e0              # double 8.119529
	.quad	0x3fe8bd2b6f19934f              # double 0.773092
	.quad	0x401108d8ec95bff0              # double 4.2586399999999998
	.quad	0x40205cb6848beb5b              # double 8.1810799999999997
	.quad	0x40231fd1dcd7060c              # double 9.5621480000000005
	.quad	0x4023e2da554b8bf0              # double 9.9430720000000008
	.quad	0x4018e84e831ad213              # double 6.2268619999999997
	.quad	0x4020bab990afe60c              # double 8.3646969999999996
	.quad	0x3ff832cf95d4e8fb              # double 1.512405
	.quad	0x401e09463cfb3312              # double 7.5090570000000003
	.quad	0x40027276fb09203a              # double 2.3058909999999999
	.quad	0x4019bea52ef911cf              # double 6.4361769999999998
	.quad	0x4002287442c7fbad              # double 2.2697530000000001
	.quad	0x3fdd676a7264a16a              # double 0.45943699999999998
	.quad	0x401c13e9ea140571              # double 7.0194470000000004
	.quad	0x3ffdd8b3f63c31df              # double 1.8654059999999999
	.quad	0x401cf0f12c27a637              # double 7.2352949999999998
	.quad	0x401eee158750c1b9              # double 7.7325039999999995
	.quad	0x3ff7620ccff21b3b              # double 1.461438
	.quad	0x4017ff077ccc0379              # double 5.9990519999999998
	.quad	0x4011b285ec3dab5c              # double 4.4243389999999998
	.quad	0x4012af6598e10cf6              # double 4.6712860000000003
	.quad	0x401d03254e6e221d              # double 7.2530720000000004
	.quad	0x3ff978e8ea39c51e              # double 1.5920190000000001
	.quad	0x4014e4d8fd5cb791              # double 5.223484
	.quad	0x4015b5a5fc7e6b40              # double 5.4273910000000001
	.quad	0x40043bc01a36e2eb              # double 2.529175
	.quad	0x4019529307af20ea              # double 6.3306389999999997
	.quad	0x400be05ccc89b0ee              # double 3.4845519999999999
	.quad	0x402105ddd2ae8e1d              # double 8.5114579999999993
	.quad	0x3ff365d28d002e23              # double 1.212359
	.quad	0x3fc579bfdf7e8039              # double 0.16777800000000001
	.quad	0x4013673ffac1d29e              # double 4.8508300000000002
	.quad	0x402029b60ae9680e              # double 8.081467
	.quad	0x3ff6ee47dc37a3db              # double 1.4331739999999999
	.quad	0x40114f8df7a4e7ab              # double 4.3276899999999996
	.quad	0x4009b89613d31b9b              # double 3.2151299999999998
	.quad	0x4022bd4a5df2239e              # double 9.3697079999999993
	.quad	0x3fa29b499d0203e6              # double 0.036340999999999998
	.quad	0x3fd6625a682b6284              # double 0.34975299999999998
	.quad	0x4020d87f66e86c66              # double 8.4228470000000009
	.quad	0x4013637329c347e9              # double 4.8471190000000002
	.quad	0x4022514a6eb91b3f              # double 9.1587709999999998
	.quad	0x3ffe1d4b6a619daa              # double 1.882152
	.quad	0x40131ef416bdb1a7              # double 4.7802280000000001
	.quad	0x401518904f6dfc5d              # double 5.2739880000000001
	.quad	0x4020137cdcca70d2              # double 8.038062
	.quad	0x3ff9ba1e3eaf6838              # double 1.607939
	.quad	0x40103c36113404ea              # double 4.0587999999999997
	.quad	0x4021d9858bc59b80              # double 8.9248469999999998
	.quad	0x401d731832fcac8e              # double 7.3623969999999996
	.quad	0x400bc73471f79421              # double 3.4722680000000001
	.quad	0x4018e31b152f3c2e              # double 6.2217830000000003
	.quad	0x4014dd9a52263d81              # double 5.2164089999999996
	.quad	0x40124c7c5260f5e4              # double 4.5746929999999999
	.quad	0x3fe73220791c4b90              # double 0.72486899999999999
	.quad	0x3fde1ffb480a5acd              # double 0.47070200000000001
	.quad	0x401ccd9cae21101b              # double 7.200793
	.quad	0x4006676ea4228998              # double 2.8005040000000001
	.quad	0x3fbcb220791c4b90              # double 0.112093
	.quad	0x400bbc5b4aa9717e              # double 3.466971
	.quad	0x3ffdc00c9539b888              # double 1.8593869999999999
	.quad	0x40142ce13f4a98ab              # double 5.0438280000000004
	.quad	0x401b1e84f09528f2              # double 6.7798040000000004
	.quad	0x40145aa3f034b0e2              # double 5.0885160000000003
	.quad	0x40212491d14e3bcd              # double 8.5714249999999996
	.quad	0x4022c395a294141f              # double 9.3820010000000007
	.quad	0x401e1a098915bdd7              # double 7.5254269999999996
	.quad	0x401e9ff8ec0f8833              # double 7.6562229999999997
	.quad	0x4014c452c59fb1e2              # double 5.1917220000000004
	.quad	0x3fede9bbf0dc768e              # double 0.934782
	.quad	0x401a400c521dda06              # double 6.5625470000000004
	.quad	0x401ee1240746455f              # double 7.7198640000000002
	.quad	0x401b8e0fc2eba27b              # double 6.8887320000000001
	.quad	0x4021c797cc39ffd6              # double 8.8898299999999999
	.quad	0x401d8ed56b00ffda              # double 7.3894859999999998
	.quad	0x3ff88de19fc2a887              # double 1.5346390000000001
	.quad	0x401fac6ac215b9a6              # double 7.9183760000000003
	.quad	0x400ee2ba5a038195              # double 3.8607070000000001
	.quad	0x401e142aed139431              # double 7.5196949999999995
	.quad	0x3fe063f141205bc0              # double 0.51219999999999999
	.quad	0x4019a5479d4d8341              # double 6.4114060000000004
	.quad	0x40162e28ed5f138c              # double 5.5450780000000002
	.quad	0x402163ef06b37868              # double 8.6951830000000001
	.quad	0x3ff2f0142f61ed5b              # double 1.183613
	.quad	0x401692af1455219b              # double 5.6432460000000004
	.quad	0x400959c304ccee5b              # double 3.1688290000000001
	.quad	0x4001a311e85fd04a              # double 2.2046239999999999
	.quad	0x4022134d48882f0e              # double 9.0376989999999999
	.quad	0x40214c9b845564b6              # double 8.6496239999999993
	.quad	0x400f2278d0cc35ce              # double 3.891832
	.quad	0x400832b2bfdb4cc2              # double 3.0247549999999999
	.quad	0x4022b9af3a14cec4              # double 9.3626649999999997
	.quad	0x3ffb9701d9f4d37c              # double 1.724367
	.quad	0x3ffd3a71151611ba              # double 1.8267679999999999
	.quad	0x40005bbc6eb0b7c3              # double 2.0447929999999999
	.quad	0x402098d7e02645e5              # double 8.2985220000000002
	.quad	0x400f20064a9cdc44              # double 3.8906369999999999
	.quad	0x3ff1ebdf4c2b51bd              # double 1.1200859999999999
	.quad	0x3fff14ab1d4f9c20              # double 1.9425460000000001
	.quad	0x4022bed56b00ffda              # double 9.3727219999999996
	.quad	0x401b99e8815e3971              # double 6.9003009999999998
	.quad	0x3ff840a6b93ccd10              # double 1.515784
	.quad	0x4022896dd26b723f              # double 9.2684160000000002
	.quad	0x400f07a17f4128bf              # double 3.8787259999999999
	.quad	0x400b688b545c78a7              # double 3.4260470000000001
	.quad	0x4018468e3ef28416              # double 6.0689019999999996
	.quad	0x401b67e90ff97247              # double 6.8514749999999998
	.quad	0x4013a7985271bcdc              # double 4.9136670000000002
	.quad	0x401c8148ba83f4ed              # double 7.1262540000000003
	.quad	0x401d83da2f8bdec6              # double 7.378762
	.quad	0x400020ebedfa43fe              # double 2.0160749999999998
	.quad	0x40180a604e1e7104              # double 6.0101329999999997
	.quad	0x3fc0d4aa10e02214              # double 0.13149
	.quad	0x40037f8012dfd695              # double 2.4372560000000001
	.quad	0x401368e3ac0c62e5              # double 4.8524310000000002
	.quad	0x40125b68cef672b9              # double 4.5892670000000004
	.quad	0x40137aefb2aae297              # double 4.8700549999999998
	.quad	0x401fbe9ab29e4d5e              # double 7.9361370000000004
	.quad	0x4006bf6d330941c8              # double 2.8434699999999999
	.quad	0x3ffcdcd8127b2cc7              # double 1.803917
	.quad	0x40142276fb09203a              # double 5.0336569999999998
	.quad	0x402322c4d65e4606              # double 9.5679079999999992
	.quad	0x3fee74afd5454153              # double 0.95174400000000003
	.quad	0x400912756861e929              # double 3.1340129999999999
	.quad	0x3ff778cc9a77e5eb              # double 1.4669920000000001
	.quad	0x3ff1b83dff3f0fe0              # double 1.1074809999999999
	.quad	0x4008d1a6d698fe69              # double 3.1023689999999999
	.quad	0x40122aa369fcf3dc              # double 4.541639
	.quad	0x4014e530ced4e4c9              # double 5.2238189999999998
	.quad	0x401bd91bc5586445              # double 6.9620199999999999
	.quad	0x4010006cca2db61c              # double 4.0004150000000003
	.quad	0x3ff5c9c0a8d21bc1              # double 1.361756
	.quad	0x3fce7bb2fec56d5d              # double 0.23815
	.quad	0x40166f41f212d773              # double 5.6086499999999999
	.quad	0x401b54df04ddb552              # double 6.8328819999999997
	.quad	0x401d289fc6da4485              # double 7.2896720000000004
	.quad	0x40114879c4113c68              # double 4.3207769999999996
	.quad	0x401a969be09bb6aa              # double 6.6470789999999997
	.quad	0x4007061e92923e5c              # double 2.8779880000000002
	.quad	0x4001c0192a737111              # double 2.218798
	.quad	0x4011366d373affb0              # double 4.3031509999999997
	.quad	0x3fe50db8bac710cb              # double 0.65792499999999998
	.quad	0x4014343c3a42f1ed              # double 5.0510109999999999
	.quad	0x40205883fd502257              # double 8.1728819999999995
	.quad	0x4013e724fa8b4bf9              # double 4.975727
	.quad	0x4011c857f3061c7a              # double 4.4456480000000003
	.quad	0x4020985015c20924              # double 8.2974859999999992
	.quad	0x3fe77fd82773e250              # double 0.73435600000000001
	.quad	0x3ff76e3715400325              # double 1.4644079999999999
	.quad	0x3ffa007789a45910              # double 1.6251139999999999
	.quad	0x4018b15c63ae2542              # double 6.1732040000000001
	.quad	0x401444a90470a809              # double 5.0670510000000002
	.quad	0x40080e0221426fe7              # double 3.00684
	.quad	0x40047138bcdfefbf              # double 2.5552839999999999
	.quad	0x40125c99ae924f22              # double 4.5904299999999996
	.quad	0x401eba45d41fa765              # double 7.6819069999999998
	.quad	0x400c2054690de093              # double 3.5157859999999999
	.quad	0x4023e11fd5885d31              # double 9.9396959999999996
	.quad	0x40124e7ff583a53c              # double 4.5766600000000004
	.quad	0x4013e6b1e5c0b999              # double 4.9752879999999999
	.quad	0x40144842b734b513              # double 5.0705669999999996
	.quad	0x4015dbf6a0dbad3a              # double 5.4648079999999997
	.quad	0x400c871e6cd29132              # double 3.565976
	.quad	0x400f91f8a0902de0              # double 3.946275
	.quad	0x401697b6bb129025              # double 5.6481579999999996
	.quad	0x4020163fdd65a145              # double 8.0434560000000008
	.quad	0x3ffe655193708aad              # double 1.8997360000000001
	.quad	0x400e0d31fcd24e16              # double 3.756443
	.quad	0x40136e6ff7dfa00e              # double 4.8578489999999999
	.quad	0x401a21a048e043a2              # double 6.5328379999999999
	.quad	0x40142ba16e7a311f              # double 5.0426080000000004
	.quad	0x401677635e74299e              # double 5.6165900000000004
	.quad	0x402334443914f484              # double 9.6020830000000003
	.quad	0x3fb2cb85a4f00ef1              # double 0.073417999999999997
	.quad	0x402365a4acf312b2              # double 9.6985220000000005
	.quad	0x400678d36b4c7f35              # double 2.8089970000000002
	.quad	0x401c03ce63a5c1c6              # double 7.003717
	.quad	0x4022cb358f2e05cd              # double 9.3968930000000003
	.quad	0x40039f38c5436b90              # double 2.4527450000000002
	.quad	0x3fc79f5dfeb8d823              # double 0.18455099999999999
	.quad	0x401900ef99806f26              # double 6.2509139999999999
	.quad	0x4020a9fcd24e160e              # double 8.3320070000000008
	.quad	0x3feff824d4cb9ecf              # double 0.99904099999999995
	.quad	0x3ff503382e44b6e9              # double 1.313286
	.quad	0x4012508feac42e99              # double 4.5786740000000004
	.quad	0x402206a67e846a5d              # double 9.0129889999999993
	.quad	0x401f9d45a5fc7e6b              # double 7.9035859999999997
	.quad	0x4018743137b07076              # double 6.1134690000000003
	.quad	0x400ca7000431bde8              # double 3.5815429999999999
	.quad	0x402142ac53b0813d              # double 8.6302210000000006
	.quad	0x400e05c03361565c              # double 3.7528079999999999
	.quad	0x400ec2b3892ee84b              # double 3.8450690000000001
	.quad	0x4010364b662fdfc2              # double 4.0530220000000003
	.quad	0x3feb6b6bf444e0db              # double 0.85686300000000004
	.quad	0x3ffd8c8366516db1              # double 1.846805
	.quad	0x4023306594af4f0e              # double 9.5945250000000009
	.quad	0x400ca37718655193              # double 3.5798169999999998
	.quad	0x4022c0574b407033              # double 9.3756660000000007
	.quad	0x3ff336d00b45ae60              # double 1.200882
	.quad	0x401fdb018611fd59              # double 7.9638730000000004
	.quad	0x4015c6adf71eafee              # double 5.4440229999999996
	.quad	0x4016a7414a4d2b2c              # double 5.663335
	.quad	0x3ff629c23b7952d2              # double 1.385195
	.quad	0x40215c4a4095f245              # double 8.6802539999999997
	.quad	0x40201528ae74f2f1              # double 8.0413259999999998
	.quad	0x3fd804b7f5a5332b              # double 0.37528800000000001
	.quad	0x40009f85d744f5d3              # double 2.0778919999999999
	.quad	0x400a5faf42784a94              # double 3.2967209999999998
	.quad	0x3ffaeee0f3cb3e57              # double 1.6833199999999999
	.quad	0x40097e5753a3ec03              # double 3.18669
	.quad	0x4020d955d5f56a7b              # double 8.4244830000000004
	.quad	0x402034a2b9d3cbc5              # double 8.1028040000000008
	.quad	0x40134e9680e06530              # double 4.826746
	.quad	0x3fe44e8533b10774              # double 0.63458499999999995
	.quad	0x40111dda8bd230ba              # double 4.2791540000000001
	.quad	0x401cdc94f69ca9ef              # double 7.2154119999999997
	.quad	0x3ffddc6195464dc2              # double 1.866304
	.quad	0x400a001d5c31593e              # double 3.2500559999999998
	.quad	0x3ff7dbc98a222d51              # double 1.4911589999999999
	.quad	0x401a7856605ee569              # double 6.6175170000000003
	.quad	0x3fea872b020c49ba              # double 0.82899999999999996
	.quad	0x40102ea52ef911cf              # double 4.0455519999999998
	.quad	0x3ffdac636b096356              # double 1.854587
	.quad	0x401ad0bdcad14a0a              # double 6.7038489999999999
	.quad	0x3fe5770153bd1676              # double 0.67077699999999996
	.quad	0x40231963779e9d0f              # double 9.5495870000000007
	.quad	0x3fe8a1a0cf1800a8              # double 0.76973000000000003
	.quad	0x4013369878316a05              # double 4.8033159999999997
	.quad	0x3ff7f4bc6a7ef9db              # double 1.49725
	.quad	0x402003964a59c066              # double 8.0070060000000005
	.quad	0x4021c767d34df04e              # double 8.8894640000000003
	.quad	0x400cfa91105e1c15              # double 3.622347
	.quad	0x4021c287e38eb032              # double 8.8799430000000008
	.quad	0x40206281ba7fc32f              # double 8.1923960000000005
	.quad	0x402019ad2135daad              # double 8.0501489999999993
	.quad	0x40145872b020c49c              # double 5.0863750000000003
	.quad	0x3ffa45cee17a02fb              # double 1.6420429999999999
	.quad	0x4003469270b06c44              # double 2.409459
	.quad	0x4015b721d53cddd7              # double 5.4288400000000001
	.quad	0x401c0ef8487b99d4              # double 7.0146189999999997
	.quad	0x400069784ec636b1              # double 2.0514990000000002
	.quad	0x40124ec95bff0457              # double 4.5769399999999996
	.quad	0x401885653c9abb02              # double 6.1302690000000002
	.quad	0x402107fa3b9ae0c1              # double 8.5155809999999992
	.quad	0x3fedd8cd20afa2f0              # double 0.93271499999999996
	.quad	0x40208396f21f6cad              # double 8.2570110000000003
	.quad	0x3ff5a69270b06c44              # double 1.353167
	.quad	0x3fe7d771001d5c31              # double 0.74504899999999996
	.quad	0x4022890385c67dfe              # double 9.2676049999999996
	.quad	0x40232f090f733a8a              # double 9.5918659999999996
	.quad	0x4020b40bdcad14a1              # double 8.3516530000000007
	.quad	0x401f5267c6b8b695              # double 7.8304739999999997
	.quad	0x400d9aaac1094a2c              # double 3.7005210000000002
	.quad	0x3feb09dcf893faf4              # double 0.84495399999999998
	.quad	0x40074bb5e0f7fcfc              # double 2.9119679999999999
	.quad	0x4011e57646ae3a3b              # double 4.4740840000000004
	.quad	0x3fd4416bdb1a6d6a              # double 0.31649300000000002
	.quad	0x401a0780baa582dc              # double 6.5073270000000001
	.quad	0x3fe456cf850df15a              # double 0.63559699999999997
	.quad	0x40212d573647baaa              # double 8.5885560000000005
	.quad	0x401b21ec0b567558              # double 6.7831270000000004
	.quad	0x4000ecf41f212d77              # double 2.1156999999999999
	.quad	0x400b4ef34d6a161e              # double 3.4135499999999999
	.quad	0x4010bb4e9813879c              # double 4.1829169999999998
	.quad	0x40128f7be121ee67              # double 4.6401209999999997
	.quad	0x402304d76ab58080              # double 9.5094560000000001
	.quad	0x400e2ae1049235f8              # double 3.770937
	.quad	0x401685a4acf312b2              # double 5.6305110000000003
	.quad	0x40113f049a9973da              # double 4.3115410000000001
	.quad	0x4022e34091c08744              # double 9.4438519999999996
	.quad	0x40212d5992428d43              # double 8.5885739999999995
	.quad	0x3fd6988d2a1f8e3b              # double 0.35306100000000001
	.quad	0x3ffd243bb40b34e7              # double 1.8213459999999999
	.quad	0x40204f7a2cecc815              # double 8.1552290000000003
	.quad	0x4001a1719f7f8ca8              # double 2.20383
	.quad	0x401520b955f7835a              # double 5.2819570000000002
	.quad	0x3fb06412cf0f9d2c              # double 0.064027000000000001
	.quad	0x3feb26d04e618ce3              # double 0.84848800000000002
	.quad	0x40227969a0ad8a11              # double 9.2371339999999993
	.quad	0x401cf1b1960fa15e              # double 7.2360290000000003
	.quad	0x40048f5a964e8b7e              # double 2.5699969999999999
	.quad	0x40223d1ce28ed5f1              # double 9.1193609999999996
	.quad	0x40225b2441fea811              # double 9.1780109999999997
	.quad	0x3ffa9625a682b628              # double 1.6616569999999999
	.quad	0x3fd442b734b51373              # double 0.31657200000000002
	.quad	0x401ab45e9185cee1              # double 6.6761419999999996
	.quad	0x40223be9c886162f              # double 9.1170179999999998
	.quad	0x3ff5a038194c0160              # double 1.3516159999999999
	.quad	0x3ffe9591cd1c7de5              # double 1.911516
	.quad	0x4009fe58e64b2314              # double 3.249193
	.quad	0x40220eb4e981387a              # double 9.0287240000000004
	.quad	0x3ff998f2e05ccc8a              # double 1.5998410000000001
	.quad	0x401976b8b69552e3              # double 6.365939
	.quad	0x40150851eb851eb8              # double 5.2581249999999997
	.quad	0x4002fd8adab9f55a              # double 2.3738000000000001
	.quad	0x3fe3586ec17ebaf1              # double 0.604545
	.quad	0x401c3901d19157ac              # double 7.0556710000000002
	.quad	0x402255d15f02c4d6              # double 9.1676129999999993
	.quad	0x4013ec7bcc2938de              # double 4.9809409999999996
	.quad	0x4016e4b167ec7864              # double 5.7233330000000002
	.quad	0x40134f10667f90da              # double 4.8272110000000001
	.quad	0x40235c08fa7a8501              # double 9.6797559999999993
	.quad	0x4001f6ca46e08f21              # double 2.2455029999999998
	.quad	0x400531904b3c3e75              # double 2.6492010000000001
	.quad	0x3feb9876188b1141              # double 0.86236100000000004
	.quad	0x40152e6ddf00abf7              # double 5.2953409999999996
	.quad	0x401db16e37154003              # double 7.4232719999999999
	.quad	0x40152390c9107faa              # double 5.284732
	.quad	0x40141d82fd75e204              # double 5.0288199999999996
	.quad	0x401140d130df9bdc              # double 4.3132979999999996
	.quad	0x4005245cbbc2b94e              # double 2.6427550000000002
	.quad	0x4000313c68661ae7              # double 2.024041
	.quad	0x400803bbc6eb0b7c              # double 3.0018229999999999
	.quad	0x3fd3ec13b9f127f6              # double 0.31128400000000001
	.quad	0x401ae4080f98fa37              # double 6.7226869999999996
	.quad	0x401b4c404a72ead9              # double 6.8244639999999999
	.quad	0x401a484be40420f7              # double 6.5706020000000001
	.quad	0x40123fd08d4bad7d              # double 4.5623189999999996
	.quad	0x4023dd8d5842b735              # double 9.9327190000000005
	.quad	0x400f0925785f8d2e              # double 3.8794659999999999
	.quad	0x4014a3b1d0c80410              # double 5.1598579999999998
	.quad	0x3ffdb291b823c85c              # double 1.856096
	.quad	0x4020fd245b291b82              # double 8.4944179999999996
	.quad	0x3fc6f733a8a3f898              # double 0.179419
	.quad	0x3fa28e2e2b8c75c5              # double 0.036241000000000002
	.quad	0x40185fdb8fde2ef5              # double 6.0936110000000001
	.quad	0x4011617ca2120e1f              # double 4.3452019999999996
	.quad	0x402216ce789e774f              # double 9.0445440000000001
	.quad	0x40224d86876e1deb              # double 9.1514170000000004
	.quad	0x4018253e2d6238da              # double 6.0363699999999998
	.quad	0x400b2bf3bea91d9b              # double 3.396461
	.quad	0x3ffc4efaa4766c6e              # double 1.769282
	.quad	0x40201102bc72e276              # double 8.0332240000000006
	.quad	0x402164c51116a8b9              # double 8.6968160000000001
	.quad	0x40010fc504816f00              # double 2.1326999999999998
	.quad	0x4014899567dbb16c              # double 5.1343589999999999
	.quad	0x400eb361565c2d28              # double 3.8375880000000002
	.quad	0x4010d4d7f0ed3d86              # double 4.2078550000000003
	.quad	0x401f7854cdb7ae58              # double 7.8675110000000003
	.quad	0x401d6ba8826aa8eb              # double 7.3551349999999998
	.quad	0x4022323d9231c64c              # double 9.0981260000000006
	.quad	0x4013b4d551d68c69              # double 4.9265949999999998
	.quad	0x4023ef28848387df              # double 9.9671059999999993
	.quad	0x402015d5b24e9c45              # double 8.0426459999999995
	.quad	0x400f868cac4b4d05              # double 3.9406979999999998
	.quad	0x400372d01c0ca601              # double 2.4310610000000001
	.quad	0x4011b5204af92296              # double 4.4268809999999998
	.quad	0x4006a1272c94b381              # double 2.8286880000000001
	.quad	0x3ff8387ad080b674              # double 1.5137890000000001
	.quad	0x402061c9b4139863              # double 8.1909919999999996
	.quad	0x401f0c6540cc78ea              # double 7.762105
	.quad	0x40137295e9e1b08a              # double 4.8619000000000003
	.quad	0x4014b74a771c970f              # double 5.1789949999999996
	.quad	0x400b57fe08aefb2b              # double 3.4179650000000001
	.quad	0x4022f8507a6bd6e9              # double 9.4849890000000005
	.quad	0x401adee88df3732a              # double 6.7176840000000002
	.quad	0x3ff8d91cd1c7de51              # double 1.5530060000000001
	.quad	0x4009cbcdfefbf402              # double 3.2245140000000001
	.quad	0x3ff319ab6d00b45b              # double 1.193767
	.quad	0x400c7bbf0dc768e0              # double 3.5604230000000001
	.quad	0x401e753dea465a57              # double 7.6144939999999996
	.quad	0x401bced41b75a74c              # double 6.951981
	.quad	0x3ffb00f98fa37692              # double 1.687738
	.quad	0x400aaf66a5508701              # double 3.335645
	.quad	0x40167828c36da87a              # double 5.617343
	.quad	0x4012b58e64b23140              # double 4.6773009999999999
	.quad	0x4008b188f42fe825              # double 3.086687
	.quad	0x40058d5cfaacd9e8              # double 2.6940249999999999
	.quad	0x40228ef6b5d462c3              # double 9.2792259999999995
	.quad	0x3fce56473471f794              # double 0.237008
	.quad	0x401772a38b8ae31d              # double 5.8619519999999996
	.quad	0x400e230caa326e11              # double 3.7671139999999999
	.quad	0x400be759253543af              # double 3.4879630000000001
	.quad	0x401393508f648c72              # double 4.8938620000000004
	.quad	0x40205815c209246c              # double 8.1720410000000001
	.quad	0x3ff9b000431bde83              # double 1.605469
	.quad	0x4015edd3bafd9770              # double 5.482253
	.quad	0x40210f42784a9479              # double 8.5298040000000004
	.quad	0x400bee6c8f755369              # double 3.4914179999999999
	.quad	0x401600cf9e3864cb              # double 5.5007919999999997
	.quad	0x4021536b4c7f3494              # double 8.6629280000000008
	.quad	0x401792ddbdb5d895              # double 5.8934240000000004
	.quad	0x4015259b3d07c84b              # double 5.2867249999999997
	.quad	0x40017cdb37c99ae9              # double 2.1859649999999999
	.quad	0x4012f49d7ba6698c              # double 4.7388820000000003
	.quad	0x400b414b599aa609              # double 3.406882
	.quad	0x401592f6e82949a5              # double 5.3935199999999996
	.quad	0x401b0949a5657fb7              # double 6.7590700000000004
	.quad	0x4011b5a038194c01              # double 4.4273689999999997
	.quad	0x4020fb0b39192642              # double 8.4903200000000005
	.quad	0x401b2b8b26394fad              # double 6.7925230000000001
	.quad	0x3f905ee568a50508              # double 0.015987000000000001
	.quad	0x4007b0917d6b65aa              # double 2.9612150000000002
	.quad	0x3ff5505af6c69b5a              # double 1.3321179999999999
	.quad	0x3ff954c22ee4191a              # double 1.5831930000000001
	.quad	0x401cc8ad68837718              # double 7.1959739999999996
	.quad	0x3ff6da61e0c5a815              # double 1.4283159999999999
	.quad	0x402046bb341e14be              # double 8.138147
	.quad	0x401426dc5d638866              # double 5.0379500000000004
	.quad	0x401c639d1b2e59b0              # double 7.0972790000000003
	.quad	0x401b52e6644d8772              # double 6.8309569999999997
	.quad	0x3ff1b7985271bcdc              # double 1.1073230000000001
	.quad	0x40221dc1c6088d6d              # double 9.0581189999999996
	.quad	0x3fe1d846e8f29d41              # double 0.55765100000000001
	.quad	0x401ad6262cba732e              # double 6.70913
	.quad	0x40202b12b1b36bd3              # double 8.0841270000000005
	.quad	0x4005fb089a027525              # double 2.7475749999999999
	.quad	0x40171b912dba4d6e              # double 5.7769209999999998
	.quad	0x3fefa2749f0e4da1              # double 0.98858100000000004
	.quad	0x4001dc393682730c              # double 2.2325309999999998
	.quad	0x40226096feb4a665              # double 9.1886519999999994
	.quad	0x4016f40c8472c0e8              # double 5.7383290000000002
	.quad	0x401a2008a697aede              # double 6.5312830000000002
	.quad	0x4006785f06f69446              # double 2.8087749999999998
	.quad	0x401d02ea52ef911d              # double 7.252847
	.quad	0x401ad060bb2bba99              # double 6.7034940000000001
	.quad	0x40001ee2435696e6              # double 2.0150800000000002
	.quad	0x40102d3cff64cf8d              # double 4.0441779999999996
	.quad	0x4016cfb69984a0e4              # double 5.7028449999999999
	.quad	0x402347cd67fd3f5b              # double 9.6402389999999993
	.quad	0x4023202730c67169              # double 9.562799
	.quad	0x4006dcf355cd91ef              # double 2.8578860000000001
	.quad	0x401a2bfd2e946801              # double 6.5429579999999996
	.quad	0x40208781626b2f23              # double 8.264659
	.quad	0x4016e851654d61b3              # double 5.7268730000000003
	.quad	0x4006380d3cff64d0              # double 2.7773690000000002
	.quad	0x40007394317acc4f              # double 2.056435
	.quad	0x401baee568a50508              # double 6.9207970000000003
	.quad	0x400eb0ba6266fd65              # double 3.836293
	.quad	0x3fe814e1a3f4666f              # double 0.75254900000000002
	.quad	0x3ff0713ad5bee3d6              # double 1.027644
	.quad	0x40221ab8e8ea39c5              # double 9.0521919999999998
	.quad	0x402280ee28672757              # double 9.2518170000000008
	.quad	0x40093e4b44a1f080              # double 3.1554169999999999
	.quad	0x400886e5471715c6              # double 3.0658669999999999
	.quad	0x4012b5e3da2f8bdf              # double 4.6776270000000002
	.quad	0x40130bef49cf56eb              # double 4.7616550000000002
	.quad	0x40159dc725c3dee8              # double 5.4040800000000004
	.quad	0x4012d2c5e2cdc012              # double 4.7058330000000002
	.quad	0x3fe96a4ca4f440af              # double 0.79422599999999999
	.quad	0x4022c59e1f3a57eb              # double 9.3859720000000006
	.quad	0x4009994d94078961              # double 3.1998549999999999
	.quad	0x4023e1e83e425aee              # double 9.9412249999999993
	.quad	0x4000cb734b51372a              # double 2.0993409999999999
	.quad	0x3ff1dd084e831ad2              # double 1.116463
	.quad	0x4012008398a65493              # double 4.500502
	.quad	0x4023f8ab92c06184              # double 9.9856839999999991
	.quad	0x401691333b96af04              # double 5.6417970000000004
	.quad	0x3fff44f7ec3547e0              # double 1.9543379999999999
	.quad	0x3ffe2ce358298cc1              # double 1.8859589999999999
	.quad	0x40088af251c193b4              # double 3.0678450000000002
	.quad	0x3fff24388ebcc6c5              # double 1.9463429999999999
	.quad	0x3fd9f672b884406c              # double 0.405667
	.quad	0x4005a66559f6ec5b              # double 2.706248
	.quad	0x4002ad74927913e8              # double 2.334695
	.quad	0x4014622b7baecd08              # double 5.0958690000000004
	.quad	0x401704ed2cbea4ec              # double 5.7548110000000001
	.quad	0x40171e586013660e              # double 5.7796339999999997
	.quad	0x3ff313b4b2fa93af              # double 1.1923109999999999
	.quad	0x400aef4a558ea7ce              # double 3.366841
	.quad	0x4015852bd3c36113              # double 5.3800499999999998
	.quad	0x402190ee49f51698              # double 8.7830680000000001
	.quad	0x4021f28ed5f138bd              # double 8.9737460000000002
	.quad	0x4003067381d7dbf5              # double 2.3781500000000002
	.quad	0x4023d9620685553f              # double 9.924576
	.quad	0x4011c4bad7d7c2ca              # double 4.4421189999999999
	.quad	0x401e4858ff759685              # double 7.5706519999999999
	.quad	0x401745ca6ca03c4b              # double 5.818155
	.quad	0x4018a8d002e23290              # double 6.1648560000000003
	.quad	0x3ff7f6f19934efcc              # double 1.497789
	.quad	0x400f0f6512a94ff0              # double 3.882517
	.quad	0x40144c91d14e3bcd              # double 5.0747749999999998
	.quad	0x3fe8f6af89c5e6ff              # double 0.78011299999999995
	.quad	0x400f2a0a0f4d7add              # double 3.895527
	.quad	0x4003e3a322af5771              # double 2.486151
	.quad	0x3ffff3ebbfd71b04              # double 1.9970509999999999
	.quad	0x402389e6256366d8              # double 9.7693340000000006
	.quad	0x3ff4c7cb70ac3a86              # double 1.298778
	.quad	0x3ff638023a6ce358              # double 1.388674
	.quad	0x4010e97e56473472              # double 4.228021
	.quad	0x4012987a072d1aeb              # double 4.6489029999999998
	.quad	0x4023423cc8de2ac3              # double 9.6293699999999997
	.quad	0x400cfbd8be7296f6              # double 3.6229719999999999
	.quad	0x4019e7ff583a53b9              # double 6.4765600000000001
	.quad	0x40067784662bae04              # double 2.8083580000000001
	.quad	0x40041a74c09c3ce2              # double 2.512918
	.quad	0x3ff18fb86f47b678              # double 1.097588
	.quad	0x3fe1384fd2a62aa2              # double 0.53812400000000005
	.quad	0x3fe7341bfbdf090f              # double 0.72511099999999995
	.quad	0x401fa3c105186db5              # double 7.9099159999999999
	.quad	0x4021a69c23b7952d              # double 8.8254099999999998
	.quad	0x3fdb48cb4aec8d5c              # double 0.42631799999999997
	.quad	0x401e67603925bb7b              # double 7.6009529999999996
	.quad	0x3ffef2a27f1b6912              # double 1.934237
	.quad	0x3fd59973d9ec7000              # double 0.33749099999999999
	.quad	0x4015703fb3743633              # double 5.3596180000000002
	.quad	0x401aebd6e8af8162              # double 6.7303119999999996
	.quad	0x4022b0509bf9c62a              # double 9.3443649999999998
	.quad	0x4001adb7281fd9ba              # double 2.209822
	.quad	0x401980b673c4f3ba              # double 6.3756959999999996
	.quad	0x40227ef8487b99d4              # double 9.2479879999999994
	.quad	0x3fd882be8bc169c2              # double 0.38297999999999999
	.quad	0x40168a7b9170d62c              # double 5.6352370000000001
	.quad	0x40104fa43fe5c91d              # double 4.0777749999999999
	.quad	0x4018db4d056c508b              # double 6.2141609999999998
	.quad	0x400a25c03361565c              # double 3.2684329999999999
	.quad	0x40239b7fe08aefb3              # double 9.8037100000000005
	.quad	0x4003092ddbdb5d89              # double 2.3794819999999999
	.quad	0x401804ccab3edd8b              # double 6.0046869999999997
	.quad	0x4023aa788db0574b              # double 9.8329509999999995
	.quad	0x401e9530489d27c4              # double 7.6456920000000004
	.quad	0x40222d30ced4e4c9              # double 9.0882629999999995
	.quad	0x40014ff08893b7d8              # double 2.1640329999999999
	.quad	0x401557dbb16c1e36              # double 5.3357989999999997
	.quad	0x401c0bb3c81908e6              # double 7.0114280000000004
	.quad	0x3ff2ed44dca8e2e3              # double 1.1829270000000001
	.quad	0x40153c6a7ef9db23              # double 5.3090000000000002
	.quad	0x40183871e6cd2913              # double 6.0551219999999999
	.quad	0x40228d028a1dfb94              # double 9.2754100000000008
	.quad	0x4011e6f60e0eb67c              # double 4.4755479999999999
	.quad	0x4000d1b0ccbc05d5              # double 2.1023879999999999
	.quad	0x40231145b6c3760c              # double 9.533735
	.quad	0x4012000eae18ac9f              # double 4.5000559999999998
	.quad	0x40149d56f32bdc27              # double 5.1536520000000001
	.quad	0x400365082cf52b91              # double 2.4243320000000002
	.quad	0x3fefabe27179bfdf              # double 0.98973199999999994
	.quad	0x4012084a515ce9e6              # double 4.5080960000000001
	.quad	0x4022d50374ff865d              # double 9.4160419999999991
	.quad	0x40130ebcc6c54bcf              # double 4.764392
	.quad	0x4023a3f727136a40              # double 9.8202449999999998
	.quad	0x401d226f1dc50ce5              # double 7.2836270000000001
	.quad	0x3ff4c0baa582dbe8              # double 1.297053
	.quad	0x3ff8fc17a89331a1              # double 1.5615460000000001
	.quad	0x3fe4134ce3de614a              # double 0.62735600000000002
	.quad	0x401171cffeb074a7              # double 4.3611449999999996
	.quad	0x4021d68fb00bcbe6              # double 8.9190649999999998
	.quad	0x4014163baba7b917              # double 5.021712
	.quad	0x4021e7240fa9c12f              # double 8.9514469999999999
	.quad	0x4003ac6d612c6ac2              # double 2.459193
	.quad	0x40221280cf9e3865              # double 9.0361390000000004
	.quad	0x402244026cc1ca3a              # double 9.1328309999999995
	.quad	0x401023b688bda943              # double 4.0348759999999997
	.quad	0x40133460aa64c2f8              # double 4.8011499999999998
	.quad	0x40086b13dfb0d520              # double 3.0522840000000002
	.quad	0x401dbe086bdf4c2b              # double 7.4355789999999997
	.quad	0x401001693c03bc4d              # double 4.0013779999999999
	.quad	0x401e5cb2d905c033              # double 7.5905259999999996
	.quad	0x400b8555c52e72da              # double 3.440105
	.quad	0x40231178d4fdf3b6              # double 9.5341249999999995
	.quad	0x4007e4de3b8a19ca              # double 2.9867520000000001
	.quad	0x401c2f383f0bae8a              # double 7.0461130000000001
	.quad	0x40033be9856a37ac              # double 2.4042539999999999
	.quad	0x402034db163baba8              # double 8.1032340000000005
	.quad	0x3fbdd3ed527e5215              # double 0.11651499999999999
	.quad	0x3ffed21ccd962068              # double 1.9262969999999999
	.quad	0x4016340789613d32              # double 5.5508100000000002
	.quad	0x4014e3760bf5d788              # double 5.2221299999999999
	.quad	0x400071d81f106680              # double 2.0555880000000002
	.quad	0x3ff58be61cffeb07              # double 1.3466549999999999
	.quad	0x3ff811cf355cd91f              # double 1.504348
	.quad	0x4014ec4c5974e65c              # double 5.2307600000000001
	.quad	0x40177c996b7670a0              # double 5.8716790000000003
	.quad	0x400dd6ffc115df65              # double 3.7299799999999999
	.quad	0x3fc4f9335d249e45              # double 0.163855
	.quad	0x400ca4230fcf80dc              # double 3.5801449999999999
	.quad	0x4012c0068db8bac7              # double 4.6875249999999999
	.quad	0x3ff6dae5ffa3b9ae              # double 1.428442
	.quad	0x4017eb9ecf638002              # double 5.9800979999999999
	.quad	0x40110a12b5e529bb              # double 4.2598370000000001
	.quad	0x4022422f6a50d6b2              # double 9.1292679999999997
	.quad	0x401aea5870da5daf              # double 6.728853
	.quad	0x3fd150092ccf6be3              # double 0.27050999999999997
	.quad	0x40231a92a3055326              # double 9.5518999999999998
	.quad	0x401b62fb5d031056              # double 6.8466620000000002
	.quad	0x400d23f67f4dbdf9              # double 3.64256
	.quad	0x4009796d0917d6b6              # double 3.1842899999999998
	.quad	0x4022d6f5232d2bb2              # double 9.4198389999999996
	.quad	0x3ff5c53198288052              # double 1.360643
	.quad	0x4015ac3a42f1ed18              # double 5.4181910000000002
	.quad	0x4012526a65cf67b2              # double 4.5804840000000002
	.quad	0x40152179bfdf7e80              # double 5.2826909999999998
	.quad	0x3ff59ff1d81f1066              # double 1.3515489999999999
	.quad	0x401c26c93a711516              # double 7.0378769999999999
	.quad	0x4017a41355475a32              # double 5.9102300000000003
	.quad	0x400589aa1d755bcd              # double 2.6922190000000001
	.quad	0x3fe29b77c02afddb              # double 0.58147800000000005
	.quad	0x4018f28fe260b2c8              # double 6.2368769999999998
	.quad	0x401571372a38b8ae              # double 5.3605619999999998
	.quad	0x401870667f90d9d7              # double 6.1097659999999996
	.quad	0x40014ee7c49fd7a1              # double 2.1635279999999999
	.quad	0x400d43ae2541d8e8              # double 3.6580469999999998
	.quad	0x40235ec13b9f127f              # double 9.6850679999999993
	.quad	0x4023ca71151611ba              # double 9.8953939999999996
	.quad	0x40036e7a311e85fd              # double 2.428944
	.quad	0x40121115df6555c5              # double 4.5166849999999998
	.quad	0x40151ca1059ea572              # double 5.2779579999999999
	.quad	0x3fe093ee5eedcc21              # double 0.51805800000000002
	.quad	0x3fe650917d6b65aa              # double 0.69733500000000004
	.quad	0x401fb15d701d9f4d              # double 7.9232079999999998
	.quad	0x401538fc0d2c386d              # double 5.3056489999999998
	.quad	0x3fce4fe36d22424a              # double 0.236813
	.quad	0x401741511dffc548              # double 5.8137860000000003
	.quad	0x40235077ee4e26d5              # double 9.6571650000000008
	.quad	0x402173a8e71476b0              # double 8.7258980000000008
	.quad	0x400708e4755ffe6d              # double 2.8793419999999998
	.quad	0x3ffc0a17b0f6ad71              # double 1.752464
	.quad	0x4007dd912556d19e              # double 2.983187
	.quad	0x3ff4ccac4b4d056c              # double 1.2999689999999999
	.quad	0x4017df9e7b80a9df              # double 5.9683780000000004
	.quad	0x4016a7aeddce7cd0              # double 5.6637529999999998
	.quad	0x4022a5ec5f394b7b              # double 9.3240689999999997
	.quad	0x4021e0af5fd47bee              # double 8.9388380000000005
	.quad	0x400f3ecb31c219eb              # double 3.9056609999999998
	.quad	0x3fbbcd466f5019f4              # double 0.108601
	.quad	0x4001c5e4606748e4              # double 2.2216269999999998
	.quad	0x4022bd2e514c22ee              # double 9.3694939999999995
	.quad	0x400127d2c7b890d6              # double 2.1444450000000002
	.quad	0x4022a7264a16a487              # double 9.3264639999999996
	.quad	0x401391a650614163              # double 4.8922359999999996
	.quad	0x4010859c49774257              # double 4.1304790000000002
	.quad	0x4012a393682730c6              # double 4.6597419999999996
	.quad	0x401318ec95bff045              # double 4.7743399999999996
	.quad	0x401da011d3671ac1              # double 7.4063179999999997
	.quad	0x4008a76102795704              # double 3.081728
	.quad	0x400eb7ad4b2745bf              # double 3.8396859999999999
	.quad	0x3fe13e07a28bb0a3              # double 0.53882200000000002
	.quad	0x40129d8409e55c10              # double 4.6538240000000002
	.quad	0x4022018655193709              # double 9.0029780000000006
	.quad	0x4014ce410b630a91              # double 5.2014199999999997
	.quad	0x3f99fba450acc0be              # double 0.025374000000000001
	.quad	0x4021bc1e7967caea              # double 8.8674199999999992
	.quad	0x401ab9635610adcd              # double 6.6810429999999998
	.quad	0x3fbb1876188b1141              # double 0.10584200000000001
	.quad	0x4022a8b1572580c3              # double 9.3294779999999999
	.quad	0x401a6d4b6a619daa              # double 6.6067330000000002
	.quad	0x4015516e37154003              # double 5.3295219999999999
	.quad	0x401c2b38d60a6330              # double 7.0422089999999997
	.quad	0x40047e9af5ba2be0              # double 2.5618189999999998
	.quad	0x40233ef75d9a0f0a              # double 9.6229809999999993
	.quad	0x400153e681a9b8cc              # double 2.1659670000000002
	.quad	0x400ca6e32e3821af              # double 3.5814879999999998
	.quad	0x401a6d8f4f93bc0a              # double 6.606992
	.quad	0x40220e2b8c75c4a8              # double 9.0276759999999996
	.quad	0x40015965b20b8067              # double 2.1686510000000001
	.quad	0x4005095421c04428              # double 2.6295549999999999
	.quad	0x3ff50d6d3b6cbd98              # double 1.3157779999999999
	.quad	0x3ff8641c3efae792              # double 1.5244409999999999
	.quad	0x4023f591ab8e8ea4              # double 9.9796270000000006
	.quad	0x40208ff69014b59a              # double 8.2811780000000006
	.quad	0x3fe4a3dd54da4ce8              # double 0.64500299999999999
	.quad	0x4016bad57bc7f77b              # double 5.682455
	.quad	0x3fdd16aceaaf35e3              # double 0.454509
	.quad	0x402158aeb80ecfa7              # double 8.6732080000000007
	.quad	0x4022d4e18266772d              # double 9.4157829999999993
	.quad	0x401bc053e2d6238e              # double 6.9378200000000003
	.quad	0x4023372b239a38fc              # double 9.6077510000000003
	.quad	0x4023ac050bd87b57              # double 9.8359760000000004
	.quad	0x4004473c4f3ba776              # double 2.534783
	.quad	0x40197add590c0ad0              # double 6.3699849999999998
	.quad	0x401f9507a6bd6e8b              # double 7.895537
	.quad	0x40188cd530489d28              # double 6.1375320000000002
	.quad	0x3ff4a9802c0a4a06              # double 1.291382
	.quad	0x3fe0bebe1650a45d              # double 0.52328399999999997
	.quad	0x401ed265f0f5a101              # double 7.7054669999999996
	.quad	0x402215df00abf76a              # double 9.0427169999999997
	.quad	0x40095953dea465a5              # double 3.1686169999999998
	.quad	0x3fcd6777079e59f3              # double 0.22972000000000001
	.quad	0x40043fcf3dc054ef              # double 2.5311569999999999
	.quad	0x4015da2403148777              # double 5.4630280000000004
	.quad	0x401769abf3387161              # double 5.8531950000000004
	.quad	0x40186c853c148345              # double 6.1059770000000002
	.quad	0x3fe9dfe974a3400c              # double 0.80858300000000005
	.quad	0x40050f3a14cec41e              # double 2.6324350000000001
	.quad	0x4020aaa0a958537e              # double 8.3332569999999997
	.quad	0x401cd7ba237cdcca              # double 7.2106709999999996
	.quad	0x4012e5e13b18dac2              # double 4.7244919999999997
	.quad	0x4022729930be0ded              # double 9.2238249999999997
	.quad	0x3fbc7f455a7d2418              # double 0.111317
	.quad	0x40211266772d5e07              # double 8.5359379999999998
	.quad	0x40171b77c02afddb              # double 5.7768240000000004
	.quad	0x4023c37fbefd0071              # double 9.8818339999999995
	.quad	0x3ffc2646f1561911              # double 1.7593449999999999
	.quad	0x4022f445aa2e3c53              # double 9.4770939999999992
	.quad	0x4018dfc7a398201d              # double 6.2185350000000001
	.quad	0x40216c85e3da2f8c              # double 8.7119590000000002
	.quad	0x400b3a5c1c6088d7              # double 3.4034960000000001
	.quad	0x4023c99d451fc4c1              # double 9.8937779999999992
	.quad	0x401fdfd2630ec311              # double 7.9685759999999996
	.quad	0x4021a1203a322af5              # double 8.8146989999999992
	.quad	0x40235aa8c9b84556              # double 9.6770689999999995
	.quad	0x4011717ca2120e1f              # double 4.3608269999999996
	.quad	0x4013452b0a6fc58b              # double 4.8175470000000002
	.quad	0x402325bb384fd2a6              # double 9.5736939999999997
	.quad	0x4014cfd3f5b5fa22              # double 5.2029569999999996
	.quad	0x40050196d8f4f93c              # double 2.6257760000000001
	.quad	0x4008aa5efe931876              # double 3.083189
	.quad	0x40196edd8b60f1b2              # double 6.3582669999999997
	.quad	0x401d98adab9f559b              # double 7.3990999999999998
	.quad	0x401efcd24e160d88              # double 7.7468959999999995
	.quad	0x3fe0288ce703afb8              # double 0.50495000000000001
	.quad	0x40227685fd04a2fd              # double 9.2314910000000001
	.quad	0x3ffc9caf2d7f950c              # double 1.7882530000000001
	.quad	0x400edb1af3a14cec              # double 3.8569849999999999
	.quad	0x3ffddcc100e6afcd              # double 1.866395
	.quad	0x4021716b11c6d1e1              # double 8.7215199999999999
	.quad	0x40002cc1ca3a4b55              # double 2.0218539999999998
	.quad	0x400b3ee88df3732a              # double 3.4057170000000001
	.quad	0x401c93293d102bc7              # double 7.1437119999999998
	.quad	0x4021d57f51efb6dd              # double 8.9169870000000007
	.quad	0x4006f2799a1fd157              # double 2.8683960000000002
	.quad	0x401808769ec2ce46              # double 6.0082649999999997
	.quad	0x4023a76af89c5e70              # double 9.8269880000000001
	.quad	0x4017e24af0bf1a5d              # double 5.9709890000000003
	.quad	0x3fc5707182235b4f              # double 0.167494
	.quad	0x40159d48882f0e0b              # double 5.4035970000000004
	.quad	0x3fd4465625a682b6              # double 0.31679299999999999
	.quad	0x401a0c2a454de7ea              # double 6.5118799999999997
	.quad	0x401ddf00ef1348b2              # double 7.4677769999999999
	.quad	0x40115a4a48f96e16              # double 4.3381740000000004
	.quad	0x3ff99e00d1b71759              # double 1.601075
	.quad	0x40203fc32ebe596d              # double 8.1245360000000009
	.quad	0x4009846950fc71d6              # double 3.189654
	.quad	0x3fe5fd3f5b5fa227              # double 0.687164
	.quad	0x4001b123c42a66dc              # double 2.2114940000000001
	.quad	0x401801971c10d7bf              # double 6.0015530000000004
	.quad	0x402155ca4b124d0a              # double 8.6675590000000007
	.quad	0x401670e8858ff759              # double 5.6102619999999996
	.quad	0x4023c3cd573647bb              # double 9.8824260000000005
	.quad	0x3fe50d3cff64cf8d              # double 0.65786599999999995
	.quad	0x40098bdc26dce39b              # double 3.1932909999999999
	.quad	0x3ff82cf7446f9b99              # double 1.5109779999999999
	.quad	0x4013cfa6defc7a3a              # double 4.9527850000000004
	.quad	0x3ff0e236c15d2d02              # double 1.0552280000000001
	.quad	0x401968eb463497b7              # double 6.3524599999999998
	.quad	0x4006da1ec4e7253e              # double 2.8565040000000002
	.quad	0x401d408c8259e1f4              # double 7.3130360000000003
	.quad	0x400e2795f676ea42              # double 3.7693289999999999
	.quad	0x400ea0285ec3dab6              # double 3.8282020000000001
	.quad	0x4021cb254e6e221d              # double 8.8967690000000008
	.quad	0x3fbb4fb549f94856              # double 0.106685
	.quad	0x401137d30ad46f58              # double 4.3045159999999996
	.quad	0x4013444523f67f4e              # double 4.8166700000000002
	.quad	0x3ff6d6b00ffda405              # double 1.427414
	.quad	0x400b01adea897636              # double 3.37582
	.quad	0x401314723aafff37              # double 4.7699670000000003
	.quad	0x40054319c5a3e39f              # double 2.6577639999999998
	.quad	0x400065f6ba0620ab              # double 2.0497869999999998
	.quad	0x401e154d1e96c3fc              # double 7.5208019999999998
	.quad	0x40216302f72b4528              # double 8.6933819999999997
	.quad	0x4010aab606b7aa26              # double 4.1667100000000001
	.quad	0x3fe73458cd20afa3              # double 0.72514000000000001
	.quad	0x3fe1a5aee631f8a1              # double 0.55147500000000005
	.quad	0x4022b41461b6d43d              # double 9.351718
	.quad	0x400cf355cd91eeaa              # double 3.6188159999999998
	.quad	0x3fea3343fa2ad3e9              # double 0.81875799999999999
	.quad	0x4023fc8e043a2164              # double 9.993271
	.quad	0x40207948fd9fd36f              # double 8.2368849999999991
	.quad	0x4002ca21a719b4dd              # double 2.348697
	.quad	0x402163b341e14be0              # double 8.6947270000000003
	.quad	0x400d03947496aad2              # double 3.6267480000000001
	.quad	0x4014e506573215fd              # double 5.2236570000000002
	.quad	0x402053a1b1960fa1              # double 8.1633429999999993
	.quad	0x40007e468cac4b4d              # double 2.061658
	.quad	0x3fe64c122749f0e5              # double 0.69678600000000002
	.quad	0x401fd06290eed02d              # double 7.9535010000000001
	.quad	0x4011247b677f6b1a              # double 4.2856269999999999
	.quad	0x4022f944aa53fc01              # double 9.4868520000000007
	.quad	0x40151e618ce2d1f2              # double 5.2796690000000002
	.quad	0x4014c5a35d67ba23              # double 5.1930059999999996
	.quad	0x40201fa658c4bd34              # double 8.0618160000000003
	.quad	0x4015cd78c436fc16              # double 5.4506560000000004
	.quad	0x400f5358f2e05ccd              # double 3.9156970000000002
	.quad	0x401bb1a005c4651f              # double 6.9234619999999998
	.quad	0x402220dae3e6c4c6              # double 9.0641700000000007
	.quad	0x4003a2541d8e8640              # double 2.4542619999999999
	.quad	0x40204ed6dc1a47aa              # double 8.1539830000000002
	.quad	0x4014b09aed56b010              # double 5.172466
	.quad	0x4021de9d0e991ff7              # double 8.9347919999999998
	.quad	0x401ec13404ea4a8c              # double 7.6886749999999999
	.quad	0x4015843632c1f42c              # double 5.3791130000000003
	.quad	0x3ffd21059ea57215              # double 1.820562
	.quad	0x4020960285ec3dab              # double 8.2929879999999994
	.quad	0x401fcdeacc92146a              # double 7.9510909999999999
	.quad	0x4006a5604189374c              # double 2.8307500000000001
	.quad	0x4021a49795b35b0c              # double 8.8214690000000004
	.quad	0x40164ff10ecb74de              # double 5.578068
	.quad	0x401a4104d551d68c              # double 6.5634949999999996
	.quad	0x3fe148605681ecd5              # double 0.54008500000000004
	.quad	0x3ff248bfc2224edf              # double 1.1427609999999999
	.quad	0x400a4670e2c12ad8              # double 3.284395
	.quad	0x401cb759ab6d00b4              # double 7.1790529999999997
	.quad	0x4009f12ec6bce853              # double 3.2427649999999999
	.quad	0x401d4986fcdee350              # double 7.3218040000000002
	.quad	0x4006196744b2b778              # double 2.7624040000000001
	.quad	0x401552d59d55e6bc              # double 5.3308929999999997
	.quad	0x40209ab6f19934f0              # double 8.3021770000000004
	.quad	0x40221e121ee67514              # double 9.0587319999999991
	.quad	0x40120c3c5bd0e12f              # double 4.5119490000000004
	.quad	0x4013d67cf1c3265b              # double 4.9594610000000001
	.quad	0x400adc58ab92c062              # double 3.3575910000000002
	.quad	0x401fd487b99d4520              # double 7.9575490000000002
	.quad	0x4004f8f8a4c1ebc8              # double 2.6215679999999999
	.quad	0x3fda71715c63ae25              # double 0.41317399999999999
	.quad	0x401ec74107314ca9              # double 7.6945839999999998
	.quad	0x400eb1dffc5479d5              # double 3.8368530000000001
	.quad	0x4021dd5ae1cde5d2              # double 8.9323340000000009
	.quad	0x40200ee568a50508              # double 8.0290940000000006
	.quad	0x4017faade657b84e              # double 5.9948040000000002
	.quad	0x400ce30d306a2b17              # double 3.610865
	.quad	0x40207d3c9abb01c9              # double 8.2446029999999997
	.quad	0x3ff818655193708b              # double 1.5059560000000001
	.quad	0x400a24a515ce9e5e              # double 3.2678929999999999
	.quad	0x3feb577318fc5048              # double 0.85442499999999999
	.quad	0x401666adb402d16c              # double 5.6002720000000004
	.quad	0x4010f58a76102795              # double 4.2397859999999996
	.quad	0x4018ebd2b6f19935              # double 6.2302960000000001
	.quad	0x4001b806f262888b              # double 2.2148569999999999
	.quad	0x4001bb19e731d2e1              # double 2.2163580000000001
	.quad	0x401785321e603d57              # double 5.8800739999999996
	.quad	0x400d1fe64f54d1e9              # double 3.6405759999999998
	.quad	0x4022f4829cfdd228              # double 9.4775589999999994
	.quad	0x4008c3a3a8e71477              # double 3.0955270000000001
	.quad	0x3ffe483fd5022574              # double 1.892639
	.quad	0x401d2e60c38f3669              # double 7.2952909999999997
	.quad	0x401d70b524399b2c              # double 7.3600659999999998
	.quad	0x3ff286f694467382              # double 1.15795
	.quad	0x400260685553ef6b              # double 2.2970739999999998
	.quad	0x401f32478854cdb8              # double 7.7991010000000002
	.quad	0x3ff71f42bb6672fc              # double 1.4451320000000001
	.quad	0x401ab18e325d4a5e              # double 6.673394
	.quad	0x400cfc46d82ba5a0              # double 3.6231819999999999
	.quad	0x401fdd46f587d6f9              # double 7.9660909999999995
	.quad	0x400355aa715831f0              # double 2.4168289999999999
	.quad	0x401fee3cd9a52264              # double 7.9826540000000001
	.quad	0x3fd0ab8a5ce5b424              # double 0.26046999999999998
	.quad	0x4001641cc532a498              # double 2.173883
	.quad	0x401dfbc2b94d9408              # double 7.4958600000000004
	.quad	0x401ffefa615a8deb              # double 7.9990019999999999
	.quad	0x401d131f46ed245b              # double 7.2686739999999999
	.quad	0x401f6c625e99e94b              # double 7.8558440000000002
	.quad	0x400429813879c411              # double 2.5202659999999999
	.quad	0x401782e8c0485a0c              # double 5.8778410000000001
	.quad	0x400b7c3dee781840              # double 3.4356650000000002
	.quad	0x4015da27ae9ab29e              # double 5.4630419999999997
	.quad	0x4022bc743201040c              # double 9.368074
	.quad	0x3fe3b93682730c67              # double 0.61635899999999999
	.quad	0x3ffb72e1693c03bc              # double 1.7155469999999999
	.quad	0x3ff2e515ce9e5e24              # double 1.1809289999999999
	.quad	0x40225a01040bfe3b              # double 9.175789
	.quad	0x4018b95c42203600              # double 6.1810159999999996
	.quad	0x40225ba190082060              # double 9.1789670000000001
	.quad	0x4011ed4ed2cbea4f              # double 4.4817460000000002
	.quad	0x40153a5e353f7cee              # double 5.3070000000000004
	.quad	0x40224878fabda0e0              # double 9.1415480000000002
	.quad	0x3ff793f5b5fa2270              # double 1.4736229999999999
	.quad	0x402057dd44135547              # double 8.1716099999999994
	.quad	0x40220cfbb949624a              # double 9.0253580000000007
	.quad	0x3ff1621fafc8b008              # double 1.0864560000000001
	.quad	0x40152121ab4b72c5              # double 5.2823549999999999
	.quad	0x401873dab5c39bcc              # double 6.1131390000000003
	.quad	0x40138f077ccc0379              # double 4.8896769999999998
	.quad	0x401792046c764ae0              # double 5.892595
	.quad	0x4014d928e0c9d9d3              # double 5.2120699999999998
	.quad	0x4017bfc2eba27aea              # double 5.9372670000000003
	.quad	0x3ffc69e7fb267c6c              # double 1.7758560000000001
	.quad	0x3fb763b256ffc116              # double 0.091365000000000001
	.quad	0x3ff212c9d16fc9bc              # double 1.1295869999999999
	.quad	0x40105163baba7b91              # double 4.0794819999999996
	.quad	0x4013f19da9c99286              # double 4.9859530000000003
	.quad	0x4002dc17225b749b              # double 2.3574660000000001
	.quad	0x400e33ba775fb2ee              # double 3.775258
	.quad	0x3fd8e8815e39713b              # double 0.38919100000000001
	.quad	0x4023232702a3486f              # double 9.5686569999999999
	.quad	0x4022e81c714fce74              # double 9.4533419999999992
	.quad	0x4020ccab1d4f9c20              # double 8.3997430000000008
	.quad	0x401993b9f127f5e8              # double 6.3942639999999997
	.quad	0x4022fc93ea2d2fe4              # double 9.493316
	.quad	0x401bc0cbf2b239a4              # double 6.9382780000000004
	.quad	0x401d5f3a14cec41e              # double 7.3429950000000002
	.quad	0x40117bf401c4fc1e              # double 4.371048
	.quad	0x4021cab649d388a9              # double 8.8959220000000006
	.quad	0x4001de3c105186db              # double 2.2335129999999999
	.quad	0x4017e39c94f69caa              # double 5.9722770000000001
	.quad	0x4016cb9fdbd2fa0d              # double 5.6988519999999996
	.quad	0x402263868fd199bb              # double 9.1943859999999997
	.quad	0x401374cf8d716d2b              # double 4.8640730000000003
	.quad	0x401e7eb78897e996              # double 7.6237469999999998
	.quad	0x401282ba16e7a312              # double 4.6276630000000001
	.quad	0x3ff643fa2ad3e921              # double 1.3915960000000001
	.quad	0x3fefa934acaff6d3              # double 0.98940499999999997
	.quad	0x40090152b0a6fc59              # double 3.1256460000000001
	.quad	0x401044052d666a98              # double 4.0664259999999999
	.quad	0x401560ec31162281              # double 5.3446509999999998
	.quad	0x3ff534b0e1b4bb5e              # double 1.325364
	.quad	0x4021b2cb85a4f00f              # double 8.8492090000000001
	.quad	0x4022e1e2584f4c6e              # double 9.4411799999999992
	.quad	0x40205d8afc47e49b              # double 8.1827009999999998
	.quad	0x3fdf591cd1c7de51              # double 0.48981400000000003
	.quad	0x3ffe363a19008206              # double 1.888239
	.quad	0x401f3c1ebc83a96d              # double 7.8087109999999997
	.quad	0x402331d8a5482385              # double 9.5973559999999995
	.quad	0x4007b7d6b65a9a80              # double 2.9647649999999999
	.quad	0x400b576b3bb83cf3              # double 3.4176850000000001
	.quad	0x40071845564b6630              # double 2.8868510000000001
	.quad	0x402326ae7d566cf4              # double 9.5755499999999997
	.quad	0x4015ac2a8869c66d              # double 5.4181309999999998
	.quad	0x4018436c58eeae9f              # double 6.0658430000000001
	.quad	0x400ac218def416be              # double 3.3447740000000001
	.quad	0x40170ccd0fe8ab50              # double 5.7625010000000003
	.quad	0x401982e7b3d8e001              # double 6.3778370000000004
	.quad	0x3ff76e49f51697f2              # double 1.464426
	.quad	0x4010f35a426bb55b              # double 4.2376490000000002
	.quad	0x401181e4b44a1f08              # double 4.376849
	.quad	0x4020a4c20d5629d8              # double 8.3217929999999996
	.quad	0x4020554cdb7ae579              # double 8.1666019999999993
	.quad	0x401825bd0e12e83a              # double 6.0368539999999999
	.quad	0x4017914057082492              # double 5.8918470000000003
	.quad	0x3ffca518f3eccc47              # double 1.7903070000000001
	.quad	0x401a5d24180d3cff              # double 6.5909579999999997
	.quad	0x401e1e9e5e247885              # double 7.5299009999999997
	.quad	0x402166e9dd7ecbb8              # double 8.701003
	.quad	0x4014f5da6a444178              # double 5.2400909999999996
	.quad	0x4023031769a91106              # double 9.5060380000000002
	.quad	0x4016593f6c2699c8              # double 5.5871560000000002
	.quad	0x4012fb6bf444e0db              # double 4.7455290000000003
	.quad	0x40037da72a7bd48d              # double 2.4363540000000001
	.quad	0x40206286d71f3626              # double 8.1924349999999997
	.quad	0x3fea985f06f69446              # double 0.83109999999999995
	.quad	0x402227f3061c79b3              # double 9.0780259999999995
	.quad	0x401badcb9a9da598              # double 6.9197220000000002
	.quad	0x4022044199fe4367              # double 9.0083129999999993
	.quad	0x3ff01c62a1b5c7ce              # double 1.0069300000000001
	.quad	0x4003cfbca1059ea5              # double 2.4764339999999998
	.quad	0x4012a8049667b5f2              # double 4.6640800000000002
	.quad	0x401bfbdd334c5da7              # double 6.9959610000000003
	.quad	0x40048c1e364bec68              # double 2.5684170000000002
	.quad	0x402044e48626f60e              # double 8.1345559999999999
	.quad	0x4021d34aec8d5c74              # double 8.9126809999999991
	.quad	0x400d8b4202d9cf14              # double 3.6929970000000001
	.quad	0x401663dff3f0fe04              # double 5.5975339999999996
	.quad	0x401ddf72b4528284              # double 7.4682110000000002
	.quad	0x401ac8576cce5f74              # double 6.695646
	.quad	0x401f359d55e6bc62              # double 7.8023579999999999
	.quad	0x3ff572367e414e7f              # double 1.340384
	.quad	0x402244d5f99c38b0              # double 9.1344449999999995
	.quad	0x3fe7bbe4473cd573              # double 0.74168599999999996
	.quad	0x3fde938151a43782              # double 0.47775299999999998
	.quad	0x400f803148776816              # double 3.9375939999999998
	.quad	0x3ffdfe38eb0318b9              # double 1.874566
	.quad	0x40213485e3da2f8c              # double 8.6025840000000002
	.quad	0x3ff8f8359bc33f85              # double 1.5605979999999999
	.quad	0x401c7b98c7e28241              # double 7.1207000000000003
	.quad	0x3fee129ed3953dea              # double 0.93977299999999997
	.quad	0x3f99d55e6bc621b8              # double 0.025228
	.quad	0x401029e7b80a9de9              # double 4.0409230000000003
	.quad	0x4015dbe2f7b17ce5              # double 5.4647329999999998
	.quad	0x400736d86ec17ebb              # double 2.90178
	.quad	0x3fca933a0407cc7d              # double 0.207618
	.quad	0x400b7b4edb2f661f              # double 3.435209
	.quad	0x401f4eb91b3f20a7              # double 7.8268779999999997
	.quad	0x4022361c79b34a45              # double 9.1056860000000004
	.quad	0x400a6485a0be5109              # double 3.299083
	.quad	0x40219ab9d3cbc48f              # double 8.8021989999999999
	.quad	0x401a80fa15db3398              # double 6.6259540000000001
	.quad	0x3febcc1a8ac5c140              # double 0.86866500000000002
	.quad	0x401a10e6299524c0              # double 6.5165030000000002
	.quad	0x4019c5f91600f345              # double 6.443333
	.quad	0x4006981b64e05469              # double 2.824271
	.quad	0x400f0f1c3265adda              # double 3.8823780000000001
	.quad	0x4006d13d74d594f2              # double 2.8521679999999998
	.quad	0x40123e1650a45d42              # double 4.560632
	.quad	0x4020ab397dd00f77              # double 8.3344229999999992
	.quad	0x401af22b7baecd08              # double 6.7364940000000004
	.quad	0x40050c28b2a6b0d9              # double 2.6309369999999999
	.quad	0x4021c0d6f544bb1b              # double 8.8766400000000001
	.quad	0x4016211c2a023209              # double 5.5323339999999996
	.quad	0x4022253aa369fcf4              # double 9.0727130000000002
	.quad	0x3fb5dd095af294dd              # double 0.085403999999999994
	.quad	0x400b7dedf1e0828c              # double 3.4364889999999999
	.quad	0x402292cfb762d83c              # double 9.2867409999999992
	.quad	0x4018523315d701da              # double 6.080273
	.quad	0x3ff13673c4f3ba77              # double 1.0757939999999999
	.quad	0x400e5eadd590c0ad              # double 3.79623
	.quad	0x3fda11a543f1c758              # double 0.40732699999999999
	.quad	0x4021114f0520d131              # double 8.5338060000000002
	.quad	0x3ff97f8012dfd695              # double 1.593628
	.quad	0x400761059ea57215              # double 2.922374
	.quad	0x3ffa8545846e8f2a              # double 1.657537
	.quad	0x3fe2de3fbbd7b203              # double 0.58962999999999999
	.quad	0x401d3b4c3c18b503              # double 7.3079080000000003
	.quad	0x4021f7d2e9468017              # double 8.9840309999999999
	.quad	0x3ff3633df2a9d628              # double 1.2117290000000001
	.quad	0x4012cd716d2aa5c6              # double 4.700628
	.quad	0x401eeec7000431be              # double 7.7331810000000001
	.quad	0x402332fb7e90ff97              # double 9.5995749999999997
	.quad	0x402380d51f81a587              # double 9.7516259999999999
	.quad	0x4010c5b8dc55000d              # double 4.1930880000000004
	.quad	0x401e6d48882f0e0b              # double 7.6067220000000004
	.quad	0x401ccd1526d8b1dd              # double 7.2002759999999997
	.quad	0x4020bcb20fb224ab              # double 8.3685460000000003
	.quad	0x400d1da5119ce076              # double 3.639475
	.quad	0x401234284dfce315              # double 4.550935
	.quad	0x402341d323fee2ca              # double 9.6285640000000008
	.quad	0x40151a311e85fd05              # double 5.2755780000000003
	.quad	0x40180b86b15f88fd              # double 6.0112560000000004
	.quad	0x4020f7b890d5a5b9              # double 8.4838299999999993
	.quad	0x4021b474fb549f95              # double 8.8524550000000009
	.quad	0x3fda4901d19157ac              # double 0.41070600000000002
	.quad	0x4009028672756862              # double 3.126233
	.quad	0x40228c606748e475              # double 9.2741729999999993
	.quad	0x3fe51628cbd1244a              # double 0.65895499999999996
	.quad	0x401c07e5eaab0425              # double 7.0077129999999999
	.quad	0x40176a8d64d7f0ed              # double 5.8540549999999998
	.quad	0x40216148dc11e42e              # double 8.6900089999999999
	.quad	0x3ffe17ff9b56323c              # double 1.8808590000000001
	.quad	0x401f7d3d859c8c93              # double 7.8723049999999998
	.quad	0x40028aa7ded6ba8c              # double 2.3177029999999998
	.quad	0x4000d4163779e9d1              # double 2.103558
	.quad	0x4001eed6777079e6              # double 2.2416200000000002
	.quad	0x4015e6aec0724b77              # double 5.475276
	.quad	0x4009379bfdf7e804              # double 3.1521530000000002
	.quad	0x401f80fdc1615ec0              # double 7.8759680000000003
	.quad	0x40220e3b03e20cd0              # double 9.0277940000000001
	.quad	0x3ff96c69b5a63f9a              # double 1.5889679999999999
	.quad	0x3ffa3fd82773e250              # double 1.640587
	.quad	0x400361e1d2178f69              # double 2.4227940000000001
	.quad	0x402252c061847f56              # double 9.1616239999999998
	.quad	0x401f779463cfb331              # double 7.8667769999999999
	.quad	0x401dda46173b85e8              # double 7.4631579999999999
	.quad	0x400379d99029ae4f              # double 2.4344969999999999
	.quad	0x40137fb69984a0e4              # double 4.8747199999999999
	.quad	0x40211586876e1deb              # double 8.5420420000000004
	.quad	0x401a42046c764ae0              # double 6.56447
	.quad	0x4006a3b2dd377e1c              # double 2.8299310000000002
	.quad	0x400cc9cc319c5a3e              # double 3.5985339999999999
	.quad	0x3fe7393250b5243a              # double 0.72573200000000004
	.quad	0x3fd84d6e47dc37a4              # double 0.37972600000000001
	.quad	0x3fc8d4fdf3b645a2              # double 0.19400000000000001
	.quad	0x3fff2976bc1effa0              # double 1.9476230000000001
	.quad	0x401b06e5471715c6              # double 6.7567339999999998
	.quad	0x40037e75bc44bf4d              # double 2.4367480000000001
	.quad	0x401de35bd512ec6c              # double 7.4720300000000002
	.quad	0x40225723465625a7              # double 9.1701910000000009
	.quad	0x3fee07ad080b673c              # double 0.93843699999999996
	.quad	0x3ff050a99b6f5caf              # double 1.019693
	.quad	0x3fe76a897635e743              # double 0.73175500000000004
	.quad	0x3fcc4cf8d716d2aa              # double 0.22109899999999999
	.quad	0x401cbd5cb790fb65              # double 7.1849239999999996
	.quad	0x401e19fd7a13c255              # double 7.5253810000000003
	.quad	0x400fbc2a8869c66d              # double 3.9668779999999999
	.quad	0x4020d1c6d1e108c4              # double 8.40972
	.quad	0x401fa3f9a49c2c1b              # double 7.9101319999999999
	.quad	0x3fe62ec8d5c74752              # double 0.69321100000000002
	.quad	0x401320d49949e881              # double 4.7820609999999997
	.quad	0x3ffad0bc7b45f17c              # double 1.675961
	.quad	0x401dc22a6f3f52fc              # double 7.4396149999999999
	.quad	0x400a076d117b5287              # double 3.2536260000000001
	.quad	0x400427e414e7ee91              # double 2.5194779999999999
	.quad	0x401b0584b1ab0857              # double 6.7553890000000001
	.quad	0x4013baa4766c6de7              # double 4.9322679999999997
	.quad	0x401c4f6be37de93a              # double 7.0775600000000001
	.quad	0x401e6d655e28aa43              # double 7.6068319999999998
	.quad	0x3fff5d6cf850df16              # double 1.9603090000000001
	.quad	0x4013182a9930be0e              # double 4.7736000000000001
	.quad	0x4017a1a500d5e8d5              # double 5.9078559999999998
	.quad	0x3fffb75fb2edfe76              # double 1.9822690000000001
	.quad	0x400af2a38b8ae31d              # double 3.3684759999999998
	.quad	0x402369a2834d26fa              # double 9.7063179999999996
	.quad	0x400522480e8c8abd              # double 2.6417389999999998
	.quad	0x4001c56c508b32cf              # double 2.2213980000000002
	.quad	0x4013c3c9abb01c93              # double 4.9411990000000001
	.quad	0x3fe072f9873ffac2              # double 0.51403500000000002
	.quad	0x401e8d30270f3882              # double 7.6378789999999999
	.quad	0x40200731f46ed246              # double 8.0140530000000005
	.quad	0x402208ea5b530ced              # double 9.0174129999999995
	.quad	0x4002d41205bc01a3              # double 2.3535499999999998
	.quad	0x401eff40a2877ee5              # double 7.7492700000000001
	.quad	0x3ffc2012599ed7c7              # double 1.75783
	.quad	0x3ff33d5454152b0a              # double 1.2024729999999999
	.quad	0x4004758afc47e49b              # double 2.5573939999999999
	.quad	0x3ffabd74d594f26b              # double 1.671254
	.quad	0x3fe0417ca2120e1f              # double 0.50799399999999995
	.quad	0x400a8f7fcfc3f812              # double 3.320068
	.quad	0x401f3c3eb7cbb3c8              # double 7.8088329999999999
	.quad	0x40185c81d3aa36a0              # double 6.0903390000000002
	.quad	0x400b4f312b1b36bd              # double 3.4136679999999999
	.quad	0x40232ffae3608d09              # double 9.5937110000000007
	.quad	0x4021f6158750c1b9              # double 8.9806329999999992
	.quad	0x4018bdb8bac710cb              # double 6.1852749999999999
	.quad	0x401e73e63e8dda49              # double 7.6131830000000002
	.quad	0x401edf1886df82b2              # double 7.717867
	.quad	0x4023af685985ad54              # double 9.8425930000000008
	.quad	0x3fe5ee392e1ef73c              # double 0.68532999999999999
	.quad	0x40135feda6612839              # double 4.84368
	.quad	0x401204d1633482bf              # double 4.5047050000000004
	.quad	0x4016693faf42784b              # double 5.6027820000000004
	.quad	0x4015eb14ec204f2b              # double 5.4795720000000001
	.quad	0x3ff5a25742dcf462              # double 1.3521339999999999
	.quad	0x401cde1693c03bc5              # double 7.2168830000000002
	.quad	0x3ffba2e09fe86834              # double 1.7272650000000001
	.quad	0x3fd86cdf266ba494              # double 0.38164500000000001
	.quad	0x3ff16ed783dff3f1              # double 1.089561
	.quad	0x401f092c8c5004fb              # double 7.7589589999999999
	.quad	0x4010543526527a20              # double 4.0822339999999997
	.quad	0x4017a903211cb03a              # double 5.9150510000000001
	.quad	0x4017538327674d16              # double 5.8315549999999998
	.quad	0x401a7bb1290257c9              # double 6.6207929999999999
	.quad	0x401fc7f6f4be835e              # double 7.9452780000000001
	.quad	0x40208777079e59f3              # double 8.2645800000000005
	.quad	0x3fb3ab324851a869              # double 0.076830999999999997
	.quad	0x402139a75cd0bb6f              # double 8.6126050000000003
	.quad	0x401b4a4b5568e821              # double 6.8225530000000001
	.quad	0x401e63dc8b86b160              # double 7.5975210000000004
	.quad	0x40219b0c88a47ed0              # double 8.8028300000000002
	.quad	0x3fcfdbf487fcb924              # double 0.24890000000000001
	.quad	0x40171f33ca31e7da              # double 5.7804710000000004
	.quad	0x40230cf56eac8605              # double 9.5253099999999992
	.quad	0x3ff1f69ec2ce464a              # double 1.1227100000000001
	.quad	0x3ffc4153bd167664              # double 1.765949
	.quad	0x4018e266fd651b0d              # double 6.2210960000000002
	.quad	0x40225ab7fe08aefb              # double 9.1771849999999997
	.quad	0x4019b2aae297396d              # double 6.42448
	.quad	0x400b1ffb480a5acd              # double 3.3906160000000001
	.quad	0x400fa642396073de              # double 3.9561809999999999
	.quad	0x3ffcc1b54195cc85              # double 1.7972919999999999
	.quad	0x401c0ac4f7ec3548              # double 7.0105170000000001
	.quad	0x3faa5614df8b1572              # double 0.051437999999999998
	.quad	0x4020aa85ec3dab5c              # double 8.3330529999999995
	.quad	0x4021fce3de6149c7              # double 8.9939260000000001
	.quad	0x401997f1737542a2              # double 6.3983819999999998
	.quad	0x4011740c4156e265              # double 4.3633280000000001
	.quad	0x401dcc54bcf0b6b7              # double 7.4495420000000001
	.quad	0x4016b163779e9d0f              # double 5.6732310000000004
	.quad	0x3fe9b402d16b97ff              # double 0.80322400000000005
	.quad	0x401e54bcf0b6b6e1              # double 7.5827520000000001
	.quad	0x40199a6117720c8d              # double 6.4007610000000001
	.quad	0x4018b4477ff151e7              # double 6.1760539999999997
	.quad	0x3fefad2135daad60              # double 0.98988399999999998
	.quad	0x4001f5e4a3832767              # double 2.2450649999999999
	.quad	0x400bafbd273d5bab              # double 3.4608099999999999
	.quad	0x400776b11c6d1e11              # double 2.9329550000000002
	.quad	0x40238cd07852f7f5              # double 9.7750280000000007
	.quad	0x401b2eaf251c193b              # double 6.7955899999999998
	.quad	0x4016a51654d61b2a              # double 5.6612179999999999
	.quad	0x4010d759ab6d00b4              # double 4.2103029999999997
	.quad	0x4023bd5b89939219              # double 9.8698390000000006
	.quad	0x400726b484d76ab6              # double 2.8938990000000002
	.quad	0x4016231832fcac8e              # double 5.5342719999999996
	.quad	0x401b1f290abb44e5              # double 6.78043
	.quad	0x402047e0ac7da1ec              # double 8.1403859999999994
	.quad	0x401519e16d6dc1a4              # double 5.2752739999999996
	.quad	0x4000492ddbdb5d89              # double 2.0357319999999999
	.quad	0x4020e7e73a365cb3              # double 8.4529359999999993
	.quad	0x401bd5f2cb641701              # double 6.9589340000000002
	.quad	0x40190efae7924af1              # double 6.2646290000000002
	.quad	0x40117503fb374363              # double 4.3642729999999998
	.quad	0x3ff30621b7e0ac7e              # double 1.1889970000000001
	.quad	0x400dde1d2178f68c              # double 3.7334540000000001
	.quad	0x40000f661f18c9fb              # double 2.0075189999999998
	.quad	0x40139234a87e38eb              # double 4.892779
	.quad	0x4021b899bf5946c3              # double 8.8605479999999996
	.quad	0x4023d4f0f9d2bf55              # double 9.9159009999999998
	.quad	0x4003962413db7f17              # double 2.4483109999999999
	.quad	0x4021d13cab81f96a              # double 8.9086660000000002
	.quad	0x4015cc7abfb9bed3              # double 5.4496869999999999
	.quad	0x40006bc1eff9f87f              # double 2.052616
	.quad	0x400fd556084a515d              # double 3.979168
	.quad	0x3fd0ad1d041cc533              # double 0.26056600000000002
	.quad	0x3ff850f301eabbcb              # double 1.519763
	.quad	0x40155f4e874c8ffc              # double 5.3430730000000004
	.quad	0x3ff33c87db2b3461              # double 1.202278
	.quad	0x3fe42775fb2edfe7              # double 0.62981699999999996
	.quad	0x4020cdadd9c27e95              # double 8.4017169999999997
	.quad	0x3fd16b54e2b063e0              # double 0.27217599999999997
	.quad	0x3fec601ffb480a5b              # double 0.88673400000000002
	.quad	0x400a0f78feef5ec8              # double 3.257555
	.quad	0x4022e5f37329c348              # double 9.4491230000000001
	.quad	0x400a5fe21d96e9bc              # double 3.296818
	.quad	0x401ab7d955714b9d              # double 6.6795400000000003
	.quad	0x402033fc86cebbba              # double 8.1015359999999994
	.quad	0x40132b5fe542e558              # double 4.7923580000000001
	.quad	0x40069c6088d6d3b7              # double 2.8263560000000001
	.quad	0x401120f238972043              # double 4.2821740000000004
	.quad	0x3ff953a604e1e710              # double 1.5829219999999999
	.quad	0x3fe78c6f7a0b5ed9              # double 0.73589300000000002
	.quad	0x4021c878d92fb19e              # double 8.8915469999999992
	.quad	0x401b1a8e71476af9              # double 6.7759340000000003
	.quad	0x40233f9ce8d972cd              # double 9.6242439999999991
	.quad	0x401dcbafd976ff3b              # double 7.4489130000000001
	.quad	0x401a61676640a6b9              # double 6.5951209999999998
	.quad	0x4010a2cf95d4e8fb              # double 4.158995
	.quad	0x401beb1ee2435697              # double 6.9796100000000001
	.quad	0x400088af81626b2f              # double 2.0667409999999999
	.quad	0x3fd6baba7b9170d6              # double 0.35514699999999999
	.quad	0x3fd29be4cd749279              # double 0.290765
	.quad	0x3ff7b605ab3aabcd              # double 1.4819389999999999
	.quad	0x401294156e264e48              # double 4.6446129999999997
	.quad	0x401a2265f0f5a101              # double 6.5335919999999996
	.quad	0x40098a8fc0d2c387              # double 3.1926570000000001
	.quad	0x400a673c4f3ba776              # double 3.300408
	.quad	0x401885aff2a172ac              # double 6.1305540000000001
	.quad	0x4010cce3de6149c7              # double 4.200088
	.quad	0x3ffab2367e414e7f              # double 1.668509
	.quad	0x3ffee69595feda66              # double 1.931295
	.quad	0x3fd6ecfa69be09bb              # double 0.35821399999999998
	.quad	0x401601aef6f8f041              # double 5.5016439999999998
	.quad	0x4016b28d8665e02f              # double 5.6743680000000003
	.quad	0x4022dfe175d13d75              # double 9.4372670000000002
	.quad	0x401c6af8df7a4e7b              # double 7.1044650000000003
	.quad	0x401303c0c1fc8f32              # double 4.7536649999999998
	.quad	0x4007f28e0c9d9d34              # double 2.9934349999999998
	.quad	0x402086a24cc68230              # double 8.2629570000000001
	.quad	0x4001e13c254a3c64              # double 2.2349779999999999
	.size	.L__constant_147x50xf64, 58800

	.type	.L__constant_148xindex,@object  # @__constant_148xindex
	.p2align	6, 0x0
.L__constant_148xindex:
	.quad	0                               # 0x0
	.quad	6                               # 0x6
	.quad	15                              # 0xf
	.quad	24                              # 0x18
	.quad	33                              # 0x21
	.quad	42                              # 0x2a
	.quad	51                              # 0x33
	.quad	58                              # 0x3a
	.quad	64                              # 0x40
	.quad	77                              # 0x4d
	.quad	90                              # 0x5a
	.quad	103                             # 0x67
	.quad	120                             # 0x78
	.quad	137                             # 0x89
	.quad	154                             # 0x9a
	.quad	171                             # 0xab
	.quad	188                             # 0xbc
	.quad	205                             # 0xcd
	.quad	222                             # 0xde
	.quad	239                             # 0xef
	.quad	256                             # 0x100
	.quad	273                             # 0x111
	.quad	290                             # 0x122
	.quad	307                             # 0x133
	.quad	323                             # 0x143
	.quad	339                             # 0x153
	.quad	355                             # 0x163
	.quad	366                             # 0x16e
	.quad	377                             # 0x179
	.quad	386                             # 0x182
	.quad	403                             # 0x193
	.quad	420                             # 0x1a4
	.quad	437                             # 0x1b5
	.quad	458                             # 0x1ca
	.quad	479                             # 0x1df
	.quad	500                             # 0x1f4
	.quad	521                             # 0x209
	.quad	542                             # 0x21e
	.quad	563                             # 0x233
	.quad	584                             # 0x248
	.quad	605                             # 0x25d
	.quad	626                             # 0x272
	.quad	647                             # 0x287
	.quad	668                             # 0x29c
	.quad	689                             # 0x2b1
	.quad	708                             # 0x2c4
	.quad	727                             # 0x2d7
	.quad	746                             # 0x2ea
	.quad	758                             # 0x2f6
	.quad	770                             # 0x302
	.quad	779                             # 0x30b
	.quad	796                             # 0x31c
	.quad	813                             # 0x32d
	.quad	830                             # 0x33e
	.quad	850                             # 0x352
	.quad	871                             # 0x367
	.quad	892                             # 0x37c
	.quad	913                             # 0x391
	.quad	934                             # 0x3a6
	.quad	955                             # 0x3bb
	.quad	976                             # 0x3d0
	.quad	997                             # 0x3e5
	.quad	1018                            # 0x3fa
	.quad	1039                            # 0x40f
	.quad	1060                            # 0x424
	.quad	1081                            # 0x439
	.quad	1100                            # 0x44c
	.quad	1119                            # 0x45f
	.quad	1138                            # 0x472
	.quad	1150                            # 0x47e
	.quad	1162                            # 0x48a
	.quad	1171                            # 0x493
	.quad	1187                            # 0x4a3
	.quad	1204                            # 0x4b4
	.quad	1221                            # 0x4c5
	.quad	1242                            # 0x4da
	.quad	1263                            # 0x4ef
	.quad	1284                            # 0x504
	.quad	1305                            # 0x519
	.quad	1326                            # 0x52e
	.quad	1347                            # 0x543
	.quad	1368                            # 0x558
	.quad	1389                            # 0x56d
	.quad	1410                            # 0x582
	.quad	1430                            # 0x596
	.quad	1451                            # 0x5ab
	.quad	1472                            # 0x5c0
	.quad	1491                            # 0x5d3
	.quad	1510                            # 0x5e6
	.quad	1529                            # 0x5f9
	.quad	1541                            # 0x605
	.quad	1553                            # 0x611
	.quad	1562                            # 0x61a
	.quad	1579                            # 0x62b
	.quad	1596                            # 0x63c
	.quad	1613                            # 0x64d
	.quad	1634                            # 0x662
	.quad	1655                            # 0x677
	.quad	1676                            # 0x68c
	.quad	1697                            # 0x6a1
	.quad	1718                            # 0x6b6
	.quad	1739                            # 0x6cb
	.quad	1759                            # 0x6df
	.quad	1780                            # 0x6f4
	.quad	1801                            # 0x709
	.quad	1822                            # 0x71e
	.quad	1843                            # 0x733
	.quad	1864                            # 0x748
	.quad	1882                            # 0x75a
	.quad	1901                            # 0x76d
	.quad	1920                            # 0x780
	.quad	1932                            # 0x78c
	.quad	1944                            # 0x798
	.quad	1953                            # 0x7a1
	.quad	1969                            # 0x7b1
	.quad	1985                            # 0x7c1
	.quad	2001                            # 0x7d1
	.quad	2020                            # 0x7e4
	.quad	2039                            # 0x7f7
	.quad	2058                            # 0x80a
	.quad	2077                            # 0x81d
	.quad	2096                            # 0x830
	.quad	2115                            # 0x843
	.quad	2134                            # 0x856
	.quad	2153                            # 0x869
	.quad	2172                            # 0x87c
	.quad	2190                            # 0x88e
	.quad	2209                            # 0x8a1
	.quad	2228                            # 0x8b4
	.quad	2245                            # 0x8c5
	.quad	2262                            # 0x8d6
	.quad	2279                            # 0x8e7
	.quad	2289                            # 0x8f1
	.quad	2299                            # 0x8fb
	.quad	2306                            # 0x902
	.quad	2317                            # 0x90d
	.quad	2328                            # 0x918
	.quad	2340                            # 0x924
	.quad	2352                            # 0x930
	.quad	2364                            # 0x93c
	.quad	2376                            # 0x948
	.quad	2388                            # 0x954
	.quad	2400                            # 0x960
	.quad	2412                            # 0x96c
	.quad	2424                            # 0x978
	.quad	2434                            # 0x982
	.quad	2444                            # 0x98c
	.quad	2449                            # 0x991
	.size	.L__constant_148xindex, 1184

	.type	.L__constant_2449xindex,@object # @__constant_2449xindex
	.p2align	6, 0x0
.L__constant_2449xindex:
	.quad	0                               # 0x0
	.quad	1                               # 0x1
	.quad	7                               # 0x7
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	0                               # 0x0
	.quad	1                               # 0x1
	.quad	2                               # 0x2
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	1                               # 0x1
	.quad	2                               # 0x2
	.quad	3                               # 0x3
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	2                               # 0x2
	.quad	3                               # 0x3
	.quad	4                               # 0x4
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	3                               # 0x3
	.quad	4                               # 0x4
	.quad	5                               # 0x5
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	4                               # 0x4
	.quad	5                               # 0x5
	.quad	6                               # 0x6
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	5                               # 0x5
	.quad	6                               # 0x6
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	0                               # 0x0
	.quad	7                               # 0x7
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	28                              # 0x1c
	.quad	0                               # 0x0
	.quad	1                               # 0x1
	.quad	7                               # 0x7
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	0                               # 0x0
	.quad	1                               # 0x1
	.quad	7                               # 0x7
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	0                               # 0x0
	.quad	1                               # 0x1
	.quad	7                               # 0x7
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	1                               # 0x1
	.quad	2                               # 0x2
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	1                               # 0x1
	.quad	2                               # 0x2
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	1                               # 0x1
	.quad	2                               # 0x2
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	2                               # 0x2
	.quad	3                               # 0x3
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	2                               # 0x2
	.quad	3                               # 0x3
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	2                               # 0x2
	.quad	3                               # 0x3
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	3                               # 0x3
	.quad	4                               # 0x4
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	3                               # 0x3
	.quad	4                               # 0x4
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	3                               # 0x3
	.quad	4                               # 0x4
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	4                               # 0x4
	.quad	5                               # 0x5
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	4                               # 0x4
	.quad	5                               # 0x5
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	4                               # 0x4
	.quad	5                               # 0x5
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	5                               # 0x5
	.quad	6                               # 0x6
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	5                               # 0x5
	.quad	6                               # 0x6
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	5                               # 0x5
	.quad	6                               # 0x6
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	6                               # 0x6
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	6                               # 0x6
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	7                               # 0x7
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	49                              # 0x31
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	8                               # 0x8
	.quad	9                               # 0x9
	.quad	10                              # 0xa
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	11                              # 0xb
	.quad	12                              # 0xc
	.quad	13                              # 0xd
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	14                              # 0xe
	.quad	15                              # 0xf
	.quad	16                              # 0x10
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	17                              # 0x11
	.quad	18                              # 0x12
	.quad	19                              # 0x13
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	20                              # 0x14
	.quad	21                              # 0x15
	.quad	22                              # 0x16
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	23                              # 0x17
	.quad	24                              # 0x18
	.quad	25                              # 0x19
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	26                              # 0x1a
	.quad	27                              # 0x1b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	28                              # 0x1c
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	70                              # 0x46
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	29                              # 0x1d
	.quad	30                              # 0x1e
	.quad	31                              # 0x1f
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	32                              # 0x20
	.quad	33                              # 0x21
	.quad	34                              # 0x22
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	35                              # 0x23
	.quad	36                              # 0x24
	.quad	37                              # 0x25
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	38                              # 0x26
	.quad	39                              # 0x27
	.quad	40                              # 0x28
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	41                              # 0x29
	.quad	42                              # 0x2a
	.quad	43                              # 0x2b
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	44                              # 0x2c
	.quad	45                              # 0x2d
	.quad	46                              # 0x2e
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	47                              # 0x2f
	.quad	48                              # 0x30
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	49                              # 0x31
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	91                              # 0x5b
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	50                              # 0x32
	.quad	51                              # 0x33
	.quad	52                              # 0x34
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	53                              # 0x35
	.quad	54                              # 0x36
	.quad	55                              # 0x37
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	56                              # 0x38
	.quad	57                              # 0x39
	.quad	58                              # 0x3a
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	59                              # 0x3b
	.quad	60                              # 0x3c
	.quad	61                              # 0x3d
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	62                              # 0x3e
	.quad	63                              # 0x3f
	.quad	64                              # 0x40
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	65                              # 0x41
	.quad	66                              # 0x42
	.quad	67                              # 0x43
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	68                              # 0x44
	.quad	69                              # 0x45
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	70                              # 0x46
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	112                             # 0x70
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	71                              # 0x47
	.quad	72                              # 0x48
	.quad	73                              # 0x49
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	74                              # 0x4a
	.quad	75                              # 0x4b
	.quad	76                              # 0x4c
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	77                              # 0x4d
	.quad	78                              # 0x4e
	.quad	79                              # 0x4f
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	80                              # 0x50
	.quad	81                              # 0x51
	.quad	82                              # 0x52
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	83                              # 0x53
	.quad	84                              # 0x54
	.quad	85                              # 0x55
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	86                              # 0x56
	.quad	87                              # 0x57
	.quad	88                              # 0x58
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	89                              # 0x59
	.quad	90                              # 0x5a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	91                              # 0x5b
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	133                             # 0x85
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	133                             # 0x85
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	133                             # 0x85
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	92                              # 0x5c
	.quad	93                              # 0x5d
	.quad	94                              # 0x5e
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	133                             # 0x85
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	95                              # 0x5f
	.quad	96                              # 0x60
	.quad	97                              # 0x61
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	98                              # 0x62
	.quad	99                              # 0x63
	.quad	100                             # 0x64
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	101                             # 0x65
	.quad	102                             # 0x66
	.quad	103                             # 0x67
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	104                             # 0x68
	.quad	105                             # 0x69
	.quad	106                             # 0x6a
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	107                             # 0x6b
	.quad	108                             # 0x6c
	.quad	109                             # 0x6d
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	146                             # 0x92
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	146                             # 0x92
	.quad	112                             # 0x70
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	133                             # 0x85
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	133                             # 0x85
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	113                             # 0x71
	.quad	114                             # 0x72
	.quad	115                             # 0x73
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	133                             # 0x85
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	116                             # 0x74
	.quad	117                             # 0x75
	.quad	118                             # 0x76
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	134                             # 0x86
	.quad	135                             # 0x87
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	119                             # 0x77
	.quad	120                             # 0x78
	.quad	121                             # 0x79
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	136                             # 0x88
	.quad	137                             # 0x89
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	122                             # 0x7a
	.quad	123                             # 0x7b
	.quad	124                             # 0x7c
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	138                             # 0x8a
	.quad	139                             # 0x8b
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	125                             # 0x7d
	.quad	126                             # 0x7e
	.quad	127                             # 0x7f
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	140                             # 0x8c
	.quad	141                             # 0x8d
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	146                             # 0x92
	.quad	128                             # 0x80
	.quad	129                             # 0x81
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	142                             # 0x8e
	.quad	143                             # 0x8f
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	146                             # 0x92
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	146                             # 0x92
	.size	.L__constant_2449xindex, 19592

	.type	.L__constant_2449xf64,@object   # @__constant_2449xf64
	.p2align	6, 0x0
.L__constant_2449xf64:
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x412d58059eb851ec              # double 961538.81000000006
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x412d58059eb851ec              # double 961538.81000000006
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x412d58056147ae14              # double 961538.68999999994
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x412d58056147ae14              # double 961538.68999999994
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x412d5804e147ae14              # double 961538.43999999994
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x412d5804e147ae14              # double 961538.43999999994
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x412d58056147ae14              # double 961538.68999999994
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x412d58056147ae14              # double 961538.68999999994
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x412d5805e147ae14              # double 961538.93999999994
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x412d5805e147ae14              # double 961538.93999999994
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x412d5805e147ae14              # double 961538.93999999994
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x412d5805e147ae14              # double 961538.93999999994
	.quad	0x4185174380000000              # double 44230768
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc13781f600000000              # double -1540598
	.quad	0x416b8284c0000000              # double 14423078
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0x412d58049eb851ec              # double 961538.31000000006
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0x3f7fff79c842fa51              # double 0.0078120000000000004
	.quad	0x3f7fff79c842fa51              # double 0.0078120000000000004
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4162570300000000              # double 9615384
	.quad	0x4156020380000000              # double 5769230
	.quad	0x3f7fff79c842fa51              # double 0.0078120000000000004
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x417b828400000000              # double 28846144
	.quad	0x3f7fff79c842fa51              # double 0.0078120000000000004
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0xbfbc000000000000              # double -0.109375
	.quad	0x3fac0010c6f7a0b6              # double 0.054688000000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xbf2ffb480a5accd5              # double -2.4399999999999999E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xbfbc000000000000              # double -0.109375
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0x3fac0010c6f7a0b6              # double 0.054688000000000001
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0x3fc3000431bde82d              # double 0.14843799999999999
	.quad	0xbfbe0008637bd05b              # double -0.117188
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xbf37fc7607c419a0              # double -3.6600000000000001E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0x3fc3000431bde82d              # double 0.14843799999999999
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfbe0008637bd05b              # double -0.117188
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0xbfc0fffbce4217d3              # double -0.13281200000000001
	.quad	0x3fc6000000000000              # double 0.171875
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x3f3ffb480a5accd5              # double 4.8799999999999999E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xbfc0fffbce4217d3              # double -0.13281200000000001
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0x3fc6000000000000              # double 0.171875
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fd47ffde7210be9              # double 0.32031199999999999
	.quad	0xbfe6000000000000              # double -0.6875
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc143f85900000000              # double -2617522
	.quad	0xbf1ffb480a5accd5              # double -1.22E-4
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x4143f85900000000              # double 2617522
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0x3fd47ffde7210be9              # double 0.32031199999999999
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b0440000000              # double -12179490
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfe6000000000000              # double -0.6875
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc1673b0360000000              # double -12179483
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc143f85900000000              # double -2617522
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac90c28f5c29              # double 502564.19
	.quad	0x3fda7ffde7210be9              # double 0.41406199999999999
	.quad	0x3fb60008637bd05b              # double 0.085938000000000001
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x4143f85900000000              # double 2617522
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x3fda7ffde7210be9              # double 0.41406199999999999
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570340000000              # double 9615386
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3fb60008637bd05b              # double 0.085938000000000001
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0x4162570380000000              # double 9615388
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc13781f600000000              # double -1540598
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x410eac907ae147ae              # double 251282.06
	.quad	0x411c4cee00000000              # double 463675.5
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0eeac90fdf3b646              # double -62820.531000000003
	.quad	0x41306ebb00000000              # double 1076923
	.quad	0x416b8284c0000000              # double 14423078
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x411c4cee00000000              # double 463675.5
	.quad	0x4191e1a340000000              # double 75000016
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570340000000              # double 9615386
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0xc13781f700000000              # double -1540599
	.quad	0x416b828500000000              # double 14423080
	.quad	0x412d58049eb851ec              # double 961538.31000000006
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x4162570300000000              # double 9615384
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0x412d58049eb851ec              # double 961538.31000000006
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf2ffb480a5accd5              # double -2.4399999999999999E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0xbfbc000000000000              # double -0.109375
	.quad	0x3fac0010c6f7a0b6              # double 0.054688000000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x4156020380000000              # double 5769230
	.quad	0xbfbc000000000000              # double -0.109375
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4162570300000000              # double 9615384
	.quad	0x417b828400000000              # double 28846144
	.quad	0x3fac0010c6f7a0b6              # double 0.054688000000000001
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf37fc7607c419a0              # double -3.6600000000000001E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fcf000431bde82d              # double 0.24218799999999999
	.quad	0xbfd87ffde7210be9              # double -0.38281199999999999
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc143f85900000000              # double -2617522
	.quad	0xbf3ffb480a5accd5              # double -4.8799999999999999E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0x3fcf000431bde82d              # double 0.24218799999999999
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfd87ffde7210be9              # double -0.38281199999999999
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x3f3ffb480a5accd5              # double 4.8799999999999999E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0xbf9800218def416c              # double -0.023438000000000001
	.quad	0x3fd6000000000000              # double 0.34375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xbf9800218def416c              # double -0.023438000000000001
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3fd6000000000000              # double 0.34375
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xbf1ffb480a5accd5              # double -1.22E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0x3fc8fffbce4217d3              # double 0.19531200000000001
	.quad	0xbfd1800218def417              # double -0.27343800000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x3f3ffb480a5accd5              # double 4.8799999999999999E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x4162570300000000              # double 9615384
	.quad	0xc1673b0360000000              # double -12179483
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x3fc8fffbce4217d3              # double 0.19531200000000001
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580400000000              # double 1923076
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xc1673b0440000000              # double -12179490
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfd1800218def417              # double -0.27343800000000001
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0xbfe1c0010c6f7a0b              # double -0.55468799999999996
	.quad	0xbfd5000000000000              # double -0.328125
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc143f85900000000              # double -2617522
	.quad	0xbf2ffb480a5accd5              # double -2.4399999999999999E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4162570340000000              # double 9615386
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xbfe1c0010c6f7a0b              # double -0.55468799999999996
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0x4162570380000000              # double 9615388
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0xbfd5000000000000              # double -0.328125
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac90c28f5c29              # double 502564.19
	.quad	0x3fe13ffef39085f5              # double 0.53906200000000004
	.quad	0x3f90000000000000              # double 0.015625
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xbf47fc7607c419a0              # double -7.3200000000000001E-4
	.quad	0x40f24228fdf3b646              # double 74786.562000000005
	.quad	0x40f2422200000000              # double 74786.125
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x4143f85900000000              # double 2617522
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4162570340000000              # double 9615386
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x3fe13ffef39085f5              # double 0.53906200000000004
	.quad	0x41a1e1a364000000              # double 150000050
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0x4162570360000000              # double 9615387
	.quad	0xc1673b0540000000              # double -12179498
	.quad	0xc143f85980000000              # double -2617523
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3f90000000000000              # double 0.015625
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f2422a00000000              # double -74786.625
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x41625703c0000000              # double 9615390
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x4156020440000000              # double 5769233
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0eeac90fdf3b646              # double -62820.531000000003
	.quad	0xc13781f700000000              # double -1540599
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x410eac907ae147ae              # double 251282.06
	.quad	0x411c4cec00000000              # double 463675
	.quad	0xbf43fd0d0678c005              # double -6.0999999999999997E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0eeac9083126e98              # double -62820.516000000003
	.quad	0x41306ebb00000000              # double 1076923
	.quad	0x41306ebb00000000              # double 1076923
	.quad	0x416b828500000000              # double 14423080
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x411c4cec00000000              # double 463675
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc13781f600000000              # double -1540598
	.quad	0x416b8284a0000000              # double 14423077
	.quad	0x412d58049eb851ec              # double 961538.31000000006
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x4162570300000000              # double 9615384
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0x412d58051eb851ec              # double 961538.56000000006
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf3ffb480a5accd5              # double -4.8799999999999999E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0x3fc8000000000000              # double 0.1875
	.quad	0x3f9800218def416c              # double 0.023438000000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x3fc8000000000000              # double 0.1875
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4162570300000000              # double 9615384
	.quad	0x417b828400000000              # double 28846144
	.quad	0x3f9800218def416c              # double 0.023438000000000001
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0xbfcf000431bde82d              # double -0.24218799999999999
	.quad	0xbfb0000000000000              # double -0.0625
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xbfcf000431bde82d              # double -0.24218799999999999
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x413d580400000000              # double 1923076
	.quad	0x4156020340000000              # double 5769229
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfb0000000000000              # double -0.0625
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x3f3ffb480a5accd5              # double 4.8799999999999999E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fca000000000000              # double 0.203125
	.quad	0xbfcf000431bde82d              # double -0.24218799999999999
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x3f4bfbdf090f733b              # double 8.5400000000000004E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580400000000              # double 1923076
	.quad	0x4156020380000000              # double 5769230
	.quad	0x3fca000000000000              # double 0.203125
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020340000000              # double 5769229
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfcf000431bde82d              # double -0.24218799999999999
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x41625702c0000000              # double 9615382
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xbf2ffb480a5accd5              # double -2.4399999999999999E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fccfffbce4217d3              # double 0.22656200000000001
	.quad	0x3fd0000000000000              # double 0.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x3fccfffbce4217d3              # double 0.22656200000000001
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0420000000              # double -12179489
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3fd0000000000000              # double 0.25
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b0420000000              # double -12179489
	.quad	0x4162570340000000              # double 9615386
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf47fc7607c419a0              # double -7.3200000000000001E-4
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0xc0f2422a00000000              # double -74786.625
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fc8000000000000              # double 0.1875
	.quad	0xbfea3ffef39085f5              # double -0.82031200000000004
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422200000000              # double 74786.125
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f24228fdf3b646              # double 74786.562000000005
	.quad	0x4162570360000000              # double 9615387
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x3fc8000000000000              # double 0.1875
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0x413d580600000000              # double 1923078
	.quad	0x4156020440000000              # double 5769233
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570380000000              # double 9615388
	.quad	0xc1673b04a0000000              # double -12179493
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580400000000              # double 1923076
	.quad	0x40f2422200000000              # double 74786.125
	.quad	0xc1673b0540000000              # double -12179498
	.quad	0x41625703c0000000              # double 9615390
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfea3ffef39085f5              # double -0.82031200000000004
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020440000000              # double 5769233
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85980000000              # double -2617523
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0xbf43fd0d0678c005              # double -6.0999999999999997E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac9100000000              # double 502564.25
	.quad	0xbfe63ffef39085f5              # double -0.69531200000000004
	.quad	0xbfc4000000000000              # double -0.15625
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x4143f85900000000              # double 2617522
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020440000000              # double 5769233
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x40f2422200000000              # double 74786.125
	.quad	0x413d580600000000              # double 1923078
	.quad	0x4156020440000000              # double 5769233
	.quad	0xbfe63ffef39085f5              # double -0.69531200000000004
	.quad	0x41a1e1a33c000000              # double 150000030
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x4156020440000000              # double 5769233
	.quad	0x417b828500000000              # double 28846160
	.quad	0xbfc4000000000000              # double -0.15625
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a33c000000              # double 150000030
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020440000000              # double 5769233
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x41625702c0000000              # double 9615382
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0eeac9083126e98              # double -62820.516000000003
	.quad	0xc13781f600000000              # double -1540598
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x410eac90f5c28f5c              # double 251282.12
	.quad	0x411c4cefc28f5c29              # double 463675.94
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0eeac91810624dd              # double -62820.546999999999
	.quad	0x41306ebc00000000              # double 1076924
	.quad	0x41306ebb00000000              # double 1076923
	.quad	0x416b8284a0000000              # double 14423077
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020440000000              # double 5769233
	.quad	0x411c4cefc28f5c29              # double 463675.94
	.quad	0x4191e1a340000000              # double 75000016
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x41625703a0000000              # double 9615389
	.quad	0xc1673b04c0000000              # double -12179494
	.quad	0xc13781f700000000              # double -1540599
	.quad	0x416b828580000000              # double 14423084
	.quad	0x412d58051eb851ec              # double 961538.56000000006
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0x412d58049eb851ec              # double 961538.31000000006
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0xbfc4000000000000              # double -0.15625
	.quad	0x3fb1fff79c842fa5              # double 0.070311999999999999
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4156020380000000              # double 5769230
	.quad	0xbfc4000000000000              # double -0.15625
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570300000000              # double 9615384
	.quad	0x417b828400000000              # double 28846144
	.quad	0x3fb1fff79c842fa5              # double 0.070311999999999999
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x3f4bfbdf090f733b              # double 8.5400000000000004E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0x3fb1fff79c842fa5              # double 0.070311999999999999
	.quad	0xbfc3000431bde82d              # double -0.14843799999999999
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b03a0000000              # double -12179485
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0x3fb1fff79c842fa5              # double 0.070311999999999999
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfc3000431bde82d              # double -0.14843799999999999
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0xbfcf000431bde82d              # double -0.24218799999999999
	.quad	0xbfd0000000000000              # double -0.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xbf2ffb480a5accd5              # double -2.4399999999999999E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b0420000000              # double -12179489
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xbfcf000431bde82d              # double -0.24218799999999999
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0420000000              # double -12179489
	.quad	0x4162570340000000              # double 9615386
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfd0000000000000              # double -0.25
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac90c28f5c29              # double 502564.19
	.quad	0x3fe3000000000000              # double 0.59375
	.quad	0xbfe3800000000000              # double -0.609375
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc143f85900000000              # double -2617522
	.quad	0xbf47fc7607c419a0              # double -7.3200000000000001E-4
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x4143f85900000000              # double 2617522
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x4162570380000000              # double 9615388
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x3fe3000000000000              # double 0.59375
	.quad	0x41a1e1a33c000000              # double 150000030
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020440000000              # double 5769233
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b04a0000000              # double -12179493
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xc1673b04a0000000              # double -12179493
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0xbfe3800000000000              # double -0.609375
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a33c000000              # double 150000030
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4162570380000000              # double 9615388
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x4156020440000000              # double 5769233
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fb60008637bd05b              # double 0.085938000000000001
	.quad	0x3ff9400000000000              # double 1.578125
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x3fb60008637bd05b              # double 0.085938000000000001
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4162570360000000              # double 9615387
	.quad	0xc1673b0440000000              # double -12179490
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x4156020440000000              # double 5769233
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3ff9400000000000              # double 1.578125
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x4156020340000000              # double 5769229
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b0440000000              # double -12179490
	.quad	0x4162570360000000              # double 9615387
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x411eac9000000000              # double 502564
	.quad	0xbff49fff79c842fa              # double -1.2890619999999999
	.quad	0xbff2000000000000              # double -1.125
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85800000000              # double 2617520
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x41625703a0000000              # double 9615389
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020340000000              # double 5769229
	.quad	0xbff49fff79c842fa              # double -1.2890619999999999
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570280000000              # double 9615380
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020300000000              # double 5769228
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b04c0000000              # double -12179494
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbff2000000000000              # double -1.125
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x41625702c0000000              # double 9615382
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020300000000              # double 5769228
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc0eeac91810624dd              # double -62820.546999999999
	.quad	0xc13781f700000000              # double -1540599
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x410eac907ae147ae              # double 251282.06
	.quad	0x411c4ce8c28f5c29              # double 463674.19
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0eeac8f9db22d0e              # double -62820.487999999998
	.quad	0x41306ebb00000000              # double 1076923
	.quad	0x41306ebc00000000              # double 1076924
	.quad	0x416b828580000000              # double 14423084
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x411c4ce8c28f5c29              # double 463674.19
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4162570280000000              # double 9615380
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0xc13781f600000000              # double -1540598
	.quad	0x416b8283e0000000              # double 14423071
	.quad	0x412d58049eb851ec              # double 961538.31000000006
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x4162570300000000              # double 9615384
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0x412d580500000000              # double 961538.5
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fe6800000000000              # double 0.703125
	.quad	0xbfc0fffbce4217d3              # double -0.13281200000000001
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x4143f85900000000              # double 2617522
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x3fe6800000000000              # double 0.703125
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580400000000              # double 1923076
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x41625702e0000000              # double 9615383
	.quad	0x417b828500000000              # double 28846160
	.quad	0xbfc0fffbce4217d3              # double -0.13281200000000001
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf2ffb480a5accd5              # double -2.4399999999999999E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fd27ffde7210be9              # double 0.28906199999999999
	.quad	0x3fd9800218def417              # double 0.39843800000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x3fd27ffde7210be9              # double 0.28906199999999999
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x4162570360000000              # double 9615387
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b03c0000000              # double -12179486
	.quad	0x41625702e0000000              # double 9615383
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3fd9800218def417              # double 0.39843800000000001
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf47fc7607c419a0              # double -7.3200000000000001E-4
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3fd0000000000000              # double 0.25
	.quad	0xbfd1800218def417              # double -0.27343800000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85880000000              # double 2617521
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x3fd0000000000000              # double 0.25
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x413d580600000000              # double 1923078
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570380000000              # double 9615388
	.quad	0xc1673b04a0000000              # double -12179493
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580400000000              # double 1923076
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b04a0000000              # double -12179493
	.quad	0x4162570380000000              # double 9615388
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbfd1800218def417              # double -0.27343800000000001
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0xbff860008637bd06              # double -1.5234380000000001
	.quad	0xbfb4000000000000              # double -0.078125
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc143f85880000000              # double -2617521
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x4143f85800000000              # double 2617520
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020440000000              # double 5769233
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4162570360000000              # double 9615387
	.quad	0xc1673b0440000000              # double -12179490
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0x413d580600000000              # double 1923078
	.quad	0x4156020400000000              # double 5769232
	.quad	0xbff860008637bd06              # double -1.5234380000000001
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020300000000              # double 5769228
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0440000000              # double -12179490
	.quad	0x4162570360000000              # double 9615387
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0xbfb4000000000000              # double -0.078125
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x41625702c0000000              # double 9615382
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac90c28f5c29              # double 502564.19
	.quad	0x3fee3ffef39085f5              # double 0.94531200000000004
	.quad	0xbff0400000000000              # double -1.015625
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422a00000000              # double 74786.625
	.quad	0xc143f85980000000              # double -2617523
	.quad	0xbf1ffb480a5accd5              # double -1.22E-4
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x4143f85980000000              # double 2617523
	.quad	0xc0f2422a00000000              # double -74786.625
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x4162570280000000              # double 9615380
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020380000000              # double 5769230
	.quad	0x3fee3ffef39085f5              # double 0.94531200000000004
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24220fdf3b646              # double -74786.062000000005
	.quad	0x413d580600000000              # double 1923078
	.quad	0x4156020440000000              # double 5769233
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020480000000              # double 5769234
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020380000000              # double 5769230
	.quad	0x417b828400000000              # double 28846144
	.quad	0xbff0400000000000              # double -1.015625
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x41560204c0000000              # double 5769235
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x4162570260000000              # double 9615379
	.quad	0x40f24220fdf3b646              # double 74786.062000000005
	.quad	0x4156020440000000              # double 5769233
	.quad	0x413d580600000000              # double 1923078
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f24220fdf3b646              # double -74786.062000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac9100000000              # double 502564.25
	.quad	0x4010d800218def41              # double 4.2109379999999996
	.quad	0x3ffa000000000000              # double 1.625
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x40f2422afdf3b646              # double 74786.687000000005
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac930a3d70a4              # double -125641.19
	.quad	0x4143f85a00000000              # double 2617524
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020300000000              # double 5769228
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x4162570280000000              # double 9615380
	.quad	0x40f2422a00000000              # double 74786.625
	.quad	0x413d580600000000              # double 1923078
	.quad	0x41560204c0000000              # double 5769235
	.quad	0x4010d800218def41              # double 4.2109379999999996
	.quad	0x41a1e1a378000000              # double 150000060
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570520000000              # double 9615401
	.quad	0xc1673b0740000000              # double -12179514
	.quad	0xc143f85a80000000              # double -2617525
	.quad	0x417b828700000000              # double 28846192
	.quad	0x4156020580000000              # double 5769238
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4156020300000000              # double 5769228
	.quad	0x413d580400000000              # double 1923076
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0xc143f85980000000              # double -2617523
	.quad	0x4156020440000000              # double 5769233
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3ffa000000000000              # double 1.625
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a364000000              # double 150000050
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f2422afdf3b646              # double -74786.687000000005
	.quad	0xc1673b05c0000000              # double -12179502
	.quad	0x4162570520000000              # double 9615401
	.quad	0x40f2422afdf3b646              # double 74786.687000000005
	.quad	0x41560205c0000000              # double 5769239
	.quad	0x413d580700000000              # double 1923079
	.quad	0xc0eeac8f9db22d0e              # double -62820.487999999998
	.quad	0xc13781f600000000              # double -1540598
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x410eac8f851eb852              # double 251281.94
	.quad	0x411c4ce8c28f5c29              # double 463674.19
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0x40f2421cfdf3b646              # double 74785.812000000005
	.quad	0x40f2422c00000000              # double 74786.75
	.quad	0xc0eeac8e1cac0831              # double -62820.440999999999
	.quad	0x41306eba00000000              # double 1076922
	.quad	0x41306ebb00000000              # double 1076923
	.quad	0x416b8283e0000000              # double 14423071
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x411c4ce8c28f5c29              # double 463674.19
	.quad	0x4191e1a240000000              # double 74999952
	.quad	0xc0f2422c00000000              # double -74786.75
	.quad	0x41625702a0000000              # double 9615381
	.quad	0xc1673b0200000000              # double -12179472
	.quad	0xc13781f500000000              # double -1540597
	.quad	0x416b8282a0000000              # double 14423061
	.quad	0x412d580500000000              # double 961538.5
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570300000000              # double 9615384
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0x412d580500000000              # double 961538.5
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac90c28f5c29              # double 502564.19
	.quad	0xbfa8000000000000              # double -0.046875
	.quad	0xbfcb000431bde82d              # double -0.21093799999999999
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020400000000              # double 5769232
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x4162570360000000              # double 9615387
	.quad	0xc1673b0460000000              # double -12179491
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xbfa8000000000000              # double -0.046875
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570320000000              # double 9615385
	.quad	0x417b828500000000              # double 28846160
	.quad	0xbfcb000431bde82d              # double -0.21093799999999999
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4162570300000000              # double 9615384
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x3f37fc7607c419a0              # double 3.6600000000000001E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac90c28f5c29              # double 502564.19
	.quad	0x3f7fff79c842fa51              # double 0.0078120000000000004
	.quad	0xbfc3000431bde82d              # double -0.14843799999999999
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x4162570380000000              # double 9615388
	.quad	0xc1673b03e0000000              # double -12179487
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x3f7fff79c842fa51              # double 0.0078120000000000004
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0xc1673b04a0000000              # double -12179493
	.quad	0x4162570320000000              # double 9615385
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0xbfc3000431bde82d              # double -0.14843799999999999
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570320000000              # double 9615385
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f24226fdf3b646              # double 74786.437000000005
	.quad	0xbf4bfbdf090f733b              # double -8.5400000000000004E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0x4143f85900000000              # double 2617522
	.quad	0x411eac907ae147ae              # double 502564.12
	.quad	0x3f9800218def416c              # double 0.023438000000000001
	.quad	0x3fd2000000000000              # double 0.28125
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x417b828500000000              # double 28846160
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x3f9800218def416c              # double 0.023438000000000001
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x4156020400000000              # double 5769232
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0380000000              # double -12179484
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3fd2000000000000              # double 0.28125
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x4156020400000000              # double 5769232
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0x41625702a0000000              # double 9615381
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xbf1ffb480a5accd5              # double -1.22E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4143f85880000000              # double 2617521
	.quad	0x411eac9000000000              # double 502564
	.quad	0x3ff29fff79c842fa              # double 1.1640619999999999
	.quad	0x3ff560008637bd06              # double 1.3359380000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x417b828400000000              # double 28846144
	.quad	0x4156020380000000              # double 5769230
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x41625702c0000000              # double 9615382
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x413d580500000000              # double 1923077
	.quad	0x4156020400000000              # double 5769232
	.quad	0x3ff29fff79c842fa              # double 1.1640619999999999
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4133900300000000              # double 1282051
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x413d580400000000              # double 1923076
	.quad	0x4156020300000000              # double 5769228
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b04c0000000              # double -12179494
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4156020300000000              # double 5769228
	.quad	0x413d580400000000              # double 1923076
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x4162570260000000              # double 9615379
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3ff560008637bd06              # double 1.3359380000000001
	.quad	0x4133900300000000              # double 1282051
	.quad	0x41a1e1a2d8000000              # double 149999980
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x4156020300000000              # double 5769228
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41625703a0000000              # double 9615389
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0x413d580500000000              # double 1923077
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f24220fdf3b646              # double 74786.062000000005
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f2422afdf3b646              # double -74786.687000000005
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x411eac903d70a3d7              # double 502564.06
	.quad	0xc008e00000000000              # double -3.109375
	.quad	0xc005c00000000000              # double -2.71875
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0x40f2421cfdf3b646              # double 74785.812000000005
	.quad	0xc143f85900000000              # double -2617522
	.quad	0xbf1ffb480a5accd5              # double -1.22E-4
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac8f0a3d70a4              # double -125640.94
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0x4143f85980000000              # double 2617523
	.quad	0x417b828500000000              # double 28846160
	.quad	0x4156020440000000              # double 5769233
	.quad	0x40f2422afdf3b646              # double 74786.687000000005
	.quad	0x4162570520000000              # double 9615401
	.quad	0xc1673b05c0000000              # double -12179502
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x413d580400000000              # double 1923076
	.quad	0x4156020300000000              # double 5769228
	.quad	0xc008e00000000000              # double -3.109375
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4133900400000000              # double 1282052
	.quad	0xc0f2422efdf3b646              # double -74786.937000000005
	.quad	0x413d580600000000              # double 1923078
	.quad	0x41560204c0000000              # double 5769235
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0xc143f85780000000              # double -2617519
	.quad	0x41560202c0000000              # double 5769227
	.quad	0xc0f2422a00000000              # double -74786.625
	.quad	0x4156020480000000              # double 5769234
	.quad	0x413d580600000000              # double 1923078
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0740000000              # double -12179514
	.quad	0x4162570520000000              # double 9615401
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x4156020300000000              # double 5769228
	.quad	0x417b828400000000              # double 28846144
	.quad	0xc005c00000000000              # double -2.71875
	.quad	0x4133900400000000              # double 1282052
	.quad	0x41a1e1a328000000              # double 150000020
	.quad	0x4143f85980000000              # double 2617523
	.quad	0x4156020440000000              # double 5769233
	.quad	0x417b828500000000              # double 28846160
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x4162570260000000              # double 9615379
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x413d580300000000              # double 1923075
	.quad	0xc0feac930a3d70a4              # double -125641.19
	.quad	0xc143f85a80000000              # double -2617525
	.quad	0x40f2422afdf3b646              # double 74786.687000000005
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0xc0f2422c00000000              # double -74786.75
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc0f2422efdf3b646              # double -74786.937000000005
	.quad	0x4143f85980000000              # double 2617523
	.quad	0x411eac9100000000              # double 502564.25
	.quad	0xc00c4fffbce4217d              # double -3.5390619999999999
	.quad	0x3ff4e0008637bd06              # double 1.3046880000000001
	.quad	0xc0feac9000000000              # double -125641
	.quad	0x40f24228fdf3b646              # double 74786.562000000005
	.quad	0x3f47fc7607c419a0              # double 7.3200000000000001E-4
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24228fdf3b646              # double -74786.562000000005
	.quad	0x4143f85a00000000              # double 2617524
	.quad	0x417b828700000000              # double 28846192
	.quad	0x41560205c0000000              # double 5769239
	.quad	0x40f2421cfdf3b646              # double 74785.812000000005
	.quad	0x41625702a0000000              # double 9615381
	.quad	0x40f2421cfdf3b646              # double 74785.812000000005
	.quad	0x413d580600000000              # double 1923078
	.quad	0x4156020440000000              # double 5769233
	.quad	0xc00c4fffbce4217d              # double -3.5390619999999999
	.quad	0x41a1e1a364000000              # double 150000050
	.quad	0x4133900200000000              # double 1282050
	.quad	0xc0f24220fdf3b646              # double -74786.062000000005
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc1673b0240000000              # double -12179474
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0x4156020580000000              # double 5769238
	.quad	0x413d580700000000              # double 1923079
	.quad	0x40f2422c00000000              # double 74786.75
	.quad	0xc1673b0200000000              # double -12179472
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x41560204c0000000              # double 5769235
	.quad	0x417b828500000000              # double 28846160
	.quad	0x3ff4e0008637bd06              # double 1.3046880000000001
	.quad	0x4133900200000000              # double 1282050
	.quad	0x41a1e1a300000000              # double 1.5E+8
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x4156020380000000              # double 5769230
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x41625701c0000000              # double 9615374
	.quad	0x40f2422000000000              # double 74786
	.quad	0x413d580400000000              # double 1923076
	.quad	0xc0eeac8e1cac0831              # double -62820.440999999999
	.quad	0xc13781f500000000              # double -1540597
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc0f24220fdf3b646              # double -74786.062000000005
	.quad	0x4143f85800000000              # double 2617520
	.quad	0x410eac9000000000              # double 251282
	.quad	0x411c4cf400000000              # double 463677
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc0eeac91810624dd              # double -62820.546999999999
	.quad	0x41306eba00000000              # double 1076922
	.quad	0x416b8282a0000000              # double 14423061
	.quad	0x40f24228fdf3b646              # double 74786.562000000005
	.quad	0x413d580400000000              # double 1923076
	.quad	0x4156020380000000              # double 5769230
	.quad	0x411c4cf400000000              # double 463677
	.quad	0x4191e1a2c0000000              # double 74999984
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc1673b04c0000000              # double -12179494
	.quad	0xc13781f700000000              # double -1540599
	.quad	0x412d580500000000              # double 961538.5
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570300000000              # double 9615384
	.quad	0x4185174380000000              # double 44230768
	.quad	0x413781f600000000              # double 1540598
	.quad	0x416b8284c0000000              # double 14423078
	.quad	0xc0feac90f5c28f5c              # double -125641.06
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0x413781f600000000              # double 1540598
	.quad	0x410eac907ae147ae              # double 251282.06
	.quad	0xc11c4ced00000000              # double -463675.25
	.quad	0xc0eeac90a3d70a3d              # double -62820.519999999997
	.quad	0xc1306ebb00000000              # double -1076923
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b0400000000              # double -12179488
	.quad	0x4162570320000000              # double 9615385
	.quad	0x416b8284c0000000              # double 14423078
	.quad	0xc11c4ced00000000              # double -463675.25
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x413781f600000000              # double 1540598
	.quad	0x416b8284c0000000              # double 14423078
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24224fdf3b646              # double 74786.312000000005
	.quad	0x3f2ffb480a5accd5              # double 2.4399999999999999E-4
	.quad	0xc0f24226fdf3b646              # double -74786.437000000005
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0xc0eeac90a3d70a3d              # double -62820.519999999997
	.quad	0x413781f600000000              # double 1540598
	.quad	0x410eac907ae147ae              # double 251282.06
	.quad	0xc11c4cea7ae147ae              # double -463674.62
	.quad	0xc0eeac8f9db22d0e              # double -62820.487999999998
	.quad	0xc1306ebb00000000              # double -1076923
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x41625702a0000000              # double 9615381
	.quad	0xc1306ebb00000000              # double -1076923
	.quad	0x416b8284c0000000              # double 14423078
	.quad	0xc11c4cea7ae147ae              # double -463674.62
	.quad	0x4191e1a2c0000000              # double 74999984
	.quad	0x413781f600000000              # double 1540598
	.quad	0x416b8283e0000000              # double 14423071
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85880000000              # double -2617521
	.quad	0x40f24222fdf3b646              # double 74786.187000000005
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0eeac8f9db22d0e              # double -62820.487999999998
	.quad	0x413781f600000000              # double 1540598
	.quad	0x410eac907ae147ae              # double 251282.06
	.quad	0xc11c4cf100000000              # double -463676.25
	.quad	0xc0eeac9160418937              # double -62820.542999999998
	.quad	0xc1306ebc00000000              # double -1076924
	.quad	0xc0f2422800000000              # double -74786.5
	.quad	0x4156020400000000              # double 5769232
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b04c0000000              # double -12179494
	.quad	0x41625703a0000000              # double 9615389
	.quad	0xc1306ebb00000000              # double -1076923
	.quad	0x416b8283e0000000              # double 14423071
	.quad	0xc11c4cf100000000              # double -463676.25
	.quad	0x4191e1a340000000              # double 75000016
	.quad	0x413781f700000000              # double 1540599
	.quad	0x416b828580000000              # double 14423084
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85900000000              # double -2617522
	.quad	0x40f2422800000000              # double 74786.5
	.quad	0xbf1ffb480a5accd5              # double -1.22E-4
	.quad	0xc0f24224fdf3b646              # double -74786.312000000005
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0eeac9160418937              # double -62820.542999999998
	.quad	0x413781f700000000              # double 1540599
	.quad	0x410eac907ae147ae              # double 251282.06
	.quad	0xc11c4ce8c28f5c29              # double -463674.19
	.quad	0xc0eeac8f9db22d0e              # double -62820.487999999998
	.quad	0xc1306ebb00000000              # double -1076923
	.quad	0xc0f24222fdf3b646              # double -74786.187000000005
	.quad	0x41560203c0000000              # double 5769231
	.quad	0x413d580500000000              # double 1923077
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0340000000              # double -12179482
	.quad	0x4162570260000000              # double 9615379
	.quad	0xc1306ebc00000000              # double -1076924
	.quad	0x416b828580000000              # double 14423084
	.quad	0xc11c4ce8c28f5c29              # double -463674.19
	.quad	0x4191e1a300000000              # double 7.5E+7
	.quad	0x413781f600000000              # double 1540598
	.quad	0x416b8283e0000000              # double 14423071
	.quad	0xc0feac8f0a3d70a4              # double -125640.94
	.quad	0xc143f85780000000              # double -2617519
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0x3f47fc7607c419a0              # double 7.3200000000000001E-4
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0f2422400000000              # double -74786.25
	.quad	0xc0eeac8f9db22d0e              # double -62820.487999999998
	.quad	0x413781f600000000              # double 1540598
	.quad	0x410eac8ef5c28f5c              # double 251281.87
	.quad	0xc11c4ce800000000              # double -463674
	.quad	0xc0eeac8e1cac0831              # double -62820.440999999999
	.quad	0xc1306eba00000000              # double -1076922
	.quad	0xc0f2422200000000              # double -74786.125
	.quad	0x41560202c0000000              # double 5769227
	.quad	0x413d580300000000              # double 1923075
	.quad	0x40f2422400000000              # double 74786.25
	.quad	0xc1673b0240000000              # double -12179474
	.quad	0x41625701c0000000              # double 9615374
	.quad	0xc1306ebb00000000              # double -1076923
	.quad	0x416b8283e0000000              # double 14423071
	.quad	0xc11c4ce800000000              # double -463674
	.quad	0x4191e1a200000000              # double 74999936
	.quad	0x413781f500000000              # double 1540597
	.quad	0x416b8282a0000000              # double 14423061
	.quad	0xc0feac9000000000              # double -125641
	.quad	0xc143f85800000000              # double -2617520
	.quad	0x40f2422000000000              # double 74786
	.quad	0x3f5001d5c31593e6              # double 9.77E-4
	.quad	0xc0f2422600000000              # double -74786.375
	.quad	0xc0eeac8e1cac0831              # double -62820.440999999999
	.quad	0x413781f500000000              # double 1540597
	.quad	0x410eac9000000000              # double 251282
	.quad	0xc11c4cf3c28f5c29              # double -463676.94
	.quad	0xc0eeac9160418937              # double -62820.542999999998
	.quad	0xc0f24228fdf3b646              # double -74786.562000000005
	.quad	0x4156020380000000              # double 5769230
	.quad	0x413d580400000000              # double 1923076
	.quad	0x40f2422600000000              # double 74786.375
	.quad	0xc1673b04c0000000              # double -12179494
	.quad	0xc1306eba00000000              # double -1076922
	.quad	0x416b8282a0000000              # double 14423061
	.quad	0xc11c4cf3c28f5c29              # double -463676.94
	.quad	0x4191e1a2c0000000              # double 74999984
	.quad	0x413781f700000000              # double 1540599
	.quad	0xc0eeac91810624dd              # double -62820.546999999999
	.quad	0xc13781f700000000              # double -1540599
	.quad	0xc0eeac9160418937              # double -62820.542999999998
	.quad	0x413781f700000000              # double 1540599
	.quad	0x40feac90f5c28f5c              # double 125641.06
	.size	.L__constant_2449xf64, 19592

	.section	".note.GNU-stack","",@progbits
