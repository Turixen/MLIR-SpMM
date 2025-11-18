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
	li	a6, 0
	li	a2, 0
	fmv.d.x	fa0, zero
	li	a3, 146
	j	.LBB0_2
.LBB0_1:                                #   in Loop: Header=BB0_2 Depth=1
	addi	a2, a2, 1
	addi	a6, a6, 1176
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	blt	a3, a2, .LBB0_5
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=1
	li	a4, 0
	mv	a0, a6
	bltz	a3, .LBB0_1
.LBB0_4:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a5, a1, a0
	fld	fa5, 0(a5)
	fadd.d	fa0, fa0, fa5
	addi	a4, a4, 1
	addi	a0, a0, 8
	bge	a3, a4, .LBB0_4
	j	.LBB0_1
.LBB0_5:
	ret
.Lfunc_end0:
	.size	compute_sum, .Lfunc_end0-compute_sum
	.cfi_endproc
                                        # -- End function
	.globl	sparse_sparse_matmul            # -- Begin function sparse_sparse_matmul
	.p2align	1
	.type	sparse_sparse_matmul,@function
sparse_sparse_matmul:                   # @sparse_sparse_matmul
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
	slli	a1, a1, 3
	sub	sp, sp, a1
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xf0, 0x00, 0x22, 0x11, 0x08, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 112 + 8 * vlenb
	li	t6, 0
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	a6, 424(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	t0, 416(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	t1, 408(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	t2, 400(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	t3, 392(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	t5, 384(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	t4, 376(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	s4, 144(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	s5, 304(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	s6, 224(a1)
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	ld	s7, 264(a1)
	li	s2, 146
	vsetivli	zero, 16, e32, m4, ta, ma
	vid.v	v4
	vsetvli	zero, zero, e64, m8, ta, ma
	vmv.v.i	v8, 0
	addi	a1, sp, 16
	vs8r.v	v8, (a1)                        # Unknown-size Folded Spill
	li	s3, 1176
	j	.LBB1_2
.LBB1_1:                                #   in Loop: Header=BB1_2 Depth=1
	addi	t6, t6, 1
.LBB1_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_5 Depth 2
                                        #       Child Loop BB1_8 Depth 3
	blt	s2, t6, .LBB1_11
# %bb.3:                                #   in Loop: Header=BB1_2 Depth=1
	slli	a1, t6, 3
	add	a1, a1, a2
	lwu	a3, 4(a1)
	lwu	a4, 0(a1)
	lwu	a5, 12(a1)
	lwu	a1, 8(a1)
	slli	a3, a3, 32
	or	s9, a3, a4
	slli	a5, a5, 32
	or	s8, a5, a1
	mul	a3, t6, s3
	add	s11, t5, a3
	j	.LBB1_5
.LBB1_4:                                #   in Loop: Header=BB1_5 Depth=2
	addi	s9, s9, 1
.LBB1_5:                                #   Parent Loop BB1_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_8 Depth 3
	bge	s9, s8, .LBB1_1
# %bb.6:                                #   in Loop: Header=BB1_5 Depth=2
	slli	a1, s9, 3
	add	a4, a7, a1
	lwu	a5, 0(a4)
	lwu	a4, 4(a4)
	add	a1, a1, s4
	fld	fa5, 0(a1)
	slli	a5, a5, 3
	slli	a4, a4, 35
	or	a4, a4, a5
	add	a4, a4, s6
	lwu	a1, 4(a4)
	lwu	a5, 0(a4)
	lwu	s0, 12(a4)
	lwu	a4, 8(a4)
	slli	a1, a1, 32
	or	a5, a5, a1
	slli	s0, s0, 32
	or	s10, s0, a4
	slli	s1, a5, 3
	add	a1, s5, s1
	add	s1, s1, s7
	sub	s0, s10, a5
	j	.LBB1_8
.LBB1_7:                                #   in Loop: Header=BB1_8 Depth=3
	vsetvli	zero, zero, e32, m4, ta, ma
	vmslt.vx	v0, v4, a4
	addi	a3, sp, 16
	vl8r.v	v8, (a3)                        # Unknown-size Folded Reload
	vmv8r.v	v16, v8
	vsetvli	zero, zero, e64, m8, ta, mu
	vle64.v	v16, (s1), v0.t
	vsll.vi	v24, v16, 3
	vmv8r.v	v16, v8
	vluxei64.v	v16, (s11), v24, v0.t
	vle64.v	v8, (a1), v0.t
	vfmul.vf	v8, v8, fa5
	vfadd.vv	v8, v16, v8
	vsoxei64.v	v8, (s11), v24, v0.t
	addi	a5, a5, 16
	addi	a1, a1, 128
	addi	s1, s1, 128
	addi	s0, s0, -16
.LBB1_8:                                #   Parent Loop BB1_2 Depth=1
                                        #     Parent Loop BB1_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	bge	a5, s10, .LBB1_4
# %bb.9:                                #   in Loop: Header=BB1_8 Depth=3
	li	a3, 16
	mv	a4, s0
	blt	s0, a3, .LBB1_7
# %bb.10:                               #   in Loop: Header=BB1_8 Depth=3
	li	a4, 16
	j	.LBB1_7
.LBB1_11:
	sd	t4, 0(a0)
	sd	t5, 8(a0)
	sd	t3, 16(a0)
	sd	t2, 24(a0)
	sd	t1, 32(a0)
	sd	t0, 40(a0)
	sd	a6, 48(a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
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
	.size	sparse_sparse_matmul, .Lfunc_end1-sparse_sparse_matmul
	.cfi_endproc
                                        # -- End function
	.section	.srodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI2_0:
	.quad	0x42bf6a6787af39d8              # double 34541863939897.844
	.text
	.globl	main
	.p2align	1
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -816
	.cfi_def_cfa_offset 816
	sd	ra, 808(sp)                     # 8-byte Folded Spill
	sd	s0, 800(sp)                     # 8-byte Folded Spill
	sd	s1, 792(sp)                     # 8-byte Folded Spill
	sd	s2, 784(sp)                     # 8-byte Folded Spill
	sd	s3, 776(sp)                     # 8-byte Folded Spill
	sd	s4, 768(sp)                     # 8-byte Folded Spill
	sd	s5, 760(sp)                     # 8-byte Folded Spill
	sd	s6, 752(sp)                     # 8-byte Folded Spill
	sd	s7, 744(sp)                     # 8-byte Folded Spill
	sd	s8, 736(sp)                     # 8-byte Folded Spill
	sd	s9, 728(sp)                     # 8-byte Folded Spill
	sd	s10, 720(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	csrr	a0, vlenb
	li	a1, 6
	mul	a0, a0, a1
	sub	sp, sp, a0
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xb0, 0x06, 0x22, 0x11, 0x06, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 816 + 6 * vlenb
	lui	a0, 42
	addiw	a0, a0, 904
	call	malloc
	mv	s0, a0
	addi	a0, a0, 63
	andi	s9, a0, -64
	addi	a0, sp, 560
	call	assemble_sparse_tensor1
	addi	a0, sp, 680
	vsetivli	zero, 4, e64, m2, ta, ma
	vle64.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 720
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 616
	vsetivli	zero, 8, e64, m4, ta, ma
	vle64.v	v8, (a0)
	addi	a0, sp, 720
	vs4r.v	v8, (a0)                        # Unknown-size Folded Spill
	ld	s2, 608(sp)
	ld	s3, 600(sp)
	ld	s4, 592(sp)
	ld	s5, 584(sp)
	ld	s6, 576(sp)
	ld	s7, 568(sp)
	ld	s8, 560(sp)
	ld	s1, 712(sp)
	addi	a0, sp, 400
	addi	s10, sp, 400
	call	assemble_sparse_tensor2
	vsetivli	zero, 16, e64, m8, ta, ma
	vle64.v	v8, (s10)
	addi	a0, sp, 528
	vsetivli	zero, 4, e64, m2, ta, ma
	vle64.v	v16, (a0)
	sd	zero, 280(sp)
	sd	s9, 272(sp)
	sd	s0, 264(sp)
	sd	s1, 96(sp)
	addi	a0, sp, 720
	vl4r.v	v20, (a0)                       # Unknown-size Folded Reload
	vsetivli	zero, 8, e64, m4, ta, ma
	vse64.v	v20, (sp)
	li	a0, 147
	vsetivli	zero, 4, e64, m2, ta, ma
	vmv.v.x	v18, a0
	li	a0, 1
	vslide1down.vx	v18, v18, a0
	addi	a0, sp, 288
	vse64.v	v18, (a0)
	addi	a0, sp, 232
	vse64.v	v16, (a0)
	addi	a0, sp, 104
	vsetivli	zero, 16, e64, m8, ta, ma
	vse64.v	v8, (a0)
	addi	a1, sp, 64
	addi	a0, sp, 344
	csrr	a2, vlenb
	slli	a2, a2, 2
	add	a2, a2, sp
	addi	a2, a2, 720
	vl2r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsetivli	zero, 4, e64, m2, ta, ma
	vse64.v	v8, (a1)
	mv	a1, s8
	mv	a2, s7
	mv	a3, s6
	mv	a4, s5
	mv	a5, s4
	mv	a6, s3
	mv	a7, s2
	call	sparse_sparse_matmul
	ld	a6, 392(sp)
	ld	a5, 384(sp)
	ld	a4, 376(sp)
	ld	a3, 368(sp)
	ld	a2, 360(sp)
	ld	a1, 352(sp)
	ld	a0, 344(sp)
	call	compute_sum
	lui	a0, %hi(.LCPI2_0)
	fld	fa5, %lo(.LCPI2_0)(a0)
	feq.d	a0, fa0, fa5
	xori	a0, a0, 1
	csrr	a1, vlenb
	li	a2, 6
	mul	a1, a1, a2
	add	sp, sp, a1
	ld	ra, 808(sp)                     # 8-byte Folded Reload
	ld	s0, 800(sp)                     # 8-byte Folded Reload
	ld	s1, 792(sp)                     # 8-byte Folded Reload
	ld	s2, 784(sp)                     # 8-byte Folded Reload
	ld	s3, 776(sp)                     # 8-byte Folded Reload
	ld	s4, 768(sp)                     # 8-byte Folded Reload
	ld	s5, 760(sp)                     # 8-byte Folded Reload
	ld	s6, 752(sp)                     # 8-byte Folded Reload
	ld	s7, 744(sp)                     # 8-byte Folded Reload
	ld	s8, 736(sp)                     # 8-byte Folded Reload
	ld	s9, 728(sp)                     # 8-byte Folded Reload
	ld	s10, 720(sp)                    # 8-byte Folded Reload
	addi	sp, sp, 816
	ret
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0                          # -- Begin function assemble_sparse_tensor1
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
	.globl	assemble_sparse_tensor1
	.p2align	1
	.type	assemble_sparse_tensor1,@function
assemble_sparse_tensor1:                # @assemble_sparse_tensor1
	.cfi_startproc
# %bb.0:
	lui	a1, %hi(.L__constant_148xindex_0)
	addi	a1, a1, %lo(.L__constant_148xindex_0)
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
	.size	assemble_sparse_tensor1, .Lfunc_end3-assemble_sparse_tensor1
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0                          # -- Begin function assemble_sparse_tensor2
.LCPI4_0:
	.quad	0                               # 0x0
	.quad	2441                            # 0x989
	.quad	1                               # 0x1
	.quad	147                             # 0x93
.LCPI4_1:
	.quad	0                               # 0x0
	.quad	2441                            # 0x989
	.quad	1                               # 0x1
	.quad	3735928559                      # 0xdeadbeef
.LCPI4_2:
	.quad	0                               # 0x0
	.quad	148                             # 0x94
	.quad	1                               # 0x1
	.quad	3735928559                      # 0xdeadbeef
	.text
	.globl	assemble_sparse_tensor2
	.p2align	1
	.type	assemble_sparse_tensor2,@function
assemble_sparse_tensor2:                # @assemble_sparse_tensor2
	.cfi_startproc
# %bb.0:
	lui	a1, %hi(.L__constant_148xindex)
	addi	a1, a1, %lo(.L__constant_148xindex)
	ld	a2, 1176(a1)
	lui	a3, %hi(.L__constant_2441xindex)
	addi	a3, a3, %lo(.L__constant_2441xindex)
	lui	a4, %hi(.L__constant_2441xf64)
	addi	a4, a4, %lo(.L__constant_2441xf64)
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
	lui	a2, %hi(.LCPI4_0)
	addi	a2, a2, %lo(.LCPI4_0)
	vsetivli	zero, 4, e64, m2, ta, ma
	vle64.v	v8, (a2)
	addi	a1, a1, -273
	sd	a1, 0(a0)
	addi	a1, a0, 96
	vse64.v	v8, (a1)
	lui	a1, %hi(.LCPI4_1)
	addi	a1, a1, %lo(.LCPI4_1)
	vle64.v	v8, (a1)
	lui	a1, %hi(.LCPI4_2)
	addi	a1, a1, %lo(.LCPI4_2)
	vle64.v	v10, (a1)
	addi	a1, a0, 56
	vse64.v	v8, (a1)
	addi	a0, a0, 16
	vse64.v	v10, (a0)
	ret
.Lfunc_end4:
	.size	assemble_sparse_tensor2, .Lfunc_end4-assemble_sparse_tensor2
	.cfi_endproc
                                        # -- End function
	.type	.L__constant_148xindex_0,@object # @__constant_148xindex_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.L__constant_148xindex_0:
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
	.size	.L__constant_148xindex_0, 1184

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
	.quad	520                             # 0x208
	.quad	541                             # 0x21d
	.quad	562                             # 0x232
	.quad	583                             # 0x247
	.quad	603                             # 0x25b
	.quad	624                             # 0x270
	.quad	645                             # 0x285
	.quad	666                             # 0x29a
	.quad	687                             # 0x2af
	.quad	706                             # 0x2c2
	.quad	725                             # 0x2d5
	.quad	744                             # 0x2e8
	.quad	756                             # 0x2f4
	.quad	768                             # 0x300
	.quad	777                             # 0x309
	.quad	794                             # 0x31a
	.quad	811                             # 0x32b
	.quad	828                             # 0x33c
	.quad	849                             # 0x351
	.quad	870                             # 0x366
	.quad	891                             # 0x37b
	.quad	912                             # 0x390
	.quad	933                             # 0x3a5
	.quad	954                             # 0x3ba
	.quad	975                             # 0x3cf
	.quad	996                             # 0x3e4
	.quad	1017                            # 0x3f9
	.quad	1038                            # 0x40e
	.quad	1059                            # 0x423
	.quad	1080                            # 0x438
	.quad	1099                            # 0x44b
	.quad	1118                            # 0x45e
	.quad	1137                            # 0x471
	.quad	1149                            # 0x47d
	.quad	1161                            # 0x489
	.quad	1170                            # 0x492
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
	.quad	1431                            # 0x597
	.quad	1452                            # 0x5ac
	.quad	1473                            # 0x5c1
	.quad	1492                            # 0x5d4
	.quad	1511                            # 0x5e7
	.quad	1530                            # 0x5fa
	.quad	1542                            # 0x606
	.quad	1554                            # 0x612
	.quad	1563                            # 0x61b
	.quad	1580                            # 0x62c
	.quad	1597                            # 0x63d
	.quad	1614                            # 0x64e
	.quad	1635                            # 0x663
	.quad	1656                            # 0x678
	.quad	1677                            # 0x68d
	.quad	1698                            # 0x6a2
	.quad	1719                            # 0x6b7
	.quad	1740                            # 0x6cc
	.quad	1761                            # 0x6e1
	.quad	1782                            # 0x6f6
	.quad	1803                            # 0x70b
	.quad	1824                            # 0x720
	.quad	1845                            # 0x735
	.quad	1866                            # 0x74a
	.quad	1885                            # 0x75d
	.quad	1904                            # 0x770
	.quad	1923                            # 0x783
	.quad	1935                            # 0x78f
	.quad	1947                            # 0x79b
	.quad	1956                            # 0x7a4
	.quad	1971                            # 0x7b3
	.quad	1986                            # 0x7c2
	.quad	2002                            # 0x7d2
	.quad	2020                            # 0x7e4
	.quad	2038                            # 0x7f6
	.quad	2057                            # 0x809
	.quad	2075                            # 0x81b
	.quad	2093                            # 0x82d
	.quad	2112                            # 0x840
	.quad	2130                            # 0x852
	.quad	2148                            # 0x864
	.quad	2167                            # 0x877
	.quad	2186                            # 0x88a
	.quad	2205                            # 0x89d
	.quad	2224                            # 0x8b0
	.quad	2240                            # 0x8c0
	.quad	2256                            # 0x8d0
	.quad	2273                            # 0x8e1
	.quad	2282                            # 0x8ea
	.quad	2291                            # 0x8f3
	.quad	2298                            # 0x8fa
	.quad	2309                            # 0x905
	.quad	2320                            # 0x910
	.quad	2332                            # 0x91c
	.quad	2344                            # 0x928
	.quad	2356                            # 0x934
	.quad	2368                            # 0x940
	.quad	2380                            # 0x94c
	.quad	2392                            # 0x958
	.quad	2404                            # 0x964
	.quad	2416                            # 0x970
	.quad	2426                            # 0x97a
	.quad	2436                            # 0x984
	.quad	2441                            # 0x989
	.size	.L__constant_148xindex, 1184

	.type	.L__constant_2441xindex,@object # @__constant_2441xindex
	.p2align	6, 0x0
.L__constant_2441xindex:
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
	.quad	130                             # 0x82
	.quad	131                             # 0x83
	.quad	132                             # 0x84
	.quad	144                             # 0x90
	.quad	145                             # 0x91
	.quad	146                             # 0x92
	.quad	110                             # 0x6e
	.quad	111                             # 0x6f
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
	.size	.L__constant_2441xindex, 19528

	.type	.L__constant_2441xf64,@object   # @__constant_2441xf64
	.p2align	6, 0x0
.L__constant_2441xf64:
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x407b1ac60029f16b              # double 433.67334
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0x4049829c7ffde721              # double 51.020401
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x8000000000000000              # double -0
	.quad	0x8000000000000000              # double -0
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0x0000000000000000              # double 0
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x8000000000000000              # double -0
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x0000000000000000              # double 0
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x8000000000000000              # double -0
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0x0000000000000000              # double 0
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0x0000000000000000              # double 0
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x0000000000000000              # double 0
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x0000000000000000              # double 0
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x0000000000000000              # double 0
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x0000000000000000              # double 0
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0xbeb0c6f7a0b5ed8d              # double -9.9999999999999995E-7
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbedd5c31593e5fb7              # double -6.9999999999999999E-6
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbee0c6f7a0b5ed8d              # double -7.9999999999999996E-6
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0xbeb0c6f7a0b5ed8d              # double -9.9999999999999995E-7
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbee0c6f7a0b5ed8d              # double -7.9999999999999996E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0x3ee0c6f7a0b5ed8d              # double 7.9999999999999996E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xbedd5c31593e5fb7              # double -6.9999999999999999E-6
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3ee0c6f7a0b5ed8d              # double 7.9999999999999996E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f05460aa64c              # double 3775.5102999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4026aaaa9f7b5aea              # double 11.333333
	.quad	0x40130c31593e5fb7              # double 4.7619069999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0x40030c308feac42f              # double 2.3809520000000002
	.quad	0x4049829c7ffde721              # double 51.020401
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x40130c31593e5fb7              # double 4.7619069999999999
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0x4049829c7ffde721              # double 51.020401
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x0000000000000000              # double 0
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x0000000000000000              # double 0
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ed92a737110e454              # double 6.0000000000000002E-6
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ed92a737110e454              # double 6.0000000000000002E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x8000000000000000              # double -0
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x8000000000000000              # double -0
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaa9f7b5aea3              # double 22.666656
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x40ad7f0504816f00              # double 3775.5097999999998
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3eed5c31593e5fb7              # double 1.4E-5
	.quad	0xbed92a737110e454              # double -6.0000000000000002E-6
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0xbee711947cfa26a2              # double -1.1E-5
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbec0c6f7a0b5ed8d              # double -1.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3eed5c31593e5fb7              # double 1.4E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbee711947cfa26a2              # double -1.1E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0xbed92a737110e454              # double -6.0000000000000002E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f05460aa64c              # double 3775.5102999999999
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0xbee711947cfa26a2              # double -1.1E-5
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ed92a737110e454              # double 6.0000000000000002E-6
	.quad	0x3eef75104d551d69              # double 1.5E-5
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x3edd5c31593e5fb7              # double 6.9999999999999999E-6
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xbee711947cfa26a2              # double -1.1E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ed92a737110e454              # double 6.0000000000000002E-6
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3edd5c31593e5fb7              # double 6.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3eef75104d551d69              # double 1.5E-5
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f0560418937              # double 3775.5104999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x3edd5c31593e5fb7              # double 6.9999999999999999E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4026aaaa7ded6ba9              # double 11.333332
	.quad	0x40130c2e33eff195              # double 4.761895
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0x40030c3009b30729              # double 2.380951
	.quad	0x40030c308feac42f              # double 2.3809520000000002
	.quad	0x4049829c7ffde721              # double 51.020401
	.quad	0x3edd5c31593e5fb7              # double 6.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x40130c2e33eff195              # double 4.761895
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc0030c3009b30729              # double -2.380951
	.quad	0x4049829c0229a5ec              # double 51.020386000000002
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ee0c6f7a0b5ed8d              # double 7.9999999999999996E-6
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ee0c6f7a0b5ed8d              # double 7.9999999999999996E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ee2dfd694ccab3f              # double 9.0000000000000002E-6
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbedd5c31593e5fb7              # double -6.9999999999999999E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ee2dfd694ccab3f              # double 9.0000000000000002E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbedd5c31593e5fb7              # double -6.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbedd5c31593e5fb7              # double -6.9999999999999999E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ee92a737110e454              # double 1.2E-5
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbedd5c31593e5fb7              # double -6.9999999999999999E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x3ee92a737110e454              # double 1.2E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ef92a737110e454              # double 2.4000000000000001E-5
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0xbef2dfd694ccab3f              # double -1.8E-5
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ef92a737110e454              # double 2.4000000000000001E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbef2dfd694ccab3f              # double -1.8E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0xbef2dfd694ccab3f              # double -1.8E-5
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3eeb43526527a205              # double 1.2999999999999999E-5
	.quad	0x3ee711947cfa26a2              # double 1.1E-5
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0x3ed92a737110e454              # double 6.0000000000000002E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbef2dfd694ccab3f              # double -1.8E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x3eeb43526527a205              # double 1.2999999999999999E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3ee711947cfa26a2              # double 1.1E-5
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f0560418937              # double 3775.5104999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaa9f7b5aea3              # double 22.666656
	.quad	0x3ee2dfd694ccab3f              # double 9.0000000000000002E-6
	.quad	0x3eef75104d551d69              # double 1.5E-5
	.quad	0xc025c71c10d7be98              # double -10.888885999999999
	.quad	0xbeeb43526527a205              # double -1.2999999999999999E-5
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x3ed92a737110e454              # double 6.0000000000000002E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ee2dfd694ccab3f              # double 9.0000000000000002E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xbeeb43526527a205              # double -1.2999999999999999E-5
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3eef75104d551d69              # double 1.5E-5
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x40ad7f0504816f00              # double 3775.5097999999998
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0xc0030c3009b30729              # double -2.380951
	.quad	0xc025c71c10d7be98              # double -10.888885999999999
	.quad	0xbeeb43526527a205              # double -1.2999999999999999E-5
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0x4026aaaa3ad18d26              # double 11.33333
	.quad	0x40130c347e8ccdd9              # double 4.7619189999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0x40030c308feac42f              # double 2.3809520000000002
	.quad	0x40030c3009b30729              # double 2.380951
	.quad	0x4049829c0229a5ec              # double 51.020386000000002
	.quad	0xbeeb43526527a205              # double -1.2999999999999999E-5
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x40130c347e8ccdd9              # double 4.7619189999999998
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0x4049829c7ffde721              # double 51.020401
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0xbeb0c6f7a0b5ed8d              # double -9.9999999999999995E-7
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbeb0c6f7a0b5ed8d              # double -9.9999999999999995E-7
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbec92a737110e454              # double -3.0000000000000001E-6
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x8000000000000000              # double -0
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xbeb0c6f7a0b5ed8d              # double -9.9999999999999995E-7
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0x8000000000000000              # double -0
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbeb0c6f7a0b5ed8d              # double -9.9999999999999995E-7
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbee4f8b588e368f1              # double -1.0000000000000001E-5
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbee2dfd694ccab3f              # double -9.0000000000000002E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xbee2dfd694ccab3f              # double -9.0000000000000002E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaa9f7b5aea3              # double 22.666656
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0x3ee0c6f7a0b5ed8d              # double 7.9999999999999996E-6
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc04bf9e6ff7dfa01              # double -55.952362000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbee4f8b588e368f1              # double -1.0000000000000001E-5
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3ee0c6f7a0b5ed8d              # double 7.9999999999999996E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f544bb1af4              # double 153.06119000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaa9f7b5aea3              # double 22.666656
	.quad	0xbee2dfd694ccab3f              # double -9.0000000000000002E-6
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbee2dfd694ccab3f              # double -9.0000000000000002E-6
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e6ff7dfa01              # double -55.952362000000001
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x40ad7f04dd2f1aa0              # double 3775.5095000000001
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f544bb1af4              # double 153.06119000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaa9f7b5aea3              # double 22.666656
	.quad	0xbf02dfd694ccab3f              # double -3.6000000000000001E-5
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0xc025c71bac2df0d4              # double -10.888883
	.quad	0x3f0040bfe3b03e21              # double 3.1000000000000001E-5
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0x404273ce7cd03537              # double 36.904738999999999
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbf02dfd694ccab3f              # double -3.6000000000000001E-5
	.quad	0x4097ea727bb2fec5              # double 1530.6117999999999
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x3f0040bfe3b03e21              # double 3.1000000000000001E-5
	.quad	0xc08189caff6d3309              # double -561.22411999999997
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x407e4b18fc504817              # double 484.6936
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x40ad7f04dd2f1aa0              # double 3775.5095000000001
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f51ac9afe2              # double 153.06117
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0xc025c71bac2df0d4              # double -10.888883
	.quad	0x3f0040bfe3b03e21              # double 3.1000000000000001E-5
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0x4026aaa9d627bf62              # double 11.333327000000001
	.quad	0x40130c27631b584b              # double 4.7618689999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c2fc69728a6              # double 4.7619009999999999
	.quad	0xc0401247fdc5931d              # double -32.142822000000002
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0x40030c3009b30729              # double 2.380951
	.quad	0x40030c308feac42f              # double 2.3809520000000002
	.quad	0x4049829c7ffde721              # double 51.020401
	.quad	0x3f0040bfe3b03e21              # double 3.1000000000000001E-5
	.quad	0xc08189caff6d3309              # double -561.22411999999997
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x40130c27631b584b              # double 4.7618689999999999
	.quad	0x4087ea71fddebd90              # double 765.30565999999999
	.quad	0xc0130c2fc69728a6              # double -4.7619009999999999
	.quad	0x4059829b7bf1e8e6              # double 102.04074
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc0030c3009b30729              # double -2.380951
	.quad	0x4049829c0229a5ec              # double 51.020386000000002
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3ef0c6f7a0b5ed8d              # double 1.5999999999999999E-5
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbee4f8b588e368f1              # double -1.0000000000000001E-5
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3ef0c6f7a0b5ed8d              # double 1.5999999999999999E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbee2dfd694ccab3f              # double -9.0000000000000002E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c75818c5d              # double -10.888889000000001
	.quad	0xbee2dfd694ccab3f              # double -9.0000000000000002E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3f02dfd694ccab3f              # double 3.6000000000000001E-5
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0xc025c71cda2b5a21              # double -10.888892
	.quad	0xbef81e03f705857b              # double -2.3E-5
	.quad	0xc04bf9e8815e3971              # double -55.952407999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c31593e5fb7              # double 4.7619069999999999
	.quad	0x404273cffeb074a7              # double 36.904784999999997
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbee4f8b588e368f1              # double -1.0000000000000001E-5
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3f02dfd694ccab3f              # double 3.6000000000000001E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbef711947cfa26a2              # double -2.1999999999999999E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cd013a92a3              # double -280.61255
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c31593e5fb7              # double -4.7619069999999999
	.quad	0x4059829d495182aa              # double 102.04085000000001
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f05460aa64c              # double 3775.5102999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x406321f5d78811b2              # double 153.06126
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71cda2b5a21              # double -10.888892
	.quad	0xbef711947cfa26a2              # double -2.1999999999999999E-5
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3f0040bfe3b03e21              # double 3.1000000000000001E-5
	.quad	0x3efe68a0d349be90              # double 2.9E-5
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c31593e5fb7              # double 4.7619069999999999
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829c779a6b51              # double 102.0408
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbef81e03f705857b              # double -2.3E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x3f0040bfe3b03e21              # double 3.1000000000000001E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c31593e5fb7              # double -4.7619069999999999
	.quad	0x4059829d7342edbb              # double 102.04086
	.quad	0xc077ea73ffac1d2a              # double -382.65332000000001
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f583a53b8e              # double 153.06121999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e8815e3971              # double -55.952407999999998
	.quad	0xc07189cd013a92a3              # double -280.61255
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3efe68a0d349be90              # double 2.9E-5
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f0560418937              # double 3775.5104999999999
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaa9f7b5aea3              # double 22.666656
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0x4097ea72ca57a787              # double 1530.6121000000001
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f544bb1af4              # double 153.06119000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ed4f8b588e368f1              # double 5.0000000000000004E-6
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x40ad7f04dd2f1aa0              # double 3775.5095000000001
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc025c71c3265adda              # double -10.888887
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3f06052502eec7c9              # double 4.1999999999999998E-5
	.quad	0xbefa36e2eb1c432d              # double -2.5000000000000001E-5
	.quad	0xc025c71da37ef5a9              # double -10.888897999999999
	.quad	0xbf00c6f7a0b5ed8d              # double -3.1999999999999999E-5
	.quad	0xc04bf9e97d06bbdc              # double -55.952438000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c319c5a3e3a              # double 4.761908
	.quad	0x404273d07c84b5dd              # double 36.904800000000002
	.quad	0x40130c308feac42f              # double 4.7619040000000004
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x3f06052502eec7c9              # double 4.1999999999999998E-5
	.quad	0x4097ea7381d7dbf5              # double 1530.6128000000001
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xbf00c6f7a0b5ed8d              # double -3.1999999999999999E-5
	.quad	0xc08189cc7e28240b              # double -561.22484999999995
	.quad	0xc07189cd013a92a3              # double -280.61255
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c319c5a3e3a              # double -4.761908
	.quad	0x4059829df1172ef1              # double 102.04089
	.quad	0x407e4b1c044284e0              # double 484.69434000000001
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f544bb1af4              # double 153.06119000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0xbefa36e2eb1c432d              # double -2.5000000000000001E-5
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f05a1cac083              # double 3775.511
	.quad	0x404bf9e8815e3971              # double 55.952407999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f6fd21ff2e              # double -306.12279999999998
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cffeb074a7              # double -36.904784999999997
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x406321f6a93f290b              # double 153.06136000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c2fc69728a6              # double -4.7619009999999999
	.quad	0xc025c71da37ef5a9              # double -10.888897999999999
	.quad	0xbf00c6f7a0b5ed8d              # double -3.1999999999999999E-5
	.quad	0x404bf9e8815e3971              # double 55.952407999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3f0711947cfa26a2              # double 4.3999999999999999E-5
	.quad	0x3f1322f2734f82f5              # double 7.2999999999999999E-5
	.quad	0xc025c71bcdbbe015              # double -10.888883999999999
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c319c5a3e3a              # double 4.761908
	.quad	0xc040124a83b1d0c8              # double -32.142899
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c2fc69728a6              # double 4.7619009999999999
	.quad	0x4059829b7bf1e8e6              # double 102.04074
	.quad	0xbf00c6f7a0b5ed8d              # double -3.1999999999999999E-5
	.quad	0xc08189cc7e28240b              # double -561.22484999999995
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3f0711947cfa26a2              # double 4.3999999999999999E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea7381d7dbf5              # double 765.30640000000005
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc0130c319c5a3e3a              # double -4.761908
	.quad	0x4059829df1172ef1              # double 102.04089
	.quad	0xc077ea73ffac1d2a              # double -382.65332000000001
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273ce7cd03537              # double 36.904738999999999
	.quad	0x407e4b18fc504817              # double 484.6936
	.quad	0x406321f51ac9afe2              # double 153.06117
	.quad	0xc0401247fdc5931d              # double -32.142822000000002
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc04bf9e97d06bbdc              # double -55.952438000000001
	.quad	0xc07189cd013a92a3              # double -280.61255
	.quad	0xc07321f6fd21ff2e              # double -306.12279999999998
	.quad	0x3f1322f2734f82f5              # double 7.2999999999999999E-5
	.quad	0x4087ea7381d7dbf5              # double 765.30640000000005
	.quad	0x40ad7f05460aa64c              # double 3775.5102999999999
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc077ea73ffac1d2a              # double -382.65332000000001
	.quad	0xc0865249fe86833c              # double -714.28612999999996
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0xc0030c3009b30729              # double -2.380951
	.quad	0xc025c71bcdbbe015              # double -10.888883999999999
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0x4026aaaa19439de5              # double 11.333329000000001
	.quad	0x40130c319c5a3e3a              # double 4.761908
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0x40030c308feac42f              # double 2.3809520000000002
	.quad	0x40030c3009b30729              # double 2.380951
	.quad	0x4049829c0229a5ec              # double 51.020386000000002
	.quad	0xbed0c6f7a0b5ed8d              # double -3.9999999999999998E-6
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x40130c319c5a3e3a              # double 4.761908
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0x4049829c0229a5ec              # double 51.020386000000002
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0x409d7f0538ef34d7              # double 1887.7551000000001
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0xc025c71c970f7b9e              # double -10.88889
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xbed4f8b588e368f1              # double -5.0000000000000004E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f05460aa64c              # double 3775.5102999999999
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c31593e5fb7              # double -4.7619069999999999
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c31593e5fb7              # double -4.7619069999999999
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc025c71c970f7b9e              # double -10.88889
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0xbee92a737110e454              # double -1.2E-5
	.quad	0xbee0c6f7a0b5ed8d              # double -7.9999999999999996E-6
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c31593e5fb7              # double 4.7619069999999999
	.quad	0x4059829d495182aa              # double 102.04085000000001
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x40130c31593e5fb7              # double 4.7619069999999999
	.quad	0x4059829d7342edbb              # double 102.04086
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbee92a737110e454              # double -1.2E-5
	.quad	0x4097ea7333333333              # double 1530.6125
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cffeb074a7              # double 36.904784999999997
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x406321f5d78811b2              # double 153.06126
	.quad	0xc04012497fa5d28d              # double -32.142868
	.quad	0xc077ea73ffac1d2a              # double -382.65332000000001
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04bf9e80389f83c              # double -55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xbee0c6f7a0b5ed8d              # double -7.9999999999999996E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f057a786c22              # double 3775.5106999999998
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0xbee711947cfa26a2              # double -1.1E-5
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbee711947cfa26a2              # double -1.1E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x3ed0c6f7a0b5ed8d              # double 3.9999999999999998E-6
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0xbef4f8b588e368f1              # double -2.0000000000000002E-5
	.quad	0xbef4f8b588e368f1              # double -2.0000000000000002E-5
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0x3ef3ec460ed80a18              # double 1.9000000000000001E-5
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c3116228134              # double 4.7619059999999998
	.quad	0x4059829cf56eac86              # double 102.04083
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbef4f8b588e368f1              # double -2.0000000000000002E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3ef3ec460ed80a18              # double 1.9000000000000001E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0xbef4f8b588e368f1              # double -2.0000000000000002E-5
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f051eb851ec              # double 3775.5100000000002
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c319c5a3e3a              # double -4.761908
	.quad	0xc04273cffeb074a7              # double -36.904784999999997
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c319c5a3e3a              # double -4.761908
	.quad	0x404012497fa5d28d              # double 32.142868
	.quad	0xc025c71cb89d6adf              # double -10.888890999999999
	.quad	0x3ef3ec460ed80a18              # double 1.9000000000000001E-5
	.quad	0x404bf9e80389f83c              # double 55.952393000000001
	.quad	0x4036aaab042528ae              # double 22.666671999999998
	.quad	0xbf0797cc39ffd60f              # double -4.5000000000000003E-5
	.quad	0xbeed5c31593e5fb7              # double -1.4E-5
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x3ef2dfd694ccab3f              # double 1.8E-5
	.quad	0xc04bf9e6ff7dfa01              # double -55.952362000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c319c5a3e3a              # double 4.761908
	.quad	0x4059829df1172ef1              # double 102.04089
	.quad	0x407e4b1afe1da7b1              # double 484.69409000000002
	.quad	0x40130c319c5a3e3a              # double 4.761908
	.quad	0x4059829df1172ef1              # double 102.04089
	.quad	0xc077ea73ffac1d2a              # double -382.65332000000001
	.quad	0x3ef3ec460ed80a18              # double 1.9000000000000001E-5
	.quad	0xc08189cc0053e2d6              # double -561.22460999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbf0797cc39ffd60f              # double -4.5000000000000003E-5
	.quad	0x4097ea72fec56d5d              # double 1530.6123
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x3ef2dfd694ccab3f              # double 1.8E-5
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x404273d07c84b5dd              # double 36.904800000000002
	.quad	0x407e4b1c044284e0              # double 484.69434000000001
	.quad	0x406321f6a93f290b              # double 153.06136000000001
	.quad	0xc040124a83b1d0c8              # double -32.142899
	.quad	0xc077ea73ffac1d2a              # double -382.65332000000001
	.quad	0xc0865249fe86833c              # double -714.28612999999996
	.quad	0xc04bf9e77d523b36              # double -55.952376999999998
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f601797cc4              # double -306.12256000000002
	.quad	0xbeed5c31593e5fb7              # double -1.4E-5
	.quad	0x4087ea72fec56d5d              # double 765.30615
	.quad	0x40ad7f05a1cac083              # double 3775.511
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x406321f544bb1af4              # double 153.06119000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0xc025c71c53f39d1b              # double -10.888888
	.quad	0x3ef2dfd694ccab3f              # double 1.8E-5
	.quad	0x404bf9e77d523b36              # double 55.952376999999998
	.quad	0x4036aaa9f7b5aea3              # double 22.666656
	.quad	0xbef3ec460ed80a18              # double -1.9000000000000001E-5
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0xc025c71bef49cf57              # double -10.888885
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x40130c30d306a2b1              # double 4.7619049999999996
	.quad	0x4059829ccb7d4174              # double 102.04082
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0x4059829c4da9003f              # double 102.04079
	.quad	0x3ef2dfd694ccab3f              # double 1.8E-5
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189cbfb15b574              # double -280.6123
	.quad	0xbef3ec460ed80a18              # double -1.9000000000000001E-5
	.quad	0x4097ea727bb2fec5              # double 1530.6117999999999
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0x407e4b18fc504817              # double 484.6936
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc04bf9e6ff7dfa01              # double -55.952362000000001
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0xc07321f4fb549f95              # double -306.12231000000003
	.quad	0x3ec0c6f7a0b5ed8d              # double 1.9999999999999999E-6
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0x40ad7f04c2f837b5              # double 3775.5093000000002
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc08652487fcb923a              # double -714.28539999999998
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0x406321f51ac9afe2              # double 153.06117
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0xc025c71bef49cf57              # double -10.888885
	.quad	0x404bf9e6ff7dfa01              # double 55.952362000000001
	.quad	0x4026aaaa19439de5              # double 11.333329000000001
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0x40030c308feac42f              # double 2.3809520000000002
	.quad	0x4049829c0229a5ec              # double 51.020386000000002
	.quad	0xc08189cb827fa1a1              # double -561.22437000000002
	.quad	0xc07189caff6d3309              # double -280.61205999999999
	.quad	0x40130c304ccee5ac              # double 4.7619030000000002
	.quad	0x4087ea7280f12c28              # double 765.30591000000004
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0x405321f57f737da6              # double 76.530608999999998
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0x40952029c779a6b5              # double 1352.0408
	.quad	0x40469e79fec99f1b              # double 45.238098000000001
	.quad	0xc06321f583a53b8e              # double -153.06121999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c3116228134              # double -4.7619059999999998
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0x40469e79fec99f1b              # double 45.238098000000001
	.quad	0x4026aaaac1094a2c              # double 11.333334000000001
	.quad	0xc043a492817763e5              # double -39.285721000000002
	.quad	0xc015c71c970f7b9e              # double -5.444445
	.quad	0xc0256db70691ea79              # double -10.714287000000001
	.quad	0x404273cf80dc3372              # double 36.904769999999999
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc086524980b24207              # double -714.28588999999999
	.quad	0xc06321f583a53b8e              # double -153.06121999999999
	.quad	0xc043a492817763e5              # double -39.285721000000002
	.quad	0x409d7f0538ef34d7              # double 1887.7551000000001
	.quad	0x40469e79fec99f1b              # double 45.238098000000001
	.quad	0xc06321f5989df117              # double -153.06122999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf80dc3372              # double -36.904769999999999
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc015c71c970f7b9e              # double -5.444445
	.quad	0x40469e79fec99f1b              # double 45.238098000000001
	.quad	0x4026aaaa9f7b5aea              # double 11.333333
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0xc015c71c10d7be98              # double -5.4444429999999997
	.quad	0xc0256db6805a2d73              # double -10.714283
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f5989df117              # double 153.06122999999999
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc0256db70691ea79              # double -10.714287000000001
	.quad	0xc06321f5989df117              # double -153.06122999999999
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0xc06321f51ac9afe2              # double -153.06117
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc015c71c10d7be98              # double -5.4444429999999997
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0x4026aaaa5c5f7c67              # double 11.333330999999999
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0xc015c71c53f39d1b              # double -5.4444439999999998
	.quad	0xc0256db6c3760bf6              # double -10.714285
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc0256db6805a2d73              # double -10.714283
	.quad	0xc06321f51ac9afe2              # double -153.06117
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x40469e79fec99f1b              # double 45.238098000000001
	.quad	0xc06321f559b3d07d              # double -153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c30d306a2b1              # double -4.7619049999999996
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc015c71c53f39d1b              # double -5.4444439999999998
	.quad	0x40469e79fec99f1b              # double 45.238098000000001
	.quad	0x4026aaaa7ded6ba9              # double 11.333332
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0xc015c71c53f39d1b              # double -5.4444439999999998
	.quad	0xc0256db6a1e81cb4              # double -10.714283999999999
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f559b3d07d              # double 153.06120000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea7304039abf              # double -382.65307999999999
	.quad	0xc0865248fd9fd36f              # double -714.28563999999994
	.quad	0xc0256db6c3760bf6              # double -10.714285
	.quad	0xc06321f559b3d07d              # double -153.06120000000001
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0x409d7f0504816f00              # double 1887.7548999999999
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0xc06321f559b3d07d              # double -153.06120000000001
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c308feac42f              # double -4.7619040000000004
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0x4040124901d19158              # double 32.142853000000002
	.quad	0xc015c71c53f39d1b              # double -5.4444439999999998
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0x4026aaaa5c5f7c67              # double 11.333330999999999
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0xc015c71c10d7be98              # double -5.4444429999999997
	.quad	0xc0256db65ecc3e32              # double -10.714282000000001
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b1a02752546              # double 484.69385
	.quad	0x406321f544bb1af4              # double 153.06119000000001
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc08652487fcb923a              # double -714.28539999999998
	.quad	0xc0256db6a1e81cb4              # double -10.714283999999999
	.quad	0xc06321f559b3d07d              # double -153.06120000000001
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0x409d7f04816f0069              # double 1887.7544
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0xc06321f51ac9afe2              # double -153.06117
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0xc04273cf0307f23d              # double -36.904755000000002
	.quad	0xbfcc71c53f39d1b3              # double -0.222222
	.quad	0xc0130c304ccee5ac              # double -4.7619030000000002
	.quad	0xc015c71c10d7be98              # double -5.4444429999999997
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0x4026aaaa19439de5              # double 11.333329000000001
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0xc015c71c10d7be98              # double -5.4444429999999997
	.quad	0x404273cf0307f23d              # double 36.904755000000002
	.quad	0x407e4b18fc504817              # double 484.6936
	.quad	0x406321f51ac9afe2              # double 153.06117
	.quad	0xc040124901d19158              # double -32.142853000000002
	.quad	0xc077ea71fddebd90              # double -382.65282999999999
	.quad	0xc0256db65ecc3e32              # double -10.714282000000001
	.quad	0xc06321f51ac9afe2              # double -153.06117
	.quad	0xc043a49203a322af              # double -39.285705999999998
	.quad	0x409d7f04816f0069              # double 1887.7544
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0xbfbc71c53f39d1b3              # double -0.111111
	.quad	0xc0030c308feac42f              # double -2.3809520000000002
	.quad	0xc015c71c10d7be98              # double -5.4444429999999997
	.quad	0x40469e7980f55de6              # double 45.238083000000003
	.quad	0x401638e2e2b8c75c              # double 5.5555529999999997
	.size	.L__constant_2441xf64, 19528

	.section	".note.GNU-stack","",@progbits
