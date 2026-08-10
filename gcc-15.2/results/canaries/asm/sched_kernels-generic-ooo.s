	.file	"sched_kernels.c"
	.option nopic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zba1p0_zbb1p0_zbc1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
# GNU C23 (GCC) version 15.2.0 (riscv64-unknown-linux-gnu)
#	compiled by GNU C version 13.3.0, GMP version 6.2.1, MPFR version 4.1.0, MPC version 1.2.1, isl version isl-0.24-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mabi=lp64d -mtune=generic-ooo -misa-spec=20191213 -mtls-dialect=trad -march=rv64imafdcv_zicsr_zifencei_zmmul_zaamo_zalrsc_zca_zcd_zba_zbb_zbc_zve32f_zve32x_zve64d_zve64f_zve64x_zvl128b_zvl256b_zvl32b_zvl64b -O3
	.text
	.align	1
	.globl	canary_load_add_chain
	.type	canary_load_add_chain, @function
canary_load_add_chain:
.LFB0:
	.cfi_startproc
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:20:   for (int i = 0; i < n; i++)
	ble	a1,zero,.L4	#, n,,
	mv	a5,a0	# ivtmp.10, p
	sh3add	a1,a1,a0	#, _15, n, ivtmp.10
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:19:   uint64_t s = 0;
	li	a0,0		# <retval>,
.L3:
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:21:     s += p[i] + (s << 1);
	ld	a4,0(a5)		# MEM[(uint64_t *)_22], MEM[(uint64_t *)_22]
	sh1add	a0,a0,a0	#, _6, <retval>, <retval>
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:20:   for (int i = 0; i < n; i++)
	addi	a5,a5,8	#, ivtmp.10, ivtmp.10
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:21:     s += p[i] + (s << 1);
	add	a0,a0,a4	# MEM[(uint64_t *)_22], <retval>, _6
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:20:   for (int i = 0; i < n; i++)
	bne	a1,a5,.L3	#, _15, ivtmp.10,
	ret	
.L4:
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:19:   uint64_t s = 0;
	li	a0,0		# <retval>,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:23: }
	ret	
	.cfi_endproc
.LFE0:
	.size	canary_load_add_chain, .-canary_load_add_chain
	.align	1
	.globl	canary_fma_chain
	.type	canary_fma_chain, @function
canary_fma_chain:
.LFB1:
	.cfi_startproc
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:27:   double x = 0.0;
	fmv.d.x	fa0,zero	# <retval>,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:28:   for (int i = 0; i < n; i++)
	ble	a3,zero,.L11	#, n,,
.L10:
	vsetvli	a5,a3,e8,mf8,ta,ma	# ivtmp_12, _11,,,,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:29:     x = a[i] * b[i] + c[i] + x;
	vle64.v	v4,0(a0)	# vect__4.21,* a
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:29:     x = a[i] * b[i] + c[i] + x;
	vle64.v	v2,0(a1)	# vect__6.24,* b
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:29:     x = a[i] * b[i] + c[i] + x;
	vle64.v	v3,0(a2)	# tmp151,* c
	vsetivli	zero,1,e64,m1,ta,ma	#,,,,
	vfmv.s.f	v1,fa0	# tmp157, <retval>
	vsetvli	zero,a5,e64,m1,ta,ma	# _11,,,,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:28:   for (int i = 0; i < n; i++)
	sub	a3,a3,a5	# ivtmp_12, ivtmp_12, _11
	sh3add	a0,a5,a0	#, a, _11, a
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:29:     x = a[i] * b[i] + c[i] + x;
	vfmadd.vv	v2,v4,v3	# tmp150, vect__4.21, tmp151,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:28:   for (int i = 0; i < n; i++)
	sh3add	a1,a5,a1	#, b, _11, b
	sh3add	a2,a5,a2	#, c, _11, c
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:29:     x = a[i] * b[i] + c[i] + x;
	vfredosum.vs	v1,v2,v1	# tmp158, tmp150, tmp157,
	vfmv.f.s	fa0,v1	# <retval>, tmp158
	bne	a3,zero,.L10	#, ivtmp_12,,
	ret	
.L11:
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:31: }
	ret	
	.cfi_endproc
.LFE1:
	.size	canary_fma_chain, .-canary_fma_chain
	.align	1
	.globl	canary_div_mix
	.type	canary_div_mix, @function
canary_div_mix:
.LFB2:
	.cfi_startproc
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:36:   for (int i = 0; i < n; i++) {
	ble	a2,zero,.L18	#, n,,
	mv	a4,a0	# ivtmp.39, a
	sh2add	a2,a2,a0	#, _35, n, ivtmp.39
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:35:   int32_t s = 1;
	li	a0,1		# <retval>,
.L17:
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:37:     s += a[i] / (b[i] | 1);
	lw	a6,0(a1)		# _6, MEM[(int32_t *)_39]
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:37:     s += a[i] / (b[i] | 1);
	lw	a5,0(a4)		# _4, MEM[(int32_t *)_40]
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:36:   for (int i = 0; i < n; i++) {
	addi	a4,a4,4	#, ivtmp.39, ivtmp.39
	addi	a1,a1,4	#, ivtmp.40, ivtmp.40
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:37:     s += a[i] / (b[i] | 1);
	ori	a3,a6,1	#, tmp152, _6
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:37:     s += a[i] / (b[i] | 1);
	divw	a3,a5,a3	# tmp152, tmp155, _4
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:38:     s ^= a[i] + b[i];
	addw	a5,a5,a6	# _6, tmp159, _4
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:37:     s += a[i] / (b[i] | 1);
	addw	a0,a3,a0	# <retval>, tmp157, tmp155
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:38:     s ^= a[i] + b[i];
	xor	a0,a0,a5	# tmp159, <retval>, tmp157
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:36:   for (int i = 0; i < n; i++) {
	bne	a2,a4,.L17	#, _35, ivtmp.39,
	ret	
.L18:
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:35:   int32_t s = 1;
	li	a0,1		# <retval>,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:41: }
	ret	
	.cfi_endproc
.LFE2:
	.size	canary_div_mix, .-canary_div_mix
	.align	1
	.globl	canary_sh1add
	.type	canary_sh1add, @function
canary_sh1add:
.LFB3:
	.cfi_startproc
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:46:   for (int i = 0; i < n; i++)
	ble	a2,zero,.L24	#, n,,
	vsetvli	a5,zero,e64,m1,ta,ma	#, tmp148,,,,
	vmv.v.i	v1,0	# vect_s_19.56,
.L23:
	vsetvli	a5,a2,e64,m1,tu,ma	# ivtmp_8, _41,,,,
	vle64.v	v3,0(a1)	# tmp149,* b
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:47:     s += (a[i] << 1) + b[i];
	vle64.v	v2,0(a0)	# tmp150,* a
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:46:   for (int i = 0; i < n; i++)
	sub	a2,a2,a5	# ivtmp_8, ivtmp_8, _41
	sh3add	a1,a5,a1	#, b, _41, b
	sh3add	a0,a5,a0	#, a, _41, a
	vadd.vv	v1,v1,v3	# vect__10.57, vect_s_19.56, tmp149,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:47:     s += (a[i] << 1) + b[i];
	vsll.vi	v2,v2,1	#, vect__5.52_30, tmp150,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:47:     s += (a[i] << 1) + b[i];
	vadd.vv	v1,v1,v2	# vect_s_19.56, vect__10.57, vect__5.52_30,
	bne	a2,zero,.L23	#, ivtmp_8,,
	vsetvli	a5,zero,e64,m1,ta,ma	#, tmp157,,,,
	vmv.s.x	v2,zero	# tmp156,
	vredsum.vs	v1,v1,v2	# tmp158, vect_s_19.56, tmp156,
	vmv.x.s	a0,v1	# <retval>, tmp158
	ret	
.L24:
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:45:   uint64_t s = 0;
	li	a0,0		# <retval>,
# /home/orangepi/gcc-tune/measurements/canaries/sched_kernels.c:49: }
	ret	
	.cfi_endproc
.LFE3:
	.size	canary_sh1add, .-canary_sh1add
	.ident	"GCC: (GNU) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
