/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin

*/

GEMM_UKR_PROT( float,    s, gemm_rviv_4vx4 )
GEMM_UKR_PROT( double,   d, gemm_rviv_4vx4 )
GEMM_UKR_PROT( scomplex, c, gemm_rviv_4vx4 )
GEMM_UKR_PROT( dcomplex, z, gemm_rviv_4vx4 )
GEMM_UKR_PROT( float,    s, gemm_rviv_2vx8 )
GEMM_UKR_PROT( double,   d, gemm_rviv_2vx8 )
GEMM_UKR_PROT( double,   d, gemm_rviv_8x8 )
