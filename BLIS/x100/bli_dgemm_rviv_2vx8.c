/*
   BLIS — dgemm rviv 2vx8 C wrapper (SpacemiT X100)
*/
#include "bli_rviv_utils.h"

void bli_dgemm_rviv_asm_2vx8
    (
             intptr_t   k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, intptr_t rs_c, intptr_t cs_c
    );

void bli_dgemm_rviv_2vx8
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	bli_static_assert( sizeof(dim_t) <= sizeof(intptr_t) &&
	                   sizeof(inc_t) <= sizeof(intptr_t) );

	const inc_t mr = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx );
	const inc_t nr = 8;

	GEMM_UKR_SETUP_CT( d, mr, nr, false );
	assert( rs_c == 1 );
	bli_dgemm_rviv_asm_2vx8( k, alpha, a, b, beta, c,
	                         get_vlenb(), cs_c * sizeof(double) );
	GEMM_UKR_FLUSH_CT( d );
}
