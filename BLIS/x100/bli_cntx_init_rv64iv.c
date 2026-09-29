/*
   BLIS rv64iv cntx — 2vx8 for VLEN>=256 (SpacemiT X100)
*/
#include "../../kernels/rviv/3/bli_rviv_utils.h"

void bli_cntx_init_rv64iv( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
	bli_cntx_init_rv64iv_ref( cntx );
	const uint32_t v = get_vlenb() / sizeof(float);

	if ( v >= 8 )
	{
		const uint32_t mr_s = 2 * v;
		const uint32_t mr_d = 8;
		const uint32_t mr_c = 2 * v;
		const uint32_t mr_z = 1 * v;

		bli_cntx_set_ukrs
		(
		  cntx,
		  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_rviv_2vx8,
		  BLIS_GEMM_UKR, BLIS_DOUBLE,   bli_dgemm_rviv_8x8,
		  BLIS_GEMM_UKR, BLIS_SCOMPLEX, bli_cgemm_rviv_4vx4,
		  BLIS_GEMM_UKR, BLIS_DCOMPLEX, bli_zgemm_rviv_4vx4,
		  BLIS_VA_END
		);
		bli_cntx_set_ukr_prefs
		(
		  cntx,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,    FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE,   FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_SCOMPLEX, FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_DCOMPLEX, FALSE,
		  BLIS_VA_END
		);

		/* BEGIN_BLKSZ */
		bli_blksz_init_easy( &blkszs[ BLIS_MR ],     mr_s,    mr_d,    mr_c,    mr_z );
		bli_blksz_init_easy( &blkszs[ BLIS_NR ],        8,       8,       4,       4 );
		bli_blksz_init_easy( &blkszs[ BLIS_MC ],  32*mr_s, 32*mr_d, 60*mr_c, 30*mr_z );
		bli_blksz_init_easy( &blkszs[ BLIS_KC ],      640,     320,     320,     160 );
		bli_blksz_init_easy( &blkszs[ BLIS_NC ],     4096,    4096,    3072,    3072 );
		/* END_BLKSZ */

		bli_cntx_set_blkszs
		(
		  cntx,
		  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
		  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
		  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
		  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
		  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
		  BLIS_VA_END
		);
	}
	else if ( v >= 4 )
	{
		const uint32_t mr_s = 4 * v;
		const uint32_t mr_d = 2 * v;
		const uint32_t mr_c = 2 * v;
		const uint32_t mr_z = v;
		bli_cntx_set_ukrs
		(
		  cntx,
		  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_rviv_4vx4,
		  BLIS_GEMM_UKR, BLIS_DOUBLE,   bli_dgemm_rviv_4vx4,
		  BLIS_GEMM_UKR, BLIS_SCOMPLEX, bli_cgemm_rviv_4vx4,
		  BLIS_GEMM_UKR, BLIS_DCOMPLEX, bli_zgemm_rviv_4vx4,
		  BLIS_VA_END
		);
		bli_cntx_set_ukr_prefs
		(
		  cntx,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,    FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE,   FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_SCOMPLEX, FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_DCOMPLEX, FALSE,
		  BLIS_VA_END
		);
		bli_blksz_init_easy( &blkszs[ BLIS_MR ],     mr_s,    mr_d,    mr_c,    mr_z );
		bli_blksz_init_easy( &blkszs[ BLIS_NR ],        4,       4,       4,       4 );
		bli_blksz_init_easy( &blkszs[ BLIS_MC ],  20*mr_s, 20*mr_d, 60*mr_c, 30*mr_z );
		bli_blksz_init_easy( &blkszs[ BLIS_KC ],      640,     320,     320,     160 );
		bli_blksz_init_easy( &blkszs[ BLIS_NC ],     3072,    3072,    3072,    3072 );
		bli_cntx_set_blkszs
		(
		  cntx,
		  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
		  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
		  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
		  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
		  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
		  BLIS_VA_END
		);
	}
}
