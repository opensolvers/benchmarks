/*
 * Minimal userspace TCM allocator for SpacemiT /dev/tcm (K1/X60 cluster 0).
 *
 * Talks directly to drivers/misc/tcm.c — no libspine_tcm.so required.
 * All sizes are rounded up to the hardware block size (128 KiB on K1).
 *
 * Run as root (or with CAP_SYS_RAWIO). Bind to a cluster-0 core before use.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef IME_TCM_H
#define IME_TCM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open /dev/tcm, optionally pin calling thread to |core_id| (>=0). */
int tcm_init(int core_id);

void tcm_shutdown(void);

int tcm_is_ready(void);

/* Hardware block size in bytes (default 128 KiB; override with TCM_BLOCK_KB). */
size_t tcm_block_size(void);

/* Number of 128 KiB blocks in one mapping (for sizing B-panels etc.). */
size_t tcm_blocks_total(void);

/*
 * Allocate |size| bytes from TCM. Rounded up to a block multiple.
 * Uses TCM_REQUEST_MEM + poll when the heap is busy; times out (TCM_POLL_MS).
 * Returns NULL on failure — never blocks forever.
 */
void *tcm_malloc(size_t size);

/* Release a mapping from tcm_malloc(). */
void tcm_free(void *ptr);

/*
 * Safe self-test: one block, touch, free. Returns 0 on success.
 * Suitable for ime-tcm-probe without wedging the board.
 */
int tcm_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* IME_TCM_H */
