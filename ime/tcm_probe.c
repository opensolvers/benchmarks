/*
 * tcm_probe.c - safe /dev/tcm probe (run with sudo on cluster-0 core).
 *
 *   sudo taskset -c 0 ./ime-tcm-probe [blocks]
 *
 * blocks: number of 128 KiB blocks to allocate (1..4, default 1).
 * Never mmap's the full 512 KiB in one shot unless TCM_PROBE_FULL=1.
 *
 * SPDX-License-Identifier: MIT
 */
#include "tcm.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void probe_libspine(void)
{
    printf("=== libspine_tcm.so (optional) ===\n");
    void *h = dlopen("libspine_tcm.so", RTLD_LAZY);
    if (!h) {
        printf("not installed (%s)\n", dlerror());
        return;
    }
    printf("present — use spine_tcm.h + libspine_tcm for production llama path\n");
    dlclose(h);
}

int main(int argc, char **argv)
{
    const int blocks_want = argc > 1 ? atoi(argv[1]) : 1;
    const int core = getenv("TCM_CORE") ? atoi(getenv("TCM_CORE")) : 0;

    probe_libspine();

    printf("\n=== ime tcm allocator (drivers/misc/tcm.c) ===\n");
    printf("core=%d blocks_want=%d TCM_BLOCK_KB=%s TCM_POLL_MS=%s\n", core, blocks_want,
           getenv("TCM_BLOCK_KB") ? getenv("TCM_BLOCK_KB") : "128",
           getenv("TCM_POLL_MS") ? getenv("TCM_POLL_MS") : "5000");

    const int err = tcm_init(core);
    if (err != 0) {
        printf("tcm_init failed: %d (%s)\n", err, strerror(-err));
        printf("hint: sudo taskset -c 0 %s\n", argv[0]);
        return 1;
    }

    printf("block_size=%zu bytes  blocks_total=%zu  capacity=%zu KiB\n", tcm_block_size(),
           tcm_blocks_total(), tcm_block_size() * tcm_blocks_total() / 1024);

    const int st = tcm_self_test();
    printf("self_test (1 block): %s\n", st == 0 ? "OK" : "FAIL");

    if (blocks_want > 1) {
        const size_t want = (size_t)blocks_want * tcm_block_size();
        void *p = tcm_malloc(want);
        printf("malloc(%zu = %d blocks): %p\n", want, blocks_want, p);
        if (p) {
            volatile char *c = (volatile char *)p;
            c[0] = 1;
            c[want - 1] = 2;
            printf("touch head/tail OK\n");
            tcm_free(p);
        }
    }

    if (getenv("TCM_PROBE_FULL")) {
        const size_t full = tcm_block_size() * tcm_blocks_total();
        printf("\nTCM_PROBE_FULL=1: trying full %zu KiB mapping...\n", full / 1024);
        void *p = tcm_malloc(full);
        printf("full mmap: %p\n", p);
        if (p)
            tcm_free(p);
    }

    tcm_shutdown();
    return st == 0 ? 0 : 1;
}
