/*
 * SpacemiT /dev/tcm userspace allocator (drivers/misc/tcm.c API).
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE

#include "tcm.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define TCM_DEV_PATH "/dev/tcm"

#define TCM_IOC_MAGIC 'c'
#define TCM_MEM_SHOW _IOR(TCM_IOC_MAGIC, 2, int)
#define TCM_REQUEST_MEM _IOR(TCM_IOC_MAGIC, 5, int)
#define TCM_RELEASE_MEM _IOR(TCM_IOC_MAGIC, 6, int)

#define TCM_DEFAULT_BLOCK (128u * 1024u)
#define TCM_DEFAULT_BLOCKS_TOTAL 4u /* 512 KiB cluster-0 TCM on K1/X60 */
#define TCM_DEFAULT_POLL_MS 5000

typedef struct tcm_map {
    void *ptr;
    size_t size;
    struct tcm_map *next;
} tcm_map;

static struct {
    int fd;
    int core_id;
    size_t block_size;
    size_t blocks_total;
    int poll_ms;
    int ready;
    tcm_map *maps;
} g_tcm;

static size_t tcm_env_u32(const char *name, size_t def)
{
    const char *s = getenv(name);
    if (!s || !s[0])
        return def;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || v == 0)
        return def;
    return (size_t)v;
}

static size_t tcm_align_up(size_t size, size_t align)
{
    if (align == 0)
        return size;
    const size_t mask = align - 1;
    return (size + mask) & ~mask;
}

static int tcm_bind_core(int core_id)
{
    if (core_id < 0)
        return 0;

#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((unsigned)core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        return -errno;
#else
    (void)core_id;
#endif
    return 0;
}

static tcm_map *tcm_find_map(void *ptr)
{
    for (tcm_map *m = g_tcm.maps; m; m = m->next) {
        if (m->ptr == ptr)
            return m;
    }
    return NULL;
}

static void *tcm_mmap_once(size_t size)
{
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_tcm.fd, 0);
    return (p == MAP_FAILED) ? NULL : p;
}

static void *tcm_mmap_sync(size_t size)
{
    void *p = tcm_mmap_once(size);
    if (p)
        return p;

    const int poll_ms = g_tcm.poll_ms > 0 ? g_tcm.poll_ms : TCM_DEFAULT_POLL_MS;

    if (ioctl(g_tcm.fd, TCM_REQUEST_MEM, &size) < 0)
        return NULL;

    struct pollfd pf = {
        .fd     = g_tcm.fd,
        .events = POLLIN | POLLERR,
    };
    const int ret = poll(&pf, 1, poll_ms);

    (void)ioctl(g_tcm.fd, TCM_RELEASE_MEM, &size);

    if (ret <= 0 || (pf.revents & POLLERR))
        return NULL;

    return tcm_mmap_once(size);
}

int tcm_init(int core_id)
{
    if (g_tcm.ready)
        return 0;

    g_tcm.block_size = tcm_env_u32("TCM_BLOCK_KB", TCM_DEFAULT_BLOCK / 1024u) * 1024u;
    g_tcm.blocks_total = tcm_env_u32("TCM_BLOCKS", TCM_DEFAULT_BLOCKS_TOTAL);
    g_tcm.poll_ms = (int)tcm_env_u32("TCM_POLL_MS", (size_t)TCM_DEFAULT_POLL_MS);
    g_tcm.core_id = core_id;

    if ((g_tcm.block_size & (g_tcm.block_size - 1)) != 0)
        return -EINVAL;

    if (tcm_bind_core(core_id) != 0)
        return -errno;

    g_tcm.fd = open(TCM_DEV_PATH, O_RDWR | O_CLOEXEC);
    if (g_tcm.fd < 0)
        return -errno;

    g_tcm.ready = 1;
    return 0;
}

void tcm_shutdown(void)
{
    while (g_tcm.maps) {
        tcm_map *m = g_tcm.maps;
        g_tcm.maps = m->next;
        munmap(m->ptr, m->size);
        free(m);
    }

    if (g_tcm.fd >= 0) {
        close(g_tcm.fd);
        g_tcm.fd = -1;
    }

    g_tcm.ready = 0;
}

int tcm_is_ready(void)
{
    return g_tcm.ready;
}

size_t tcm_block_size(void)
{
    return g_tcm.block_size ? g_tcm.block_size : TCM_DEFAULT_BLOCK;
}

size_t tcm_blocks_total(void)
{
    return g_tcm.blocks_total ? g_tcm.blocks_total : TCM_DEFAULT_BLOCKS_TOTAL;
}

void *tcm_malloc(size_t size)
{
    if (!g_tcm.ready || size == 0)
        return NULL;

    const size_t block = tcm_block_size();
    const size_t aligned = tcm_align_up(size, block);
    const size_t max_bytes = block * tcm_blocks_total();

    if (aligned == 0 || aligned > max_bytes)
        return NULL;

    void *p = tcm_mmap_sync(aligned);
    if (!p)
        return NULL;

    tcm_map *m = (tcm_map *)calloc(1, sizeof(*m));
    if (!m) {
        munmap(p, aligned);
        return NULL;
    }

    m->ptr  = p;
    m->size = aligned;
    m->next = g_tcm.maps;
    g_tcm.maps = m;
    return p;
}

void tcm_free(void *ptr)
{
    if (!ptr)
        return;

    tcm_map *m = tcm_find_map(ptr);
    if (!m)
        return;

    if (m == g_tcm.maps)
        g_tcm.maps = m->next;
    else {
        tcm_map *prev = g_tcm.maps;
        while (prev && prev->next != m)
            prev = prev->next;
        if (prev)
            prev->next = m->next;
    }

    munmap(m->ptr, m->size);
    free(m);
}

int tcm_self_test(void)
{
    if (!g_tcm.ready)
        return -EINVAL;

    const size_t block = tcm_block_size();
    void *p = tcm_malloc(block);
    if (!p)
        return -ENOMEM;

    volatile unsigned char *b = (volatile unsigned char *)p;
    b[0] = 0xa5;
    b[block - 1] = 0x5a;

    const int ok = (b[0] == 0xa5 && b[block - 1] == 0x5a);
    tcm_free(p);
    return ok ? 0 : -EIO;
}
