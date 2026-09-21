/* Trace the host call stack of CUDA launches, to see below-Python kernel enqueues.
 *
 * v4. What the previous attempts established:
 *   - the library loads under LD_PRELOAD (the constructor prints), and
 *   - torch never calls the `cudaLaunchKernel` I define, even for a 116 MB uint8 copy_, so
 *     interposing only that entry point is not enough. v3's header comment claimed
 *     cudaLaunchKernelExC was covered; the code did not define it. It does now.
 *   - v2 passed shmem where args belongs, corrupting every launch it forwarded. Both
 *     entry points here use the real runtime layout.
 *
 *   DCOPY_TRACE_OUT       output file (default stderr)
 *   DCOPY_TRACE_LIMIT     maximum stack dumps (default 200)
 *   DCOPY_TRACE_BLOCK_X   only dump launches with this blockX (default 128, 0 = any)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    unsigned x, y, z;
} dim3_t;

typedef struct {
    dim3_t gridDim;
    dim3_t blockDim;
    size_t dynamicSmemBytes;
    void *stream;
    void *attrs;
    unsigned numAttrs;
} launch_config_t;

typedef int (*launch_t)(const void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned, void **, size_t, void *);
typedef int (*launch_exc_t)(const launch_config_t *, const void *, void **);

static launch_t real_launch = NULL;
static launch_exc_t real_launch_exc = NULL;
static FILE *out = NULL;
static int ready = 0;
static long launches = 0;
static long dumps = 0;
static long limit = 200;
static unsigned want_block_x = 128;

__attribute__((constructor)) static void onload(void) {
    fprintf(stderr, "[dcopy_trace] loaded pid=%d\n", (int)getpid());
    fflush(stderr);
}

static void init(void) {
    if (ready) {
        return;
    }
    ready = 1;
    real_launch = (launch_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    real_launch_exc = (launch_exc_t)dlsym(RTLD_NEXT, "cudaLaunchKernelExC");
    const char *p = getenv("DCOPY_TRACE_OUT");
    out = p ? fopen(p, "a") : stderr;
    if (!out) {
        out = stderr;
    }
    const char *li = getenv("DCOPY_TRACE_LIMIT");
    const char *bx = getenv("DCOPY_TRACE_BLOCK_X");
    if (li) limit = strtol(li, NULL, 10);
    if (bx) want_block_x = (unsigned)strtoul(bx, NULL, 10);
    fprintf(out, "[dcopy_trace] init pid=%d plain=%p exc=%p block_x=%u\n", (int)getpid(),
            (void *)real_launch, (void *)real_launch_exc, want_block_x);
    fflush(out);
}

static void dump_stack(const char *which, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                       unsigned by, unsigned bz) {
    if (dumps >= limit || (want_block_x != 0 && bx != want_block_x)) {
        return;
    }
    dumps++;
    void *bt[40];
    int n = backtrace(bt, 40);
    char **sym = backtrace_symbols(bt, n);
    fprintf(out, "=== dump#%ld %s launch#%ld grid=(%u,%u,%u) block=(%u,%u,%u)\n", dumps, which,
            launches, gx, gy, gz, bx, by, bz);
    for (int i = 0; i < n; i++) {
        fprintf(out, "    %s\n", sym ? sym[i] : "?");
    }
    fflush(out);
    if (sym) {
        free(sym);
    }
}

int cudaLaunchKernel(const void *func, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                     unsigned by, unsigned bz, void **args, size_t shmem, void *stream) {
    init();
    launches++;
    dump_stack("plain", gx, gy, gz, bx, by, bz);
    if (real_launch == NULL) {
        fprintf(stderr, "[dcopy_trace] FATAL: real cudaLaunchKernel missing\n");
        fflush(stderr);
        abort();
    }
    return real_launch(func, gx, gy, gz, bx, by, bz, args, shmem, stream);
}

int cudaLaunchKernelExC(const launch_config_t *cfg, const void *func, void **args) {
    init();
    launches++;
    if (cfg != NULL) {
        dump_stack("exc", cfg->gridDim.x, cfg->gridDim.y, cfg->gridDim.z, cfg->blockDim.x,
                   cfg->blockDim.y, cfg->blockDim.z);
    }
    if (real_launch_exc == NULL) {
        fprintf(stderr, "[dcopy_trace] FATAL: real cudaLaunchKernelExC missing\n");
        fflush(stderr);
        abort();
    }
    return real_launch_exc(cfg, func, args);
}

/* CUDA-graph replay entry point. If the decode step is captured, this is where its kernels
 * actually execute: graph replay does not call cudaLaunchKernel or cudaLaunchKernelExC once
 * per node, which is why an interposer on those two entry points sees nothing no matter
 * which process loads it. Counting this, without filtering, is the cheap test of that.
 */
typedef int (*graph_launch_t)(void *graphExec, void *stream);
static graph_launch_t real_graph_launch = NULL;
static long graph_launches = 0;

int cudaGraphLaunch(void *graphExec, void *stream) {
    init();
    graph_launches++;
    if (graph_launches <= 2) {
        dump_stack("graphlaunch", 0, 0, 0, 0, 0, 0);
    } else if (graph_launches % 25 == 0) {
        fprintf(out, "[dcopy_trace] cudaGraphLaunch total=%ld\n", graph_launches);
        fflush(out);
    }
    if (real_graph_launch == NULL) {
        real_graph_launch = (graph_launch_t)dlsym(RTLD_NEXT, "cudaGraphLaunch");
    }
    if (real_graph_launch == NULL) {
        fprintf(stderr, "[dcopy_trace] FATAL: real cudaGraphLaunch missing\n");
        fflush(stderr);
        abort();
    }
    return real_graph_launch(graphExec, stream);
}

/* CUDA-graph node dump. A launch-API hook cannot see the copies because replay does not call
 * cudaLaunchKernel once per node, so the graph itself has to be read. cudaGraphInstantiateWithFlags
 * is handed the graph before instantiation; cudaGraphGetNodes plus cudaGraphKernelNodeGetParams
 * then give each kernel node's func pointer, grid and block. The func pointer is opaque, so the
 * join to a kernel name is by geometry: compare the dump against a named kernel list (nsys or the
 * torch profiler in eager mode) and match on grid/block. cudaGraphNodeTypeKernel == 0.
 */
struct kernel_node_params {
    void *func;
    dim3_t gridDim;
    dim3_t blockDim;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
};

typedef int (*graph_instantiate_t)(void **graphExec, void *graph, unsigned long long flags);
typedef int (*graph_get_nodes_t)(void *graph, void **nodes, size_t *numNodes);
typedef int (*graph_node_get_type_t)(void *node, int *type);
typedef int (*graph_kernel_node_get_params_t)(void *node, struct kernel_node_params *params);

static graph_instantiate_t real_graph_instantiate = NULL;
static int dumped_graphs = 0;

static void dump_graph_nodes(void *graph) {
    if (graph == NULL || dumped_graphs >= 3) {
        return;
    }
    dumped_graphs++;
    graph_get_nodes_t get_nodes = (graph_get_nodes_t)dlsym(RTLD_NEXT, "cudaGraphGetNodes");
    graph_node_get_type_t get_type = (graph_node_get_type_t)dlsym(RTLD_NEXT, "cudaGraphNodeGetType");
    graph_kernel_node_get_params_t get_params =
        (graph_kernel_node_get_params_t)dlsym(RTLD_NEXT, "cudaGraphKernelNodeGetParams");
    if (get_nodes == NULL || get_type == NULL || get_params == NULL) {
        fprintf(out, "[dcopy_trace] graph node APIs unavailable (get_nodes=%p)\n",
                (void *)get_nodes);
        fflush(out);
        return;
    }
    size_t n = 0;
    if (get_nodes(graph, NULL, &n) != 0) {
        fprintf(out, "[dcopy_trace] cudaGraphGetNodes(count) failed\n");
        fflush(out);
        return;
    }
    fprintf(out, "[dcopy_trace] graph #%d: %zu nodes\n", dumped_graphs, n);
    fflush(out);
    void **nodes = (void **)calloc(n > 0 ? n : 1, sizeof(void *));
    if (nodes == NULL) {
        return;
    }
    if (get_nodes(graph, nodes, &n) != 0) {
        free(nodes);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        int type = -1;
        if (get_type(nodes[i], &type) != 0 || type != 0) {
            continue;
        }
        struct kernel_node_params params;
        memset(&params, 0, sizeof(params));
        if (get_params(nodes[i], &params) != 0) {
            continue;
        }
        fprintf(out, "  node %zu func=%p grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u\n", i,
                params.func, params.gridDim.x, params.gridDim.y, params.gridDim.z,
                params.blockDim.x, params.blockDim.y, params.blockDim.z,
                params.sharedMemBytes);
    }
    fflush(out);
    free(nodes);
}

int cudaGraphInstantiateWithFlags(void **graphExec, void *graph, unsigned long long flags) {
    init();
    dump_graph_nodes(graph);
    if (real_graph_instantiate == NULL) {
        real_graph_instantiate =
            (graph_instantiate_t)dlsym(RTLD_NEXT, "cudaGraphInstantiateWithFlags");
    }
    if (real_graph_instantiate == NULL) {
        fprintf(stderr, "[dcopy_trace] FATAL: real cudaGraphInstantiateWithFlags missing\n");
        fflush(stderr);
        abort();
    }
    return real_graph_instantiate(graphExec, graph, flags);
}

/* libtorch_cuda.so references these with a version tag, so an unversioned definition
 * binds to nothing and the shim is never entered. Verified in the image:
 *     U cudaLaunchKernel@libcudart.so.13
 *     U cudaLaunchKernelExC@libcudart.so.13
 * The version node has to be declared, which .symver alone cannot do (it fails to link
 * with "version node not found"), so the definitions are tagged by libcudart.vers and
 * the link line passes -Wl,--version-script=libcudart.vers. libcudart's internal alias
 * __cudaLaunchKernel@libcudart.so.13 points at the same entry point and is not covered;
 * add it to that script if a trace still comes back empty.
 */

