/* h2d_bw.c -- where does the correlator's H2D bandwidth actually go?
 *
 * Measured facts this program exists to explain:
 *   - a plain linear cudaMemcpyAsync reaches ~27 GB/s on this link (PCIe 4.0 x16)
 *   - the full correlator pipeline only achieves 12-16 GB/s of input
 *   - 15 stations need 12.0 GB/s and 24 stations need 19.2 GB/s
 *
 * So either the strided 2D copy the pipeline uses is much slower than a linear
 * one, or the copies are not overlapping with compute. This isolates the first
 * question by moving the same bytes three ways, using the real geometry:
 *
 *   n_station stations, each owning a block of n_cum frames, copied chunk by
 *   chunk exactly as gpu_process_block() does it.
 *
 * usage: h2d_bw [N_STATION] [N_FREQ] [N_CUM] [CHUNK_FRAMES] [BLOCKS]
 */

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main(int argc, char **argv) {
    const int n_station = argc > 1 ? atoi(argv[1]) : 24;
    const int n_freq = argc > 2 ? atoi(argv[2]) : 4096;
    const int n_cum = argc > 3 ? atoi(argv[3]) : 1000;
    const int chunk_frames = argc > 4 ? atoi(argv[4]) : 128;
    const int blocks = argc > 5 ? atoi(argv[5]) : 50;

    const size_t frame_stride = (size_t)2 * n_freq;
    const size_t block_stride = frame_stride * (size_t)n_cum; /* per station */
    const size_t chunk_bytes = frame_stride * (size_t)chunk_frames;
    const int n_chunks = (n_cum + chunk_frames - 1) / chunk_frames;
    const size_t total = block_stride * (size_t)n_station;

    printf("h2d_bw: %d stations, n_freq=%d n_cum=%d chunk=%d frames\n", n_station,
           n_freq, n_cum, chunk_frames);
    printf("        frame=%zu B, station block=%.1f MiB, chunk=%.1f MiB, %d chunks, "
           "total=%.1f MiB/block, %d blocks\n",
           frame_stride, block_stride / 1048576.0, chunk_bytes / 1048576.0, n_chunks,
           total / 1048576.0, blocks);

    CK(cudaSetDevice(0));

    /* Host side mirrors the real staging layout: [station][chunk]. */
    void *host = NULL;
    CK(cudaHostAlloc(&host, total, cudaHostAllocDefault));
    memset(host, 0x5a, total);
    void *devp = NULL;
    CK(cudaMalloc(&devp, total));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    /* Ramp the link so we do not measure a downclocked state. */
    CK(cudaMemcpyAsync(devp, host, host ? chunk_bytes : chunk_bytes,
                       cudaMemcpyHostToDevice, s));
    CK(cudaStreamSynchronize(s));

    const double moved = (double)total * blocks;

    /* --- 2d: one cudaMemcpy2DAsync per chunk, the current pipeline's pattern --- */
    {
        const double t0 = now_s();
        for (int b = 0; b < blocks; ++b) {
            for (int c = 0; c < n_chunks; ++c) {
                const int ft = (c + 1) * chunk_frames <= n_cum
                                   ? chunk_frames
                                   : n_cum - c * chunk_frames;
                const size_t w = frame_stride * (size_t)ft;
                CK(cudaMemcpy2DAsync(devp, w,
                                     (const char *)host + (size_t)c * chunk_bytes,
                                     block_stride, w, (size_t)n_station,
                                     cudaMemcpyHostToDevice, s));
            }
        }
        CK(cudaStreamSynchronize(s));
        printf("  2d       %6.2f GB/s   (%.4f s)\n", moved / (now_s() - t0) / 1e9,
               now_s() - t0);
    }

    /* --- per-row: the same bytes as n_station linear copies per chunk --- */
    {
        const double t0 = now_s();
        for (int b = 0; b < blocks; ++b) {
            for (int c = 0; c < n_chunks; ++c) {
                const int ft = (c + 1) * chunk_frames <= n_cum
                                   ? chunk_frames
                                   : n_cum - c * chunk_frames;
                const size_t w = frame_stride * (size_t)ft;
                for (int g = 0; g < n_station; ++g) {
                    CK(cudaMemcpyAsync((char *)devp + ((size_t)c * n_station + g) * w,
                                       (const char *)host + (size_t)g * block_stride +
                                           (size_t)c * chunk_bytes,
                                       w, cudaMemcpyHostToDevice, s));
                }
            }
        }
        CK(cudaStreamSynchronize(s));
        printf("  per-row  %6.2f GB/s   (%.4f s)\n", moved / (now_s() - t0) / 1e9,
               now_s() - t0);
    }

    /* --- linear: one copy per station block, i.e. the upper bound for this
     *     layout with no chunking at all --- */
    {
        const double t0 = now_s();
        for (int b = 0; b < blocks; ++b) {
            for (int g = 0; g < n_station; ++g) {
                CK(cudaMemcpyAsync((char *)devp + (size_t)g * block_stride,
                                   (const char *)host + (size_t)g * block_stride,
                                   block_stride, cudaMemcpyHostToDevice, s));
            }
        }
        CK(cudaStreamSynchronize(s));
        printf("  linear   %6.2f GB/s   (%.4f s)\n", moved / (now_s() - t0) / 1e9,
               now_s() - t0);
    }

    cudaFree(devp);
    cudaFreeHost(host);
    cudaStreamDestroy(s);
    return 0;
}
