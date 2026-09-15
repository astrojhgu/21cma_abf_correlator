/* multi_h2d_bw.c -- how fast can N cards pull the same staging bytes at once?
 *
 * This is the number the whole multi-card design turns on. A card that holds a
 * copy of every station must pull every station's stream, so with S stations and
 * N cards the host link has to carry S*N*0.8 GB/s, and the pipeline only keeps up
 * if the machine can actually do that. Everything else -- the split, the pair
 * assignment, whether a covering design is needed at all -- follows from it.
 *
 * It deliberately contains no pipeline: no FFT, no correlation, no sockets. Just
 * the copy pattern the engine uses (one 2-D copy per chunk per device, from one
 * shared portable pinned buffer, all devices concurrently), so the result is a
 * property of the machine rather than of the correlator.
 *
 * usage: multi_h2d_bw [n_devices] [n_station] [seconds]
 */

#include "corr_cuda.h"

#include <cuda_runtime_api.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x)                                                                     \
    do {                                                                          \
        cudaError_t e_ = (x);                                                     \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "multi_h2d_bw: %s: %s\n", #x, cudaGetErrorString(e_)); \
            exit(2);                                                              \
        }                                                                         \
    } while (0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    int device;
    const int8_t *host;
    size_t station_stride;   /* bytes per station in the shared staging */
    size_t slot_stride;      /* bytes per staging slot, per station     */
    int n_station;
    int frames_per_slot;
    int chunk_frames;
    int block_slots;
    double seconds;
    double bytes;            /* out: bytes this device pulled */
    int transfers;           /* out */
} Job;

static void *worker(void *arg) {
    Job *j = (Job *)arg;
    CK(cudaSetDevice(j->device));

    const size_t frame_stride =
        j->slot_stride / (size_t)j->frames_per_slot; /* = 2*n_freq */
    const size_t chunk_bytes = frame_stride * (size_t)j->chunk_frames;
    const int n_chunks = (j->frames_per_slot + j->chunk_frames - 1) / j->chunk_frames;

    void *devp = NULL;
    CK(cudaMalloc(&devp, chunk_bytes * (size_t)j->n_station));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    /* Ramp once so the first measurement is not a downclocked link. */
    CK(cudaMemcpyAsync(devp, j->host, chunk_bytes, cudaMemcpyHostToDevice, s));
    CK(cudaStreamSynchronize(s));

    const double deadline = now_s() + j->seconds;
    const int8_t *slot_base;
    int slot = 0;
    while (now_s() < deadline) {
        slot = (slot + 1) % j->block_slots;
        slot_base = j->host + (size_t)slot * j->slot_stride;
        for (int c = 0; c < n_chunks; ++c) {
            const size_t w = chunk_bytes;
            CK(cudaMemcpy2DAsync((char *)devp + (size_t)c * j->n_station * w, w,
                                 slot_base + (size_t)c * chunk_bytes, j->station_stride,
                                 w, (size_t)j->n_station, cudaMemcpyHostToDevice, s));
            j->bytes += (double)w * (double)j->n_station;
            ++j->transfers;
        }
    }
    CK(cudaStreamSynchronize(s));
    CK(cudaFree(devp));
    CK(cudaStreamDestroy(s));
    return NULL;
}

int main(int argc, char **argv) {
    int visible = 0;
    CK(cudaGetDeviceCount(&visible));

    int n_devices = argc > 1 ? atoi(argv[1]) : visible;
    const int n_station = argc > 2 ? atoi(argv[2]) : 24;
    const double seconds = argc > 3 ? atof(argv[3]) : 5.0;
    const int n_freq = 4096;
    const int frames_per_slot = 256;
    const int chunk_frames = 256;
    const int block_slots = 3;

    if (n_devices < 1 || n_devices > visible) {
        fprintf(stderr, "multi_h2d_bw: %d device(s) requested, %d visible\n", n_devices,
                visible);
        return n_devices == 0 ? 0 : 2;
    }

    const size_t frame_stride = (size_t)2 * n_freq;
    const size_t slot_stride = frame_stride * (size_t)frames_per_slot;
    const size_t station_stride = slot_stride * (size_t)block_slots;
    const size_t total = station_stride * (size_t)n_station;

    printf("multi_h2d_bw: %d stations x %d device(s), %.1f s each; staging %.0f MiB "
           "(%d slots of %d frames)\n",
           n_station, n_devices, seconds, total / 1048576.0, block_slots, frames_per_slot);
    printf("              demand if every device holds every station: %.1f GB/s aggregate "
           "(%.2f GB/s per device)\n",
           (double)n_station * n_devices * frame_stride * 800000000.0 / (double)(2 * n_freq) /
               1e9,
           (double)n_station * frame_stride * 800000000.0 / (double)(2 * n_freq) / 1e9);

    /* One shared portable pinned buffer, exactly as the correlator allocates it:
     * the receive threads write it once and every device reads it. */
    void *host = NULL;
    CK(cudaHostAlloc(&host, total, cudaHostAllocPortable));
    memset(host, 0x5a, total);

    pthread_t th[8];
    Job jobs[8];
    for (int d = 0; d < n_devices; ++d) {
        jobs[d].device = d;
        jobs[d].host = (const int8_t *)host;
        jobs[d].station_stride = station_stride;
        jobs[d].slot_stride = slot_stride;
        jobs[d].n_station = n_station;
        jobs[d].frames_per_slot = frames_per_slot;
        jobs[d].chunk_frames = chunk_frames;
        jobs[d].block_slots = block_slots;
        jobs[d].seconds = seconds;
        jobs[d].bytes = 0.0;
        jobs[d].transfers = 0;
    }

    const double t0 = now_s();
    for (int d = 0; d < n_devices; ++d)
        if (pthread_create(&th[d], NULL, worker, &jobs[d]) != 0) {
            fprintf(stderr, "multi_h2d_bw: cannot create thread %d\n", d);
            return 2;
        }
    for (int d = 0; d < n_devices; ++d) pthread_join(th[d], NULL);
    const double dt = now_s() - t0;

    double sum = 0.0;
    int transfers = 0;
    for (int d = 0; d < n_devices; ++d) {
        printf("  dev %d  %6.2f GB/s   (%.2f GB moved, %d transfers)\n", jobs[d].device,
               jobs[d].bytes / dt / 1e9, jobs[d].bytes / 1e9, jobs[d].transfers);
        sum += jobs[d].bytes;
        transfers += jobs[d].transfers;
    }
    printf("  AGGREGATE %6.2f GB/s over %d device(s), %.2f s, %d transfers\n", sum / dt / 1e9,
           n_devices, dt, transfers);

    CK(cudaFreeHost(host));
    return 0;
}
