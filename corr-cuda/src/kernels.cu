// kernels.cu -- the GPU work behind corr_row_accumulate() and friends.
//
// Per staging slot, per chunk of `chunk_frames` frames:
//
//   pinned int8 staging
//     --cudaMemcpy2DAsync-->      d_int8   [station][frame][sample]
//     --cudaMemcpy2DAsync-->      d_mask   [station][bit per frame of the slot]
//     --zero_invalid_frames-->    d_int8   frames the mask calls missing := 0
//     --int8_to_float-------->    d_real   [station][frame][sample]  float32
//     --cufftExecR2C (batched)--> d_y      [station][frame][n_freq+1] complex64
//     --corr_kernel---------->    acc[row] [baseline][n_freq]        += ...
//
// Three decisions here come from measurement, not taste:
//
//  * Missing frames are zeroed HERE, not on the receive thread. The host used to
//    write the zeros, and at nch=8192 that is ~5 GB/s of memset on the same thread
//    that has to keep reading its socket -- the amplification loop
//    docs/DESIGN-zero-fill-decoupling.md exists to break. What the host sends
//    instead is one bit per frame, and the zeroing costs nothing when nothing is
//    missing.
//
//  * int8 -> float32 happens on the DEVICE. 15 stations is 12.0 GB/s of int8;
//    converting on the host first would mean 48.1 GB/s over PCIe, and the link
//    tops out at 27 GB/s.
//
//  * The correlation kernel stages Y in shared memory and gives each thread one
//    baseline across a tile of frequencies. The naive alternative -- one thread
//    per (baseline, frequency) looping over frames straight out of global
//    memory -- re-reads Y once per station, about 7.9 GB per block at 15
//    stations, which is at the L20's memory bandwidth limit.
//
//  * Copies run on a separate stream from the kernels. A plain H2D reaches
//    27 GB/s on this link and the strided 2D copy is just as fast (26.9 GB/s
//    measured), but with everything on one stream the pipeline only moved
//    12-16 GB/s because the copy and the compute took turns. Overlapping them is
//    the difference between one GPU keeping up with 24 stations and not.
//
// The accumulator is a stack of `rows_per_fetch` rows, so the integration length
// and the host transfer cadence are independent.

#include "gpu_internal.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace corr {
namespace {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

corr_status gpu_fail(corr_status st, const std::string &msg) {
    last_error() = msg;
    return st;
}

corr_status cuda_check(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return CORR_OK;
    return gpu_fail(CORR_ERR_CUDA, std::string(what) + ": " + cudaGetErrorName(e) +
                                       ": " + cudaGetErrorString(e));
}

const char *cufft_result_name(cufftResult r) {
    switch (r) {
    case CUFFT_SUCCESS:                   return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:              return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:              return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:              return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:             return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:            return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:               return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:              return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:              return "CUFFT_INVALID_SIZE";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:            return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:               return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:              return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:           return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED:             return "CUFFT_NOT_SUPPORTED";
    default:                              return "CUFFT_UNKNOWN";
    }
}

corr_status cufft_check(cufftResult r, const char *what) {
    if (r == CUFFT_SUCCESS) return CORR_OK;
    return gpu_fail(CORR_ERR_CUFFT, std::string(what) + ": " + cufft_result_name(r));
}

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/// Frequencies handled by one correlation thread block. Each thread keeps this
/// many complex accumulators in registers for the whole chunk.
constexpr int kChannelsPerBlock = 16;

/// Frames staged into shared memory per __syncthreads() round.
constexpr int kStagedFrames = 4;

/// Threads per correlation block: one per baseline. Hence the n_station <= 44 cap
/// (44*45/2 = 990) enforced in gpu_create.
constexpr int kMaxBaselines = 1024;

/// Device-side chunk buffers, i.e. how many host-to-device copies may be in
/// flight ahead of the kernels. Three is enough to keep the copy engine busy
/// while two chunks compute.
constexpr int kDeviceBuffers = 3;

constexpr int kConvertThreads = 256;
constexpr int kScaleThreads = 256;

/// Threads per block in the frame-zeroing kernel. One block covers
/// kZeroThreads*16 bytes of a frame, so a 8192-sample frame takes two blocks.
constexpr int kZeroThreads = 256;

inline int ceil_div(long long a, long long b) { return (int)((a + b - 1) / b); }

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

__global__ void int8_to_float_kernel(const int8_t *__restrict__ in,
                                     float *__restrict__ out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        // Two's complement int8 straight to float. data_shift_count is front-end
        // metadata and never enters this conversion.
        out[i] = (float)in[i];
    }
}

/// Zero every frame of the chunk whose validity bit is clear.
///
/// The host no longer fills missing frames with zeros -- the receive thread only
/// says which frames arrived (see docs/DESIGN-zero-fill-decoupling.md). This is
/// where the silence is actually written, which is what lets a station that fell
/// behind keep reading its socket instead of writing gigabytes of zeros into the
/// staging it is also trying to keep up with.
///
/// Grid : (blocks over one frame's samples, frames in this chunk, stations)
/// Block: threads striding over the frame's samples.
///
/// One block per frame rather than an elementwise test: a frame is either wholly
/// there or wholly zero, so the mask is read once per frame instead of once per
/// sample, and the common case (nothing missing) costs one early return per
/// block. A frame is `fft_len` *int8* samples -- `fft_len` is an element count, not
/// a byte count, and the two happen to agree here only because a sample is one
/// byte -- and n_freq is a multiple of 2048, so fft_len is a multiple of 4096 and
/// the frame is a whole number of int4 stores.
__global__ void zero_invalid_frames_kernel(int8_t *__restrict__ d_int8,
                                           const unsigned char *__restrict__ d_mask,
                                           int mask_bytes_per_slot, int f0,
                                           int fft_len, long long w) {
    const int f = f0 + (int)blockIdx.y;
    const int st = (int)blockIdx.z;
    const unsigned char *m = d_mask + (size_t)st * (size_t)mask_bytes_per_slot;
    if (m[f >> 3] & (1u << (f & 7))) return; // present: leave the samples alone

    int4 *p = reinterpret_cast<int4 *>(d_int8 + (size_t)st * (size_t)w +
                                       (size_t)blockIdx.y * (size_t)fft_len);
    const int n4 = fft_len / (int)sizeof(int4);
    for (int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; i < n4;
         i += (int)gridDim.x * (int)blockDim.x)
        p[i] = make_int4(0, 0, 0, 0);
}

/// acc[row][ch] *= s, used to divide a completed integration row by its frame
/// count exactly once.
__global__ void scale_row_kernel(float2 *__restrict__ row, long long n, float s) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        row[i].x *= s;
        row[i].y *= s;
    }
}

/// acc[row][b][ch] += sum_frames Y[j][f][ch] * conj(Y[k][f][ch])
///
/// Grid : one block per group of kChannelsPerBlock frequencies.
/// Block: one thread per baseline.
///
/// A (station, frame) slice of the channel tile is staged in shared memory, so
/// global memory is read once regardless of how many baselines touch a station.
__global__ void corr_kernel(const float2 *__restrict__ Y,
                            const int2 *__restrict__ pairs,
                            float2 *__restrict__ acc_row,
                            int n_station,
                            int chunk_frames,
                            int y_frame_stride,
                            int n_freq,
                            int n_baseline) {
    extern __shared__ float2 smem[];

    const int ch0 = blockIdx.x * kChannelsPerBlock;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float2 a[kChannelsPerBlock];
#pragma unroll
    for (int c = 0; c < kChannelsPerBlock; ++c) a[c] = make_float2(0.0f, 0.0f);

    const bool active = tid < n_baseline;
    const int j = active ? pairs[tid].x : 0;
    const int k = active ? pairs[tid].y : 0;

    for (int f0 = 0; f0 < chunk_frames; f0 += kStagedFrames) {
        const int ft = min(kStagedFrames, chunk_frames - f0);

        // Cooperative load of [station][staged frame][channel in tile].
        const int total = n_station * ft * kChannelsPerBlock;
        for (int idx = tid; idx < total; idx += nthreads) {
            const int c = idx % kChannelsPerBlock;
            const int t = (idx / kChannelsPerBlock) % ft;
            const int g = idx / (kChannelsPerBlock * ft);
            const int freq = ch0 + c;
            float2 v = make_float2(0.0f, 0.0f);
            if (freq < n_freq) {
                v = Y[((size_t)g * chunk_frames + (size_t)(f0 + t)) * y_frame_stride +
                      (size_t)freq];
            }
            smem[(g * ft + t) * kChannelsPerBlock + c] = v;
        }
        __syncthreads();

        if (active) {
            for (int t = 0; t < ft; ++t) {
                const float2 *rowj = &smem[(j * ft + t) * kChannelsPerBlock];
                const float2 *rowk = &smem[(k * ft + t) * kChannelsPerBlock];
#pragma unroll
                for (int c = 0; c < kChannelsPerBlock; ++c) {
                    const float2 x = rowj[c];
                    const float2 y = rowk[c];
                    // x * conj(y) = (xr*yr + xi*yi) + i(xi*yr - xr*yi)
                    a[c].x += x.x * y.x + x.y * y.y;
                    a[c].y += x.y * y.x - x.x * y.y;
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        float2 *out = &acc_row[(size_t)tid * n_freq + ch0];
#pragma unroll
        for (int c = 0; c < kChannelsPerBlock; ++c) {
            if (ch0 + c < n_freq) {
                // This block is the only writer for these frequencies, but the
                // kernel runs once per chunk, so earlier chunks' partial sums
                // must be preserved.
                out[c].x += a[c].x;
                out[c].y += a[c].y;
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// GpuState
// ---------------------------------------------------------------------------

struct DeviceBuf {
    int8_t *d_int8 = nullptr;  // [station][frame][sample] for one chunk
    float *d_real = nullptr;   // same shape, float32
    float2 *d_y = nullptr;     // [station][frame][n_freq+1] complex64
    /// Validity mask of the chunk's frames, [station][validity_bytes_per_slot],
    /// straight from the staging: one bit per frame of the *slot*, so the bit for
    /// the frame at chunk offset `fi` is `f0 + fi`. Per device buffer rather than
    /// per slot because it is overwritten by the next chunk's copy that uses this
    /// buffer, and the copy already waits for this buffer's earlier kernels.
    unsigned char *d_mask = nullptr;
};

struct GpuState {
    corr_config cfg{};
    int device = -1;

    /// The stations this device holds a copy of, in local order: `resident[i]`
    /// is the global station index whose frames land in row `i` of this device's
    /// buffers, and the pair table's indices refer to `i`.
    std::vector<int32_t> resident;
    /// True when `resident` is exactly 0,1,2,... -- the common case, and the one
    /// the two-dimensional copy is built for.
    bool resident_is_identity = true;

    int fft_len = 0;   // 2 * n_freq
    int bins = 0;      // n_freq + 1, what cuFFT writes
    int n_station_local = 0;  // rows in d_int8/d_real/d_y, = resident.size()
    int n_baseline = 0;       // baselines *this device* owns
    int32_t first_baseline = 0; // its first baseline in the global numbering
    int rows_per_fetch = 1;
    int cur_row = 0;   // row of the stack being accumulated
    int pending = 0;   // rows closed but not yet fetched

    /// Cumulative traffic, for the run report and for the efficiency question
    /// this whole split exists to answer: how many bytes each card actually
    /// pulled over its host link. Written only by the scheduler thread that
    /// drives the context, so no atomics.
    uint64_t bytes_h2d = 0;
    uint64_t bytes_d2h = 0;
    uint64_t transfers = 0;

    DeviceBuf dbuf[kDeviceBuffers];

    // Accumulator stack: rows_per_fetch x n_baseline x n_freq, zero when the
    // corresponding row is not in use.
    float2 *d_acc = nullptr;
    size_t row_elems = 0;   // n_baseline * n_freq complex
    size_t acc_bytes = 0;
    /// Where another device's row lands before it is added in. Only allocated on
    /// the device that sums the partials, and only for the time-slice split.
    float2 *d_peer = nullptr;
    size_t d_peer_bytes = 0;

    int2 *d_pairs = nullptr;
    size_t d_pairs_bytes = 0;

    cufftHandle plan = 0;
    bool plan_valid = false;

    /// Rotates over the device buffer ring. It must be global rather than the
    /// chunk index within a slot: with one chunk per slot, or any slot length
    /// that shares a factor with kDeviceBuffers, a per-slot index would keep
    /// landing on the same buffer and the copy would have to wait for the
    /// previous chunk's kernels every time -- silently collapsing the overlap
    /// the separate streams exist to provide.
    uint64_t chunk_seq = 0;

    cudaStream_t copy_stream = nullptr;     // host -> device
    cudaStream_t compute_stream = nullptr;  // convert, FFT, correlate, scale
    cudaStream_t fetch_stream = nullptr;    // device -> host
    /// Carries nothing but `cudaStreamWaitEvent` and the host callback queued
    /// behind it, for callers that would rather be told than wait.
    ///
    /// A stream of its own, not the one that recorded the event: the copy stream
    /// has several slots' copies in flight at once, and a callback queued behind
    /// them would fire only after every later copy had finished too -- which is
    /// not the condition any caller is waiting for. Waiting on the *event*
    /// instead is what makes "this slot is done" mean this slot.
    cudaStream_t notify_stream = nullptr;

    std::vector<cudaEvent_t> h2d_done;      // per staging slot
    std::vector<cudaEvent_t> copy_done;     // per device buffer
    std::vector<cudaEvent_t> compute_done;  // per device buffer
    cudaEvent_t row_done = nullptr;         // last corr_row_end() finished
    cudaEvent_t fetch_done = nullptr;       // last D2H finished
};

namespace {

void free_all(GpuState *g) {
    if (g == nullptr) return;
    if (g->plan_valid) {
        cufftDestroy(g->plan);
        g->plan_valid = false;
    }
    if (g->row_done) cudaEventDestroy(g->row_done);
    if (g->fetch_done) cudaEventDestroy(g->fetch_done);
    for (cudaEvent_t e : g->h2d_done)
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : g->copy_done)
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : g->compute_done)
        if (e) cudaEventDestroy(e);
    if (g->notify_stream) cudaStreamDestroy(g->notify_stream);
    if (g->fetch_stream) cudaStreamDestroy(g->fetch_stream);
    if (g->compute_stream) cudaStreamDestroy(g->compute_stream);
    if (g->copy_stream) cudaStreamDestroy(g->copy_stream);
    for (DeviceBuf &b : g->dbuf) {
        cudaFree(b.d_int8);
        cudaFree(b.d_real);
        cudaFree(b.d_y);
        cudaFree(b.d_mask);
    }
    cudaFree(g->d_acc);
    cudaFree(g->d_peer);
    cudaFree(g->d_pairs);
    delete g;
}

/// Record an event on `s` so that a later wait on it does not block.
corr_status prime_event(cudaEvent_t e, cudaStream_t s) {
    return cuda_check(cudaEventRecord(e, s), "cudaEventRecord(prime)");
}

/// `dst += src`, elementwise over complex64 samples.
///
/// The row-sum the time-slice split needs when its partial rows are added
/// together. It exists as a kernel rather than as a host loop because the host
/// loop measured 10.6 GB/s against the 26 GB/s the transfer beside it ran at,
/// and because doing it here is what lets the whole fetch stay off the CPU.
__global__ void add_rows_kernel(float2 *dst, const float2 *src, size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i].x += src[i].x;
        dst[i].y += src[i].y;
    }
}

/// Queue `fn(user)` on the notify stream, to run once `ev` has completed.
///
/// The event is what is waited on, not the stream that recorded it: see the
/// comment on `GpuState::notify_stream`.
///
/// `fn` is handed to CUDA as the host function itself rather than through a
/// trampoline, because `corr_notify_fn` is already the `void (*)(void *)` that
/// `cudaLaunchHostFunc` takes -- on this platform its calling convention is the
/// default one, so the cast is a formality. That leaves nothing to allocate and
/// nothing to free on a thread that is not ours.
corr_status gpu_notify_on(GpuState *g, cudaEvent_t ev, corr_notify_fn fn, void *user) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_notify_on: null state");
    if (fn == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_notify_on: no callback");
    corr_status st = cuda_check(cudaStreamWaitEvent(g->notify_stream, ev, 0),
                                "cudaStreamWaitEvent(notify)");
    if (st != CORR_OK) return st;
    cudaError_t e = cudaLaunchHostFunc(g->notify_stream,
                                       reinterpret_cast<cudaHostFn_t>(fn), user);
    if (e != cudaSuccess)
        return gpu_fail(CORR_ERR_CUDA,
                        std::string("cudaLaunchHostFunc: ") + cudaGetErrorString(e));
    return CORR_OK;
}

} // namespace

// ---------------------------------------------------------------------------
// create / destroy
// ---------------------------------------------------------------------------

corr_status gpu_create(const corr_config &cfg, const GpuPlacement &pl,
                       const StagingLayout &layout, GpuState **out) {
    if (out == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_create: out is NULL");
    *out = nullptr;
    if (pl.device < 0) return gpu_fail(CORR_ERR_ARG, "gpu_create: no device in placement");
    if (pl.stations.empty())
        return gpu_fail(CORR_ERR_ARG, "gpu_create: placement holds no stations");
    if (pl.pairs.empty())
        return gpu_fail(CORR_ERR_ARG, "gpu_create: placement owns no baselines");

    GpuState *g = new (std::nothrow) GpuState();
    if (g == nullptr) return gpu_fail(CORR_ERR_NOMEM, "gpu_create: out of host memory");
    g->cfg = cfg;
    g->device = pl.device;
    g->fft_len = 2 * cfg.n_freq;
    g->bins = cfg.n_freq + 1;
    g->resident = pl.stations;
    g->n_station_local = pl.n_station_local();
    g->n_baseline = pl.n_baseline_local();
    g->first_baseline = pl.baselines.empty() ? 0 : pl.baselines.front();
    g->rows_per_fetch = cfg.rows_per_fetch;
    g->row_elems = (size_t)g->n_baseline * (size_t)cfg.n_freq;
    g->acc_bytes = g->row_elems * sizeof(float2) * (size_t)cfg.rows_per_fetch;
    g->d_pairs_bytes = (size_t)g->n_baseline * sizeof(int2);

    // A placement that holds every station in order needs no gather: the staging
    // rows are already contiguous and one two-dimensional copy moves the chunk.
    g->resident_is_identity = (g->n_station_local == cfg.n_station);
    for (int i = 0; i < g->n_station_local && g->resident_is_identity; ++i)
        if (g->resident[(size_t)i] != i) g->resident_is_identity = false;

    corr_status st = CORR_OK;

    if (g->n_baseline > kMaxBaselines) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "gpu_create: %d baselines on one device, but the correlation "
                      "kernel maps one thread per baseline and supports at most %d "
                      "(n_station <= 44; split the baselines over more devices)",
                      g->n_baseline, kMaxBaselines);
        free_all(g);
        return gpu_fail(CORR_ERR_ARG, buf);
    }
    // The baseline list is the authority, whether or not it is contiguous; check
    // it against the global numbering rather than trusting the caller's intent.
    for (int32_t b : pl.baselines) {
        if (b < 0 || b >= corr_n_baseline(cfg.n_station)) {
            free_all(g);
            return gpu_fail(CORR_ERR_ARG,
                            "gpu_create: a baseline index is outside the global numbering");
        }
    }
    // And the pairs must refer to stations this device actually holds, or the
    // correlation kernel would read another station's spectrum as its own.
    for (const int2 &p : pl.pairs) {
        if (p.x < 0 || p.y < p.x || p.y >= g->n_station_local) {
            free_all(g);
            return gpu_fail(CORR_ERR_ARG,
                            "gpu_create: a pair refers to a station this device does not hold");
        }
    }

    const size_t chunk_bytes = (size_t)cfg.chunk_frames * (size_t)g->fft_len;
    const size_t chunk_samples = (size_t)g->n_station_local * chunk_bytes;
    // One bit per frame of the slot per resident station. The staging's own mask
    // is per *station*, of which this device may hold a subset, but the bits are
    // copied per station either way, so the buffer is sized like the sample
    // buffers: one row per resident station.
    const size_t chunk_mask_bytes =
        (size_t)g->n_station_local * layout.validity_bytes_per_slot;

    if (layout.validity_bytes_per_slot == 0) {
        free_all(g);
        return gpu_fail(CORR_ERR_ARG,
                        "gpu_create: the staging layout has no validity mask; the "
                        "GPU would have to guess which frames are missing");
    }

    for (DeviceBuf &b : g->dbuf) {
        if ((st = cuda_check(cudaMalloc(reinterpret_cast<void **>(&b.d_int8),
                                        chunk_samples),
                             "cudaMalloc(int8)")) != CORR_OK ||
            (st = cuda_check(cudaMalloc(reinterpret_cast<void **>(&b.d_real),
                                        chunk_samples * sizeof(float)),
                             "cudaMalloc(real)")) != CORR_OK ||
            (st = cuda_check(cudaMalloc(reinterpret_cast<void **>(&b.d_y),
                                        (size_t)g->n_station_local * cfg.chunk_frames *
                                            (size_t)g->bins * sizeof(float2)),
                             "cudaMalloc(Y)")) != CORR_OK ||
            (st = cuda_check(cudaMalloc(reinterpret_cast<void **>(&b.d_mask),
                                        chunk_mask_bytes),
                             "cudaMalloc(mask)")) != CORR_OK) {
            free_all(g);
            return st;
        }
    }

    if ((st = cuda_check(cudaMalloc(reinterpret_cast<void **>(&g->d_acc), g->acc_bytes),
                         "cudaMalloc(acc)")) != CORR_OK ||
        (st = cuda_check(cudaMalloc(reinterpret_cast<void **>(&g->d_pairs),
                                    g->d_pairs_bytes),
                         "cudaMalloc(pairs)")) != CORR_OK) {
        free_all(g);
        return st;
    }

    // The baseline (j,k) table is built once here so the kernel never has to
    // invert the triangular index, and so this numbering cannot drift from
    // corr_baseline_pairs(). The indices are *local* to this device's resident
    // list, which is what the correlation kernel's station axis uses; the global
    // baseline numbering only reappears when the row slices are stitched back
    // together on the host.
    {
        if ((st = cuda_check(cudaMemcpy(g->d_pairs, pl.pairs.data(), g->d_pairs_bytes,
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy(pairs)")) != CORR_OK) {
            free_all(g);
            return st;
        }
    }

    if ((st = cuda_check(cudaMemset(g->d_acc, 0, g->acc_bytes), "cudaMemset(acc)")) !=
        CORR_OK) {
        free_all(g);
        return st;
    }
    for (DeviceBuf &b : g->dbuf) {
        // Keep the tail of a partial last chunk finite: the batched FFT always
        // transforms a full chunk, and stale device memory there must not be a
        // trap for denormals.
        if ((st = cuda_check(cudaMemset(b.d_real, 0, chunk_samples * sizeof(float)),
                             "cudaMemset(real)")) != CORR_OK) {
            free_all(g);
            return st;
        }
    }

    // Batched R2C: `batch` transforms of length fft_len, contiguous float32 in,
    // contiguous complex64 out with n_freq+1 bins each.
    {
        int n[1] = {g->fft_len};
        int inembed[1] = {g->fft_len};
        int onembed[1] = {g->bins};
        const int batch = g->n_station_local * cfg.chunk_frames;
        cufftResult r = cufftPlanMany(&g->plan, 1, n, inembed, 1, g->fft_len, onembed, 1,
                                      g->bins, CUFFT_R2C, batch);
        if (r != CUFFT_SUCCESS) {
            free_all(g);
            return cufft_check(r, "cufftPlanMany");
        }
        g->plan_valid = true;
    }

    if ((st = cuda_check(cudaStreamCreateWithFlags(&g->copy_stream, cudaStreamNonBlocking),
                         "cudaStreamCreate(copy)")) != CORR_OK ||
        (st = cuda_check(
                 cudaStreamCreateWithFlags(&g->compute_stream, cudaStreamNonBlocking),
                 "cudaStreamCreate(compute)")) != CORR_OK ||
        (st = cuda_check(cudaStreamCreateWithFlags(&g->fetch_stream, cudaStreamNonBlocking),
                         "cudaStreamCreate(fetch)")) != CORR_OK ||
        (st = cuda_check(cudaStreamCreateWithFlags(&g->notify_stream, cudaStreamNonBlocking),
                         "cudaStreamCreate(notify)")) != CORR_OK) {
        free_all(g);
        return st;
    }
    // cuFFT runs entirely on the compute stream; the same plan cannot be active
    // on two streams, which is why every FFT goes there.
    if ((st = cufft_check(cufftSetStream(g->plan, g->compute_stream), "cufftSetStream")) !=
        CORR_OK) {
        free_all(g);
        return st;
    }

    g->h2d_done.assign((size_t)cfg.block_slots, nullptr);
    g->copy_done.assign(kDeviceBuffers, nullptr);
    g->compute_done.assign(kDeviceBuffers, nullptr);
    for (int s = 0; s < cfg.block_slots; ++s)
        if ((st = cuda_check(
                 cudaEventCreateWithFlags(&g->h2d_done[(size_t)s], cudaEventDisableTiming),
                 "cudaEventCreate(h2d)")) != CORR_OK) {
            free_all(g);
            return st;
        }
    for (int i = 0; i < kDeviceBuffers; ++i) {
        if ((st = cuda_check(
                 cudaEventCreateWithFlags(&g->copy_done[(size_t)i], cudaEventDisableTiming),
                 "cudaEventCreate(copy)")) != CORR_OK ||
            (st = cuda_check(cudaEventCreateWithFlags(&g->compute_done[(size_t)i],
                                                      cudaEventDisableTiming),
                             "cudaEventCreate(compute)")) != CORR_OK) {
            free_all(g);
            return st;
        }
    }
    if ((st = cuda_check(cudaEventCreateWithFlags(&g->row_done, cudaEventDisableTiming),
                         "cudaEventCreate(row)")) != CORR_OK ||
        (st = cuda_check(cudaEventCreateWithFlags(&g->fetch_done, cudaEventDisableTiming),
                         "cudaEventCreate(fetch)")) != CORR_OK) {
        free_all(g);
        return st;
    }

    // Prime the "previous use finished" events so the first wait on each of them
    // does not block for ever.
    for (int i = 0; i < kDeviceBuffers; ++i)
        if ((st = prime_event(g->compute_done[(size_t)i], g->compute_stream)) != CORR_OK) {
            free_all(g);
            return st;
        }
    for (int s = 0; s < cfg.block_slots; ++s)
        if ((st = prime_event(g->h2d_done[(size_t)s], g->copy_stream)) != CORR_OK) {
            free_all(g);
            return st;
        }
    if ((st = prime_event(g->row_done, g->compute_stream)) != CORR_OK ||
        (st = prime_event(g->fetch_done, g->fetch_stream)) != CORR_OK) {
        free_all(g);
        return st;
    }

    *out = g;
    return CORR_OK;
}

corr_status gpu_destroy(GpuState *g) {
    if (g == nullptr) return CORR_OK;
    // Drain first: every allocation below may still be referenced by in-flight
    // work, and unmapping under a running transfer is a fault, not an error code.
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        const std::string msg =
            std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
        free_all(g);
        return gpu_fail(CORR_ERR_CUDA, "cudaDeviceSynchronize during destroy: " + msg);
    }
    free_all(g);
    return CORR_OK;
}

// ---------------------------------------------------------------------------
// accumulate
// ---------------------------------------------------------------------------

corr_status gpu_accumulate(GpuState *g, const StagingLayout &layout, int32_t slot,
                           int32_t frames) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_accumulate: null state");
    if (slot < 0 || slot >= g->cfg.block_slots)
        return gpu_fail(CORR_ERR_ARG, "gpu_accumulate: slot out of range");
    if (layout.base == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_accumulate: null staging");
    if (frames < 1 || frames > g->cfg.frames_per_slot)
        return gpu_fail(CORR_ERR_ARG,
                        "gpu_accumulate: frames must be within [1, frames_per_slot]; "
                        "the slot holds stale bytes past the caller's frame count");

    const corr_config &cfg = g->cfg;
    const int n_station = g->n_station_local;
    const int n_frames = frames;
    const int chunk_frames = cfg.chunk_frames;
    const int n_chunks = (n_frames + chunk_frames - 1) / chunk_frames;

    float2 *acc_row = g->d_acc + (size_t)g->cur_row * g->row_elems;
    corr_status st = CORR_OK;

    // Staging is [station][frame][sample] with the stations separated by
    // station_stride; one 2D copy per chunk puts each station's frames
    // contiguously on the device. Measured at 26.9 GB/s, the same as a plain
    // linear copy, so the strided descriptor costs nothing.
    const int8_t *slot_base = layout.base + (size_t)slot * layout.slot_stride;

    for (int c = 0; c < n_chunks; ++c) {
        const int d = (int)(g->chunk_seq++ % (uint64_t)kDeviceBuffers);
        DeviceBuf &b = g->dbuf[d];

        const int f0 = c * chunk_frames;
        const int ft = std::min(chunk_frames, n_frames - f0);
        const size_t w = (size_t)ft * (size_t)g->fft_len;
        const size_t elems = (size_t)n_station * w;

        // This device buffer may still be in use by the compute of an earlier
        // chunk; the copy must not start until that has finished reading it.
        if ((st = cuda_check(cudaStreamWaitEvent(g->copy_stream, g->compute_done[(size_t)d]),
                             "cudaStreamWaitEvent(compute_done)")) != CORR_OK)
            return st;

        g->bytes_h2d += (uint64_t)elems;
        ++g->transfers;

        if (g->resident_is_identity) {
            // The device holds every station, in order: one descriptor over all
            // of them, which is the pattern the 26.9 GB/s figure was measured on.
            if ((st = cuda_check(
                     cudaMemcpy2DAsync(b.d_int8, w,
                                       slot_base + (size_t)f0 * layout.frame_stride,
                                       layout.station_stride, w, (size_t)n_station,
                                       cudaMemcpyHostToDevice, g->copy_stream),
                     "cudaMemcpy2DAsync(H2D)")) != CORR_OK)
                return st;
        } else {
            // A subset of the stations, so the source rows are not adjacent: one
            // copy per resident station. Within a station the frames of a chunk
            // *are* adjacent, so each of these is a plain contiguous copy of
            // `w` bytes -- the two-dimensional descriptor above is only needed
            // when several stations are being gathered in one go. This is the
            // shape pair covering needs: a card holds a mix of stations, and it
            // pays one descriptor per resident station per chunk.
            for (int i = 0; i < n_station; ++i) {
                const int8_t *src =
                    layout.base +
                    (size_t)g->resident[(size_t)i] * layout.station_stride +
                    (size_t)slot * layout.slot_stride +
                    (size_t)f0 * layout.frame_stride;
                if ((st = cuda_check(
                         cudaMemcpyAsync(b.d_int8 + (size_t)i * w, src, w,
                                         cudaMemcpyHostToDevice, g->copy_stream),
                         "cudaMemcpyAsync(H2D, per station)")) != CORR_OK)
                    return st;
            }
        }

        // The validity mask rides with the samples, as a second small strided
        // copy on the same stream: one bit per frame of the slot per station, so
        // 32 bytes per station at frames_per_slot=256 -- 768 bytes for 24
        // stations against 6 MB of samples. It is issued before copy_done is
        // recorded, so every kernel that waits for this chunk's samples has
        // waited for this mask too.
        {
            const unsigned char *mask_base =
                reinterpret_cast<const unsigned char *>(layout.base) +
                layout.validity_offset +
                (size_t)slot * layout.validity_bytes_per_slot;
            const size_t mb = layout.validity_bytes_per_slot;
            g->bytes_h2d += (uint64_t)(mb * (size_t)n_station);
            if (g->resident_is_identity) {
                if ((st = cuda_check(
                         cudaMemcpy2DAsync(b.d_mask, mb, mask_base,
                                           layout.station_stride, mb,
                                           (size_t)n_station, cudaMemcpyHostToDevice,
                                           g->copy_stream),
                         "cudaMemcpy2DAsync(H2D, mask)")) != CORR_OK)
                    return st;
            } else {
                for (int i = 0; i < n_station; ++i) {
                    const unsigned char *src =
                        mask_base +
                        (size_t)g->resident[(size_t)i] * layout.station_stride;
                    if ((st = cuda_check(
                             cudaMemcpyAsync(b.d_mask + (size_t)i * mb, src, mb,
                                             cudaMemcpyHostToDevice, g->copy_stream),
                             "cudaMemcpyAsync(H2D, mask per station)")) != CORR_OK)
                        return st;
                }
            }
        }

        if ((st = cuda_check(cudaEventRecord(g->copy_done[(size_t)d], g->copy_stream),
                             "cudaEventRecord(copy_done)")) != CORR_OK)
            return st;

        // The kernels must wait for this chunk's copy, but nothing else: that is
        // what lets the next chunk's copy proceed in parallel.
        if ((st = cuda_check(
                 cudaStreamWaitEvent(g->compute_stream, g->copy_done[(size_t)d]),
                 "cudaStreamWaitEvent(copy_done)")) != CORR_OK)
            return st;

        // Silence for the frames that never arrived, written on the device before
        // the conversion reads them. This runs on the compute stream, so it costs
        // one launch when nothing is missing and a frame's worth of stores when
        // something is.
        {
            const dim3 grid(std::max(1, ceil_div((long long)g->fft_len / 16,
                                                  (long long)kZeroThreads)),
                            (unsigned)ft, (unsigned)n_station);
            zero_invalid_frames_kernel<<<grid, kZeroThreads, 0, g->compute_stream>>>(
                b.d_int8, b.d_mask, (int)layout.validity_bytes_per_slot, f0,
                g->fft_len, (long long)w);
            if ((st = cuda_check(cudaGetLastError(), "zero_invalid_frames_kernel")) != CORR_OK)
                return st;
        }

        {
            const int blocks = std::max(1, ceil_div((long long)elems, kConvertThreads));
            int8_to_float_kernel<<<blocks, kConvertThreads, 0, g->compute_stream>>>(
                b.d_int8, b.d_real, (long long)elems);
            if ((st = cuda_check(cudaGetLastError(), "int8_to_float_kernel")) != CORR_OK)
                return st;
        }

        // The plan's batch is a full chunk. A partial last chunk therefore
        // transforms a few frames of stale-but-finite data whose outputs the
        // correlation kernel never reads, because chunk_frames is passed as ft.
        if ((st = cufft_check(cufftExecR2C(g->plan, b.d_real,
                                           reinterpret_cast<cufftComplex *>(b.d_y)),
                              "cufftExecR2C")) != CORR_OK)
            return st;

        {
            const int n_blocks = ceil_div(cfg.n_freq, kChannelsPerBlock);
            const size_t smem = (size_t)n_station * kStagedFrames * kChannelsPerBlock *
                                sizeof(float2);
            corr_kernel<<<n_blocks, g->n_baseline, smem, g->compute_stream>>>(
                b.d_y, g->d_pairs, acc_row, n_station, ft, g->bins, cfg.n_freq,
                g->n_baseline);
            if ((st = cuda_check(cudaGetLastError(), "corr_kernel")) != CORR_OK) return st;
        }

        if ((st = cuda_check(
                 cudaEventRecord(g->compute_done[(size_t)d], g->compute_stream),
                 "cudaEventRecord(compute_done)")) != CORR_OK)
            return st;
    }

    // Recorded after the last chunk's copy, and the copy stream is ordered, so
    // waiting on it means the whole slot has left host memory.
    return cuda_check(cudaEventRecord(g->h2d_done[(size_t)slot], g->copy_stream),
                      "cudaEventRecord(h2d_done)");
}

corr_status gpu_slot_free(GpuState *g, int32_t slot) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_slot_free: null state");
    if (slot < 0 || slot >= g->cfg.block_slots)
        return gpu_fail(CORR_ERR_ARG, "gpu_slot_free: slot out of range");
    return cuda_check(cudaEventSynchronize(g->h2d_done[(size_t)slot]),
                      "cudaEventSynchronize(h2d_done)");
}

corr_status gpu_slot_free_submit(GpuState *g, int32_t slot, corr_notify_fn fn, void *user) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_slot_free_submit: null state");
    if (slot < 0 || slot >= g->cfg.block_slots)
        return gpu_fail(CORR_ERR_ARG, "gpu_slot_free_submit: slot out of range");
    return gpu_notify_on(g, g->h2d_done[(size_t)slot], fn, user);
}

// ---------------------------------------------------------------------------
// row lifecycle
// ---------------------------------------------------------------------------

corr_status gpu_row_end(GpuState *g, uint64_t frames) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_row_end: null state");
    if (frames == 0) return gpu_fail(CORR_ERR_ARG, "gpu_row_end: frames must be > 0");
    if (g->pending >= g->rows_per_fetch) {
        return gpu_fail(CORR_ERR_BUSY,
                        "gpu_row_end: all rows are full and unfetched; the caller has "
                        "stopped fetching and continuing would overwrite data");
    }

    float2 *acc_row = g->d_acc + (size_t)g->cur_row * g->row_elems;
    const long long n = (long long)g->row_elems;
    const int blocks = std::max(1, ceil_div(n, kScaleThreads));
    scale_row_kernel<<<blocks, kScaleThreads, 0, g->compute_stream>>>(
        acc_row, n, 1.0f / (float)frames);
    corr_status st = cuda_check(cudaGetLastError(), "scale_row_kernel");
    if (st != CORR_OK) return st;

    if ((st = cuda_check(cudaEventRecord(g->row_done, g->compute_stream),
                         "cudaEventRecord(row_done)")) != CORR_OK)
        return st;

    ++g->pending;
    ++g->cur_row;
    if (g->cur_row >= g->rows_per_fetch) g->cur_row = 0;
    return CORR_OK;
}

corr_status gpu_rows_wait(GpuState *g) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_rows_wait: null state");
    // The scale kernels all run on one ordered stream, so the last recorded
    // event covers every earlier row as well.
    return cuda_check(cudaEventSynchronize(g->row_done), "cudaEventSynchronize(row_done)");
}

corr_status gpu_rows_wait_submit(GpuState *g, corr_notify_fn fn, void *user) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_rows_wait_submit: null state");
    return gpu_notify_on(g, g->row_done, fn, user);
}

corr_status gpu_rows_ready(GpuState *g) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_rows_ready: null state");
    // `cudaEventQuery` is the non-blocking form of the wait above: it answers
    // "has it happened yet", and -- because a host callback has no way to report
    // a failure -- this is also the only place a failed scaling kernel can be
    // seen after the fact.
    cudaError_t e = cudaEventQuery(g->row_done);
    if (e == cudaSuccess) return CORR_OK;
    if (e == cudaErrorNotReady) return CORR_ERR_BUSY;
    return gpu_fail(CORR_ERR_CUDA, std::string("cudaEventQuery(row_done): ") +
                                       cudaGetErrorString(e));
}

corr_status gpu_rows_pending(GpuState *g, int32_t *out_pending) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_rows_pending: null state");
    if (out_pending == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_rows_pending: out is NULL");
    *out_pending = g->pending;
    return CORR_OK;
}

corr_status gpu_row_fetch(GpuState *g, void *out, size_t out_bytes, size_t dst_pitch,
                          int32_t *out_rows) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_row_fetch: null state");
    if (out == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_row_fetch: out is NULL");
    if (out_rows == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_row_fetch: out_rows is NULL");

    const size_t slice_bytes = g->row_elems * sizeof(float2);
    const size_t need = slice_bytes * (size_t)g->rows_per_fetch;
    if (out_bytes < need)
        return gpu_fail(CORR_ERR_ARG, "gpu_row_fetch: output buffer too small");

    if (g->pending == 0) {
        *out_rows = 0;
        return CORR_OK;
    }

    // Rows are always claimed from index 0 upward, so the pending rows are
    // contiguous at the front of the stack.
    const size_t bytes = slice_bytes * (size_t)g->pending;
    corr_status st = CORR_OK;
    if (dst_pitch == 0 || dst_pitch == slice_bytes) {
        st = cuda_check(
            cudaMemcpyAsync(out, g->d_acc, bytes, cudaMemcpyDeviceToHost, g->fetch_stream),
            "cudaMemcpyAsync(D2H)");
    } else {
        // Several devices write interleaved slices of one global row buffer, so
        // the destination rows are further apart than this device's slice.
        st = cuda_check(cudaMemcpy2DAsync(out, dst_pitch, g->d_acc, slice_bytes, slice_bytes,
                                          (size_t)g->pending, cudaMemcpyDeviceToHost,
                                          g->fetch_stream),
                        "cudaMemcpy2DAsync(D2H)");
    }
    if (st != CORR_OK) return st;
    g->bytes_d2h += (uint64_t)bytes;

    if ((st = cuda_check(cudaEventRecord(g->fetch_done, g->fetch_stream),
                         "cudaEventRecord(fetch_done)")) != CORR_OK)
        return st;

    // The caller expects the data to be in `out` when this returns.
    if ((st = cuda_check(cudaStreamSynchronize(g->fetch_stream),
                         "cudaStreamSynchronize(fetch)")) != CORR_OK)
        return st;

    // Clear only now: zeroing before the copy completed would hand back zeros.
    // The memset goes on the compute stream so that it is ordered ahead of the
    // kernels of the next row, and it waits on fetch_done so that it cannot
    // clobber the accumulator while the copy engine is still reading it.
    if ((st = cuda_check(cudaStreamWaitEvent(g->compute_stream, g->fetch_done),
                         "cudaStreamWaitEvent(fetch_done)")) != CORR_OK)
        return st;
    if ((st = cuda_check(
             cudaMemsetAsync(g->d_acc, 0, bytes, g->compute_stream),
             "cudaMemsetAsync(acc)")) != CORR_OK)
        return st;

    *out_rows = g->pending;
    g->pending = 0;
    g->cur_row = 0;
    return CORR_OK;
}

corr_status gpu_peer_row_reserve(GpuState *g, size_t bytes) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_peer_row_reserve: null state");
    if (bytes == 0) return gpu_fail(CORR_ERR_ARG, "gpu_peer_row_reserve: bytes must be > 0");
    if (g->d_peer != nullptr && g->d_peer_bytes >= bytes) return CORR_OK;
    // The allocation belongs to this device, and the caller may have made another
    // one current -- every stream and pointer below is this device's.
    corr_status st = cuda_check(cudaSetDevice(g->device), "cudaSetDevice(peer row)");
    if (st != CORR_OK) return st;
    if (g->d_peer != nullptr) {
        cudaFree(g->d_peer);
        g->d_peer = nullptr;
        g->d_peer_bytes = 0;
    }
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&g->d_peer), bytes);
    if (e != cudaSuccess)
        return gpu_fail(CORR_ERR_NOMEM,
                        std::string("cudaMalloc(peer row ") + std::to_string(bytes) +
                            ") failed: " + cudaGetErrorString(e));
    g->d_peer_bytes = bytes;
    return CORR_OK;
}

corr_status gpu_row_add_peer_row(GpuState *dst, const GpuState *src,
                                 const void *host_src, size_t bytes) {
    if (dst == nullptr || src == nullptr || host_src == nullptr)
        return gpu_fail(CORR_ERR_ARG, "gpu_row_add_peer_row: null argument");
    if (bytes == 0) return gpu_fail(CORR_ERR_ARG, "gpu_row_add_peer_row: bytes must be > 0");
    if (dst->d_peer == nullptr || dst->d_peer_bytes < bytes)
        return gpu_fail(CORR_ERR_INTERNAL, "gpu_row_add_peer_row: no room reserved");

    // Everything below -- the stream, the destination buffer, the kernel -- is
    // device `dst`'s, and the caller has just made `src`'s device current,
    // because that is whose row it fetched. A launch on another device's stream
    // is what `cudaErrorInvalidResourceHandle` means.
    corr_status st = cuda_check(cudaSetDevice(dst->device), "cudaSetDevice(peer add)");
    if (st != CORR_OK) return st;

    // The bytes are where they are because of `src`'s own fetch, which ran on
    // another device's stream. Waiting on its *event* rather than on its stream
    // is what makes that work across devices, and it is why those events are
    // created with `cudaEventDisableTiming` -- a cross-device wait requires it.
    st = cuda_check(cudaStreamWaitEvent(dst->fetch_stream, src->fetch_done, 0),
                    "cudaStreamWaitEvent(peer fetch_done)");
    if (st != CORR_OK) return st;
    if ((st = cuda_check(cudaMemcpyAsync(dst->d_peer, host_src, bytes,
                                         cudaMemcpyHostToDevice, dst->fetch_stream),
                         "cudaMemcpyAsync(peer row)")) != CORR_OK)
        return st;

    const size_t elems = bytes / sizeof(float2);
    const int threads = 256;
    int blocks = static_cast<int>((elems + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    add_rows_kernel<<<blocks, threads, 0, dst->fetch_stream>>>(dst->d_acc, dst->d_peer,
                                                               elems);
    return cuda_check(cudaGetLastError(), "add_rows_kernel launch");
}

corr_status gpu_row_fetch_scattered(GpuState *g, const int32_t *global_baselines,
                                     void *out, size_t row_stride, size_t baseline_bytes,
                                     void *scratch, size_t scratch_bytes,
                                     int32_t *out_rows) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_row_fetch_scattered: null state");
    if (global_baselines == nullptr || out == nullptr || out_rows == nullptr)
        return gpu_fail(CORR_ERR_ARG, "gpu_row_fetch_scattered: null argument");

    const size_t slice_bytes = g->row_elems * sizeof(float2);
    const size_t need = slice_bytes * (size_t)g->rows_per_fetch;
    if (scratch_bytes < need)
        return gpu_fail(CORR_ERR_ARG, "gpu_row_fetch_scattered: scratch too small");

    // Fetch the slice contiguously first: one transfer, then place each baseline.
    int32_t rows = 0;
    corr_status st = gpu_row_fetch(g, scratch, scratch_bytes, 0, &rows);
    if (st != CORR_OK) return st;
    if (rows == 0) {
        *out_rows = 0;
        return CORR_OK;
    }

    const char *src = static_cast<const char *>(scratch);
    char *dst = static_cast<char *>(out);
    for (int r = 0; r < rows; ++r) {
        const char *row_src = src + (size_t)r * slice_bytes;
        for (int b = 0; b < g->n_baseline; ++b) {
            const int32_t gb = global_baselines[b];
            if (gb < 0) return gpu_fail(CORR_ERR_INTERNAL,
                                        "gpu_row_fetch_scattered: negative baseline index");
            std::memcpy(dst + (size_t)gb * baseline_bytes + (size_t)r * row_stride,
                        row_src + (size_t)b * baseline_bytes, baseline_bytes);
        }
    }
    *out_rows = rows;
    return CORR_OK;
}

corr_status gpu_stats(GpuState *g, uint64_t *out_h2d, uint64_t *out_d2h,
                      uint64_t *out_transfers) {
    if (g == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_stats: null state");
    if (out_h2d != nullptr) *out_h2d = g->bytes_h2d;
    if (out_d2h != nullptr) *out_d2h = g->bytes_d2h;
    if (out_transfers != nullptr) *out_transfers = g->transfers;
    return CORR_OK;
}

corr_status gpu_out_bytes(const corr_config &cfg, size_t *out) {
    if (out == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_out_bytes: out is NULL");
    const int nbl = corr_n_baseline(cfg.n_station);
    if (nbl < 0) return gpu_fail(CORR_ERR_ARG, "gpu_out_bytes: bad n_station");
    if (cfg.rows_per_fetch < 1)
        return gpu_fail(CORR_ERR_ARG, "gpu_out_bytes: rows_per_fetch must be >= 1");
    *out = (size_t)nbl * (size_t)cfg.n_freq * sizeof(float2) *
           (size_t)cfg.rows_per_fetch;
    return CORR_OK;
}

corr_status gpu_device_bytes(const corr_config &cfg, size_t *out) {
    if (out == nullptr) return gpu_fail(CORR_ERR_ARG, "gpu_device_bytes: out is NULL");
    const size_t fft_len = (size_t)2 * cfg.n_freq;
    const size_t chunk = (size_t)cfg.n_station * cfg.chunk_frames * fft_len;
    const size_t per_buf = chunk + chunk * sizeof(float) +
                           (size_t)cfg.n_station * cfg.chunk_frames *
                               ((size_t)cfg.n_freq + 1) * sizeof(float2);
    const int nbl = cfg.n_station * (cfg.n_station + 1) / 2;
    const size_t acc = (size_t)nbl * cfg.n_freq * sizeof(float2) *
                       (size_t)std::max(1, cfg.rows_per_fetch);
    *out = per_buf * kDeviceBuffers + acc + (size_t)nbl * sizeof(int2);
    return CORR_OK;
}

} // namespace corr
