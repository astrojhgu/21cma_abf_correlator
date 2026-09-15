// gpu_internal.h -- interface between the C ABI layer (corr_cuda.cpp) and the
// CUDA implementation (kernels.cu).
//
// Kept separate from the public corr_cuda.h so that the ABI layer stays pure C++
// and readable, while the device code lives in a translation unit nvcc owns.

#ifndef CORR_GPU_INTERNAL_H
#define CORR_GPU_INTERNAL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <vector_types.h> // int2, for the pair tables

#include "corr_cuda.h"

namespace corr {

/// Thread-local error message owned by the C ABI layer and shared with the CUDA
/// translation unit, so that a failure inside a kernel launch reaches Rust
/// through corr_last_error() rather than being lost.
std::string &last_error();

/// Opaque GPU-side state: device buffers, cuFFT plan, streams, events.
struct GpuState;

/// One device's share of the run: which stations it holds a copy of, and which
/// baselines it owns.
///
/// `stations` is in *local* order, i.e. `stations[0]` is the row of the device's
/// int8 buffer that the first station's frames land in, and the pair table's
/// indices refer to positions in this vector rather than to global station
/// indices. Keeping the device's view local is what lets the kernels stay
/// unchanged when a device holds only a subset of the array.
struct GpuPlacement {
    int           device = -1;
    std::vector<int32_t> stations;      /// global station indices, local order
    /// Global indices of the baselines this device owns, in its own local order.
    /// A *list* rather than a range, because a device that holds only some of the
    /// stations owns the pairs among those, and those are scattered through the
    /// global row-major numbering rather than contiguous in it.
    std::vector<int32_t> baselines;
    std::vector<int2> pairs;            /// local (j,k) pairs, j <= k, same order
    /// True when `baselines` happens to be a contiguous ascending run starting at
    /// `baselines[0]`, which is what the single-device and full-residency splits
    /// produce and what lets the fetch stay one two-dimensional copy.
    bool baselines_are_contiguous = true;

    int n_station_local() const { return static_cast<int>(stations.size()); }
    int n_baseline_local() const { return static_cast<int>(pairs.size()); }
};

/// Internal view of the pinned staging ring.
///
/// The public corr_staging is per-station and hides the station stride, but the
/// 2D host-to-device copy needs it: the source rows are one station apart. So the
/// ABI layer keeps this and derives the public struct from it.
struct StagingLayout {
    int8_t *base = nullptr;        /// start of station 0
    size_t frame_stride = 0;       /// bytes per frame  = 2*n_freq
    size_t slot_stride = 0;        /// bytes per slot   = frames_per_slot*frame_stride
    /// Validity mask per the public ABI: one bit per frame of a slot, appended
    /// after a station's samples so the station's extent stays contiguous.
    size_t validity_offset = 0;      /// from a station's base, = samples only
    size_t validity_bytes_per_slot = 0;
    size_t station_stride = 0;     /// bytes per station = block_slots*slot_stride
    size_t total_bytes = 0;        /// n_station * station_stride
};

/// Allocate device memory, build the cuFFT plan, create the streams and events,
/// and zero the accumulator for one device's share of the work.
///
/// `cfg` still describes the *whole* run (n_station is the global count, which
/// is what the staging layout and the two-dimensional copies are built from);
/// `pl` says which slice of it this device owns. The caller must have made
/// `pl.device` current with cudaSetDevice().
corr_status gpu_create(const corr_config &cfg, const GpuPlacement &pl, GpuState **out);

/// Free everything. Safe on nullptr.
corr_status gpu_destroy(GpuState *g);

/// Queue the copy and correlation of one staging slot into the accumulator.
/// Asynchronous.
corr_status gpu_accumulate(GpuState *g, const StagingLayout &layout, int32_t slot,
                           int32_t frames);

/// Block until the host-to-device copy of `slot` has completed, which is when
/// the receive thread may write that slot again.
corr_status gpu_slot_free(GpuState *g, int32_t slot);

/// Divide the accumulator by `frames` on the compute stream and record the row.
corr_status gpu_row_end(GpuState *g, uint64_t frames);

/// Block until every closed row is complete.
corr_status gpu_rows_wait(GpuState *g);

/// How many closed rows are waiting to be fetched, in [0, rows_per_fetch].
corr_status gpu_rows_pending(GpuState *g, int32_t *out_pending);

/// Copy the pending rows out and clear them, in that order.
///
/// `dst_pitch` is the distance in bytes between consecutive rows *at the
/// destination*, which is not the same as this device's row size when several
/// devices write interleaved slices of one global row buffer. Pass 0 for a
/// contiguous destination (the single-device case).
corr_status gpu_row_fetch(GpuState *g, void *out, size_t out_bytes, size_t dst_pitch,
                          int32_t *out_rows);

/// Same, but for a device whose baselines are scattered through the global
/// numbering: the slice is fetched contiguously into `scratch` and then copied
/// out one baseline at a time to out[global_baseline * row_bytes_each]. The
/// scatter is a host memcpy of the row's worth of spectra -- 26.9 MB per row at
/// 40 stations, which at a one-second integration is 1% of the D2H rate.
corr_status gpu_row_fetch_scattered(GpuState *g, const int32_t *global_baselines,
                                    void *out, size_t row_stride, size_t baseline_bytes,
                                    void *scratch, size_t scratch_bytes, int32_t *out_rows);

/// Cumulative bytes this device has pulled from the host and handed back, plus
/// how many host-to-device transfers it took. Any of the pointers may be NULL.
corr_status gpu_stats(GpuState *g, uint64_t *out_h2d, uint64_t *out_d2h,
                      uint64_t *out_transfers);

/// Bytes of one fetch's worth of spectrum, i.e. rows_per_fetch rows.
corr_status gpu_out_bytes(const corr_config &cfg, size_t *out);

/// Bytes of device memory the given configuration will need, for logging.
corr_status gpu_device_bytes(const corr_config &cfg, size_t *out);

} // namespace corr

#endif // CORR_GPU_INTERNAL_H
