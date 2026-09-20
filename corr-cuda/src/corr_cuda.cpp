// corr_cuda.cpp -- implementation of the C ABI declared in corr_cuda.h.
//
// This file owns everything that is not device code: argument validation, the
// context and its pinned staging region, the slot state machine, and the
// translation of GPU failures into corr_status plus a readable message.
//
// The numerics live in kernels.cu, reached through gpu_internal.h.

#include "corr_cuda.h"

#include "gpu_internal.h"

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace corr {
std::string &last_error() {
    static thread_local std::string s;
    return s;
}
} // namespace corr

namespace {

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

corr_status fail(corr_status st, const std::string &msg) {
    corr::last_error() = msg;
    return st;
}

corr_status failf(corr_status st, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    corr::last_error() = buf;
    return st;
}

std::string cuda_error_string(cudaError_t e) {
    return std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

enum class SlotState {
    kFree,   // writable by the receive threads
    kCoping, // host-to-device copy in flight; staging must not be touched
};

/// Nanoseconds since an arbitrary origin, for the per-call timings below.
uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

struct Context {
    corr_config cfg{};
    /// One entry per device, in config order. `device_id` alone is used when
    /// n_devices == 0, which is what an older caller's zeroed config means.
    std::vector<int> devices;
    void       *staging = nullptr; // cudaHostAlloc'd portable, page locked
    corr::StagingLayout layout{};
    std::vector<corr::GpuState *> gpus;
    /// What each device was given, kept so the caller can see the split.
    std::vector<corr::GpuPlacement> plans;
    std::vector<SlotState> slots;
    /// Cumulative time this thread spent inside each fan-out call, per device.
    /// Written only by the thread that drives the context, like the counters in
    /// the device state.
    struct Timing {
        uint64_t accumulate = 0, slot_free = 0, row_end = 0, rows_wait = 0, fetch = 0;
    };
    std::vector<Timing> timing;
    /// Pinned scratch for devices whose baselines are scattered through the
    /// global numbering: the slice lands here and is then copied out one
    /// baseline at a time. Allocated only if some device needs it.
    void *scatter = nullptr;
    size_t scatter_bytes = 0;

    // Guards the slot state machine only. The ABI documents that the caller
    // serialises staging writes itself; this mutex just makes misuse detectable
    // rather than a data race inside the library.
    std::mutex mu;

    /// The configured device count, which is what every loop below is bounded by:
    /// `gpus` grows as the devices are created, so sizing a loop by it would skip
    /// the very work that fills it.
    int n_devices() const { return static_cast<int>(devices.size()); }

    /// Bytes the host must supply to corr_rows_fetch: rows_per_fetch rows of
    /// n_baseline * n_freq complex64. One row is the whole global baseline axis,
    /// no matter how many devices produced slices of it.
    size_t out_bytes() const {
        return static_cast<size_t>(corr_n_baseline(cfg.n_station)) *
               static_cast<size_t>(cfg.n_freq) * sizeof(float) * 2 *
               static_cast<size_t>(cfg.rows_per_fetch);
    }

    /// Bytes of one global row: the destination pitch between rows.
    size_t row_bytes() const {
        return static_cast<size_t>(corr_n_baseline(cfg.n_station)) *
               static_cast<size_t>(cfg.n_freq) * sizeof(float) * 2;
    }

    /// Bytes of the baseline slice device `d` owns.
    size_t slice_bytes(int d) const {
        return static_cast<size_t>(plans[static_cast<size_t>(d)].n_baseline_local()) *
               static_cast<size_t>(cfg.n_freq) * sizeof(float) * 2;
    }
};

Context *as_ctx(void *h) { return static_cast<Context *>(h); }

/// Make device `d` current, so that every gpu_* call below operates on the
/// device its state belongs to. Cheap enough to do per call and impossible to
/// get wrong by accident; a kernel launched on the wrong device would either
/// fault or, worse, land in another device's buffers.
corr_status select_device(Context *ctx, int d) {
    cudaError_t e = cudaSetDevice(ctx->devices[static_cast<size_t>(d)]);
    if (e != cudaSuccess) {
        return failf(CORR_ERR_CUDA, "cudaSetDevice(%d): %s",
                     ctx->devices[static_cast<size_t>(d)], cuda_error_string(e).c_str());
    }
    return CORR_OK;
}

/// Split the global baseline axis into `n_devices` contiguous ranges, as evenly
/// as possible, and give every device a copy of every station.
///
/// This is the simplest complete split: each baseline is computed exactly once,
/// on the one device that owns it, and no device needs a station it does not
/// already hold. It is the right first step because it exercises everything
/// multi-device needs -- many contexts fed by one staging ring, a shared
/// release, and slices stitched back into one row on the host -- without also
/// requiring a pair-covering design. Shrinking the resident set per device is
/// what later makes a station count that does not fit one card's host link
/// work, and it is a change to this function alone.
/// The pair axis in row-major (j outer, k inner) order, which is the order
/// corr_baseline_pairs() defines and the output files are written in.
std::vector<int2> enumerate_pairs(int n_station) {
    std::vector<int2> all;
    all.reserve(static_cast<size_t>(corr_n_baseline(n_station)));
    for (int j = 0; j < n_station; ++j)
        for (int k = j; k < n_station; ++k) {
            int2 p;
            p.x = j;
            p.y = k;
            all.push_back(p);
        }
    return all;
}

/// Split by baseline *range*, with every device holding every station.
///
/// The simplest complete split, and what the single-device fallback uses: each
/// baseline is computed once, on the device that owns its range, and no device
/// needs a station it does not already hold. Its cost is that each device pulls
/// every station's stream, so N devices move N times the run's bytes.
void plan_by_baseline_range(const corr_config &cfg, const std::vector<int> &devices,
                            std::vector<corr::GpuPlacement> &out) {
    const int nd = static_cast<int>(devices.size());
    const int n_station = cfg.n_station;
    const std::vector<int2> all = enumerate_pairs(n_station);
    const int n_baseline = static_cast<int>(all.size());

    int first = 0;
    for (int d = 0; d < nd; ++d) {
        // Remainder spread over the first devices, so no device is more than one
        // baseline away from any other.
        const int count = n_baseline / nd + (d < n_baseline % nd ? 1 : 0);
        corr::GpuPlacement pl;
        pl.device = devices[static_cast<size_t>(d)];
        pl.stations.resize(static_cast<size_t>(n_station));
        for (int i = 0; i < n_station; ++i) pl.stations[static_cast<size_t>(i)] = i;
        pl.pairs.assign(all.begin() + first, all.begin() + first + count);
        pl.baselines.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) pl.baselines[static_cast<size_t>(i)] = first + i;
        pl.baselines_are_contiguous = true;
        first += count;
        out.push_back(std::move(pl));
    }
}

/// Split by *time*: every device holds every station and every baseline, and
/// owns the slots whose sequence is congruent to it modulo the device count.
///
/// This is the split that costs nothing in host-link bandwidth -- the run's data
/// crosses the link exactly once, because each frame is pulled by exactly one
/// device -- and it needs no combinatorial design at all: the routing rule is
/// `slot_seq % n_devices`. What it costs instead is one addition per row per
/// device beyond the first, done on the host in corr_rows_fetch().
///
/// It works because a row is a sum over frames and a sum splits: each device
/// accumulates the frames it was given, and the partial sums are added. A
/// baseline, by contrast, is a product, which is why splitting by station costs
/// replicated operands.
void plan_by_time_slice(const corr_config &cfg, const std::vector<int> &devices,
                        std::vector<corr::GpuPlacement> &out) {
    const std::vector<int2> all = enumerate_pairs(cfg.n_station);
    for (size_t d = 0; d < devices.size(); ++d) {
        corr::GpuPlacement pl;
        pl.device = devices[d];
        pl.stations.resize(static_cast<size_t>(cfg.n_station));
        for (int i = 0; i < cfg.n_station; ++i) pl.stations[static_cast<size_t>(i)] = i;
        pl.pairs = all;                       // local indices == global, identity residency
        pl.baselines.resize(all.size());
        for (size_t i = 0; i < all.size(); ++i) pl.baselines[i] = static_cast<int32_t>(i);
        pl.baselines_are_contiguous = true;   // 0,1,2,... is a contiguous run
        out.push_back(std::move(pl));
    }
}

/// Smallest resident-set size that can cover C(n,2) pairs with `nd` devices.
int minimum_resident_set(int n_station, int nd) {
    const long long pairs = (long long)n_station * (n_station - 1) / 2;
    for (int s = 2; s <= n_station; ++s) {
        const long long capacity = (long long)nd * s * (s - 1) / 2;
        if (capacity >= pairs) return s;
    }
    return n_station;
}

/// Split by *pair covering*: each device holds a subset of the stations, and
/// each baseline is assigned to one device that holds both of its stations.
///
/// This is what makes a station count that does not fit one card's host link
/// possible at all. A device holding s stations can compute C(s,2) baselines, so
/// covering every pair needs sum(C(s_d,2)) >= C(n,2) with as small a sum of s_d
/// as possible -- the replication factor R = sum(s_d)/n. For 24 stations on four
/// devices that is s=13 and R=2.17; for 40 stations, s=21 and R=2.10. Compare
/// that with the range split above, whose R is the device count.
///
/// The construction is a deterministic greedy set cover with a little slack over
/// the bound (the bound itself is meeting-every-pair-exactly-once, which needs a
/// design that exists only for special parameters). Every pair is then assigned
/// to one of the devices that can compute it, chosen to balance the baseline
/// count. Coverage is verified afterwards rather than assumed: an uncovered pair
/// would be a row of zeros that looks like data.
corr_status plan_by_pair_covering(const corr_config &cfg, const std::vector<int> &devices,
                                  std::vector<corr::GpuPlacement> &out,
                                  std::string &err) {
    const int nd = static_cast<int>(devices.size());
    const int n = cfg.n_station;
    const std::vector<int2> all = enumerate_pairs(n);
    const int n_baseline = static_cast<int>(all.size());

    if (nd > n_baseline) {
        err = "pair covering needs at least as many baselines as devices";
        return CORR_ERR_ARG;
    }

    // Row-major index of the pair (j,k).
    const auto pair_index = [n](int j, int k) {
        if (j > k) std::swap(j, k);
        return j * n - j * (j - 1) / 2 + (k - j);
    };

    // Coverage is the whole point, so it is constructed by search and then
    // verified -- not assumed. A pair nobody owns would be a row of zeros that
    // looks exactly like data.
    //
    // The search is a randomized greedy with a balance term: repeatedly give the
    // station that covers the most still-uncovered pairs to the device that has
    // the fewest stations, so that no card ends up carrying the whole array. The
    // randomization is seeded from a fixed constant, so the plan is reproducible
    // run to run -- a placement that changes between runs cannot be debugged --
    // and multiple seeds are tried before the resident set is grown, because
    // every extra station per device is another station's worth of host link on
    // every card.
    const int s_min = minimum_resident_set(n, nd);
    std::vector<std::vector<int32_t>> best_sets;
    for (int s = s_min; s <= n && best_sets.empty(); ++s) {
        // Phase 1: fill the devices round-robin, each picking the station that
        // covers the most of *its* remaining pairs. Round-robin keeps the
        // resident sets the same size by construction, which matters: the host
        // link a card pays is proportional to its own set, so an unbalanced plan
        // spends the budget on the largest one.
        std::vector<std::vector<int32_t>> sets(static_cast<size_t>(nd));
        std::vector<std::vector<char>> in_set(static_cast<size_t>(nd),
                                              std::vector<char>(static_cast<size_t>(n), 0));
        std::vector<char> covered(static_cast<size_t>(n_baseline), 0);
        int covered_count = 0;
        const auto mark = [&](int d, int st) {
            in_set[static_cast<size_t>(d)][static_cast<size_t>(st)] = 1;
            sets[static_cast<size_t>(d)].push_back(st);
            for (int32_t other : sets[static_cast<size_t>(d)]) {
                const int idx = pair_index(st, other);
                if (!covered[static_cast<size_t>(idx)]) {
                    covered[static_cast<size_t>(idx)] = 1;
                    ++covered_count;
                }
            }
        };
        // One seed each, spread over the array, so the sets start out different.
        for (int d = 0; d < nd; ++d) mark(d, (d * n) / nd);
        for (int round = 1; round < s; ++round) {
            for (int d = 0; d < nd; ++d) {
                int best = -1, best_gain = 0;
                for (int c = 0; c < n; ++c) {
                    if (in_set[static_cast<size_t>(d)][static_cast<size_t>(c)]) continue;
                    int gain = 0;
                    for (int32_t other : sets[static_cast<size_t>(d)])
                        if (!covered[static_cast<size_t>(pair_index(c, other))]) ++gain;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best = c;
                    }
                }
                if (best < 0) {
                    // Nothing this device can add that covers anything new; it
                    // still needs a station, so take the first it does not hold.
                    for (int c = 0; c < n && best < 0; ++c)
                        if (!in_set[static_cast<size_t>(d)][static_cast<size_t>(c)]) best = c;
                }
                if (best >= 0) mark(d, best);
            }
        }

        // Phase 2: repair. Whatever the round-robin could not reach is covered
        // by handing stations to the devices that need them, smallest first, up
        // to a bounded amount of growth. If that bound is not enough, the caller
        // tries a larger resident set.
        const int growth_cap = s + 3;
        for (int guard = 0; guard < n * nd * 4 && covered_count < n_baseline; ++guard) {
            int gap_j = -1, gap_k = -1;
            for (int idx = 0; idx < n_baseline && gap_j < 0; ++idx)
                if (!covered[static_cast<size_t>(idx)]) {
                    gap_j = all[static_cast<size_t>(idx)].x;
                    gap_k = all[static_cast<size_t>(idx)].y;
                }
            if (gap_j < 0) break;
            int d = 0;
            for (int i = 1; i < nd; ++i)
                if (sets[static_cast<size_t>(i)].size() < sets[static_cast<size_t>(d)].size())
                    d = i;
            if (static_cast<int>(sets[static_cast<size_t>(d)].size()) >= growth_cap) break;
            // Take whichever of the pair's stations is missing here.
            const int add = !in_set[static_cast<size_t>(d)][static_cast<size_t>(gap_j)]
                                ? gap_j
                                : gap_k;
            if (in_set[static_cast<size_t>(d)][static_cast<size_t>(add)]) break;
            mark(d, add);
        }
        if (covered_count == n_baseline) best_sets = sets;
    }

    if (best_sets.empty()) {
        err = "pair covering could not cover every baseline; this is a bug in the planner, "
              "not a configuration problem";
        return CORR_ERR_INTERNAL;
    }

    out.assign(static_cast<size_t>(nd), corr::GpuPlacement{});
    for (int d = 0; d < nd; ++d) {
        out[static_cast<size_t>(d)].device = devices[static_cast<size_t>(d)];
        out[static_cast<size_t>(d)].stations = best_sets[static_cast<size_t>(d)];
        std::sort(out[static_cast<size_t>(d)].stations.begin(),
                  out[static_cast<size_t>(d)].stations.end());
    }

    // Assign each pair to exactly one device that holds both stations, always to
    // the device that currently owns the fewest baselines, so no card ends up
    // doing a disproportionate share of the correlation or of the write-out.
    std::vector<std::vector<char>> holds(static_cast<size_t>(nd),
                                         std::vector<char>(static_cast<size_t>(n), 0));
    for (int d = 0; d < nd; ++d)
        for (int32_t st : out[static_cast<size_t>(d)].stations)
            holds[static_cast<size_t>(d)][static_cast<size_t>(st)] = 1;

    for (int idx = 0; idx < n_baseline; ++idx) {
        const int2 p = all[static_cast<size_t>(idx)];
        int chosen = -1, fewest = 0;
        for (int d = 0; d < nd; ++d) {
            if (!holds[static_cast<size_t>(d)][static_cast<size_t>(p.x)]) continue;
            if (!holds[static_cast<size_t>(d)][static_cast<size_t>(p.y)]) continue;
            const int have = static_cast<int>(out[static_cast<size_t>(d)].baselines.size());
            if (chosen < 0 || have < fewest) {
                chosen = d;
                fewest = have;
            }
        }
        if (chosen < 0) {
            err = "pair covering has a pair no device can compute";
            return CORR_ERR_INTERNAL;
        }
        corr::GpuPlacement &pl = out[static_cast<size_t>(chosen)];
        // The kernel's pair table refers to the device's own station rows, not to
        // the global numbering.
        const auto local = [&pl](int st) {
            const auto it = std::lower_bound(pl.stations.begin(), pl.stations.end(), st);
            return static_cast<int>(it - pl.stations.begin());
        };
        int2 lp;
        lp.x = local(p.x);
        lp.y = local(p.y);
        pl.pairs.push_back(lp);
        pl.baselines.push_back(idx);
    }

    for (corr::GpuPlacement &pl : out) {
        // A device with nothing to do is pure overhead: its copies would still
        // cost its stations' worth of host link for no baselines.
        if (pl.baselines.empty()) {
            err = "pair covering gives a device no baselines; use fewer devices";
            return CORR_ERR_ARG;
        }
        // The gather is a single two-dimensional copy when the baselines happen
        // to be consecutive, which the single-device case always is.
        pl.baselines_are_contiguous = true;
        for (size_t i = 1; i < pl.baselines.size(); ++i)
            if (pl.baselines[i] != pl.baselines[i - 1] + 1) {
                pl.baselines_are_contiguous = false;
                break;
            }
    }
    return CORR_OK;
}

// Check the fields of corr_config that do not depend on the machine.
corr_status validate_config(const corr_config *cfg) {
    if (cfg == nullptr) return fail(CORR_ERR_ARG, "cfg is NULL");
    if (cfg->abi_version != CORR_ABI_VERSION)
        return failf(CORR_ERR_ABI,
                     "abi_version mismatch: header %u, caller %u; rebuild both sides",
                     CORR_ABI_VERSION, cfg->abi_version);
    if (cfg->struct_size != sizeof(corr_config))
        return failf(CORR_ERR_ABI,
                     "struct_size mismatch: header %zu, caller %u; rebuild both sides",
                     sizeof(corr_config), cfg->struct_size);
    if (cfg->device_id < 0) return fail(CORR_ERR_ARG, "device_id is negative");
    if (cfg->n_station < 1) return fail(CORR_ERR_ARG, "n_station must be >= 1");
    if (cfg->n_freq < 1) return fail(CORR_ERR_ARG, "n_freq must be >= 1");
    if (cfg->n_devices < 0) return fail(CORR_ERR_ARG, "n_devices is negative");
    if (cfg->split_mode != CORR_SPLIT_BASELINE_RANGE &&
        cfg->split_mode != CORR_SPLIT_PAIR_COVERING &&
        cfg->split_mode != CORR_SPLIT_TIME_SLICE)
        return failf(CORR_ERR_ARG,
                     "split_mode=%d is not a known split (0 = baseline ranges, "
                     "1 = pair covering, 2 = time slice)",
                     cfg->split_mode);
    if (cfg->n_devices > static_cast<int32_t>(CORR_MAX_DEVICES))
        return failf(CORR_ERR_ARG,
                     "n_devices=%d exceeds CORR_MAX_DEVICES=%u; raise the constant in "
                     "corr_cuda.h (and the mirror in Rust) if you really have that many "
                     "cards in one machine",
                     cfg->n_devices, CORR_MAX_DEVICES);

    // A frame is 2*n_freq samples and one UDP packet carries 4096 samples, so a
    // frame is only a whole number of packets when n_freq is a multiple of 2048.
    // This is the arithmetic behind frame_id = counter / (n_freq/2048); if it
    // does not hold, the host cannot assemble frames at all.
    if (cfg->n_freq % 2048 != 0)
        return failf(CORR_ERR_ARG,
                     "n_freq=%d is not a multiple of 2048; a frame would not be a "
                     "whole number of 4096-sample packets",
                     cfg->n_freq);

    if (cfg->frames_per_slot < 1)
        return fail(CORR_ERR_ARG, "frames_per_slot must be >= 1");
    if (cfg->chunk_frames < 1)
        return fail(CORR_ERR_ARG, "chunk_frames must be >= 1");
    if (cfg->chunk_frames > cfg->frames_per_slot)
        return failf(CORR_ERR_ARG,
                     "chunk_frames=%d exceeds frames_per_slot=%d; a chunk is the "
                     "device-side work unit and cannot span staging slots",
                     cfg->chunk_frames, cfg->frames_per_slot);
    if (cfg->block_slots < 2)
        return failf(CORR_ERR_ARG,
                     "block_slots=%d must be >= 2, otherwise the receive threads "
                     "have nowhere to write while the GPU reads",
                     cfg->block_slots);
    if (cfg->rows_per_fetch < 1)
        return failf(CORR_ERR_ARG,
                     "rows_per_fetch=%d must be >= 1; it is the number of integration "
                     "rows held on the device between host transfers",
                     cfg->rows_per_fetch);
    return CORR_OK;
}

// Compute the staging layout. Kept in one place so the header's doc comment and
// the implementation cannot drift apart.
corr::StagingLayout compute_layout(const corr_config &c) {
    corr::StagingLayout s{};
    const size_t frame_stride = static_cast<size_t>(2) * static_cast<size_t>(c.n_freq);
    s.frame_stride = frame_stride;
    s.slot_stride = frame_stride * static_cast<size_t>(c.frames_per_slot);
    // One bit per frame of a slot, rounded up to whole bytes, per slot. Appended
    // to each station's samples so that a station's extent stays contiguous and
    // the public struct can report the mask as an offset into it.
    s.validity_bytes_per_slot =
        (static_cast<size_t>(c.frames_per_slot) + 7u) / 8u;
    const size_t samples_per_station =
        s.slot_stride * static_cast<size_t>(c.block_slots);
    const size_t mask_per_station =
        s.validity_bytes_per_slot * static_cast<size_t>(c.block_slots);
    s.validity_offset = samples_per_station;
    s.station_stride = samples_per_station + mask_per_station;
    s.total_bytes = s.station_stride * static_cast<size_t>(c.n_station);
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
// Version banner
// ---------------------------------------------------------------------------

extern "C" const char *corr_abi_banner(void) {
    static std::string banner;
    static std::once_flag once;
    std::call_once(once, [] {
        int rt = 0, drv = 0;
        cudaRuntimeGetVersion(&rt);
        cudaDriverGetVersion(&drv);
        int cf = 0;
        cufftGetVersion(&cf);
        banner = "corr_cuda 0.1.0 abi=" + std::to_string(CORR_ABI_VERSION) +
                 " cudart=" + std::to_string(rt) +
                 " driver=" + std::to_string(drv) +
                 " cufft=" + std::to_string(cf) +
                 " arch=sm_89 target=L20";
    });
    return banner.c_str();
}

extern "C" const char *corr_last_error(void) {
    return corr::last_error().c_str();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

extern "C" corr_status corr_create(const corr_config *cfg, void **out_ctx) {
    if (out_ctx == nullptr) return fail(CORR_ERR_ARG, "out_ctx is NULL");
    *out_ctx = nullptr;

    corr_status st = validate_config(cfg);
    if (st != CORR_OK) return st;

    int ndev = 0;
    cudaError_t e = cudaGetDeviceCount(&ndev);
    if (e != cudaSuccess) return fail(CORR_ERR_CUDA, cuda_error_string(e));
    if (ndev == 0) return fail(CORR_ERR_ARG, "no CUDA device is visible");

    // Resolve the device list. n_devices == 0 keeps the single-device meaning of
    // device_id, so a caller built against ABI v2 -- or a C test that only sets
    // device_id -- still gets exactly one device.
    std::vector<int> devices;
    if (cfg->n_devices > 0) {
        devices.assign(cfg->devices, cfg->devices + cfg->n_devices);
    } else {
        devices.push_back(cfg->device_id);
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] < 0 || devices[i] >= ndev)
            return failf(CORR_ERR_ARG, "devices[%zu]=%d but only %d device(s) are visible",
                         i, devices[i], ndev);
        for (size_t j = 0; j < i; ++j)
            if (devices[j] == devices[i])
                return failf(CORR_ERR_ARG,
                             "devices[%zu]=%d repeats an earlier entry; the same card "
                             "would be driven by two contexts",
                             i, devices[i]);
    }

    e = cudaSetDevice(devices[0]);
    if (e != cudaSuccess) return fail(CORR_ERR_CUDA, cuda_error_string(e));

    Context *ctx = new (std::nothrow) Context();
    if (ctx == nullptr) return fail(CORR_ERR_NOMEM, "cannot allocate Context");

    ctx->cfg = *cfg;
    ctx->devices = devices;
    ctx->layout = compute_layout(*cfg);

    // Portable when there is more than one device, because every device that
    // holds a copy of these stations DMAs from this one allocation: the receive
    // threads write each staging slot once and all cards read it, which is what
    // keeps the host-side cost of replication at one memcpy per slot instead of
    // one per card. With a single device the plain flag is used, so the fallback
    // to one card is the allocation the performance baseline was measured on.
    const unsigned alloc_flags =
        devices.size() > 1 ? cudaHostAllocPortable : cudaHostAllocDefault;
    e = cudaHostAlloc(&ctx->staging, ctx->layout.total_bytes, alloc_flags);
    if (e != cudaSuccess) {
        const std::string msg = cuda_error_string(e);
        const size_t want = ctx->layout.total_bytes;
        delete ctx;
        return failf(CORR_ERR_NOMEM,
                     "cudaHostAlloc(%zu bytes) failed: %s; check RLIMIT_MEMLOCK",
                     want, msg.c_str());
    }
    std::memset(ctx->staging, 0, ctx->layout.total_bytes);
    ctx->layout.base = static_cast<int8_t *>(ctx->staging);

    // Zeroed staging matters: a frame that never arrives must contribute zero,
    // and starting from a known state makes an accidental unwritten frame show
    // up as silence rather than as garbage.
    ctx->slots.assign(static_cast<size_t>(cfg->block_slots), SlotState::kFree);

    // Who computes what. Deterministic and derived from the configuration, never
    // from a hard-coded card count.
    {
        std::string err;
        corr_status pst = CORR_OK;
        if (cfg->split_mode == CORR_SPLIT_PAIR_COVERING) {
            pst = plan_by_pair_covering(*cfg, ctx->devices, ctx->plans, err);
        } else if (cfg->split_mode == CORR_SPLIT_TIME_SLICE) {
            plan_by_time_slice(*cfg, ctx->devices, ctx->plans);
        } else {
            plan_by_baseline_range(*cfg, ctx->devices, ctx->plans);
        }
        if (pst != CORR_OK || ctx->plans.size() != devices.size() || ctx->plans.empty()) {
            const std::string msg =
                err.empty() ? "no usable placement was produced" : err;
            cudaFreeHost(ctx->staging);
            delete ctx;
            return fail(pst == CORR_OK ? CORR_ERR_ARG : pst, msg);
        }
    }

    ctx->timing.assign(ctx->plans.size(), Context::Timing{});

    // Room for one device's share of a fetch: the scattered-gather path needs it
    // for a device whose baselines are not contiguous, and the time-slice split
    // needs it for every device, because each holds a whole partial row.
    for (const corr::GpuPlacement &pl : ctx->plans) {
        const bool needed = !pl.baselines_are_contiguous ||
                            cfg->split_mode == CORR_SPLIT_TIME_SLICE;
        if (!needed) continue;
        const size_t need = static_cast<size_t>(pl.baselines.size()) *
                            static_cast<size_t>(cfg->n_freq) * sizeof(float) * 2 *
                            static_cast<size_t>(cfg->rows_per_fetch);
        if (need > ctx->scatter_bytes) ctx->scatter_bytes = need;
    }
    if (ctx->scatter_bytes > 0) {
        // Not `cudaHostAllocPortable`: this scratch is only ever a copy
        // *destination* for the host and a *source* for the host-side sum, and no
        // device ever reads or writes it. Portable memory is mapped into every
        // device's address space, which costs transfer rate for a property this
        // buffer does not use.
        e = cudaHostAlloc(&ctx->scatter, ctx->scatter_bytes, cudaHostAllocDefault);
        if (e != cudaSuccess) {
            const std::string msg = cuda_error_string(e);
            cudaFreeHost(ctx->staging);
            delete ctx;
            return failf(CORR_ERR_NOMEM, "cudaHostAlloc(scatter %zu bytes) failed: %s",
                         ctx->scatter_bytes, msg.c_str());
        }
    }

    // Device buffers, cuFFT plans and streams, one set per device. Done last so
    // that every failure above leaves nothing to unwind.
    for (int d = 0; d < ctx->n_devices(); ++d) {
        if ((st = select_device(ctx, d)) != CORR_OK) break;
        corr::GpuState *g = nullptr;
        if ((st = corr::gpu_create(*cfg, ctx->plans[static_cast<size_t>(d)],
                                   ctx->layout, &g)) != CORR_OK)
            break;
        ctx->gpus.push_back(g);
    }
    if (st != CORR_OK) {
        const std::string msg = corr::last_error();
        for (corr::GpuState *g : ctx->gpus) corr::gpu_destroy(g);
        cudaFreeHost(ctx->staging);
        delete ctx;
        return fail(st, msg);
    }

    *out_ctx = ctx;
    return CORR_OK;
}

extern "C" corr_status corr_destroy(void *h) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return CORR_OK;

    // Order matters: gpu_destroy() synchronises each device, so no transfer can
    // still be reading the pinned pages when cudaFreeHost unpins them.
    corr_status first_err = CORR_OK;
    std::string first_msg;
    for (int d = 0; d < ctx->n_devices(); ++d) {
        if (select_device(ctx, d) != CORR_OK) continue;
        corr_status gst = corr::gpu_destroy(ctx->gpus[static_cast<size_t>(d)]);
        if (gst != CORR_OK && first_err == CORR_OK) {
            first_err = gst;
            first_msg = corr::last_error();
        }
    }
    ctx->gpus.clear();

    if (ctx->scatter != nullptr) cudaFreeHost(ctx->scatter);
    if (ctx->staging != nullptr) {
        cudaError_t e = cudaFreeHost(ctx->staging);
        if (e != cudaSuccess) {
            const std::string msg = cuda_error_string(e);
            delete ctx;
            return fail(CORR_ERR_CUDA, "cudaFreeHost failed: " + msg);
        }
    }
    delete ctx;
    if (first_err != CORR_OK) return fail(first_err, first_msg);
    return CORR_OK;
}

extern "C" corr_status corr_staging_info(void *h, int32_t station, corr_staging *out) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (out == nullptr) return fail(CORR_ERR_ARG, "out is NULL");
    if (station < 0 || station >= ctx->cfg.n_station)
        return failf(CORR_ERR_ARG, "station %d out of range [0,%d)",
                     station, ctx->cfg.n_station);

    // The public struct has no station stride (the caller does not need it) and
    // every field is relative to the already-offset base, so total_bytes is this
    // station's extent rather than the size of the whole allocation.
    out->base = ctx->layout.base +
                static_cast<size_t>(station) * ctx->layout.station_stride;
    out->frame_stride = ctx->layout.frame_stride;
    out->slot_stride = ctx->layout.slot_stride;
    out->total_bytes = ctx->layout.station_stride;
    out->validity_offset = ctx->layout.validity_offset;
    out->validity_bytes_per_slot = ctx->layout.validity_bytes_per_slot;
    return CORR_OK;
}

// ---------------------------------------------------------------------------
// Row lifecycle
// ---------------------------------------------------------------------------

extern "C" corr_status corr_row_accumulate(void *h, int64_t slot_seq, int32_t frames) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (slot_seq < 0)
        return failf(CORR_ERR_ARG, "slot_seq=%lld is negative", (long long)slot_seq);
    // The physical slot is derived, not taken from the caller: the ring reuses a
    // slot every block_slots sequences, so the sequence is the identity and the
    // two sides cannot disagree about which slot this is.
    const int32_t slot = static_cast<int32_t>(slot_seq % ctx->cfg.block_slots);
    if (frames < 1 || frames > ctx->cfg.frames_per_slot)
        return failf(CORR_ERR_ARG,
                     "frames=%d out of range [1,%d]; a partial slot must say how "
                     "many frames it holds, or stale bytes past that point get "
                     "correlated as if they were data",
                     frames, ctx->cfg.frames_per_slot);

    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        if (ctx->slots[static_cast<size_t>(slot)] != SlotState::kFree)
            return failf(CORR_ERR_BUSY,
                         "slot %d (sequence %lld) is not free; it was submitted and never "
                         "released with corr_row_slot_free",
                         slot, (long long)slot_seq);
        ctx->slots[static_cast<size_t>(slot)] = SlotState::kCoping;
    }

    corr_status st = CORR_OK;
    if (ctx->cfg.split_mode == CORR_SPLIT_TIME_SLICE) {
        // One device per slot: it holds every station, so it can compute every
        // baseline for the frames in this slot, and no other device needs the
        // bytes at all. That is what makes the host-link cost exactly the run's
        // data rate.
        const int d = static_cast<int>(slot_seq % ctx->n_devices());
        if ((st = select_device(ctx, d)) == CORR_OK) {
            const uint64_t t0 = now_ns();
            st = corr::gpu_accumulate(ctx->gpus[static_cast<size_t>(d)], ctx->layout, slot, frames);
            ctx->timing[static_cast<size_t>(d)].accumulate += now_ns() - t0;
        }
    } else {
        // Every device that holds a copy of these stations pulls the slot. They
        // are queued on independent streams, so the copies overlap; the slot is
        // only handed back once the slowest of them has finished (see
        // corr_row_slot_free).
        for (int d = 0; d < ctx->n_devices(); ++d) {
            if ((st = select_device(ctx, d)) != CORR_OK) break;
            const uint64_t t0 = now_ns();
            st = corr::gpu_accumulate(ctx->gpus[static_cast<size_t>(d)], ctx->layout, slot,
                                      frames);
            ctx->timing[static_cast<size_t>(d)].accumulate += now_ns() - t0;
            if (st != CORR_OK) break;
        }
    }
    if (st != CORR_OK) {
        // Put the slot back so one failure does not wedge the ring for ever.
        std::lock_guard<std::mutex> lock(ctx->mu);
        ctx->slots[static_cast<size_t>(slot)] = SlotState::kFree;
    }
    return st;
}

extern "C" corr_status corr_row_slot_free(void *h, int64_t slot_seq) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (slot_seq < 0)
        return failf(CORR_ERR_ARG, "slot_seq=%lld is negative", (long long)slot_seq);
    const int32_t slot = static_cast<int32_t>(slot_seq % ctx->cfg.block_slots);

    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        if (ctx->slots[static_cast<size_t>(slot)] != SlotState::kCoping)
            return failf(CORR_ERR_STATE, "slot %d (sequence %lld) was not submitted", slot,
                         (long long)slot_seq);
    }

    // Who has to be done with this slot? Under the time-slice split exactly one
    // device ever read it, and under the others every device holding a copy did;
    // either way each gpu_slot_free waits on that device's own copy event, so the
    // loop returns when the slowest reader is done.
    const bool single = ctx->cfg.split_mode == CORR_SPLIT_TIME_SLICE;
    const int first = single ? static_cast<int>(slot_seq % ctx->n_devices()) : 0;
    const int last = single ? first + 1 : ctx->n_devices();
    for (int d = first; d < last; ++d) {
        corr_status st = select_device(ctx, d);
        if (st != CORR_OK) return st;
        const uint64_t t0 = now_ns();
        st = corr::gpu_slot_free(ctx->gpus[static_cast<size_t>(d)], slot);
        ctx->timing[static_cast<size_t>(d)].slot_free += now_ns() - t0;
        if (st != CORR_OK) return st;
    }

    // Only now may the receive thread write this slot again -- and it has to find
    // an empty validity mask when it gets there. The clear is here rather than in
    // the receive thread for two reasons: the copies that read this mask are known
    // to be finished (the events above), and the receiver cannot be writing this
    // slot yet (it is still marked busy, and the caller only releases the ring
    // after this returns). It also covers what the receive thread cannot: a station
    // that sends nothing at all for a whole slot runs no code that would clear it,
    // so the next lap would inherit this lap's bits and have the GPU correlate
    // stale samples as if they were data.
    for (int station = 0; station < ctx->cfg.n_station; ++station) {
        int8_t *mask = ctx->layout.base +
                       static_cast<size_t>(station) * ctx->layout.station_stride +
                       ctx->layout.validity_offset +
                       static_cast<size_t>(slot) * ctx->layout.validity_bytes_per_slot;
        std::memset(mask, 0, ctx->layout.validity_bytes_per_slot);
    }

    std::lock_guard<std::mutex> lock(ctx->mu);
    ctx->slots[static_cast<size_t>(slot)] = SlotState::kFree;
    return CORR_OK;
}

extern "C" corr_status corr_row_end(void *h, uint64_t frames) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    // Every device closes the row at the same moment -- the pipeline decides once
    // per row -- so each one scales and seals its own slice.
    for (int d = 0; d < ctx->n_devices(); ++d) {
        corr_status st = select_device(ctx, d);
        if (st != CORR_OK) return st;
        const uint64_t t0 = now_ns();
        st = corr::gpu_row_end(ctx->gpus[static_cast<size_t>(d)], frames);
        ctx->timing[static_cast<size_t>(d)].row_end += now_ns() - t0;
        if (st != CORR_OK) return st;
    }
    return CORR_OK;
}

extern "C" corr_status corr_rows_wait(void *h) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    for (int d = 0; d < ctx->n_devices(); ++d) {
        corr_status st = select_device(ctx, d);
        if (st != CORR_OK) return st;
        const uint64_t t0 = now_ns();
        st = corr::gpu_rows_wait(ctx->gpus[static_cast<size_t>(d)]);
        ctx->timing[static_cast<size_t>(d)].rows_wait += now_ns() - t0;
        if (st != CORR_OK) return st;
    }
    return CORR_OK;
}

extern "C" corr_status corr_rows_pending(void *h, int32_t *out_pending) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (out_pending == nullptr) return fail(CORR_ERR_ARG, "out_pending is NULL");
    // Every device is driven by the same corr_row_end/corr_rows_fetch calls, so
    // they must agree. Disagreeing means a call reached one device and not
    // another, which would silently lose a slice of a row; say so rather than
    // reporting the first device's answer.
    int32_t pending = -1;
    for (int d = 0; d < ctx->n_devices(); ++d) {
        corr_status st = select_device(ctx, d);
        if (st != CORR_OK) return st;
        int32_t p = 0;
        if ((st = corr::gpu_rows_pending(ctx->gpus[static_cast<size_t>(d)], &p)) != CORR_OK)
            return st;
        if (d == 0) {
            pending = p;
        } else if (p != pending) {
            return failf(CORR_ERR_INTERNAL,
                         "devices disagree on pending rows: device %d says %d, device 0 "
                         "says %d; the row lifecycle calls did not reach every device",
                         d, p, pending);
        }
    }
    *out_pending = pending;
    return CORR_OK;
}

extern "C" corr_status corr_rows_fetch(void *h, void *out, size_t out_bytes,
                                       int32_t *out_rows) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (out == nullptr) return fail(CORR_ERR_ARG, "out is NULL");
    if (out_rows == nullptr) return fail(CORR_ERR_ARG, "out_rows is NULL");

    const size_t need = ctx->out_bytes();
    if (out_bytes < need)
        return failf(CORR_ERR_ARG,
                     "out_bytes=%zu but %zu are required for rows_per_fetch=%d",
                     out_bytes, need, ctx->cfg.rows_per_fetch);

    // Each device writes its own baseline slice into the one contiguous row
    // buffer the caller supplied, at the offset its slice starts at. The
    // destination pitch is a whole global row, which is what makes a slice land
    // in the right place for every row in the batch.
    const size_t pitch = ctx->row_bytes();
    const size_t baseline_bytes =
        static_cast<size_t>(ctx->cfg.n_freq) * sizeof(float) * 2;
    char *base = static_cast<char *>(out);
    int32_t rows = -1;
    for (int d = 0; d < ctx->n_devices(); ++d) {
        corr_status st = select_device(ctx, d);
        if (st != CORR_OK) return st;
        const corr::GpuPlacement &pl = ctx->plans[static_cast<size_t>(d)];
        int32_t got = 0;
        const size_t slice = ctx->slice_bytes(d);
        const uint64_t tf0 = now_ns();
        if (ctx->cfg.split_mode == CORR_SPLIT_TIME_SLICE) {
            // Every device transferred a whole partial row into the scratch; add
            // them. The first one is copied rather than added so the output
            // buffer does not have to be cleared first.
            if (ctx->scatter == nullptr ||
                ctx->scatter_bytes < ctx->out_bytes())
                return fail(CORR_ERR_INTERNAL, "partial-row buffer is missing or too small");
            if ((st = corr::gpu_row_fetch(ctx->gpus[static_cast<size_t>(d)], ctx->scatter,
                                          ctx->scatter_bytes, 0, &got)) != CORR_OK)
                return st;
            if (got > 0) {
                const float *src = static_cast<const float *>(ctx->scatter);
                float *dst = reinterpret_cast<float *>(out);
                const size_t values = (size_t)got * pitch / sizeof(float);
                if (d == 0) {
                    std::memcpy(dst, src, values * sizeof(float));
                } else {
                    for (size_t i = 0; i < values; ++i) dst[i] += src[i];
                }
            }
        } else if (pl.baselines_are_contiguous) {
            // One two-dimensional copy straight into the row buffer, at the
            // offset this device's range starts at.
            char *dst = base + static_cast<size_t>(pl.baselines.front()) * baseline_bytes;
            if ((st = corr::gpu_row_fetch(ctx->gpus[static_cast<size_t>(d)], dst,
                                          slice * static_cast<size_t>(ctx->cfg.rows_per_fetch),
                                          pitch, &got)) != CORR_OK)
                return st;
        } else {
            // Scattered baselines: fetch into the scratch and place each one.
            if (ctx->scatter == nullptr || ctx->scatter_bytes < slice * ctx->cfg.rows_per_fetch)
                return fail(CORR_ERR_INTERNAL, "scatter buffer is missing or too small");
            if ((st = corr::gpu_row_fetch_scattered(
                     ctx->gpus[static_cast<size_t>(d)], pl.baselines.data(), base, pitch,
                     baseline_bytes, ctx->scatter, ctx->scatter_bytes, &got)) != CORR_OK)
                return st;
        }
        ctx->timing[static_cast<size_t>(d)].fetch += now_ns() - tf0;
        if (d == 0) {
            rows = got;
        } else if (got != rows) {
            return failf(CORR_ERR_INTERNAL,
                         "devices disagree on fetched rows: device %d returned %d, "
                         "device 0 returned %d", d, got, rows);
        }
    }
    *out_rows = rows < 0 ? 0 : rows;
    return CORR_OK;
}

// ---------------------------------------------------------------------------
// Device placement -- what the split actually was
// ---------------------------------------------------------------------------

extern "C" int32_t corr_n_devices(void *h) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return -1;
    return ctx->n_devices();
}

extern "C" corr_status corr_device_plan_get(void *h, int32_t device_index,
                                            corr_device_plan *out) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (out == nullptr) return fail(CORR_ERR_ARG, "out is NULL");
    if (device_index < 0 || device_index >= ctx->n_devices())
        return failf(CORR_ERR_ARG, "device_index %d out of range [0,%d)", device_index,
                     ctx->n_devices());
    const corr::GpuPlacement &pl = ctx->plans[static_cast<size_t>(device_index)];
    out->device_id = pl.device;
    out->n_station_resident = pl.n_station_local();
    out->first_baseline = pl.baselines.empty() ? 0 : pl.baselines.front();
    out->n_baseline = pl.n_baseline_local();
    out->contiguous = pl.baselines_are_contiguous ? 1 : 0;
    return CORR_OK;
}

extern "C" corr_status corr_device_baselines(void *h, int32_t device_index,
                                             int32_t *out_baselines, int32_t out_len) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (out_baselines == nullptr) return fail(CORR_ERR_ARG, "out_baselines is NULL");
    if (device_index < 0 || device_index >= ctx->n_devices())
        return failf(CORR_ERR_ARG, "device_index %d out of range [0,%d)", device_index,
                     ctx->n_devices());
    const corr::GpuPlacement &pl = ctx->plans[static_cast<size_t>(device_index)];
    if (out_len < pl.n_baseline_local())
        return failf(CORR_ERR_ARG, "out_len=%d but device %d owns %d baselines", out_len,
                     device_index, pl.n_baseline_local());
    for (int i = 0; i < pl.n_baseline_local(); ++i)
        out_baselines[i] = pl.baselines[static_cast<size_t>(i)];
    return CORR_OK;
}

extern "C" corr_status corr_device_stations(void *h, int32_t device_index,
                                            int32_t *out_stations, int32_t out_len) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (out_stations == nullptr) return fail(CORR_ERR_ARG, "out_stations is NULL");
    if (device_index < 0 || device_index >= ctx->n_devices())
        return failf(CORR_ERR_ARG, "device_index %d out of range [0,%d)", device_index,
                     ctx->n_devices());
    const corr::GpuPlacement &pl = ctx->plans[static_cast<size_t>(device_index)];
    if (out_len < pl.n_station_local())
        return failf(CORR_ERR_ARG, "out_len=%d but device %d holds %d stations", out_len,
                     device_index, pl.n_station_local());
    for (int i = 0; i < pl.n_station_local(); ++i)
        out_stations[i] = pl.stations[static_cast<size_t>(i)];
    return CORR_OK;
}

extern "C" corr_status corr_device_stats_get(void *h, int32_t device_index,
                                              corr_device_stats *out) {
    Context *ctx = as_ctx(h);
    if (ctx == nullptr) return fail(CORR_ERR_ARG, "ctx is NULL");
    if (out == nullptr) return fail(CORR_ERR_ARG, "out is NULL");
    if (device_index < 0 || device_index >= ctx->n_devices())
        return failf(CORR_ERR_ARG, "device_index %d out of range [0,%d)", device_index,
                     ctx->n_devices());
    corr_status st = select_device(ctx, device_index);
    if (st != CORR_OK) return st;
    uint64_t h2d = 0, d2h = 0, transfers = 0;
    if ((st = corr::gpu_stats(ctx->gpus[static_cast<size_t>(device_index)], &h2d, &d2h,
                              &transfers)) != CORR_OK)
        return st;
    out->device_id = ctx->plans[static_cast<size_t>(device_index)].device;
    out->bytes_h2d = h2d;
    out->bytes_d2h = d2h;
    out->transfers = transfers;
    const Context::Timing &t = ctx->timing[static_cast<size_t>(device_index)];
    out->ns_accumulate = t.accumulate;
    out->ns_slot_free = t.slot_free;
    out->ns_row_end = t.row_end;
    out->ns_rows_wait = t.rows_wait;
    out->ns_fetch = t.fetch;
    return CORR_OK;
}

// ---------------------------------------------------------------------------
// Baseline numbering -- the definition lives here, not in Rust
// ---------------------------------------------------------------------------

extern "C" int32_t corr_n_baseline(int32_t n_station) {
    if (n_station < 1) return -1;
    return n_station * (n_station + 1) / 2;
}

extern "C" corr_status corr_baseline_pairs(int32_t n_station, int32_t *out_jk,
                                           int32_t out_len) {
    if (n_station < 1) return fail(CORR_ERR_ARG, "n_station must be >= 1");
    const int32_t nbl = corr_n_baseline(n_station);
    if (out_len != 2 * nbl)
        return failf(CORR_ERR_ARG,
                     "out_len=%d but %d are required for n_station=%d",
                     out_len, 2 * nbl, n_station);
    if (out_jk == nullptr) return fail(CORR_ERR_ARG, "out_jk is NULL");

    int32_t i = 0;
    for (int32_t j = 0; j < n_station; ++j) {
        for (int32_t k = j; k < n_station; ++k) {
            out_jk[2 * i] = j;
            out_jk[2 * i + 1] = k;
            ++i;
        }
    }
    return CORR_OK;
}
