/*
 * corr_cuda.h -- C ABI of the GPU side of the real-time correlator.
 *
 * This header is the *only* contract between corr-rs (Rust) and corr-cuda
 * (CMake/CUDA). Rust links against libcorr_cuda.so and never sees a CUDA type,
 * a stream, or a device pointer.
 *
 * Division of labour
 * ------------------
 *   corr-rs   : multicast sockets, frame assembly, slot scheduling, output files
 *   corr-cuda : device accumulation buffer, int8 -> float32 conversion, cuFFT R2C,
 *               baseline accumulation, 1/frames scaling, streams and events
 *
 * The accumulation model
 * ----------------------
 * One output row (one integration) is produced by repeatedly handing the GPU
 * batches of frames. Inside the library there is a stack of
 * `rows_per_fetch * n_baseline * n_freq` complex values in device memory. Every
 * batch is correlated with `+=` into the row currently being filled. Closing a
 * row divides it by its frame count and moves to the next slot of the stack;
 * when the caller fetches, the whole stack is copied to the host in one transfer
 * and cleared.
 *
 *   create                      -> stack zeroed, row 0 active
 *   accumulate(slot) * K        -> acc[row] += correlation of that slot's frames
 *   end(frames)                 -> acc[row] /= frames; row advances
 *   ... repeated rows_per_fetch times ...
 *   rows_wait()                 -> every closed row is complete
 *   rows_fetch(out)             -> one D2H of the whole stack, then clear
 *
 * Why a stack rather than one accumulator: the integration length and the host
 * transfer cadence are then independent. A 10 ms integration that is fetched
 * every row would push n_baseline*n_freq*8 bytes across PCIe 100 times a second
 * (983 MB/s at 24 stations) and wake the writer 100 times a second. With
 * rows_per_fetch chosen so that the fetch happens about once a second, the
 * integration length can be anything from 1 ms to 10 s without changing the
 * host-side cost at all.
 *
 * Frame staging
 * -------------
 * The receive threads never call into this library. They write int8 frames into
 * a ring of page-locked staging slots reported by corr_staging_info(); the
 * scheduler hands a filled slot to the GPU with corr_row_accumulate() and must
 * not touch that slot again until corr_row_slot_free() says the copy is done.
 *
 * A slot holds `frames_per_slot` frames per station, so the pinned footprint is
 * independent of the integration length: a 1 second integration and a 10 ms
 * integration need the same staging.
 *
 * What a slot does *not* have to be is complete. Each slot carries a validity
 * mask (see corr_staging) saying which of its frames arrived whole, and the
 * missing ones are zeroed on the device. Zeroing there rather than on the receive
 * thread is what keeps a station that fell behind from having to write gigabytes
 * of silence through the same thread that needs to be reading its socket.
 *
 * Overlap
 * -------
 * The library owns two streams. Host-to-device copies run on a copy stream and
 * the conversion / FFT / correlation work runs on a compute stream, so the copy
 * of slot k+1 proceeds while the kernels chew on slot k. Measured on this
 * machine: a plain H2D reaches 27 GB/s while the serialised pipeline only moved
 * 12-16 GB/s, so this overlap is the difference between one GPU keeping up with
 * 24 stations and not.
 *
 * Thread safety
 * -------------
 * A context is not internally locked and is driven by one scheduler thread. The
 * receive threads only write to the pinned staging bytes; that is the point of
 * the design, and it is why a slot must stay untouched between its
 * corr_row_accumulate() and its corr_row_slot_free().
 */

#ifndef CORR_CUDA_H
#define CORR_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bump on any layout or semantic change of the structures below.
 *
 * v7: the validity mask became load bearing. From this version the GPU reads it
 *     and zeroes every frame whose bit is clear, and corr_row_slot_free() clears
 *     it as it hands the slot back. A v6 library linked against a v7 host would
 *     ignore the mask while the host had stopped zero filling, i.e. it would
 *     correlate whatever the previous lap left in the staging and report it as
 *     data -- silently. That is the one failure this version number exists to
 *     refuse. See docs/DESIGN-zero-fill-decoupling.md.
 */
#define CORR_ABI_VERSION 7u

/*
 * How the pair space is divided over the devices.
 *
 * CORR_SPLIT_BASELINE_RANGE: every device holds every station and owns a
 *   contiguous range of baselines. Simple, complete, and N times the run's bytes
 *   over the host links -- right for one device, and for several when the station
 *   count is small enough that N times its data still fits.
 *
 * CORR_SPLIT_PAIR_COVERING: each device holds a subset of the stations and owns
 *   every baseline it is assigned among those. Total host traffic is
 *   (sum of resident set sizes) / n_station times the run's bytes, which for 24
 *   stations over four devices is 2.17x instead of 4x, and for 40 stations 2.10x.
 *   That is what makes a station count too large for one card's host link fit at
 *   all. The price is that a device's baselines are scattered through the global
 *   numbering, so the row is assembled on the host per baseline rather than by
 *   one contiguous slice copy.
 */
#define CORR_SPLIT_BASELINE_RANGE 0
#define CORR_SPLIT_PAIR_COVERING 1
/*
 * CORR_SPLIT_TIME_SLICE: every device holds every station, but only for its
 * share of the *frames*, and computes every baseline for them.
 *
 * A baseline is a product, so splitting it needs both operands in one place; a
 * row is a sum over frames, and a sum splits for free. So instead of replicating
 * operands (which costs the host links R times the run's bytes), this splits the
 * summation axis: each device accumulates its own partial row over the frames it
 * was given, and the partials are added on the host when the row is fetched.
 *
 * The arithmetic: at N devices the aggregate host traffic is exactly the run's
 * data rate -- R = 1, so 40 stations need 32 GB/s instead of 67 (pair covering
 * at its bound) or 128 (baseline range) -- and each device pulls 1/N of it.
 * Every device holds all stations, so the fast two-dimensional copy per slot and
 * the plain contiguous row layout both survive; the only new work is the sum of
 * N partial rows per fetch, which at a one-second integration is 107 MB/s for
 * four devices and is batched to that same cadence by rows_per_fetch whatever
 * the integration length.
 *
 * Which device gets a slot is decided by the library from the slot sequence:
 * slot_seq % n_devices. The caller does not need to know.
 */
#define CORR_SPLIT_TIME_SLICE 2

/* How many CUDA devices one context may span. Four is the machine's card count;
   the limit is only about the fixed-size array in corr_config. */
#define CORR_MAX_DEVICES 8u

typedef enum corr_status {
    CORR_OK           = 0,
    CORR_ERR_ARG      = 1, /* bad or inconsistent argument                     */
    CORR_ERR_ABI      = 2, /* abi_version / struct_size mismatch               */
    CORR_ERR_CUDA     = 3, /* CUDA runtime or driver error                     */
    CORR_ERR_CUFFT    = 4, /* cuFFT error                                      */
    CORR_ERR_STATE    = 5, /* call out of order for this context or slot       */
    CORR_ERR_BUSY     = 6, /* slot still owned by the GPU                      */
    CORR_ERR_NOMEM    = 7, /* host or device allocation failed                 */
    CORR_ERR_NOT_IMPL = 8, /* deliberately unimplemented in this build         */
    CORR_ERR_INTERNAL = 9  /* should not happen                                */
} corr_status;

/*
 * Static description of one correlator instance.
 *
 * Invariants enforced at corr_create() (rejected with CORR_ERR_ARG):
 *   abi_version     == CORR_ABI_VERSION
 *   struct_size     == sizeof(corr_config)
 *   device_id       >= 0 and < number of visible CUDA devices
 *   n_station       >= 1, and small enough that n_station*(n_station+1)/2 <= 1024
 *   n_freq          >= 1 and a multiple of 2048, so a frame of 2*n_freq
 *                      samples is a whole number of 4096-sample UDP packets
 *   frames_per_slot >= 1
 *   chunk_frames    >= 1 and <= frames_per_slot
 *   block_slots     >= 2
 *   rows_per_fetch  >= 1
 */
typedef struct corr_config {
    uint32_t abi_version;    /* must be CORR_ABI_VERSION                      */
    uint32_t struct_size;    /* must be sizeof(corr_config)                   */
    int32_t  device_id;      /* CUDA device ordinal; used when n_devices == 0  */
    int32_t  n_station;      /* N; N*(N+1)/2 baselines including autocorrelation */
    int32_t  n_freq;         /* retained bins; FFT length is 2*n_freq         */
    int32_t  frames_per_slot;/* frames one staging slot holds, per station    */
    int32_t  chunk_frames;   /* frames per FFT + correlate launch             */
    int32_t  block_slots;    /* staging slots in the ring, >= 2               */
    int32_t  rows_per_fetch; /* integration rows held on the device before one
                                host transfer; decouples integration length
                                from transfer cadence                        */

    /* --- ABI v3: the context may span several devices --------------------- */
    int32_t  n_devices;      /* 1..CORR_MAX_DEVICES, or 0 to mean "just
                                device_id" (the v2 behaviour, which is what a
                                zero-initialised config from an older caller
                                gets)                                        */
    int32_t  devices[CORR_MAX_DEVICES]; /* CUDA ordinals; the first n_devices
                                entries are used, the rest are ignored. They
                                must be distinct: the same device twice would
                                mean two contexts fighting over one card.     */

    /* --- ABI v4: how the pair space is split ----------------------------- */
    int32_t  split_mode;     /* CORR_SPLIT_*; 0 is the ABI v3 behaviour        */
} corr_config;

/*
 * Where one device's share of the work lives.
 *
 * With several devices the pair space is split: every device owns a contiguous
 * range of baselines, and it can only compute baselines whose two stations it
 * holds a copy of. In this version every device holds every station (`n_station
 * _resident == n_station` for all of them) and only the baseline axis is split,
 * which is the simplest complete split: it produces every baseline exactly once
 * and needs no pair-covering design. Later versions will shrink the resident set
 * per device, which is what makes 40 stations fit in the cards' host-link budget.
 */
typedef struct corr_device_plan {
    int32_t device_id;           /* CUDA ordinal this plan describes          */
    int32_t n_station_resident;  /* stations this device holds a copy of      */
    int32_t first_baseline;      /* its lowest-numbered baseline              */
    int32_t n_baseline;          /* how many baselines it owns                */
    /* 1 when its baselines are exactly [first_baseline, first_baseline +
       n_baseline); 0 when they are scattered through the global numbering, which
       is what pair covering produces. Use corr_device_baselines() for the list
       itself. */
    int32_t contiguous;
} corr_device_plan;

/*
 * Running totals for one device, cumulative since corr_create().
 *
 * `bytes_h2d` is the interesting one: with every device holding a copy of every
 * station, it is (stations held) x (staging bytes per station per second) -- the
 * number that decides how many stations a card can take before its host link is
 * the limit. Divide by the elapsed seconds to get the achieved rate.
 */
typedef struct corr_device_stats {
    int32_t  device_id;
    uint64_t bytes_h2d;   /* staging bytes pulled from the host               */
    uint64_t bytes_d2h;   /* spectrum bytes handed back                       */
    uint64_t transfers;   /* host-to-device transfers issued (chunks)         */

    /* Nanoseconds the calling thread spent inside each call, cumulatively. These
       say where a slow run's time goes: a large `ns_slot_free` means the receive
       ring is waiting on this device's copy, a large `ns_accumulate` means the
       thread itself is the bottleneck (it only queues work), and a large
       `ns_fetch`/`ns_rows_wait` means the pipeline is waiting for results. */
    uint64_t ns_accumulate;
    uint64_t ns_slot_free;
    uint64_t ns_row_end;
    uint64_t ns_rows_wait;
    uint64_t ns_fetch;
} corr_device_stats;

/*
 * One station's staging area. Everything is relative to `base`, which
 * corr_staging_info() has already advanced to the requested station, so the
 * bytes a receive thread may legally touch are exactly
 * [base, base + total_bytes).
 *
 * Frame f of the slot currently being filled:
 *
 *   base + slot*slot_stride + (f % frames_per_slot)*frame_stride
 *
 * A frame is exactly 2*n_freq int8 samples, two's complement, assembled by
 * concatenating consecutive UDP packet payloads in packet-counter order. Frames
 * whose packets did not all arrive are left as they are by the host -- missing
 * ones are declared in the validity mask below and zeroed on the device.
 */
typedef struct corr_staging {
    int8_t *base;          /* start of THIS station's region                 */
    size_t  frame_stride;  /* bytes per frame = 2*n_freq                     */
    size_t  slot_stride;   /* bytes per slot  = frames_per_slot*frame_stride */
    size_t  total_bytes;   /* this station's whole extent, samples + mask     */

    /*
     * Validity mask, one bit per frame of a slot, `validity_bytes_per_slot`
     * bytes per slot starting `validity_offset` bytes into this station's region.
     * Bit `f` of slot `s` is set when frame `f` of that slot is present.
     *
     * This is the only statement of which frames are real: corr_row_accumulate()
     * uploads it with the samples, and a frame whose bit is clear is zeroed on
     * the device before the FFT, exactly as if the host had filled it with zeros.
     * A frame that is present must still be filled in full -- a bit set over
     * half-written samples is the one wrong answer this scheme can produce, so
     * the writer sets the bit only once every packet of that frame has landed.
     *
     * The region starts at all-zero, i.e. "nothing present". That default is
     * deliberate: a consumer that reads a slot before the frame assembler has
     * written its mask sees an entirely missing slot, which fails loudly in a
     * phase test, instead of a slot that is silently claimed valid.
     *
     * Bits are set by the receive thread that owns the station; the slot's whole
     * mask is cleared by corr_row_slot_free(), which is the moment the slot stops
     * being the GPU's and starts being the next writer's. Clearing it there is
     * what covers a station that sends nothing at all for a whole slot: without
     * it the next lap would inherit the previous lap's bits and correlate stale
     * samples as data.
     */
    size_t  validity_offset;
    size_t  validity_bytes_per_slot;
} corr_staging;

/*
 * Version banner of the loaded shared object, e.g.
 *   "corr_cuda 0.2.0 abi=2 cudart=12040 driver=12040 cufft=11201 arch=sm_89"
 * Rust compares this at startup rather than trusting the link.
 */
const char *corr_abi_banner(void);

/*
 * Thread-local message describing the most recent failure on this thread.
 * Never returns NULL; returns "" when nothing has failed yet.
 */
const char *corr_last_error(void);

/*
 * Create an instance: validate the configuration, select the device, allocate the
 * pinned staging ring and the device workspace, create the streams and events,
 * build the cuFFT plan, and zero the accumulator.
 *
 * On success *out_ctx is a handle owned by the caller; release it with
 * corr_destroy(). On failure *out_ctx is set to NULL.
 */
corr_status corr_create(const corr_config *cfg, void **out_ctx);

/* Tear down a context. Passing NULL is a no-op and returns CORR_OK. */
corr_status corr_destroy(void *ctx);

/*
 * Report where the receive thread for `station` writes frames. Valid from
 * corr_create() until corr_destroy().
 */
corr_status corr_staging_info(void *ctx, int32_t station, corr_staging *out);

/*
 * Hand the frames held in staging slot `slot_seq % block_slots` to the GPU and add
 * their correlation into the current row's accumulator. Returns as soon as the
 * work is queued.
 *
 * The caller passes the *sequence*, not the physical slot: the sequence is what
 * identifies the frames (the ring reuses a physical slot every block_slots
 * sequences), and under CORR_SPLIT_TIME_SLICE it is also what decides which
 * device the slot belongs to. Deriving the physical slot here means the two
 * sides cannot disagree about it.
 *
 * `frames` is how many frames the caller actually filled, 1 <= frames <=
 * frames_per_slot. It has to be passed rather than inferred from the slot size:
 * the last slot of a row (and the last slot of a run) is usually partial, and a
 * slot that is being reused still holds the previous batch's bytes past
 * `frames`. Processing the full slot in that case would silently fold stale
 * samples into the correlation -- a wrong answer that looks perfectly reasonable.
 *
 * The caller must not modify that slot's staging again until
 * corr_row_slot_free(slot) has returned.
 *
 * The slot's validity mask is uploaded with the samples and decides which frames
 * are zeroed on the device (see corr_staging). Frames the caller did not fill
 * must have their bit clear; the bits of frames past `frames` are not read.
 *
 * Returns CORR_ERR_ARG for a slot outside [0, block_slots) or a frame count
 * outside [1, frames_per_slot].
 */
corr_status corr_row_accumulate(void *ctx, int64_t slot_seq, int32_t frames);

/*
 * Block until the staging bytes of `slot_seq` may be written again, i.e. until
 * every device that reads them has finished its host-to-device copy. This is
 * deliberately much earlier than corr_row_wait(): the copy finishes in
 * milliseconds while the row may run for a second, and holding the receive ring
 * that long would stall reception.
 *
 * Returning also hands the slot's validity mask back: every bit of it is cleared,
 * because the next writer of that physical slot starts from "nothing present" and
 * sets a bit per frame it actually completes. The clear happens after the copy is
 * known to be done and before the caller may release the slot to the receive
 * side, so a writer that inherits the slot can never see last lap's bits.
 */
corr_status corr_row_slot_free(void *ctx, int64_t slot_seq);

/*
 * Close the current row. `frames` is the number of FFT frames that were actually
 * accumulated into it, and the row is divided by it.
 *
 * It is deliberately the caller's number and not the sum of the slot frame
 * counts: a row that ends short (shutdown, or a watermark timeout) is divided by
 * what really went in, while a row containing zero filled frames is divided by
 * the full count, so the amplitude bias stays explainable by the recorded flag
 * rather than being silently corrected per baseline.
 *
 * Returns CORR_ERR_BUSY if all rows_per_fetch slots are full and unclaimed: that
 * means the caller has stopped fetching, and overwriting a row that has not been
 * copied out yet would lose data silently.
 */
corr_status corr_row_end(void *ctx, uint64_t frames);

/*
 * Block until every closed but not yet fetched row is complete, i.e. until all
 * of their scaling kernels have run.
 */
corr_status corr_rows_wait(void *ctx);

/*
 * How many closed rows are sitting in the stack waiting to be fetched, in
 * [0, rows_per_fetch]. Rows are always claimed from index 0 upward, so this is
 * also the count corr_rows_fetch() would transfer right now.
 */
corr_status corr_rows_pending(void *ctx, int32_t *out_pending);

/*
 * Copy every pending row to the host in a single transfer and clear them.
 *
 * out must hold rows_per_fetch * n_baseline * n_freq * 8 bytes; the transfer
 * writes only the pending rows, laid out row-major:
 *
 *   out[(row*n_baseline + baseline)*n_freq + bin]
 *
 * out_rows receives how many rows were actually written (0 is a valid answer:
 * a caller flushing at shutdown may find nothing pending).
 *
 * Each row is complex64 with the real part first (little endian). Bin 0 is DC
 * and bin n_freq-1 the last retained bin; the Nyquist bin produced by the R2C
 * transform is dropped. No FFT normalisation is applied, and the row sum has
 * already been divided by the frame count passed to corr_row_end().
 *
 * The rows are cleared only after the copy has actually completed, so
 * accumulating the next row immediately afterwards is safe.
 *
 * Under CORR_SPLIT_BASELINE_RANGE and CORR_SPLIT_PAIR_COVERING each device owns
 * a different set of baselines and its transfer lands directly in its own place
 * in the row. Under CORR_SPLIT_TIME_SLICE every device holds a *partial* row over
 * the frames it was given, and the library adds them together here, so `out` is
 * still one complete row. The addition is exact up to the order of the frame
 * sum, which is the same freedom a single device already has.
 */
corr_status corr_rows_fetch(void *ctx, void *out, size_t out_bytes,
                            int32_t *out_rows);

/*
 * Number of baselines for n_station stations, i.e. n_station*(n_station+1)/2.
 * Returns -1 for n_station < 1.
 *
 * Exposed so that Rust and CUDA cannot disagree about the count; the .so is the
 * single source of truth for the numbering.
 */
int32_t corr_n_baseline(int32_t n_station);

/*
 * Fill out_jk with the (j, k) pair of every baseline with j <= k, in row-major
 * order (j outer, k inner):
 *
 *   out_jk[2*i]     = j
 *   out_jk[2*i + 1] = k
 *
 * out_len must be 2*corr_n_baseline(n_station) int32 values. This is the
 * definition behind the baseline axis of the output files, and is written
 * verbatim into the run metadata.
 */
corr_status corr_baseline_pairs(int32_t n_station, int32_t *out_jk, int32_t out_len);

/*
 * --- ABI v3: the device list behind one context -------------------------
 *
 * None of the calls above changed when the context started spanning several
 * devices: corr_row_accumulate() still means "this staging slot, all stations",
 * corr_row_slot_free() waits until every device holding a copy has finished
 * reading it, and corr_rows_fetch() still fills one contiguous row buffer in
 * baseline order. The split is the library's business.
 *
 * These three calls exist so that the caller can *see* the split rather than
 * guess at it: a plan is performance-relevant (it decides how many bytes each
 * card must pull over its host link) and a mis-assigned baseline shows up as a
 * slice of the row that is quietly zero, which is the kind of failure that has
 * to be visible before it happens rather than after.
 */

/* How many devices this context runs on, or -1 if ctx is NULL. */
int32_t corr_n_devices(void *ctx);

/* Describe device `device_index` in [0, corr_n_devices(ctx)). */
corr_status corr_device_plan_get(void *ctx, int32_t device_index,
                                 corr_device_plan *out);

/*
 * The global station indices device `device_index` holds a copy of, in the
 * local order it holds them (which is the order the correlation kernel's station
 * axis uses). out_len must be at least that device's n_station_resident.
 */
corr_status corr_device_stations(void *ctx, int32_t device_index,
                                 int32_t *out_stations, int32_t out_len);

/*
 * The global indices of the baselines device `device_index` owns, in the order
 * its own row slice holds them. out_len must be at least its n_baseline.
 */
corr_status corr_device_baselines(void *ctx, int32_t device_index,
                                  int32_t *out_baselines, int32_t out_len);

/* Cumulative traffic counters for device `device_index`. */
corr_status corr_device_stats_get(void *ctx, int32_t device_index,
                                  corr_device_stats *out);

/*
 * Host memory for a row buffer, pinned, and its matching free.
 *
 * Why this exists: corr_rows_fetch() transfers into whatever the caller passes,
 * and the transfer rate depends on what that is. Measured on this machine for a
 * 1.31 MiB row: 22.5 GB/s into page-locked memory, 12 GB/s into ordinary
 * pageable memory, because the pageable case is staged through a bounce buffer.
 * At a one-second integration that difference is invisible (the fetch is 0.2% of
 * the run); at a 40.96 us row the fetch is most of the row period, and it is the
 * difference between 6333 and roughly 9000 rows per second.
 *
 * Nothing in the library requires the buffer to be pinned, so a caller that does
 * not care can keep a Vec. Handing the same buffer to a writer afterwards is
 * fine: page-locked memory is ordinary cached memory as far as the CPU is
 * concerned, except that a large page-locked allocation counts against
 * RLIMIT_MEMLOCK, which is why the free is not optional.
 */
corr_status corr_host_alloc(size_t bytes, void **out);
void corr_host_free(void *p);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CORR_CUDA_H */
