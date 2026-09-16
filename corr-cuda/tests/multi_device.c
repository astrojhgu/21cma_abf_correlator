/* multi_device.c -- does splitting the pair space over N devices change the
 * numbers?
 *
 * It must not. Every baseline is computed on exactly one device, from the same
 * frames, in the same order, by the same kernels, so a run over N devices has to
 * produce the *same bytes* as a run over one -- not merely something close. That
 * makes this the strongest check available for the multi-device path, and it
 * needs no reference implementation: the single-device run is the reference, and
 * it is itself checked against numpy by golden.c.
 *
 * What it does not check: throughput. The point here is that four cards produce
 * one correct answer; whether four cards can pull the bytes fast enough is a
 * measurement, not a test (tools/ has the script for that).
 */

#include "corr_cuda.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *what) {
    fprintf(stderr, "multi_device: %s: %s\n", what, corr_last_error());
    exit(2);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (p == NULL) {
        fprintf(stderr, "multi_device: out of memory (%zu bytes)\n", n);
        exit(2);
    }
    return p;
}

/* Deterministic input, so a failure is reproducible. The values only have to be
 * small enough that no frame overflows anything; the interesting property is
 * that the same bytes go to every configuration. */
static void fill_input(int8_t *p, size_t n, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        p[i] = (int8_t)((int)((s >> 16) & 0x1f) - 16);
    }
}

typedef struct Params {
    int n_station;
    int n_freq;
    int frames_per_slot;
    int chunk_frames;
    int block_slots;
    int rows_per_fetch;
    int n_cum;
    int rows;
} Params;

static corr_config make_config(const Params *p, int n_devices, const int *devices,
                               int split_mode) {
    corr_config c;
    memset(&c, 0, sizeof c);
    c.abi_version = CORR_ABI_VERSION;
    c.struct_size = (uint32_t)sizeof(corr_config);
    c.device_id = 0;
    c.n_station = p->n_station;
    c.n_freq = p->n_freq;
    c.frames_per_slot = p->frames_per_slot;
    c.chunk_frames = p->chunk_frames;
    c.block_slots = p->block_slots;
    c.rows_per_fetch = p->rows_per_fetch;
    c.n_devices = n_devices;
    for (int i = 0; i < n_devices; ++i) c.devices[i] = devices[i];
    c.split_mode = split_mode;
    return c;
}

/* Drive `rows` rows of `n_cum` frames exactly as the scheduler does, then fetch
 * them. `out` must hold rows * n_baseline * n_freq complex64 values. */
static void run(const Params *p, int n_devices, const int *devices, int split_mode,
                const int8_t *input, float *out) {
    corr_config cfg = make_config(p, n_devices, devices, split_mode);
    void *ctx = NULL;
    if (corr_create(&cfg, &ctx) != CORR_OK) die("corr_create");

    const size_t frame_stride = (size_t)2 * p->n_freq;

    /* The placement must cover the baseline axis exactly once between the
     * devices: no gaps and no duplicates. A missing baseline shows up as a row of
     * zeros that looks like data, and a duplicated one as a row that is scaled
     * twice, so this is checked here rather than trusted. */
    const int n_baseline = corr_n_baseline(p->n_station);
    const bool time_slice = (split_mode == CORR_SPLIT_TIME_SLICE);
    int32_t seen_devices = corr_n_devices(ctx);
    if (seen_devices != n_devices) die("corr_n_devices disagrees with the config");
    int32_t *owner = (int32_t *)xmalloc(sizeof(int32_t) * (size_t)n_baseline);
    for (int i = 0; i < n_baseline; ++i) owner[i] = -1;
    int32_t next_range = 0;
    for (int d = 0; d < n_devices; ++d) {
        corr_device_plan plan;
        if (corr_device_plan_get(ctx, d, &plan) != CORR_OK) die("corr_device_plan_get");
        if (plan.device_id != devices[d]) die("plan reports the wrong device");
        if (plan.n_baseline <= 0) {
            fprintf(stderr, "multi_device: device %d owns no baselines\n", d);
            exit(1);
        }
        /* What a placement must look like depends on the split: the range split
         * and the time-slice split both give every device every station (they
         * differ in which frames and which baselines the device owns), while pair
         * covering gives each device a subset. */
        const bool full_residency =
            (split_mode == CORR_SPLIT_BASELINE_RANGE) || time_slice;
        if (full_residency && plan.n_station_resident != p->n_station) {
            fprintf(stderr, "multi_device: device %d holds %d of %d stations\n", d,
                    plan.n_station_resident, p->n_station);
            exit(1);
        }
        if (!full_residency && plan.n_station_resident >= p->n_station) {
            fprintf(stderr, "multi_device: pair covering gave device %d all %d stations, "
                            "so nothing is being replicated\n", d, p->n_station);
            exit(1);
        }
        int32_t *list = (int32_t *)xmalloc(sizeof(int32_t) * (size_t)plan.n_station_resident);
        if (corr_device_stations(ctx, d, list, plan.n_station_resident) != CORR_OK)
            die("corr_device_stations");
        if (full_residency) {
            for (int i = 0; i < plan.n_station_resident; ++i)
                if (list[i] != i) {
                    fprintf(stderr, "multi_device: device %d holds station %d in slot %d\n",
                            d, list[i], i);
                    exit(1);
                }
        } else {
            for (int i = 1; i < plan.n_station_resident; ++i)
                if (list[i] <= list[i - 1]) {
                    fprintf(stderr, "multi_device: device %d station list is not sorted and "
                                    "distinct\n", d);
                    exit(1);
                }
            for (int i = 0; i < plan.n_station_resident; ++i)
                if (list[i] < 0 || list[i] >= p->n_station) {
                    fprintf(stderr, "multi_device: device %d holds station %d out of range\n",
                            d, list[i]);
                    exit(1);
                }
        }
        free(list);

        int32_t *bl = (int32_t *)xmalloc(sizeof(int32_t) * (size_t)plan.n_baseline);
        if (corr_device_baselines(ctx, d, bl, plan.n_baseline) != CORR_OK)
            die("corr_device_baselines");
        for (int i = 0; i < plan.n_baseline; ++i) {
            const int32_t b = bl[i];
            if (b < 0 || b >= n_baseline) {
                fprintf(stderr, "multi_device: device %d owns baseline %d out of range\n", d, b);
                exit(1);
            }
            if (time_slice) {
                /* Under the time-slice split every device computes every baseline,
                 * each as a partial sum over its own frames; the library adds the
                 * partials when the row is fetched. So duplicates are expected --
                 * what must hold is that each device covers the whole axis. */
                if (b != i) {
                    fprintf(stderr, "multi_device: time-slice device %d owns baseline %d in "
                                    "slot %d; it must own all of them in order\n", d, b, i);
                    exit(1);
                }
            } else {
                if (owner[b] >= 0) {
                    fprintf(stderr, "multi_device: baseline %d is owned by devices %d and %d\n",
                            b, owner[b], d);
                    exit(1);
                }
                owner[b] = d;
            }
        }
        if (full_residency && !time_slice) {
            /* The range split tiles the axis: each device owns a consecutive run. */
            if (!plan.contiguous || plan.first_baseline != next_range) {
                fprintf(stderr, "multi_device: baseline-range split is not contiguous at "
                                "device %d\n", d);
                exit(1);
            }
            next_range = plan.first_baseline + plan.n_baseline;
        }
        if (time_slice) {
            /* The time-slice split gives every device the whole axis: each owns a
             * partial sum over its own frames, and they are added on fetch. */
            if (!plan.contiguous || plan.first_baseline != 0 ||
                plan.n_baseline != n_baseline) {
                fprintf(stderr, "multi_device: time-slice device %d owns [%d..%d) of %d "
                                "baselines; it must own all of them\n", d,
                        plan.first_baseline, plan.first_baseline + plan.n_baseline,
                        n_baseline);
                exit(1);
            }
        }
        free(bl);
    }
    if (!time_slice)
        for (int b = 0; b < n_baseline; ++b)
            if (owner[b] < 0) {
                fprintf(stderr, "multi_device: baseline %d is owned by no device\n", b);
                exit(1);
            }
    if (split_mode == CORR_SPLIT_BASELINE_RANGE && next_range != n_baseline) {
        fprintf(stderr, "multi_device: slices cover %d of %d baselines\n", next_range,
                n_baseline);
        exit(1);
    }
    free(owner);

    int submits = 0;
    for (int r = 0; r < p->rows; ++r) {
        for (int f0 = 0; f0 < p->n_cum; f0 += p->frames_per_slot) {
            const int ft = p->n_cum - f0 < p->frames_per_slot ? p->n_cum - f0
                                                              : p->frames_per_slot;
            const int slot = submits % cfg.block_slots;
            if (submits >= cfg.block_slots) {
                if (corr_row_slot_free(ctx, submits) != CORR_OK) die("corr_row_slot_free");
            }
            for (int g = 0; g < p->n_station; ++g) {
                corr_staging st;
                if (corr_staging_info(ctx, g, &st) != CORR_OK) die("corr_staging_info");
                /* Every frame of this slot is real, and the validity mask is what
                 * says so: a clear bit would have the device zero the frame, and
                 * this test compares spectra with numpy. The release above emptied
                 * the mask, so the bits go in after the samples, as the receive
                 * thread does it. */
                unsigned char *mask = (unsigned char *)st.base + st.validity_offset +
                                      (size_t)slot * st.validity_bytes_per_slot;
                for (int f = 0; f < ft; ++f) {
                    const int8_t *src =
                        input + (((size_t)r * p->n_station + g) * p->n_cum + f0 + f) *
                                    frame_stride;
                    memcpy(st.base + (size_t)slot * st.slot_stride +
                               (size_t)f * st.frame_stride,
                           src, frame_stride);
                    mask[f >> 3] |= (unsigned char)(1u << (f & 7));
                }
            }
            if (corr_row_accumulate(ctx, submits, ft) != CORR_OK) die("corr_row_accumulate");
            ++submits;
        }
        if (corr_row_end(ctx, (uint64_t)p->n_cum) != CORR_OK) die("corr_row_end");
    }

    const int first_held = submits > cfg.block_slots ? submits - cfg.block_slots : 0;
    for (int sIdx = submits - 1; sIdx >= first_held; --sIdx) {
        if (corr_row_slot_free(ctx, sIdx) != CORR_OK)
            die("corr_row_slot_free(final)");
    }

    if (corr_rows_wait(ctx) != CORR_OK) die("corr_rows_wait");
    int32_t pending = -1;
    if (corr_rows_pending(ctx, &pending) != CORR_OK) die("corr_rows_pending");
    if (pending != p->rows) {
        fprintf(stderr, "multi_device: %d pending rows, expected %d\n", pending, p->rows);
        exit(1);
    }

    const size_t row_bytes =
        (size_t)corr_n_baseline(p->n_station) * (size_t)p->n_freq * sizeof(float) * 2;
    int32_t fetched = -1;
    if (corr_rows_fetch(ctx, out, row_bytes * (size_t)p->rows, &fetched) != CORR_OK)
        die("corr_rows_fetch");
    if (fetched != p->rows) {
        fprintf(stderr, "multi_device: fetched %d rows, expected %d\n", fetched, p->rows);
        exit(1);
    }
    if (corr_destroy(ctx) != CORR_OK) die("corr_destroy");
}

/* The planner, checked at the sizes the design actually cares about: 24 stations
 * (what flows today) and 40 (the target). No data path -- just the placement --
 * so this is fast and says exactly what the replication factor is. */
static void check_plan_shape(int n_station, int n_devices, int split_mode) {
    Params p = {0};
    p.n_station = n_station;
    p.n_freq = 2048;
    p.frames_per_slot = 32;
    p.chunk_frames = 32;
    p.block_slots = 2;
    p.rows_per_fetch = 1;
    p.n_cum = 32;
    p.rows = 1;
    int devices[CORR_MAX_DEVICES];
    for (int i = 0; i < n_devices; ++i) devices[i] = i;

    corr_config cfg = make_config(&p, n_devices, devices, split_mode);
    void *ctx = NULL;
    corr_status st = corr_create(&cfg, &ctx);
    if (st != CORR_OK) {
        /* Fewer visible devices than asked for is not a planner failure. */
        printf("  %2d stations / %d devices: SKIPPED (%s)\n", n_station, n_devices,
               corr_last_error());
        return;
    }

    const int n_baseline = corr_n_baseline(n_station);
    const bool ts = (split_mode == CORR_SPLIT_TIME_SLICE);
    int *owner = (int *)xmalloc(sizeof(int) * (size_t)n_baseline);
    for (int i = 0; i < n_baseline; ++i) owner[i] = -1;
    int total_resident = 0, min_res = 0, max_res = 0, min_bl = 0, max_bl = 0;
    for (int d = 0; d < n_devices; ++d) {
        corr_device_plan plan;
        if (corr_device_plan_get(ctx, d, &plan) != CORR_OK) die("corr_device_plan_get");
        total_resident += plan.n_station_resident;
        if (d == 0) {
            min_res = max_res = plan.n_station_resident;
            min_bl = max_bl = plan.n_baseline;
        } else {
            if (plan.n_station_resident < min_res) min_res = plan.n_station_resident;
            if (plan.n_station_resident > max_res) max_res = plan.n_station_resident;
            if (plan.n_baseline < min_bl) min_bl = plan.n_baseline;
            if (plan.n_baseline > max_bl) max_bl = plan.n_baseline;
        }
        if (!ts && plan.n_station_resident >= n_station) {
            fprintf(stderr, "plan_shape: device %d holds every station, so the split is "
                            "not a covering\n", d);
            exit(1);
        }
        if (ts) {
            if (plan.n_station_resident != n_station || plan.n_baseline != n_baseline ||
                !plan.contiguous || plan.first_baseline != 0) {
                fprintf(stderr, "plan_shape: time-slice device %d must hold every station "
                                "and the whole baseline axis\n", d);
                exit(1);
            }
            total_resident += 0; /* R is reported below; every device reads its own frames */
        }
        int *bl = (int *)xmalloc(sizeof(int) * (size_t)plan.n_baseline);
        if (corr_device_baselines(ctx, d, bl, plan.n_baseline) != CORR_OK)
            die("corr_device_baselines");
        for (int i = 0; i < plan.n_baseline; ++i) {
            if (ts) continue; /* every device holds all of them, by construction */
            if (owner[bl[i]] >= 0) {
                fprintf(stderr, "plan_shape: baseline %d owned twice\n", bl[i]);
                exit(1);
            }
            owner[bl[i]] = d;
        }
        free(bl);
    }
    if (!ts)
        for (int i = 0; i < n_baseline; ++i)
            if (owner[i] < 0) {
                fprintf(stderr, "plan_shape: baseline %d owned by nobody\n", i);
                exit(1);
            }
    const double r = ts ? 1.0 : (double)total_resident / (double)n_station;
    printf("  %2d stations / %d devices: resident %d..%d per device, R = %.2f, "
           "baselines %d..%d per device, all %d covered\n",
           n_station, n_devices, min_res, max_res, r, min_bl, max_bl, n_baseline);
    if (max_res - min_res > 1) {
        fprintf(stderr, "plan_shape: resident sets differ by %d stations, which spends the "
                        "host-link budget on the largest card\n", max_res - min_res);
        exit(1);
    }
    free(owner);
    corr_destroy(ctx);
}

int main(void) {
    int visible = 0;
    if (cudaGetDeviceCount(&visible) != cudaSuccess || visible < 1) {
        fprintf(stderr, "multi_device: no CUDA device is visible\n");
        return 2;
    }
    if (visible < 2) {
        printf("multi_device: SKIPPED -- only %d device is visible, and the point of "
               "this test is the difference between one and several\n", visible);
        return 0;
    }

    Params p;
    p.n_station = 8;
    p.n_freq = 2048;
    p.frames_per_slot = 32;
    p.chunk_frames = 16;
    p.block_slots = 3;
    p.rows_per_fetch = 3; /* == rows: fetch once at the end, as the driver below does */
    p.n_cum = 96; /* 3 slots of 32, so a row is a whole number of slots */
    p.rows = 3;

    int n_devices = visible < (int)CORR_MAX_DEVICES ? visible : (int)CORR_MAX_DEVICES;
    int devices[CORR_MAX_DEVICES];
    for (int i = 0; i < n_devices; ++i) devices[i] = i;

    const size_t frame_stride = (size_t)2 * p.n_freq;
    const size_t in_bytes =
        (size_t)p.rows * (size_t)p.n_station * (size_t)p.n_cum * frame_stride;
    int8_t *input = (int8_t *)xmalloc(in_bytes);
    fill_input(input, in_bytes, 20260914u);

    const size_t row_bytes =
        (size_t)corr_n_baseline(p.n_station) * (size_t)p.n_freq * sizeof(float) * 2;
    const size_t out_bytes = row_bytes * (size_t)p.rows;
    float *one = (float *)xmalloc(out_bytes);
    float *many = (float *)xmalloc(out_bytes);

    printf("multi_device: %d stations, %d baselines, %d devices: 1 vs %d (baseline range) "
           "vs %d (pair covering)\n", p.n_station, corr_n_baseline(p.n_station), n_devices,
           n_devices, n_devices);

    run(&p, 1, devices, CORR_SPLIT_BASELINE_RANGE, input, one);
    run(&p, n_devices, devices, CORR_SPLIT_BASELINE_RANGE, input, many);

    /* And again with the pair space covered rather than merely sliced: each
     * device holds only part of the array, so a baseline can only be computed by
     * a device that holds both of its stations. The answer must not change. */
    float *covered = (float *)xmalloc(out_bytes);
    run(&p, n_devices, devices, CORR_SPLIT_PAIR_COVERING, input, covered);

    /* And once more split by *time*: every device holds every station but only
     * its share of the frames, so each accumulates a partial row and the library
     * adds the partials when the row is fetched. This is the split whose host
     * traffic is the run's data rate and nothing more, so it is the one that has
     * to be exactly right. */
    float *tslice = (float *)xmalloc(out_bytes);
    run(&p, n_devices, devices, CORR_SPLIT_TIME_SLICE, input, tslice);
    /* The sharpest check on the reduction is against another time-slice run with a
     * different number of partials: same frames, same kernels, so the only
     * difference is how many partial sums were added and in what order. Comparing
     * against the single-device row instead only bounds the scale, because a sum
     * split in four is honestly a different (less accurate) accumulation order. */
    float *tslice2 = NULL;
    int tslice2_devices = n_devices > 2 ? 2 : 0;
    if (tslice2_devices > 0) {
        tslice2 = (float *)xmalloc(out_bytes);
        run(&p, tslice2_devices, devices, CORR_SPLIT_TIME_SLICE, input, tslice2);
    }

    /* Compare against the single-device reference.
     *
     * The baseline-range split must reproduce it *bit for bit*: the same
     * baselines, the same frames, the same kernels, the same layout, so a single
     * differing bit means a slice landed in the wrong place, was scaled twice, or
     * was fetched from the wrong device.
     *
     * The pair-covering split is held to a tight tolerance rather than to bit
     * equality, and the reason is worth stating: a device that holds a subset of
     * the stations transforms a different *number* of spectra per cuFFT call, and
     * cuFFT does not promise identical rounding for different batch sizes. A
     * layout bug would be orders of magnitude larger than float32 rounding, which
     * is what the tolerance distinguishes. */
    const size_t n_values = out_bytes / sizeof(float);
    size_t bad = 0, first_bad = 0, bad_covered = 0;
    double sum_ref = 0.0, max_abs_covered = 0.0;
    for (size_t i = 0; i < n_values; ++i) {
        if (memcmp(&one[i], &many[i], sizeof(float)) != 0) {
            if (bad == 0) first_bad = i;
            ++bad;
        }
        const double d = fabs((double)one[i] - (double)covered[i]);
        if (d != 0.0) ++bad_covered;
        if (d > max_abs_covered) max_abs_covered = d;
        sum_ref += fabs((double)one[i]);
    }
    const double mean_ref = sum_ref / (double)(n_values ? n_values : 1);
    if (bad != 0) {
        const size_t per_row = row_bytes / sizeof(float);
        const size_t row = first_bad / per_row;
        const size_t off = first_bad % per_row;
        fprintf(stderr,
                "multi_device: baseline-range split differs from one device in %zu of %zu "
                "values; first at row %zu baseline %d bin %d: %g vs %g\n",
                bad, n_values, row, (int)(off / (size_t)p.n_freq),
                (int)(off % (size_t)p.n_freq), (double)one[first_bad], (double)many[first_bad]);
        return 1;
    }
    double max_abs_tslice = 0.0;
    size_t bad_tslice = 0;
    for (size_t i = 0; i < n_values; ++i) {
        const double d = fabs((double)one[i] - (double)tslice[i]);
        if (d != 0.0) ++bad_tslice;
        if (d > max_abs_tslice) max_abs_tslice = d;
    }
    const double rel_tslice = mean_ref > 0.0 ? max_abs_tslice / mean_ref : max_abs_tslice;
    const double rel_covered = mean_ref > 0.0 ? max_abs_covered / mean_ref : max_abs_covered;
    if (rel_covered > 1e-5) {
        fprintf(stderr,
                "multi_device: pair covering differs from one device by %.3e relative "
                "(%zu values, max |d| = %.3e, mean |value| = %.3e); that is far more than "
                "float32 rounding, so the placement is wrong\n",
                rel_covered, bad_covered, max_abs_covered, mean_ref);
        return 1;
    }
    /* The time-slice result differs by construction: the frames are summed in a
     * different order, because N partial sums are added instead of one long
     * accumulation. That is float32 rounding, not an approximation -- but it is
     * the reason this split cannot be checked bit for bit the way the range split
     * is, and the tolerance is what catches a reduction that is actually wrong
     * (which would be off by a factor, not by 1e-7). */
    /* Scale check: a reduction that dropped or double counted a partial would be
     * off by tens of percent, not by 1e-5. */
    if (rel_tslice > 1e-3) {
        fprintf(stderr,
                "multi_device: time-slice split differs from one device by %.3e relative "
                "(%zu values, max |d| = %.3e, mean |value| = %.3e); the partial rows are "
                "not being summed correctly\n",
                rel_tslice, bad_tslice, max_abs_tslice, mean_ref);
        return 1;
    }
    if (tslice2 != NULL) {
        double worst = 0.0;
        size_t differing = 0;
        for (size_t i = 0; i < n_values; ++i) {
            const double d = fabs((double)tslice[i] - (double)tslice2[i]);
            if (d != 0.0) ++differing;
            if (d > worst) worst = d;
        }
        const double rel2 = mean_ref > 0.0 ? worst / mean_ref : worst;
        if (rel2 > 1e-5) {
            fprintf(stderr,
                    "multi_device: %d-way and %d-way time-slice splits differ by %.3e "
                    "relative (%zu values, max |d| = %.3e); the reduction depends on the "
                    "device count, which it must not\n",
                    n_devices, tslice2_devices, rel2, differing, worst);
            return 1;
        }
        printf("multi_device: %d-way and %d-way time-slice splits agree to %.2e relative "
               "(%zu values differ)\n", n_devices, tslice2_devices, rel2, differing);
    }
    printf("multi_device: pair covering agrees with one device to %.2e relative "
           "(%zu values differ, max |d| = %.3e, cuFFT batch rounding)\n",
           rel_covered, bad_covered, max_abs_covered);
    printf("multi_device: time-slice split agrees with one device to %.2e relative "
           "(%zu values differ, max |d| = %.3e, partial-row sum rounding)\n",
           rel_tslice, bad_tslice, max_abs_tslice);

    /* A row of all zeros would compare equal and prove nothing. */
    int nonzero = 0;
    for (size_t i = 0; i < out_bytes / sizeof(float); ++i)
        if (many[i] != 0.0f) ++nonzero;
    if (nonzero < (int)(out_bytes / sizeof(float)) / 2) {
        fprintf(stderr, "multi_device: only %d of %zu values are nonzero; the comparison "
                        "is vacuous\n", nonzero, out_bytes / sizeof(float));
        return 1;
    }

    printf("multi_device: planning at the sizes that matter\n");
    int visible_now = 0;
    cudaGetDeviceCount(&visible_now);
    const int plan_devices = visible_now < 4 ? visible_now : 4;
    if (plan_devices >= 2) {
        check_plan_shape(8, plan_devices, CORR_SPLIT_PAIR_COVERING);
        check_plan_shape(24, plan_devices, CORR_SPLIT_PAIR_COVERING);
        check_plan_shape(40, plan_devices, CORR_SPLIT_PAIR_COVERING);
        check_plan_shape(40, plan_devices, CORR_SPLIT_TIME_SLICE);
    }

    printf("multi_device: OK -- %d devices reproduce the single-device row byte for byte "
           "(%d nonzero values)\n", n_devices, nonzero);
    free(input);
    free(one);
    free(many);
    free(covered);
    free(tslice);
    free(tslice2);
    return 0;
}
