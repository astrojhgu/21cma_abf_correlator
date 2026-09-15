/* golden.c -- run libcorr_cuda on a numpy-generated input and compare.
 *
 * The reference comes from tests/gen_golden.py; this program drives the public C
 * ABI the way the Rust scheduler will, so a pass means the whole device pipeline
 * agrees with numpy:
 *
 *   chunked H2D -> int8->float32 -> cuFFT R2C -> correlate into the row stack
 *   -> per-row 1/n_cum scale -> one host transfer of the whole stack
 *
 * It exercises the parts most likely to be wrong and least likely to be noticed:
 *
 *   * a row spans several staging slots, and the last slot is partial;
 *   * several rows share the device stack and are fetched in one transfer;
 * * the baseline numbering matches libcorr_cuda;
 *   * the autocorrelation of a real signal has a negligible imaginary part,
 *     which is what a wrong conjugation or a transposed sample layout breaks.
 *
 * usage: golden DATADIR
 */

#include "corr_cuda.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Relative RMS error budget. Measured on this machine: 1.6e-07 for
 * 4 stations / n_freq=2048 and 1.7e-07 for 15 stations / n_freq=4096, which is
 * float32 rounding and nothing else. The bar sits three orders of magnitude
 * above that, so ordinary drift cannot fail the test but a structural mistake
 * (wrong stride, a kept Nyquist bin, wrong or missing normalisation, a
 * transposed baseline axis, rows written to the wrong stack slot) cannot pass.
 */
static const double kRelRmsTolerance = 1e-4;
static const int kMaxBaselines = 1024;

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (p == NULL) {
        fprintf(stderr, "golden: out of memory (%zu bytes)\n", n);
        exit(2);
    }
    return p;
}

static void *xread(const char *dir, const char *name, size_t expect_bytes) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "golden: cannot open %s\n", path);
        exit(2);
    }
    if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); exit(2); }
    const long sz = ftell(f);
    rewind(f);
    if (sz < 0) { perror("ftell"); exit(2); }
    if ((size_t)sz != expect_bytes) {
        fprintf(stderr, "golden: %s is %ld bytes, expected %zu\n", path, sz,
                expect_bytes);
        exit(2);
    }
    void *p = xmalloc((size_t)sz);
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "golden: short read on %s\n", path);
        exit(2);
    }
    fclose(f);
    return p;
}

static void die(const char *what) {
    fprintf(stderr, "golden: %s: %s\n", what, corr_last_error());
    exit(1);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s DATADIR\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    char path[4096];

    /* --- meta.txt: header line, then "index j k" per baseline --- */
    snprintf(path, sizeof path, "%s/meta.txt", dir);
    FILE *mf = fopen(path, "r");
    if (mf == NULL) {
        fprintf(stderr, "golden: cannot open %s\n", path);
        return 2;
    }
    int n_station = 0, n_freq = 0, n_cum = 0, frames_per_slot = 0, rows = 0;
    if (fscanf(mf, "%d %d %d %d %d", &n_station, &n_freq, &n_cum, &frames_per_slot,
               &rows) != 5) {
        fprintf(stderr, "golden: malformed meta.txt header\n");
        return 2;
    }
    const int n_baseline = corr_n_baseline(n_station);
    if (n_baseline < 0 || n_baseline > kMaxBaselines || rows < 1 ||
        frames_per_slot < 1 || n_cum < 1) {
        fprintf(stderr, "golden: implausible meta.txt values\n");
        return 2;
    }

    int32_t lib_pairs[2 * kMaxBaselines];
    if (corr_baseline_pairs(n_station, lib_pairs, 2 * n_baseline) != CORR_OK)
        die("corr_baseline_pairs");

    for (int b = 0; b < n_baseline; ++b) {
        int idx = -1, j = -1, k = -1;
        if (fscanf(mf, "%d %d %d", &idx, &j, &k) != 3 || idx != b) {
            fprintf(stderr, "golden: malformed meta.txt baseline row %d\n", b);
            return 2;
        }
        if (lib_pairs[2 * b] != j || lib_pairs[2 * b + 1] != k) {
            fprintf(stderr,
                    "golden: baseline %d is (%d,%d) in meta.txt but (%d,%d) in "
                    "libcorr_cuda; the numbering definitions have diverged\n",
                    b, j, k, lib_pairs[2 * b], lib_pairs[2 * b + 1]);
            return 1;
        }
    }
    fclose(mf);

    const int n_slots_per_row = (n_cum + frames_per_slot - 1) / frames_per_slot;
    printf("golden: %d stations, n_freq=%d, %d baselines | row: %d frames = %d slots "
           "of %d | %d rows per fetch\n",
           n_station, n_freq, n_baseline, n_cum, n_slots_per_row, frames_per_slot, rows);

    /* --- context --- */
    corr_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = CORR_ABI_VERSION;
    cfg.struct_size = (uint32_t)sizeof(corr_config);
    cfg.device_id = 0;
    cfg.n_station = n_station;
    cfg.n_freq = n_freq;
    cfg.frames_per_slot = frames_per_slot;
    cfg.chunk_frames = frames_per_slot < 128 ? frames_per_slot : 128;
    cfg.block_slots = 3;
    cfg.rows_per_fetch = rows;

    void *ctx = NULL;
    if (corr_create(&cfg, &ctx) != CORR_OK) die("corr_create");

    const size_t frame_stride = (size_t)2 * n_freq;
    const int8_t *input =
        (const int8_t *)xread(dir, "in.bin",
                              (size_t)rows * n_station * n_cum * frame_stride);

    /* --- drive the rows exactly as the scheduler will ---
     *
     * `submits` counts slot handovers across the whole run, not within a row:
     * the staging ring is global, so a slot must be released before its next use
     * no matter which row that use belongs to. Resetting the counter per row is
     * the bug this comment exists to prevent -- it leaves the first block_slots
     * submissions unreleased for ever, and the ring then refuses the second row.
     */
    int submits = 0;
    for (int r = 0; r < rows; ++r) {
        for (int f0 = 0; f0 < n_cum; f0 += frames_per_slot) {
            const int ft = n_cum - f0 < frames_per_slot ? n_cum - f0 : frames_per_slot;
            const int slot = submits % cfg.block_slots;

            if (submits >= cfg.block_slots) {
                if (corr_row_slot_free(ctx, submits) != CORR_OK) die("corr_row_slot_free");
            }

            for (int g = 0; g < n_station; ++g) {
                corr_staging st;
                if (corr_staging_info(ctx, g, &st) != CORR_OK)
                    die("corr_staging_info");
                for (int f = 0; f < ft; ++f) {
                    const int8_t *src =
                        input + (((size_t)r * n_station + g) * n_cum + f0 + f) *
                                    frame_stride;
                    memcpy(st.base + (size_t)slot * st.slot_stride +
                               (size_t)f * st.frame_stride,
                           src, frame_stride);
                }
            }

            if (corr_row_accumulate(ctx, submits, ft) != CORR_OK) die("corr_row_accumulate");
            ++submits;
        }

        /* n_cum frames, not the sum of the slots: the frozen decision is to
         * divide by the frame count the row was supposed to contain. */
        if (corr_row_end(ctx, (uint64_t)n_cum) != CORR_OK) die("corr_row_end");
    }

    /* Release whichever slots are still held, so nothing is left in flight.
     * That is the last min(block_slots, submits) handovers. */
    {
        const int first_held = submits > cfg.block_slots ? submits - cfg.block_slots : 0;
        for (int sIdx = submits - 1; sIdx >= first_held; --sIdx) {
            if (corr_row_slot_free(ctx, sIdx) != CORR_OK)
                die("corr_row_slot_free(final)");
        }
    }

    if (corr_rows_wait(ctx) != CORR_OK) die("corr_rows_wait");

    int32_t pending = -1;
    if (corr_rows_pending(ctx, &pending) != CORR_OK) die("corr_rows_pending");
    if (pending != rows) {
        fprintf(stderr, "golden: expected %d pending rows, got %d\n", rows, pending);
        return 1;
    }

    const size_t row_bytes = (size_t)n_baseline * n_freq * sizeof(float) * 2;
    float *got = (float *)xmalloc(row_bytes * (size_t)rows);
    int32_t fetched = -1;
    if (corr_rows_fetch(ctx, got, row_bytes * (size_t)rows, &fetched) != CORR_OK)
        die("corr_rows_fetch");
    if (fetched != rows) {
        fprintf(stderr, "golden: fetched %d rows, expected %d\n", fetched, rows);
        return 1;
    }

    const float *expect = (const float *)xread(dir, "expect.bin", row_bytes * (size_t)rows);

    /* --- compare --- */
    double num = 0.0, den = 0.0, worst = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < row_bytes * (size_t)rows / sizeof(float); ++i) {
        const double d = (double)got[i] - (double)expect[i];
        num += d * d;
        den += (double)expect[i] * (double)expect[i];
        if (fabs(d) > worst) {
            worst = fabs(d);
            worst_i = i;
        }
    }
    const double rel_rms = den > 0.0 ? sqrt(num / den) : (num > 0.0 ? INFINITY : 0.0);
    const size_t elem = worst_i / 2;
    const size_t per_row = (size_t)n_baseline * n_freq;

    printf("golden: relative RMS error = %.3e (tolerance %.1e)\n", rel_rms,
           kRelRmsTolerance);
    printf("golden: worst element |delta| = %.6g at row %d baseline %d bin %d "
           "(got %.6g, expected %.6g)\n",
           worst, (int)(elem / per_row), (int)((elem % per_row) / n_freq),
           (int)(elem % n_freq), (double)got[worst_i], (double)expect[worst_i]);

    /* --- structural check: an autocorrelation must be real --- */
    double max_imag = 0.0, max_abs = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < n_baseline; ++b) {
            if (lib_pairs[2 * b] != lib_pairs[2 * b + 1]) continue;
            for (int c = 0; c < n_freq; ++c) {
                const size_t i = (((size_t)r * n_baseline + b) * n_freq + c) * 2;
                if (fabs(got[i + 1]) > max_imag) max_imag = fabs(got[i + 1]);
                if (fabs(got[i]) > max_abs) max_abs = fabs(got[i]);
            }
        }
    }
    printf("golden: autocorrelation max |Re| = %.6g, max |Im| = %.3g\n", max_abs,
           max_imag);

    corr_destroy(ctx);
    free((void *)input);
    free((void *)expect);
    free(got);

    if (max_abs > 0.0 && max_imag > 1e-3 * max_abs) {
        fprintf(stderr,
                "golden: FAILED, autocorrelation has an imaginary part; the "
                "conjugation or the sample layout is wrong\n");
        return 1;
    }
    if (!(rel_rms < kRelRmsTolerance)) {
        fprintf(stderr, "golden: FAILED, relative RMS %.3e exceeds %.1e\n", rel_rms,
                kRelRmsTolerance);
        return 1;
    }
    printf("golden: OK\n");
    return 0;
}
