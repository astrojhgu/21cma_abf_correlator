/* bench.c -- can the GPU pipeline keep up with the front end?
 *
 * The receive path cannot be wired up responsibly without knowing the answer, so
 * this measures it: drive integration rows through the C ABI exactly as the
 * scheduler will, and compare the achieved frame rate against the frame rate the
 * front end produces.
 *
 * The front end delivers one FFT frame every 2*n_freq samples at 800 MHz, i.e.
 * 800e6/(2*n_freq) frames per second, which is the budget this has to beat. With
 * n_freq=4096 that is 97,656.25 frames/s, and one second of integration is
 * 97,656 frames.
 *
 * What is timed is the GPU side only: the staging buffers are filled once up
 * front. Refilling them is the receive threads' job and is measured separately,
 * on the Rust side, where it actually runs in parallel.
 *
 * usage: bench N_STATION N_FREQ FRAMES_PER_SLOT CHUNK_FRAMES ROWS_PER_FETCH
 *              FRAMES_PER_ROW ROWS
 */

#include "corr_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *what) {
    fprintf(stderr, "bench: %s: %s\n", what, corr_last_error());
    exit(1);
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s N_STATION N_FREQ FRAMES_PER_SLOT CHUNK_FRAMES ROWS_PER_FETCH "
                "FRAMES_PER_ROW ROWS\n",
                argv[0]);
        return 2;
    }
    const int n_station = atoi(argv[1]);
    const int n_freq = atoi(argv[2]);
    const int frames_per_slot = atoi(argv[3]);
    const int chunk_frames = atoi(argv[4]);
    const int rows_per_fetch = atoi(argv[5]);
    const int frames_per_row = atoi(argv[6]);
    const int rows = atoi(argv[7]);

    const double frame_rate = 800.0e6 / (2.0 * n_freq);
    const int n_baseline = corr_n_baseline(n_station);

    corr_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = CORR_ABI_VERSION;
    cfg.struct_size = (uint32_t)sizeof(corr_config);
    cfg.device_id = 0;
    cfg.n_station = n_station;
    cfg.n_freq = n_freq;
    cfg.frames_per_slot = frames_per_slot;
    cfg.chunk_frames = chunk_frames;
    cfg.block_slots = 3;
    cfg.rows_per_fetch = rows_per_fetch;

    printf("bench: %d stations -> %d baselines, n_freq=%d\n", n_station, n_baseline,
           n_freq);
    printf("bench: slot=%d frames chunk=%d slots=%d rows_per_fetch=%d, "
           "row=%d frames (%.3f ms)\n",
           frames_per_slot, cfg.chunk_frames, cfg.block_slots, rows_per_fetch,
           frames_per_row, frames_per_row / frame_rate * 1e3);
    printf("bench: front end delivers %.1f frames/s = %.2f GB/s of int8\n", frame_rate,
           n_station * 2.0 * n_freq * frame_rate / 1e9);
    printf("bench: H2D per slot = %.1f MiB, per fetch = %.1f MiB\n",
           (double)n_station * frames_per_slot * 2 * n_freq / 1048576.0,
           (double)n_baseline * n_freq * 8 * rows_per_fetch / 1048576.0);

    void *ctx = NULL;
    if (corr_create(&cfg, &ctx) != CORR_OK) die("corr_create");

    /* Fill the staging once with a deterministic non-trivial pattern. The
     * content does not change the cost of the pipeline, only the geometry does. */
    for (int g = 0; g < n_station; ++g) {
        corr_staging s;
        if (corr_staging_info(ctx, g, &s) != CORR_OK) die("corr_staging_info");
        for (size_t i = 0; i < s.total_bytes; ++i)
            s.base[i] = (int8_t)((i * 31 + (size_t)g * 7) % 9 - 4);
    }

    const size_t fetch_bytes =
        (size_t)n_baseline * n_freq * 8 * (size_t)rows_per_fetch;
    unsigned char *out = malloc(fetch_bytes);
    if (out == NULL) die("malloc");

    double *row_ms = malloc((size_t)rows * sizeof *row_ms);
    if (row_ms == NULL) die("malloc");

    /* Prime the pipeline so the first measured row is not dominated by a cold
     * start: one full slot in flight is enough to get the copy engine going. */
    if (corr_row_accumulate(ctx, 0, frames_per_slot) != CORR_OK)
        die("corr_row_accumulate(prime)");

    long long processed_frames = 0;
    const double t_start = now_s();
    int submits = 1;

    for (int r = 0; r < rows; ++r) {
        const double row_t0 = now_s();

        int done = 0;
        while (done < frames_per_row) {
            const int ft = frames_per_row - done < frames_per_slot
                               ? frames_per_row - done
                               : frames_per_slot;
            /* Release the slot whose physical buffer this sequence reuses.
             *
             * The ring maps sequence to buffer by `sequence % block_slots`, so
             * submitting sequence `s` needs buffer `s % block_slots` free, which
             * means its previous user -- sequence `s - block_slots` -- has been
             * released. Releasing `s - 1` instead (which is what this did) leaks
             * that buffer for ever and the engine stops accepting sequences after
             * `block_slots` of them. The real pipeline releases the oldest slot it
             * holds; this mirrors that. */
            if (submits >= cfg.block_slots) {
                if (corr_row_slot_free(ctx, submits - cfg.block_slots) != CORR_OK)
                    die("corr_row_slot_free");
            }
            if (corr_row_accumulate(ctx, submits, ft) != CORR_OK) die("corr_row_accumulate");
            ++submits;
            done += ft;
            processed_frames += ft;
        }

        if (corr_row_end(ctx, (uint64_t)frames_per_row) != CORR_OK) die("corr_row_end");

        int32_t pending = 0;
        if (corr_rows_pending(ctx, &pending) != CORR_OK) die("corr_rows_pending");
        if (pending >= rows_per_fetch) {
            if (corr_rows_wait(ctx) != CORR_OK) die("corr_rows_wait");
            int32_t got = 0;
            if (corr_rows_fetch(ctx, out, fetch_bytes, &got) != CORR_OK)
                die("corr_rows_fetch");
            if (got != pending) {
                fprintf(stderr, "bench: fetched %d rows, expected %d\n", got, pending);
                return 1;
            }
        }
        row_ms[r] = (now_s() - row_t0) * 1e3;
    }

    /* Drain whatever the last partial fetch batch left behind. */
    if (corr_rows_wait(ctx) != CORR_OK) die("corr_rows_wait(final)");
    int32_t got = 0;
    if (corr_rows_fetch(ctx, out, fetch_bytes, &got) != CORR_OK) die("corr_rows_fetch(final)");
    const double elapsed = now_s() - t_start;

    /* Release the slots still held, then tear down. */
    {
        const int first_held = submits > cfg.block_slots ? submits - cfg.block_slots : 0;
        for (int i = submits - 1; i >= first_held; --i)
            corr_row_slot_free(ctx, i);
    }
    corr_destroy(ctx);

    const double achieved = processed_frames / elapsed;
    double sum = 0.0, worst = 0.0;
    for (int i = 0; i < rows; ++i) {
        sum += row_ms[i];
        if (row_ms[i] > worst) worst = row_ms[i];
    }

    printf("bench: %d rows, %lld frames in %.3f s\n", rows, processed_frames, elapsed);
    printf("bench: achieved %.0f frames/s  (budget %.0f frames/s)\n", achieved,
           frame_rate);
    printf("bench: mean row %.3f ms  worst row %.3f ms  (row budget %.3f ms)\n",
           sum / rows, worst, frames_per_row / frame_rate * 1e3);
    printf("bench: GPU utilisation %.1f%%  (100%% means it exactly keeps up)\n",
           frame_rate / achieved * 100.0);
    printf("bench: H2D sustained %.2f GB/s\n",
           processed_frames * n_station * 2.0 * n_freq / elapsed / 1e9);

    free(row_ms);
    free(out);
    return 0;
}
