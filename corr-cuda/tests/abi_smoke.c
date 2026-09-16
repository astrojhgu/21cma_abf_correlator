/* abi_smoke.c -- exercise libcorr_cuda.so without any Rust involved.
 *
 * This exists so that a failure in the Rust integration test can be attributed:
 * if this test passes, the shared object, its ABI, the CUDA runtime link and the
 * pinned allocation all work, and the problem is on the Rust side.
 *
 * build/run: cmake --build build && ctest --test-dir build --output-on-failure
 */

#include "corr_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);                      \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static const char *status_name(corr_status s) {
    switch (s) {
    case CORR_OK:           return "CORR_OK";
    case CORR_ERR_ARG:      return "CORR_ERR_ARG";
    case CORR_ERR_ABI:      return "CORR_ERR_ABI";
    case CORR_ERR_CUDA:     return "CORR_ERR_CUDA";
    case CORR_ERR_CUFFT:    return "CORR_ERR_CUFFT";
    case CORR_ERR_STATE:    return "CORR_ERR_STATE";
    case CORR_ERR_BUSY:     return "CORR_ERR_BUSY";
    case CORR_ERR_NOMEM:    return "CORR_ERR_NOMEM";
    case CORR_ERR_NOT_IMPL: return "CORR_ERR_NOT_IMPL";
    case CORR_ERR_INTERNAL: return "CORR_ERR_INTERNAL";
    }
    return "?";
}

/* Mirrors the shipped correlator.yaml at 4 stations, with the buffers shrunk so
 * the pinned allocation stays small in a test. Two rows per fetch so the row
 * stack is exercised rather than trivially bypassed. */
static corr_config make_config(void) {
    corr_config c;
    memset(&c, 0, sizeof c);
    c.abi_version     = CORR_ABI_VERSION;
    c.struct_size     = (uint32_t)sizeof(corr_config);
    c.device_id       = 0;
    c.n_station       = 4;
    c.n_freq          = 4096;
    c.frames_per_slot = 16;
    c.chunk_frames    = 8;
    c.block_slots     = 3;
    c.rows_per_fetch  = 2;
    return c;
}

static void test_banner_and_errors(void) {
    printf("banner / error reporting\n");
    const char *b = corr_abi_banner();
    CHECK(b != NULL && strlen(b) > 0, "banner is empty");
    printf("  banner: %s\n", b);

    CHECK(corr_last_error() != NULL, "corr_last_error returned NULL");
    CHECK(corr_last_error()[0] == '\0', "error string should start empty");

    /* A failing call must leave a message behind for the caller to print. */
    corr_status s = corr_create(NULL, NULL);
    CHECK(s == CORR_ERR_ARG, "expected CORR_ERR_ARG, got %s", status_name(s));
    CHECK(strlen(corr_last_error()) > 0, "failure did not set an error message");
}

static void test_baseline_numbering(void) {
    printf("baseline numbering\n");
    CHECK(corr_n_baseline(0) == -1, "n_station=0 should return -1");
    CHECK(corr_n_baseline(1) == 1, "n_station=1 should give 1 baseline");
    CHECK(corr_n_baseline(4) == 10, "n_station=4 should give 10 baselines");
    CHECK(corr_n_baseline(15) == 120, "n_station=15 should give 120 baselines");
    CHECK(corr_n_baseline(24) == 300, "n_station=24 should give 300 baselines");

    int32_t pairs[2 * 10];
    corr_status s = corr_baseline_pairs(4, pairs, 2 * 10);
    CHECK(s == CORR_OK, "corr_baseline_pairs failed: %s", status_name(s));

    /* j <= k, row major: (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3) */
    static const int32_t expect[20] = {0, 0, 0, 1, 0, 2, 0, 3, 1, 1,
                                       1, 2, 1, 3, 2, 2, 2, 3, 3, 3};
    CHECK(memcmp(pairs, expect, sizeof expect) == 0, "baseline order is wrong");
    for (int i = 0; i < 10; ++i)
        CHECK(pairs[2 * i] <= pairs[2 * i + 1], "baseline %d violates j <= k", i);

    /* Wrong out_len must be rejected rather than silently truncated. */
    s = corr_baseline_pairs(4, pairs, 2 * 10 - 1);
    CHECK(s == CORR_ERR_ARG, "short out_len should be CORR_ERR_ARG, got %s",
          status_name(s));
}

static void test_config_rejection(void) {
    printf("config validation\n");
    void *ctx = (void *)0x1;

    corr_config c = make_config();
    c.abi_version = CORR_ABI_VERSION + 1;
    corr_status s = corr_create(&c, &ctx);
    CHECK(s == CORR_ERR_ABI, "bad abi_version should be CORR_ERR_ABI, got %s",
          status_name(s));
    CHECK(ctx == NULL, "out_ctx must be NULL after failure");

    c = make_config();
    c.n_freq = 4096 + 2048 / 2; /* not a multiple of 2048 */
    s = corr_create(&c, &ctx);
    CHECK(s == CORR_ERR_ARG, "non multiple-of-2048 n_freq should be rejected, got %s",
          status_name(s));

    c = make_config();
    c.block_slots = 1;
    s = corr_create(&c, &ctx);
    CHECK(s == CORR_ERR_ARG, "block_slots=1 should be rejected, got %s", status_name(s));

    c = make_config();
    c.chunk_frames = c.frames_per_slot + 1;
    s = corr_create(&c, &ctx);
    CHECK(s == CORR_ERR_ARG, "chunk_frames > frames_per_slot should be rejected, got %s",
          status_name(s));

    c = make_config();
    c.rows_per_fetch = 0;
    s = corr_create(&c, &ctx);
    CHECK(s == CORR_ERR_ARG, "rows_per_fetch=0 should be rejected, got %s",
          status_name(s));

    c = make_config();
    c.device_id = 64;
    s = corr_create(&c, &ctx);
    CHECK(s == CORR_ERR_ARG, "out-of-range device_id should be rejected, got %s",
          status_name(s));
}

/* Mark every frame of `slot` valid in every station, which is what a receive
 * thread that received everything writes. Without it the device zeroes the whole
 * slot (a clear bit means "this frame did not arrive"), and any check of the
 * samples would pass for the wrong reason. */
static void mark_all_valid(void *ctx, const corr_config *c, int32_t slot) {
    for (int32_t g = 0; g < c->n_station; ++g) {
        corr_staging st;
        if (corr_staging_info(ctx, g, &st) != CORR_OK) return;
        unsigned char *mask = (unsigned char *)st.base + st.validity_offset +
                              (size_t)slot * st.validity_bytes_per_slot;
        memset(mask, 0xFF, st.validity_bytes_per_slot);
    }
}

static void test_staging_layout(void) {
    printf("staging layout\n");
    corr_config c = make_config();
    void *ctx = NULL;
    corr_status s = corr_create(&c, &ctx);
    if (s != CORR_OK) {
        printf("  FAIL corr_create: %s (%s)\n", status_name(s), corr_last_error());
        ++g_failures;
        return;
    }

    const size_t frame_stride = (size_t)2 * c.n_freq;
    const size_t slot_stride = frame_stride * (size_t)c.frames_per_slot;
    const size_t samples_stride = slot_stride * (size_t)c.block_slots;
    /* One bit per frame of a slot, rounded up to whole bytes, appended after the
     * samples. It is part of the station's extent: the device reads this to decide
     * which frames are silence, so a caller that sizes its own buffer from
     * total_bytes minus the mask would be handing the library a buffer it will
     * read past. */
    const size_t mask_bytes_per_slot = ((size_t)c.frames_per_slot + 7) / 8;
    const size_t mask_stride = mask_bytes_per_slot * (size_t)c.block_slots;
    const size_t station_stride = samples_stride + mask_stride;

    for (int32_t g = 0; g < c.n_station; ++g) {
        corr_staging st;
        s = corr_staging_info(ctx, g, &st);
        CHECK(s == CORR_OK, "corr_staging_info(%d) failed: %s", g, status_name(s));
        CHECK(st.frame_stride == frame_stride, "frame_stride %zu != %zu",
              st.frame_stride, frame_stride);
        CHECK(st.slot_stride == slot_stride, "slot_stride %zu != %zu",
              st.slot_stride, slot_stride);
        CHECK(st.validity_bytes_per_slot == mask_bytes_per_slot,
              "validity_bytes_per_slot %zu != %zu", st.validity_bytes_per_slot,
              mask_bytes_per_slot);
        CHECK(st.validity_offset == samples_stride,
              "validity_offset %zu should follow the samples at %zu",
              st.validity_offset, samples_stride);
        /* base is already offset to this station, so total_bytes is this
         * station's extent, not the size of the whole allocation. Getting this
         * wrong is how a caller walks off the end of the buffer. */
        CHECK(st.total_bytes == station_stride,
              "total_bytes %zu should be station_stride %zu", st.total_bytes,
              station_stride);
        CHECK(st.base != NULL, "station %d has a NULL base", g);
        if (g > 0) {
            corr_staging prev;
            corr_staging_info(ctx, g - 1, &prev);
            CHECK(st.base == prev.base + (ptrdiff_t)station_stride,
                  "station %d base is not previous+station_stride", g);
        }
    }

    corr_staging st;
    s = corr_staging_info(ctx, c.n_station, &st);
    CHECK(s == CORR_ERR_ARG, "station out of range should be CORR_ERR_ARG, got %s",
          status_name(s));

    /* The pinned region must really be writable by plain stores: this is the
     * exact access pattern the receive threads use. */
    corr_staging s0;
    corr_staging_info(ctx, 0, &s0);
    const size_t off = (size_t)2 * slot_stride + (size_t)7 * frame_stride + 4096;
    s0.base[off] = (int8_t)-128;
    s0.base[off + 1] = (int8_t)127;
    CHECK(s0.base[off] == (int8_t)-128, "pinned write of -128 did not stick");
    CHECK(s0.base[off + 1] == (int8_t)127, "pinned write of 127 did not stick");
    /* Put them back: a later section submits this staging and requires an
     * all-zero spectrum, which is only meaningful if the input is all zero. */
    s0.base[off] = 0;
    s0.base[off + 1] = 0;

    int nonzero = 0;
    for (size_t i = 0; i < s0.total_bytes; i += 997)
        if (s0.base[i] != 0) ++nonzero;
    CHECK(nonzero == 0, "staging was not zero initialised (%d nonzero samples)", nonzero);

    /* --- the validity mask's lifecycle ---
     *
     * It starts all-zero ("nothing present"), the receive thread sets a bit per
     * frame that arrived whole, and corr_row_slot_free clears the whole slot as it
     * hands it back. That last clear is what keeps a station that sends nothing at
     * all for a whole slot from inheriting the previous lap's bits, which the
     * device would then read as real data. */
    {
        unsigned char *mask = (unsigned char *)s0.base + s0.validity_offset;
        size_t set = 0;
        for (size_t i = 0; i < mask_stride; ++i)
            if (mask[i] != 0) ++set;
        CHECK(set == 0, "the validity mask should start all-zero, %zu bytes are set", set);

        memset(mask, 0xAA, mask_bytes_per_slot);
        s = corr_row_accumulate(ctx, 0, c.frames_per_slot);
        CHECK(s == CORR_OK, "accumulating a marked slot failed: %s", corr_last_error());
        s = corr_row_slot_free(ctx, 0);
        CHECK(s == CORR_OK, "slot_free failed: %s", corr_last_error());
        set = 0;
        for (size_t i = 0; i < mask_stride; ++i)
            if (mask[i] != 0) ++set;
        CHECK(set == 0,
              "slot_free must hand the slot back with an empty mask, %zu bytes are set",
              set);
    }

    /* --- row lifecycle --- */
    const size_t row_bytes = (size_t)corr_n_baseline(c.n_station) * (size_t)c.n_freq * 8;
    void *out = malloc(row_bytes * (size_t)c.rows_per_fetch);
    CHECK(out != NULL, "cannot allocate the spectrum buffer");

    /* Out-of-order use must be reported, not passed through. */
    s = corr_row_slot_free(ctx, 1);
    CHECK(s == CORR_ERR_STATE, "releasing an unsubmitted slot should be CORR_ERR_STATE, got %s",
          status_name(s));

    /* A negative sequence is a caller bug, not a slot that happens to be busy. */
    s = corr_row_accumulate(ctx, -1, 1);
    CHECK(s == CORR_ERR_ARG, "a negative slot sequence should be CORR_ERR_ARG, got %s",
          status_name(s));

    s = corr_rows_fetch(ctx, out, row_bytes * (size_t)c.rows_per_fetch, NULL);
    CHECK(s == CORR_ERR_ARG, "NULL out_rows should be CORR_ERR_ARG, got %s", status_name(s));

    int32_t rows = -1;
    s = corr_rows_fetch(ctx, out, 8, &rows);
    CHECK(s == CORR_ERR_ARG, "too-small output should be CORR_ERR_ARG, got %s",
          status_name(s));

    s = corr_rows_pending(ctx, &rows);
    CHECK(s == CORR_OK && rows == 0, "a fresh context should have 0 pending rows");

    s = corr_rows_fetch(ctx, out, row_bytes * (size_t)c.rows_per_fetch, &rows);
    CHECK(s == CORR_OK && rows == 0, "fetching with nothing pending should succeed and report 0");

    s = corr_row_accumulate(ctx, 0, 0);
    CHECK(s == CORR_ERR_ARG, "frames=0 should be CORR_ERR_ARG, got %s", status_name(s));

    s = corr_row_end(ctx, 0);
    CHECK(s == CORR_ERR_ARG, "row_end(0) should be CORR_ERR_ARG, got %s", status_name(s));

    /* Staging is all zeros, so a real row must come out all zeros: anything
     * else means uninitialised device memory reached the output. The mask is set
     * all-valid first so that the samples are actually correlated -- otherwise the
     * device would zero the whole slot and this would pass without reading a
     * sample. */
    for (int32_t i = 0; i < c.block_slots; ++i) {
        const int64_t seq = i;
        const int32_t slot = (int32_t)(seq % c.block_slots);
        mark_all_valid(ctx, &c, slot);
        s = corr_row_accumulate(ctx, seq, c.frames_per_slot);
        CHECK(s == CORR_OK, "accumulate(%lld) failed: %s", (long long)seq, corr_last_error());

        s = corr_row_accumulate(ctx, seq, c.frames_per_slot);
        CHECK(s == CORR_ERR_BUSY, "double accumulate should be CORR_ERR_BUSY, got %s",
              status_name(s));

        s = corr_row_slot_free(ctx, seq);
        CHECK(s == CORR_OK, "slot_free(%lld) failed: %s", (long long)seq, corr_last_error());

        s = corr_row_slot_free(ctx, seq);
        CHECK(s == CORR_ERR_STATE, "second slot_free should be CORR_ERR_STATE, got %s",
              status_name(s));

        if (slot % 2 == 1) {
            s = corr_row_end(ctx, c.frames_per_slot);
            CHECK(s == CORR_OK, "row_end failed: %s", corr_last_error());
        }
    }

    s = corr_rows_pending(ctx, &rows);
    CHECK(s == CORR_OK && rows == 1, "expected 1 pending row, got %d", rows);

    /* The stack is rows_per_fetch deep; filling it without fetching must be
     * refused rather than silently overwriting a row. */
    s = corr_row_end(ctx, c.frames_per_slot);
    CHECK(s == CORR_OK, "second row_end failed: %s", corr_last_error());
    s = corr_row_end(ctx, c.frames_per_slot);
    CHECK(s == CORR_ERR_BUSY,
          "a third row_end with rows_per_fetch=2 should be CORR_ERR_BUSY, got %s",
          status_name(s));

    s = corr_rows_wait(ctx);
    CHECK(s == CORR_OK, "rows_wait failed: %s", corr_last_error());

    s = corr_rows_fetch(ctx, out, row_bytes * (size_t)c.rows_per_fetch, &rows);
    CHECK(s == CORR_OK, "rows_fetch failed: %s", corr_last_error());
    CHECK(rows == 2, "expected 2 rows fetched, got %d", rows);

    {
        const float *f = (const float *)out;
        const size_t n = row_bytes * 2 / sizeof(float);
        size_t nz = 0;
        for (size_t i = 0; i < n; ++i)
            if (f[i] != 0.0f) ++nz;
        CHECK(nz == 0, "zero input produced %zu nonzero output values", nz);
    }

    /* Fetching clears the rows, so the pipeline can cycle. */
    for (int32_t i = 0; i < c.block_slots; ++i) {
        const int64_t seq = c.block_slots + i; /* carry on where the block above stopped */
        mark_all_valid(ctx, &c, (int32_t)(seq % c.block_slots));
        CHECK(corr_row_accumulate(ctx, seq, c.frames_per_slot) == CORR_OK,
              "reuse accumulate failed");
        CHECK(corr_row_slot_free(ctx, seq) == CORR_OK, "reuse slot_free failed");
    }
    CHECK(corr_row_end(ctx, c.frames_per_slot) == CORR_OK, "reuse row_end failed");
    CHECK(corr_rows_wait(ctx) == CORR_OK, "reuse rows_wait failed");
    CHECK(corr_rows_fetch(ctx, out, row_bytes * (size_t)c.rows_per_fetch, &rows) == CORR_OK,
          "reuse rows_fetch failed");
    CHECK(rows == 1, "reuse should have fetched 1 row, got %d", rows);

    free(out);

    s = corr_destroy(ctx);
    CHECK(s == CORR_OK, "corr_destroy failed: %s", status_name(s));
    CHECK(corr_destroy(NULL) == CORR_OK, "corr_destroy(NULL) must be a no-op");
}

int main(void) {
    printf("corr_cuda ABI smoke test\n");
    test_banner_and_errors();
    test_baseline_numbering();
    test_config_rejection();
    test_staging_layout();

    if (g_failures != 0) {
        printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
