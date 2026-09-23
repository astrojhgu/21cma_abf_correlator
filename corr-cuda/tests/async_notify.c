/* async_notify.c -- prove the non-blocking half of the ABI, without Rust.
 *
 * The claim under test is not that the numbers are right (that is the golden
 * test) but the shape a task-based caller is built on:
 *
 *   1. corr_row_slot_free_submit() returns without waiting for the copy, while
 *      the blocking call on the same work does wait;
 *   2. the callback fires once the copy is done, on a thread this library does
 *      not own;
 *   3. nothing is wedged while it is pending -- other CUDA calls still work;
 *   4. corr_row_slot_free_finish() completes the release, mask and all;
 *   5. the counting is balanced: one callback per submit no matter how many
 *      devices read the slot, and N submits deliver exactly N callbacks.
 *
 * What it deliberately does NOT assert is that the callback arrives after
 * submit() returned. A copy can finish first, and the implementation is required
 * to cope with that -- asserting the other order would be asserting a race.
 *
 * build/run: cmake --build build && ctest --test-dir build --output-on-failure
 */

#include "corr_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* Written from CUDA's callback thread, read from this one. */
static int g_callbacks = 0;
static void on_done(void *user) {
    (void)user;
    /* Deliberately nothing else: this runs on a thread the library does not own,
     * and the whole contract is that it only signals. */
    __atomic_fetch_add(&g_callbacks, 1, __ATOMIC_SEQ_CST);
}

static int callbacks(void) { return __atomic_load_n(&g_callbacks, __ATOMIC_SEQ_CST); }

/* Wait for `want` callbacks, keeping the library busy from this thread while we
 * wait: that is the point of the exercise, and it is also how a wedged library
 * shows up as a failure rather than as a hang. */
static void *g_ctx = NULL;
static int wait_for(int want, double seconds) {
    const uint64_t deadline = now_ns() + (uint64_t)(seconds * 1e9);
    while (now_ns() < deadline) {
        if (callbacks() >= want) return 0;
        int32_t pending = -1;
        if (corr_rows_pending(g_ctx, &pending) != CORR_OK) return -1;
        usleep(50);
    }
    return -1;
}

static void mark_all_valid(void *ctx, const corr_config *c, int32_t slot) {
    for (int32_t g = 0; g < c->n_station; ++g) {
        corr_staging st;
        if (corr_staging_info(ctx, g, &st) != CORR_OK) return;
        unsigned char *mask = (unsigned char *)st.base + st.validity_offset +
                              (size_t)slot * st.validity_bytes_per_slot;
        memset(mask, 0xFF, st.validity_bytes_per_slot);
    }
}

/* Big enough that a host-to-device copy is measurable, small enough that the
 * pinned staging stays modest. One slot is n_station * frames_per_slot * 2*n_freq
 * = 2 * 256 * 16384 = 8 MiB, so the ring is 24 MiB. */
static corr_config make_config(int n_devices) {
    corr_config c;
    memset(&c, 0, sizeof c);
    c.abi_version     = CORR_ABI_VERSION;
    c.struct_size     = (uint32_t)sizeof(corr_config);
    c.device_id       = 0;
    c.n_station       = 2;
    c.n_freq          = 8192;
    c.frames_per_slot = 256;
    c.chunk_frames    = 64;
    c.block_slots     = 3;
    c.rows_per_fetch  = 1;
    c.split_mode      = CORR_SPLIT_BASELINE_RANGE;
    if (n_devices > 1) {
        c.n_devices = n_devices;
        for (int i = 0; i < n_devices; ++i) c.devices[i] = i;
    }
    return c;
}

/* The blocking call, for the comparison in step 1. */
static uint64_t time_blocking_release(void *ctx, const corr_config *c, int64_t seq) {
    mark_all_valid(ctx, c, (int32_t)(seq % c->block_slots));
    if (corr_row_accumulate(ctx, seq, c->frames_per_slot) != CORR_OK) return 0;
    const uint64_t t0 = now_ns();
    if (corr_row_slot_free(ctx, seq) != CORR_OK) return 0;
    return now_ns() - t0;
}

static void test_submit_does_not_wait(const char *what, int n_devices) {
    printf("submit / finish, %s\n", what);
    corr_config c = make_config(n_devices);
    void *ctx = NULL;
    corr_status s = corr_create(&c, &ctx);
    if (s != CORR_OK) {
        printf("  FAIL corr_create: %s (%s)\n", status_name(s), corr_last_error());
        ++g_failures;
        return;
    }
    g_ctx = ctx;

    /* Warm the path up: the first accumulate pays for the cuFFT plan's first
     * use and for page mapping, and neither belongs in a timing comparison. */
    time_blocking_release(ctx, &c, 0);

    /* --- 1. the blocking call waits, the submit does not ------------------- */
    uint64_t block_ns = 0;
    for (int i = 1; i <= 5; ++i) block_ns += time_blocking_release(ctx, &c, i);

    uint64_t submit_ns = 0;
    int64_t seq = 10;
    for (int i = 0; i < 5; ++i, ++seq) {
        mark_all_valid(ctx, &c, (int32_t)(seq % c.block_slots));
        if (corr_row_accumulate(ctx, seq, c.frames_per_slot) != CORR_OK) {
            CHECK(0, "accumulate failed: %s", corr_last_error());
            break;
        }
        __atomic_store_n(&g_callbacks, 0, __ATOMIC_SEQ_CST);
        const uint64_t t0 = now_ns();
        s = corr_row_slot_free_submit(ctx, seq, on_done, NULL);
        submit_ns += now_ns() - t0;
        CHECK(s == CORR_OK, "slot_free_submit failed: %s", corr_last_error());
        if (s != CORR_OK) break;
        /* 2 and 3: the callback arrives, and the library is not wedged while it
         * is pending -- `wait_for` calls corr_rows_pending() on the way. */
        CHECK(wait_for(1, 5.0) == 0, "no callback within 5 s for sequence %lld",
              (long long)seq);
        CHECK(callbacks() == 1, "expected exactly 1 callback, got %d", callbacks());
        /* 4: and the release is only complete after finish. */
        s = corr_row_slot_free_finish(ctx, seq);
        CHECK(s == CORR_OK, "slot_free_finish failed: %s", corr_last_error());
    }
    printf("  blocking release %llu ns/5, submit %llu ns/5 (%s)\n",
           (unsigned long long)block_ns, (unsigned long long)submit_ns,
           submit_ns * 4 < block_ns ? "submit is far cheaper" : "TOO CLOSE TO CALL");
    CHECK(submit_ns * 4 < block_ns,
          "submit (%llu ns) should be far cheaper than the blocking release (%llu ns)",
          (unsigned long long)submit_ns, (unsigned long long)block_ns);

    /* --- 4b. the release really happened: the mask is clear, slot is free --- */
    corr_staging st;
    corr_staging_info(ctx, 0, &st);
    uint64_t set = 0;
    for (int32_t slot = 0; slot < c.block_slots; ++slot) {
        const unsigned char *mask =
            (const unsigned char *)st.base + st.validity_offset +
            (size_t)slot * st.validity_bytes_per_slot;
        for (size_t i = 0; i < st.validity_bytes_per_slot; ++i)
            if (mask[i] != 0) ++set;
    }
    CHECK(set == 0, "%llu validity bytes are still set after the releases",
          (unsigned long long)set);

    s = corr_row_slot_free_finish(ctx, seq - 1);
    CHECK(s == CORR_ERR_STATE,
          "finishing an already-released slot should be CORR_ERR_STATE, got %s",
          status_name(s));

    /* --- 5. balanced counting ---------------------------------------------
     *
     * Two shapes, because they exercise different things. First as many
     * submissions outstanding at once as the ring holds, which is the only way
     * the notify stream has a queue to serialise rather than one callback at a
     * time. Then many in turn, each fully released before the next, which is
     * what a run actually does. */
    __atomic_store_n(&g_callbacks, 0, __ATOMIC_SEQ_CST);
    int64_t first = seq;
    for (int i = 0; i < c.block_slots; ++i, ++seq) {
        mark_all_valid(ctx, &c, (int32_t)(seq % c.block_slots));
        CHECK(corr_row_accumulate(ctx, seq, c.frames_per_slot) == CORR_OK,
              "accumulate %lld failed: %s", (long long)seq, corr_last_error());
        CHECK(corr_row_slot_free_submit(ctx, seq, on_done, NULL) == CORR_OK,
              "submit %lld failed: %s", (long long)seq, corr_last_error());
    }
    CHECK(wait_for(c.block_slots, 10.0) == 0, "%d of %d callbacks arrived",
          callbacks(), c.block_slots);
    CHECK(callbacks() == c.block_slots, "expected %d callbacks, got %d",
          c.block_slots, callbacks());
    for (int i = 0; i < c.block_slots; ++i)
        CHECK(corr_row_slot_free_finish(ctx, first + i) == CORR_OK,
              "finish %lld failed: %s", (long long)(first + i), corr_last_error());
    printf("  %d submissions outstanding at once -> %d callbacks\n", c.block_slots,
           callbacks());

    const int many = 60;
    for (int i = 0; i < many; ++i, ++seq) {
        mark_all_valid(ctx, &c, (int32_t)(seq % c.block_slots));
        if (corr_row_accumulate(ctx, seq, c.frames_per_slot) != CORR_OK) {
            CHECK(0, "accumulate %lld failed: %s", (long long)seq, corr_last_error());
            break;
        }
        if (corr_row_slot_free_submit(ctx, seq, on_done, NULL) != CORR_OK) {
            CHECK(0, "submit %lld failed: %s", (long long)seq, corr_last_error());
            break;
        }
        if (wait_for(c.block_slots + i + 1, 5.0) != 0) {
            CHECK(0, "no callback for sequence %lld", (long long)seq);
            break;
        }
        if (corr_row_slot_free_finish(ctx, seq) != CORR_OK) {
            CHECK(0, "finish %lld failed: %s", (long long)seq, corr_last_error());
            break;
        }
    }
    CHECK(callbacks() == c.block_slots + many, "expected %d callbacks, got %d",
          c.block_slots + many, callbacks());
    printf("  then %d cycles, each released before the next, %d device(s) each\n",
           many, n_devices);

    s = corr_destroy(ctx);
    CHECK(s == CORR_OK, "corr_destroy failed: %s", status_name(s));
    g_ctx = NULL;
}

static void test_rows_wait(const char *what, int n_devices) {
    printf("rows wait without waiting, %s\n", what);
    corr_config c = make_config(n_devices);
    void *ctx = NULL;
    corr_status s = corr_create(&c, &ctx);
    if (s != CORR_OK) {
        printf("  FAIL corr_create: %s (%s)\n", status_name(s), corr_last_error());
        ++g_failures;
        return;
    }
    g_ctx = ctx;

    for (int i = 0; i < c.block_slots; ++i) {
        mark_all_valid(ctx, &c, i);
        CHECK(corr_row_accumulate(ctx, i, c.frames_per_slot) == CORR_OK,
              "accumulate failed: %s", corr_last_error());
        CHECK(corr_row_slot_free(ctx, i) == CORR_OK, "slot_free failed: %s",
              corr_last_error());
    }
    CHECK(corr_row_end(ctx, c.frames_per_slot) == CORR_OK, "row_end failed: %s",
          corr_last_error());

    __atomic_store_n(&g_callbacks, 0, __ATOMIC_SEQ_CST);
    s = corr_rows_wait_submit(ctx, on_done, NULL);
    CHECK(s == CORR_OK, "rows_wait_submit failed: %s", corr_last_error());
    CHECK(wait_for(1, 10.0) == 0, "no rows-wait callback within 10 s");
    CHECK(callbacks() == 1, "expected exactly 1 rows-wait callback, got %d", callbacks());

    /* The callback cannot report a failure, so the caller has to ask. */
    s = corr_rows_ready(ctx);
    CHECK(s == CORR_OK, "rows_ready after the callback: %s", status_name(s));

    /* And the row is really there to fetch. */
    const size_t row_bytes = (size_t)corr_n_baseline(c.n_station) * (size_t)c.n_freq * 8;
    void *out = malloc(row_bytes * (size_t)c.rows_per_fetch);
    CHECK(out != NULL, "cannot allocate the spectrum buffer");
    if (out != NULL) {
        int32_t rows = -1;
        s = corr_rows_fetch(ctx, out, row_bytes * (size_t)c.rows_per_fetch, &rows);
        CHECK(s == CORR_OK && rows == 1, "rows_fetch after the wait: %s, %d rows",
              status_name(s), rows);
        free(out);
    }

    CHECK(corr_destroy(ctx) == CORR_OK, "corr_destroy failed");
    g_ctx = NULL;
}

/* Fill every station's staging with a deterministic non-trivial pattern, and
 * mark every frame valid. A fetch of all-zero input would agree with itself no
 * matter what it did. */
static void fill_pattern(void *ctx, const corr_config *c) {
    for (int32_t g = 0; g < c->n_station; ++g) {
        corr_staging st;
        if (corr_staging_info(ctx, g, &st) != CORR_OK) return;
        for (size_t i = 0; i < st.validity_offset; ++i)
            st.base[i] = (int8_t)((i * 31 + (size_t)g * 7) % 9 - 4);
        memset(st.base + st.validity_offset, 0xFF, st.total_bytes - st.validity_offset);
    }
}

/* One row's worth of work: every slot accumulated and released, then closed. */
static corr_status accumulate_one_row(void *ctx, const corr_config *c, int64_t first_seq) {
    for (int32_t i = 0; i < c->block_slots; ++i) {
        const int64_t seq = first_seq + i;
        /* Re-mark every lap: releasing a slot clears its validity mask, and an
         * unmarked slot is silence. Without this both rows would be all zeros and
         * the comparison below would pass for the wrong reason. */
        mark_all_valid(ctx, c, (int32_t)(seq % c->block_slots));
        corr_status s = corr_row_accumulate(ctx, seq, c->frames_per_slot);
        if (s != CORR_OK) return s;
        s = corr_row_slot_free(ctx, seq);
        if (s != CORR_OK) return s;
    }
    return corr_row_end(ctx, (uint64_t)c->frames_per_slot * (uint64_t)c->block_slots);
}

/* The claim: the fetch can hand the row back without the caller waiting for it,
 * and the bytes it produces are the bytes the blocking call produces. */
static void test_fetch(const char *what, int n_devices, int split) {
    printf("fetch without waiting, %s\n", what);
    corr_config c = make_config(n_devices);
    c.split_mode = split;
    void *ctx = NULL;
    corr_status s = corr_create(&c, &ctx);
    if (s != CORR_OK) {
        printf("  FAIL corr_create: %s (%s)\n", status_name(s), corr_last_error());
        ++g_failures;
        return;
    }
    g_ctx = ctx;
    fill_pattern(ctx, &c);

    const size_t row_bytes = (size_t)corr_n_baseline(c.n_station) * (size_t)c.n_freq * 8;
    const size_t bytes = row_bytes * (size_t)c.rows_per_fetch;
    void *out = malloc(bytes);
    void *ref = malloc(bytes);
    CHECK(out != NULL && ref != NULL, "cannot allocate the spectrum buffers");
    if (out == NULL || ref == NULL) { free(out); free(ref); corr_destroy(ctx); return; }

    /* The blocking call first, as the reference for what the row should be. */
    int32_t ref_rows = -1;
    CHECK(accumulate_one_row(ctx, &c, 0) == CORR_OK, "accumulate failed: %s",
          corr_last_error());
    s = corr_rows_fetch(ctx, ref, bytes, &ref_rows);
    CHECK(s == CORR_OK, "the blocking fetch failed: %s", corr_last_error());
    CHECK(ref_rows == c.rows_per_fetch, "the blocking fetch returned %d rows, expected %d",
          ref_rows, c.rows_per_fetch);

    /* The same work again, fetched without waiting for it. */
    CHECK(accumulate_one_row(ctx, &c, c.block_slots) == CORR_OK, "accumulate failed: %s",
          corr_last_error());
    memset(out, 0xA5, bytes);
    __atomic_store_n(&g_callbacks, 0, __ATOMIC_SEQ_CST);
    int32_t rows = -1;
    s = corr_rows_fetch_submit(ctx, out, bytes, &rows, on_done, NULL);
    CHECK(s == CORR_OK, "rows_fetch_submit failed: %s", corr_last_error());
    /* The count is knowable before the copy starts, so a caller can size whatever
     * depends on it while the transfer is in flight. */
    CHECK(rows == c.rows_per_fetch, "out_rows must be filled at submit: got %d, expected %d",
          rows, c.rows_per_fetch);
    CHECK(wait_for(1, 10.0) == 0, "no fetch callback within 10 s");
    CHECK(callbacks() == 1, "expected exactly 1 fetch callback, got %d", callbacks());
    s = corr_rows_fetch_ready(ctx);
    CHECK(s == CORR_OK, "rows_fetch_ready after the callback: %s", status_name(s));

    /* And a third round through the blocking call, because "the row fetched
     * without waiting is the row" is a claim about the *path*, not about two
     * laps agreeing with each other. Comparing against a fresh blocking fetch of
     * an equivalent row tests the path; comparing against lap 1 would also be
     * testing that two laps produce identical spectra, which is a different
     * question and not this test's. */
    void *ref2 = malloc(bytes);
    CHECK(ref2 != NULL, "cannot allocate the third buffer");
    int32_t ref2_rows = -1;
    if (ref2 != NULL) {
        CHECK(accumulate_one_row(ctx, &c, 2 * c.block_slots) == CORR_OK,
              "accumulate failed: %s", corr_last_error());
        s = corr_rows_fetch(ctx, ref2, bytes, &ref2_rows);
        CHECK(s == CORR_OK && ref2_rows == c.rows_per_fetch,
              "the third fetch failed: %s", corr_last_error());
    }

    /* Compare as samples, not as bytes.
     *
     * Under the time-slice split two laps are not byte-identical by construction:
     * which device a frame is grouped with changes from lap to lap, and summing
     * the same frames in a different order rounds differently. That is a property
     * of the split, not of the path under test -- multi_device makes exactly this
     * comparison and allows exactly this difference. What this test is about is
     * whether fetching without waiting hands back the row that fetching with
     * waiting does, so the tolerance is the same one, and the first pair below is
     * printed to show the size of the difference the split alone accounts for. */
    {
        const float *f1 = (const float *)ref, *f3 = (const float *)ref2,
                    *fa = (const float *)out;
        const size_t n = bytes / sizeof(float);
        double lap = 0, path = 0, scale = 0;
        for (size_t i = 0; i < n; ++i) {
            double a = f1[i] < 0 ? -f1[i] : f1[i];
            double d_lap = f3[i] - f1[i], d_path = fa[i] - f3[i];
            if (d_lap < 0) d_lap = -d_lap;
            if (d_path < 0) d_path = -d_path;
            if (d_lap > lap) lap = d_lap;
            if (d_path > path) path = d_path;
            if (a > scale) scale = a;
        }
        /* Two blocking laps are the baseline for what this split can be asked to
         * reproduce. Under the time-slice split they are not bit-identical: which
         * device a frame is grouped with changes from lap to lap, and the row is
         * the sum of the partials. Whatever that costs in reproducibility, the
         * path under test may not cost more -- and this pair of numbers says
         * whether it does. Under a split with one device, or one where each device
         * owns its own baselines, the baseline is zero and so is the tolerance. */
        printf("    two blocking laps differ by at most %.3e (this split's own spread)\n",
               lap);
        printf("    without waiting vs with waiting differ by at most %.3e\n", path);
        if (lap == 0.0) {
            CHECK(path == 0.0,
                  "the two fetches disagree although two blocking laps do not: %.3e",
                  path);
        } else {
            CHECK(path <= lap,
                  "fetching without waiting differs from fetching with waiting by "
                  "%.3e, more than two blocking laps differ from each other (%.3e)",
                  path, lap);
        }
        CHECK(scale > 0.0, "the row is all zeros, so this proved nothing");
    }
    free(ref2);

    free(out);
    free(ref);
    CHECK(corr_destroy(ctx) == CORR_OK, "corr_destroy failed");
    g_ctx = NULL;
}

/* The pair-covering split places baselines into the row on the host after the
 * transfer, so there is no completion to signal. It has to say so rather than
 * hand back a row that is still being assembled. */
static void test_scattered_fetch_is_refused(void) {
    printf("fetch without waiting, pair covering\n");
    corr_config c = make_config(2);
    c.split_mode = CORR_SPLIT_PAIR_COVERING;
    void *ctx = NULL;
    if (corr_create(&c, &ctx) != CORR_OK) {
        printf("  FAIL corr_create: %s\n", corr_last_error());
        ++g_failures;
        return;
    }
    const size_t row_bytes = (size_t)corr_n_baseline(c.n_station) * (size_t)c.n_freq * 8;
    void *out = malloc(row_bytes);
    int32_t rows = -1;
    corr_status s = corr_rows_fetch_submit(ctx, out, row_bytes, &rows, on_done, NULL);
    CHECK(s == CORR_ERR_NOT_IMPL,
          "the scattered fetch should be refused with CORR_ERR_NOT_IMPL, got %s",
          status_name(s));
    free(out);
    CHECK(corr_destroy(ctx) == CORR_OK, "corr_destroy failed");
}

static void test_argument_checking(void) {
    printf("argument checking\n");
    corr_config c = make_config(1);
    void *ctx = NULL;
    if (corr_create(&c, &ctx) != CORR_OK) {
        printf("  FAIL corr_create: %s\n", corr_last_error());
        ++g_failures;
        return;
    }
    CHECK(corr_row_slot_free_submit(ctx, 0, NULL, NULL) == CORR_ERR_ARG,
          "a NULL callback should be CORR_ERR_ARG");
    CHECK(corr_row_slot_free_submit(NULL, 0, on_done, NULL) == CORR_ERR_ARG,
          "a NULL context should be CORR_ERR_ARG");
    CHECK(corr_row_slot_free_submit(ctx, -1, on_done, NULL) == CORR_ERR_ARG,
          "a negative sequence should be CORR_ERR_ARG");
    /* Not submitted yet, so there is nothing to release -- and the same is true
     * of the finish half, which must not be a way around the state machine. */
    CHECK(corr_row_slot_free_submit(ctx, 0, on_done, NULL) == CORR_ERR_STATE,
          "submitting an unsubmitted slot should be CORR_ERR_STATE");
    CHECK(corr_row_slot_free_finish(ctx, 0) == CORR_ERR_STATE,
          "finishing an unsubmitted slot should be CORR_ERR_STATE");
    CHECK(corr_rows_wait_submit(ctx, NULL, NULL) == CORR_ERR_ARG,
          "a NULL callback should be CORR_ERR_ARG");
    CHECK(corr_destroy(ctx) == CORR_OK, "corr_destroy failed");
}

int main(void) {
    printf("corr_cuda async notification test\n");
    test_argument_checking();
    test_submit_does_not_wait("1 device", 1);
    test_rows_wait("1 device", 1);
    test_fetch("1 device, baseline range", 1, CORR_SPLIT_BASELINE_RANGE);
    /* One device under the time-slice split: the same row arithmetic with no peer
     * to add, which is what isolates the combine from the rest. */
    test_fetch("1 device, time slice", 1, CORR_SPLIT_TIME_SLICE);
    /* Two devices, baselines split between them: no cross-device sum at all, so
     * this is the exact case that exercises the counting path. */
    test_fetch("2 devices, baseline range", 2, CORR_SPLIT_BASELINE_RANGE);
    /* Two devices, frames split: the peers' partial rows are summed, and the sum
     * is where the split's own lap-to-lap spread comes from. */
    test_fetch("2 devices, time slice", 2, CORR_SPLIT_TIME_SLICE);
    test_scattered_fetch_is_refused();
    /* Several devices read the same slot under this split, so this is the case
     * where the counted notification has to collapse them into one. */
    test_submit_does_not_wait("2 devices", 2);

    if (g_failures != 0) {
        printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
