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
