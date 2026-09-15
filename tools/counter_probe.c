// Probe whether packet counters from different stations share one global clock.
//
// One thread per station, each pinned to its own core, each reading as fast as
// the kernel will hand out packets. Every thread publishes its newest counter
// plus the time it was read; the main thread samples the whole set and prints
// the pairwise differences. A shared clock shows constant differences; per-board
// free-running counters show a monotone drift. Counter gaps are counted so we
// can tell "thread fell behind" apart from "network dropped packets".
//
// build: gcc -O2 -pthread -o /tmp/counter_probe /tmp/counter_probe.c

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define PAYLOAD 4104
#define NSLOT 16

struct station {
    const char *name;
    const char *group;
    int port;
    int cpu;
    int fd;
};

static struct station st[NSLOT];
static int nst;

struct shard {
    _Atomic uint64_t counter;   // newest counter seen by this thread
    _Atomic uint64_t stamp_ns;  // when it was read
    _Atomic uint64_t total;     // packets consumed
    _Atomic uint64_t gaps;      // counter jumps > 1
    _Atomic uint64_t gap_sum;   // total packets lost inside those jumps
    _Atomic uint64_t reorder;   // counter went backwards
    _Atomic int ready;
} sh[NSLOT];

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static int join_group(int fd, const char *group, const char *iface, int port) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, group, &addr.sin_addr);
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) < 0) {
        perror("bind");
        return -1;
    }
    struct ip_mreqn mreq;
    memset(&mreq, 0, sizeof mreq);
    inet_pton(AF_INET, group, &mreq.imr_multiaddr);
    inet_pton(AF_INET, iface, &mreq.imr_address);
    mreq.imr_ifindex = 0;
    if (setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof mreq) < 0) {
        perror("IP_ADD_MEMBERSHIP");
        return -1;
    }
    return 0;
}

static void *reader(void *arg) {
    int idx = (int)(intptr_t)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(st[idx].cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);

    int fd = st[idx].fd;
    static __thread unsigned char buf[PAYLOAD + 64] __attribute__((aligned(64)));
    uint64_t last = 0, have = 0, total = 0, gaps = 0, gap_sum = 0, reorder = 0;

    sh[idx].ready = 1;
    for (;;) {
        ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n < 8) continue;
        uint64_t c;
        memcpy(&c, buf, 8);
        if (have) {
            if (c <= last) reorder++;
            else if (c != last + 1) { gaps++; gap_sum += c - last - 1; }
        }
        last = c; have = 1; total++;
        // publish only every 16th packet: keeps the atomics off the hot path
        if ((total & 15) == 0) {
            atomic_store_explicit(&sh[idx].counter, c, memory_order_relaxed);
            atomic_store_explicit(&sh[idx].stamp_ns, now_ns(), memory_order_relaxed);
            atomic_store_explicit(&sh[idx].total, total, memory_order_relaxed);
            atomic_store_explicit(&sh[idx].gaps, gaps, memory_order_relaxed);
            atomic_store_explicit(&sh[idx].gap_sum, gap_sum, memory_order_relaxed);
            atomic_store_explicit(&sh[idx].reorder, reorder, memory_order_relaxed);
        }
    }
    sh[idx].total = total;
    sh[idx].gaps = gaps;
    sh[idx].gap_sum = gap_sum;
    sh[idx].reorder = reorder;
    return NULL;
}

int main(int argc, char **argv) {
    double dur = argc > 1 ? atof(argv[1]) : 10.0;

    struct { const char *name; const char *group; int port; int cpu; } spec[] = {
        {"W17", "239.1.137.3", 10037, 8},
        {"S01", "239.1.161.3", 10061, 9},
        {"S06", "239.1.166.3", 10066, 10},
        {"N04", "239.1.144.3", 10044, 11},
    };
    nst = sizeof spec / sizeof spec[0];

    for (int i = 0; i < nst; i++) {
        st[i].name = spec[i].name;
        st[i].group = spec[i].group;
        st[i].port = spec[i].port;
        st[i].cpu = spec[i].cpu;
        st[i].fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (st[i].fd < 0) { perror("socket"); return 1; }
        int rcv = 64 << 20;
        setsockopt(st[i].fd, SOL_SOCKET, SO_RCVBUF, &rcv, sizeof rcv);
        int one = 1;
        setsockopt(st[i].fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (join_group(st[i].fd, st[i].group, "172.23.0.11", st[i].port) < 0) return 1;
        pthread_t th;
        pthread_create(&th, NULL, reader, (void *)(intptr_t)i);
        pthread_detach(th);
    }
    for (int i = 0; i < nst; i++) while (!sh[i].ready) usleep(1000);
    usleep(300000);

    uint64_t t0 = now_ns(), base[NSLOT];
    int have_base = 0;
    printf("%8s", "t(s)");
    for (int i = 0; i < nst; i++) printf("  %18s", st[i].name);
    printf("   |  diffs vs t0 (packets)   |  age(us)\n");

    while ((now_ns() - t0) < (uint64_t)(dur * 1e9)) {
        uint64_t c[NSLOT], ts[NSLOT];
        for (int i = 0; i < nst; i++) {
            c[i] = atomic_load_explicit(&sh[i].counter, memory_order_relaxed);
            ts[i] = atomic_load_explicit(&sh[i].stamp_ns, memory_order_relaxed);
        }
        if (!have_base) { memcpy(base, c, sizeof base); have_base = 1; }
        printf("%8.2f", (now_ns() - t0) / 1e9);
        for (int i = 0; i < nst; i++) printf("  %18llu", (unsigned long long)c[i]);
        printf("   | ");
        for (int i = 0; i < nst; i++) printf("%+8lld ", (long long)(c[i] - base[i]));
        printf(" | ");
        uint64_t now = now_ns();
        for (int i = 0; i < nst; i++) printf("%6llu ", (unsigned long long)((now - ts[i]) / 1000));
        printf("\n");
        fflush(stdout);
        usleep(200000);
    }

    printf("\n%-6s %12s %10s %10s %12s %12s\n", "stat", "packets", "gaps", "lost", "reorder", "pps");
    for (int i = 0; i < nst; i++) {
        uint64_t total = atomic_load(&sh[i].total);
        uint64_t span = atomic_load(&sh[i].counter) - 0; // informational
        (void)span;
        printf("%-6s %12llu %10llu %10llu %12llu %12.0f\n", st[i].name,
               (unsigned long long)total, (unsigned long long)atomic_load(&sh[i].gaps),
               (unsigned long long)atomic_load(&sh[i].gap_sum),
               (unsigned long long)atomic_load(&sh[i].reorder), total / dur);
    }
    return 0;
}
