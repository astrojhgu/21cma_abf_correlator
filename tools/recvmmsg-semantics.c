/* Does plain recvmmsg on a blocking socket wait for a FULL batch, or return
 * whatever is queued?
 *
 * This decides how the A/B in bench/ab-no-recvmmsg.sh has to be read. If plain
 * recvmmsg blocks until it has vlen messages, then the baseline's `recvmmsg` is
 * not "drain what is queued in one syscall" -- it is "sleep until 32 packets have
 * arrived", and the batching of the caller's per-iteration work comes for free
 * from that sleep rather than from amortising a syscall.
 *
 * Queue 3 datagrams, then call recvmmsg(vlen=8) on a blocking socket with a
 * 300 ms SO_RCVTIMEO. If it returns 3 quickly, it drains what is queued. If it
 * returns 3 after ~300 ms, it waited for a full batch and only the timeout ended
 * the wait.
 *
 *   gcc -O2 -o tools/recvmmsg-semantics tools/recvmmsg-semantics.c && tools/recvmmsg-semantics
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <time.h>

#define VLEN 8
#define QUEUED 3

static double ms_since(struct timespec t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

int main(void) {
    int rx = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons(39999);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(rx, (struct sockaddr *)&a, sizeof a);
    struct timeval tv = {0, 300000};
    setsockopt(rx, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    int tx = socket(AF_INET, SOCK_DGRAM, 0);
    char msg[16];
    memset(msg, 'x', sizeof msg);

    char bufs[VLEN][64];
    struct iovec iov[VLEN];
    struct mmsghdr h[VLEN];
    memset(h, 0, sizeof h);
    for (int i = 0; i < VLEN; i++) {
        iov[i].iov_base = bufs[i];
        iov[i].iov_len = sizeof bufs[i];
        h[i].msg_hdr.msg_iov = &iov[i];
        h[i].msg_hdr.msg_iovlen = 1;
    }

    for (int i = 0; i < QUEUED; i++)
        sendto(tx, msg, sizeof msg, 0, (struct sockaddr *)&a, sizeof a);
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int n = recvmmsg(rx, h, VLEN, 0, NULL);
    printf("plain recvmmsg : n=%d in %7.1f ms   (%d datagrams queued, vlen=%d)\n",
           n, ms_since(t0), QUEUED, VLEN);

    for (int i = 0; i < QUEUED; i++)
        sendto(tx, msg, sizeof msg, 0, (struct sockaddr *)&a, sizeof a);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    n = recvmmsg(rx, h, VLEN, MSG_WAITFORONE, NULL);
    printf("MSG_WAITFORONE : n=%d in %7.1f ms\n", n, ms_since(t0));
    return 0;
}
