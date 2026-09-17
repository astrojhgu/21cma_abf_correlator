// raw_capture.c -- write one station's UDP payloads to a file, verbatim.
//
// The correlator's own output has been through framing, an FFT and an average;
// when a station's spectrum looks wrong the first question is whether the raw
// stream already looks like that. This joins one group and appends every packet
// it receives, exactly as it arrived: 8-byte little-endian counter, then 4096
// int8 samples. Nothing is reordered or repaired, so a gap in the counters is
// visible in the file and the reader decides what to do about it.
//
// In C rather than Python on purpose: this machine's front end sends about 195
// kpps per station and a Python socket loop tops out near 35 kpps (measured, see
// tools/README.md), which would silently turn a spectrum into a measurement of
// Python. The FFT is still done in Python afterwards, where it is easy to read.
//
// build: gcc -O2 -o /tmp/raw_capture tools/raw_capture.c
// usage: raw_capture GROUP PORT IFACE_IP SECONDS OUTFILE
//
// prints one line at the end: packets, counter gaps, packets lost across gaps,
// bytes written, and the achieved rate.

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define DATAGRAM 4104 /* 8-byte counter + 4096 int8 samples */

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s GROUP PORT IFACE_IP SECONDS OUTFILE\n", argv[0]);
        return 2;
    }
    const char *group = argv[1];
    const int port = atoi(argv[2]);
    const char *iface_ip = argv[3];
    const double seconds = atof(argv[4]);
    const char *out = argv[5];

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { perror("socket"); return 1; }

    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) < 0) {
        perror("SO_REUSEADDR");
        return 1;
    }
    // Other readers hold this group (the correlator itself, and on this machine a
    // long-running tenant that sits on fifteen front-end groups): asking for a
    // large buffer is what keeps this capture from measuring its own scheduler.
    int rcvbuf = 64 << 20;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf) < 0)
        perror("SO_RCVBUF (continuing)");

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) < 0) { perror("bind"); return 1; }

    struct ip_mreq mreq;
    memset(&mreq, 0, sizeof mreq);
    mreq.imr_multiaddr.s_addr = inet_addr(group);
    mreq.imr_interface.s_addr = inet_addr(iface_ip);
    if (setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof mreq) < 0) {
        perror("IP_ADD_MEMBERSHIP");
        return 1;
    }

    FILE *f = fopen(out, "wb");
    if (f == NULL) { perror("fopen"); return 1; }

    static unsigned char buf[DATAGRAM];
    const double t0 = now_s();
    const double deadline = t0 + seconds;
    uint64_t packets = 0, last = 0, gaps = 0, lost_across_gaps = 0;
    int have_last = 0;

    while (now_s() < deadline) {
        ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("recv");
            break;
        }
        if (n != DATAGRAM) continue; /* truncated or short: not a sample packet */
        uint64_t counter;
        memcpy(&counter, buf, 8);
        if (have_last) {
            if (counter > last + 1) {
                gaps++;
                lost_across_gaps += counter - last - 1;
            }
            if (counter > last) last = counter;
        } else {
            last = counter;
            have_last = 1;
        }
        if (fwrite(buf, 1, DATAGRAM, f) != DATAGRAM) { perror("fwrite"); break; }
        packets++;
    }
    const double elapsed = now_s() - t0;
    fflush(f);
    fclose(f);
    close(fd);

    printf("packets %llu  run %.3f s  rate %.1f pps  gaps %llu  packets across gaps %llu  "
           "bytes %llu\n",
           (unsigned long long)packets, elapsed, packets / elapsed,
           (unsigned long long)gaps, (unsigned long long)lost_across_gaps,
           (unsigned long long)packets * DATAGRAM);
    return 0;
}
