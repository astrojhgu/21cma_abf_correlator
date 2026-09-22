//! The socket layer: one UDP multicast socket per station, batched receives.
//!
//! Why these particular options, in one place:
//!
//! * `SO_RCVBUF` is set to 256 MiB. `net.core.rmem_max` on this host is
//!   2147483647, so no privilege is needed. A big receive buffer is what turns a
//!   scheduling hiccup into a short backlog instead of lost packets: the measured
//!   four-station run over 120 s had zero gaps with a 64 MiB buffer.
//! * `SO_BUSY_POLL` makes the kernel spin in the driver for a few tens of
//!   microseconds before sleeping, which removes the wake-up latency that
//!   otherwise shows up as jitter at 195 kpps. `net.core.busy_poll` defaults to 0
//!   but the per-socket value is not bound by it.
//! * **one `recvfrom` per packet.** This used to be `recvmmsg`, amortising one
//!   syscall over a batch of up to 32 packets, on the argument that at 195 kpps a
//!   syscall per packet spends a measurable fraction of a core on entry and exit
//!   alone. That argument was made for 1500-byte frames and a packet rate that no
//!   longer exists: the front end sends 4104-byte datagrams over jumbo frames, so
//!   the per-packet syscall is amortised over 2.7x the bytes it was sized for.
//!   The measurement that decided it is in `docs/DESIGN-no-recvmmsg.md`.
//! * `SO_RCVTIMEO` is deliberately short. A receive thread must not block for
//!   ever: when a station goes quiet it has to notice and honour the scheduler's
//!   request to declare the missing frames up to the next slot boundary.
//!
//! `IP_ADD_MEMBERSHIP` binds the group to `receiver.interface_ip`, not to
//! `INADDR_ANY`: with several interfaces on the host, joining on the wrong one
//! silently receives nothing.

use std::io;
use std::net::Ipv4Addr;
use std::os::unix::io::RawFd;

use crate::config::ReceiverCfg;

/// Size of the packet counter that prefixes every payload.
pub const COUNTER_BYTES: usize = 8;
/// Sample bytes in one packet.
pub const PAYLOAD_BYTES: usize = 4096;
/// What a whole packet must be able to hold.
pub const PACKET_BYTES: usize = COUNTER_BYTES + PAYLOAD_BYTES;

/// `SO_BUSY_POLL` is 46 on Linux; spelled out because the libc crate does not
/// expose it on every target.
const SO_BUSY_POLL: libc::c_int = 46;

#[derive(Debug, Default, Clone, Copy)]
pub struct RecvStats {
    /// Times the receive call came back empty: the receive timeout expired, or a
    /// non-blocking socket had nothing to give.
    pub empty_polls: u64,
    /// Packets the kernel reported as truncated, i.e. the buffer was too small.
    pub truncated: u64,
}

/// One received packet: the counter that prefixes it, and the samples behind it.
///
/// Borrows the socket's receive buffer, which the next receive overwrites, so a
/// packet cannot outlive the call that produced it. That is the point: there is no
/// second buffer, no batch, and nothing to index.
pub struct Packet<'a> {
    /// The front end's packet counter, little endian on the wire. It is a global
    /// counter shared by the boards, not a per-station one.
    pub counter: u64,
    /// Sample bytes, as many as arrived. Shorter than a full sample block when the
    /// datagram was short or truncated; the assembler rejects anything that cannot
    /// fill one.
    pub samples: &'a [u8],
}

/// One UDP multicast socket, with room for one packet.
pub struct MulticastSocket {
    fd: RawFd,
    /// One datagram, with a little slack over the nominal payload so an oversized
    /// frame is detected as truncated rather than as a short read.
    buf: Vec<u8>,
    /// Datagram length reported by the last `recvfrom`. `MSG_TRUNC` makes this the
    /// datagram's real length even when it did not fit, which is what makes
    /// truncation visible instead of silently arriving as a short packet.
    len: usize,
    stats: RecvStats,
    /// Whether the kernel accepted `SO_BUSY_POLL`. Reported rather than assumed:
    /// it needs a privilege this host's user does not have.
    busy_poll_granted: bool,
}

impl MulticastSocket {
    /// Open and join `group`:`port` on `cfg.interface_ip`.
    pub fn open(cfg: &ReceiverCfg, group: Ipv4Addr, port: u16) -> io::Result<Self> {
        // A little slack over the nominal payload so an oversized frame is
        // detected as truncated rather than as a short read.
        let stride = PACKET_BYTES.next_multiple_of(64) + 64;

        // SAFETY: a plain socket(2) call.
        let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut sock = MulticastSocket {
            fd,
            buf: vec![0u8; stride],
            len: 0,
            stats: RecvStats::default(),
            busy_poll_granted: false,
        };
        // On failure `sock` simply drops, which closes the descriptor.
        sock.configure(cfg, group, port)?;
        Ok(sock)
    }

    fn configure(&mut self, cfg: &ReceiverCfg, group: Ipv4Addr, port: u16) -> io::Result<()> {
        set_int_opt(
            self.fd,
            libc::SOL_SOCKET,
            libc::SO_REUSEADDR,
            1,
            "SO_REUSEADDR",
        )?;
        set_int_opt(
            self.fd,
            libc::SOL_SOCKET,
            libc::SO_RCVBUF,
            cfg.recv_buffer_bytes as libc::c_int,
            "SO_RCVBUF",
        )?;
        if cfg.busy_poll_us > 0 {
            // Best effort. Measured on this host: EPERM without privileges, so a
            // hard failure here would make the correlator unusable as jhgu for a
            // latency optimisation it can live without.
            self.busy_poll_granted = set_int_opt(
                self.fd,
                libc::SOL_SOCKET,
                SO_BUSY_POLL,
                cfg.busy_poll_us as libc::c_int,
                "SO_BUSY_POLL",
            )
            .is_ok();
        }
        // A short receive timeout is what lets a quiet station be noticed.
        let tv = libc::timeval {
            tv_sec: 0,
            tv_usec: (cfg.recv_timeout_ms * 1000) as libc::suseconds_t,
        };
        // SAFETY: tv is a properly initialised timeval.
        if unsafe {
            libc::setsockopt(
                self.fd,
                libc::SOL_SOCKET,
                libc::SO_RCVTIMEO,
                &tv as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }

        let addr = libc::sockaddr_in {
            sin_family: libc::AF_INET as libc::sa_family_t,
            sin_port: port.to_be(),
            sin_addr: libc::in_addr {
                s_addr: u32::from(group).to_be(),
            },
            sin_zero: [0; 8],
        };
        // SAFETY: addr is a fully initialised sockaddr_in of the given length.
        if unsafe {
            libc::bind(
                self.fd,
                &addr as *const _ as *const libc::sockaddr,
                std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }

        // Joining on the configured interface, never INADDR_ANY: with several
        // interfaces the group would otherwise be joined on the wrong one and the
        // station would look silent.
        let mreq = libc::ip_mreqn {
            imr_multiaddr: libc::in_addr {
                s_addr: u32::from(group).to_be(),
            },
            imr_address: libc::in_addr {
                s_addr: u32::from(cfg.interface_ip).to_be(),
            },
            imr_ifindex: 0,
        };
        // SAFETY: mreq is a fully initialised ip_mreqn.
        if unsafe {
            libc::setsockopt(
                self.fd,
                libc::IPPROTO_IP,
                libc::IP_ADD_MEMBERSHIP,
                &mreq as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::ip_mreqn>() as libc::socklen_t,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// The receive buffer actually granted by the kernel, which can be smaller
    /// than what was asked for.
    pub fn rcvbuf(&self) -> io::Result<usize> {
        let mut v: libc::c_int = 0;
        let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
        // SAFETY: v and len are valid.
        if unsafe {
            libc::getsockopt(
                self.fd,
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &mut v as *mut _ as *mut libc::c_void,
                &mut len,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(v as usize)
    }

    /// Whether `SO_BUSY_POLL` was accepted. False means the receive thread falls
    /// back on the normal wake-up path.
    pub fn busy_poll_granted(&self) -> bool {
        self.busy_poll_granted
    }

    pub fn stats(&self) -> RecvStats {
        self.stats
    }

    /// Receive one packet, waiting for one for up to `SO_RCVTIMEO`.
    ///
    /// `Ok(None)` means there was nothing to place: the socket stayed empty for
    /// the whole timeout, or a datagram arrived too short to carry the counter
    /// every packet is prefixed with. An empty socket is not an error here -- it
    /// is the normal way a receive thread notices that it should service other
    /// work, such as the scheduler's request to zero fill a station that has gone
    /// quiet.
    pub fn recv_one(&mut self) -> io::Result<Option<Packet<'_>>> {
        match self.recv_from() {
            Ok(()) => Ok(self.packet()),
            Err(e) if is_empty_socket(&e) => {
                self.stats.empty_polls += 1;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Receive one packet, reporting an empty socket as `WouldBlock`.
    ///
    /// The same single `recvfrom` as [`Self::recv_one`]. It exists separately
    /// because the readiness-driven backend cannot take a borrowed packet, and
    /// neither reason for that is a preference:
    ///
    /// * a reactor (epoll, and therefore anything built on it) clears an
    ///   edge-triggered readiness only when the caller says the socket ran dry, so
    ///   an empty socket has to reach it as an `Err` rather than as "no packet";
    /// * the packet has to be carried out of a `try_io` closure and across a
    ///   `select!` before it can be used, which a borrow of this socket cannot do.
    ///
    /// So the bytes stay in the socket's buffer and are read back with
    /// [`Self::packet`], which is what that backend already did.
    ///
    /// Requires the socket to be non-blocking ([`Self::set_nonblocking`]),
    /// otherwise an empty socket blocks for `SO_RCVTIMEO` before it says so.
    pub(crate) fn recv_one_nowait(&mut self) -> io::Result<()> {
        self.recv_from()
    }

    /// The syscall itself: one datagram into this socket's buffer.
    fn recv_from(&mut self) -> io::Result<()> {
        let mut src: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut slen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;

        // SAFETY: `buf` is `buf.len()` bytes of storage this socket owns, `src` is
        // a fully sized sockaddr_storage, and `slen` says so. `MSG_TRUNC` makes the
        // return value the datagram's real length even when it did not fit, so an
        // oversized frame is counted instead of arriving as a silent short packet.
        let n = unsafe {
            libc::recvfrom(
                self.fd,
                self.buf.as_mut_ptr() as *mut libc::c_void,
                self.buf.len(),
                libc::MSG_TRUNC,
                &mut src as *mut _ as *mut libc::sockaddr,
                &mut slen,
            )
        };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        self.len = n as usize;
        if self.len > self.buf.len() {
            self.stats.truncated += 1;
        }
        Ok(())
    }

    #[cfg_attr(not(feature = "async-recv"), allow(dead_code))]
    // Unused only when `async-recv` is off (`--no-default-features`): it exists for
    // the readiness-driven backend, and the test below exercises it either way.
    /// Put the socket in non-blocking mode.
    ///
    /// Needed by any readiness-based receiver: with `SO_RCVTIMEO` still in force
    /// (see [`Self::configure`]) an empty socket would still park the calling
    /// thread for the timeout instead of returning `EAGAIN` at once, and the poll
    /// would then be a wait the reactor did not know about. Calling this twice is
    /// a no-op, and `SO_BUSY_POLL` keeps working: it is the socket receive path
    /// that spins in the driver, not the syscall.
    pub(crate) fn set_nonblocking(&self) -> io::Result<()> {
        // SAFETY: fcntl(F_GETFL/F_SETFL) on a descriptor this socket owns.
        let flags = unsafe { libc::fcntl(self.fd, libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if flags & libc::O_NONBLOCK != 0 {
            return Ok(());
        }
        if unsafe { libc::fcntl(self.fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// The packet the last receive put in this socket's buffer.
    ///
    /// `None` means that datagram was too short to hold the counter every packet
    /// starts with. There is no counter to report and the samples behind it
    /// cannot be placed without one, so such a packet is dropped -- the same
    /// outcome the batched receive path had, which returned `None` here for the
    /// same reason.
    pub fn packet(&self) -> Option<Packet<'_>> {
        // A truncated datagram is longer than the buffer: clip it rather than
        // index past the end. The assembler then sees a short payload and rejects
        // it, which is the same outcome `recvmmsg` produced from `msg_flags`.
        let len = self.len.min(self.buf.len());
        if len < COUNTER_BYTES {
            return None;
        }
        let bytes = &self.buf[..len];
        let mut c = [0u8; 8];
        c.copy_from_slice(&bytes[..COUNTER_BYTES]);
        // Little endian, verified against the front end: the counter is
        // contiguous as a uint64 LE and not as a BE.
        Some(Packet {
            counter: u64::from_le_bytes(c),
            samples: &bytes[COUNTER_BYTES..],
        })
    }

    pub fn fd(&self) -> RawFd {
        self.fd
    }
}

// `MulticastSocket` needs no `unsafe impl Send`: it owns its descriptor and its
// packet buffer, both of which are `Send`, so moving the whole struct to another
// thread moves the only references to it. One thread receiving per socket is a
// design rule, not a safety one -- every receive entry point takes `&mut self`.

impl Drop for MulticastSocket {
    fn drop(&mut self) {
        // SAFETY: fd came from socket(2) and is closed exactly once.
        unsafe { libc::close(self.fd) };
    }
}

/// Whether a receive error means "the socket had nothing", as opposed to a real
/// failure. `SO_RCVTIMEO` expiring and a non-blocking socket finding nothing are
/// the same situation seen from two different sockets, and both callers treat it
/// the same way.
fn is_empty_socket(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

fn set_int_opt(
    fd: RawFd,
    level: libc::c_int,
    name: libc::c_int,
    value: libc::c_int,
    what: &str,
) -> io::Result<()> {
    // SAFETY: value is a valid c_int of the given length.
    if unsafe {
        libc::setsockopt(
            fd,
            level,
            name,
            &value as *const _ as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    } != 0
    {
        return Err(io::Error::new(
            io::Error::last_os_error().kind(),
            format!("setsockopt({what}): {}", io::Error::last_os_error()),
        ));
    }
    Ok(())
}

/// Pin the calling thread to one CPU.
///
/// The host has no real-time priority available without `CAP_SYS_NICE`, so this
/// is the only scheduling control there is; it at least keeps a receive thread
/// from migrating between NUMA nodes mid-run.
pub fn pin_to_cpu(cpu: u32) -> io::Result<()> {
    // SAFETY: cpu_set_t is a plain bitmask and sched_setaffinity(0, ...) acts on
    // the calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu as usize, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_receiver_cfg() -> ReceiverCfg {
        ReceiverCfg {
            interface: "lo".to_string(),
            interface_ip: Ipv4Addr::new(127, 0, 0, 1),
            recv_buffer_bytes: 1 << 20,
            busy_poll_us: 0,
            recvmmsg_batch: 8,
            recv_timeout_ms: 20,
        }
    }

    #[test]
    fn opening_and_closing_a_multicast_socket_works_unprivileged() {
        // 239.255.0.1 is in the administratively scoped range and is not used by
        // anything else on this host, so this neither needs privileges nor
        // disturbs the front end.
        let cfg = test_receiver_cfg();
        let s = MulticastSocket::open(&cfg, Ipv4Addr::new(239, 255, 0, 1), 29999);
        match s {
            Ok(sock) => {
                assert!(sock.fd() >= 0);
                let granted = sock.rcvbuf().expect("SO_RCVBUF should be readable");
                assert!(granted > 0, "kernel granted a zero receive buffer");
            }
            Err(e) => {
                // A host with no multicast route at all is not a test failure of
                // this code; report it rather than pretending to pass.
                eprintln!("skipping: cannot open a multicast socket here: {e}");
            }
        }
    }

    #[test]
    fn an_empty_socket_polls_without_blocking_for_ever() {
        let cfg = test_receiver_cfg();
        let mut sock = match MulticastSocket::open(&cfg, Ipv4Addr::new(239, 255, 0, 2), 29998) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: cannot open a multicast socket here: {e}");
                return;
            }
        };
        let t0 = std::time::Instant::now();
        let got = sock.recv_one().expect("an empty poll is not an error");
        assert!(got.is_none(), "nothing was sent, so nothing should arrive");
        let dt = t0.elapsed();
        assert!(
            dt < std::time::Duration::from_secs(2),
            "SO_RCVTIMEO should have cut the wait short, took {dt:?}"
        );
        assert_eq!(sock.stats().empty_polls, 1);
    }

    /// The whole point of the direct call: one datagram in, its counter and its
    /// samples out.
    ///
    /// Nothing covered the success path before -- the batched receive was only
    /// ever exercised by a real run -- so a refactor of it had nothing to fail
    /// against. This sends one datagram to the group the socket joined, over the
    /// interface it joined it on, and reads it back.
    #[test]
    fn one_datagram_arrives_with_its_counter_and_samples() {
        let cfg = test_receiver_cfg();
        let group = Ipv4Addr::new(239, 255, 0, 4);
        let port: u16 = 29996;
        let mut sock = match MulticastSocket::open(&cfg, group, port) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: cannot open a multicast socket here: {e}");
                return;
            }
        };

        // A counter that is not palindromic in either byte order, so reading it big
        // endian instead of little endian cannot pass by accident.
        let counter: u64 = 0x0123_4567_89ab_cdef;
        let payload: Vec<u8> = (0..64u8).collect();
        let mut msg = counter.to_le_bytes().to_vec();
        msg.extend_from_slice(&payload);

        // SAFETY: a plain sending socket, its options, and one sendto; every
        // buffer passed is owned by this function and sized as the call is told.
        let sent = unsafe {
            let tx = libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0);
            if tx < 0 {
                eprintln!("skipping: cannot create a sending socket here");
                return;
            }
            let one: libc::c_int = 1;
            let loopback = libc::in_addr {
                s_addr: u32::from(Ipv4Addr::LOCALHOST).to_be(),
            };
            libc::setsockopt(
                tx,
                libc::IPPROTO_IP,
                libc::IP_MULTICAST_LOOP,
                &one as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            libc::setsockopt(
                tx,
                libc::IPPROTO_IP,
                libc::IP_MULTICAST_IF,
                &loopback as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::in_addr>() as libc::socklen_t,
            );
            let dst = libc::sockaddr_in {
                sin_family: libc::AF_INET as libc::sa_family_t,
                sin_port: port.to_be(),
                sin_addr: libc::in_addr {
                    s_addr: u32::from(group).to_be(),
                },
                sin_zero: [0; 8],
            };
            let n = libc::sendto(
                tx,
                msg.as_ptr() as *const libc::c_void,
                msg.len(),
                0,
                &dst as *const _ as *const libc::sockaddr,
                std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
            );
            let err = io::Error::last_os_error();
            libc::close(tx);
            if n < 0 {
                eprintln!("skipping: cannot send to a multicast group here: {err}");
                return;
            }
            n as usize
        };
        assert_eq!(sent, msg.len());

        // The test socket's `SO_RCVTIMEO` is 20 ms, so this polls rather than
        // waiting on one long call -- and it must fail rather than hang if the
        // datagram never comes back.
        for _ in 0..20 {
            match sock.recv_one() {
                Ok(Some(pkt)) => {
                    assert_eq!(pkt.counter, counter, "the counter is little endian");
                    assert_eq!(pkt.samples, &payload[..], "the samples follow it");
                    return;
                }
                Ok(None) => {}
                Err(e) => panic!("receive error: {e}"),
            }
        }
        panic!("a datagram sent to the group this socket joined never arrived");
    }

    #[test]
    fn pinning_to_a_cpu_that_exists_succeeds() {
        // CPU 0 always exists; the kernel clamps an affinity mask to the online
        // set, so this is safe on any host.
        pin_to_cpu(0).expect("sched_setaffinity(0) should work unprivileged");
    }

    /// The reactor's contract: an empty non-blocking socket says `WouldBlock`
    /// *immediately*, while the blocking flavour still folds that into `Ok(None)`.
    /// A reactor that got `Ok(None)` instead could not tell "empty" from "a
    /// zero-length datagram arrived" and would never clear its readiness, which is
    /// a busy loop that looks like a hot socket.
    #[test]
    fn a_nonblocking_socket_reports_an_empty_poll_as_would_block() {
        let cfg = test_receiver_cfg();
        let mut sock = match MulticastSocket::open(&cfg, Ipv4Addr::new(239, 255, 0, 3), 29997) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: cannot open a multicast socket here: {e}");
                return;
            }
        };
        sock.set_nonblocking().expect("O_NONBLOCK is unprivileged");
        sock.set_nonblocking().expect("setting it twice is a no-op");

        let t0 = std::time::Instant::now();
        let e = sock
            .recv_one_nowait()
            .expect_err("nothing was sent, so the reactor must be told the socket is dry");
        assert_eq!(e.kind(), io::ErrorKind::WouldBlock);
        assert!(
            t0.elapsed() < std::time::Duration::from_millis(5),
            "a non-blocking poll must not wait for SO_RCVTIMEO, took {:?}",
            t0.elapsed()
        );

        // The blocking-flavoured entry point still reports the same empty poll as
        // `Ok(None)` and counts it, which is what the thread-per-station path
        // relies on to notice that it should service other work.
        let before = sock.stats().empty_polls;
        assert!(sock
            .recv_one()
            .expect("an empty poll is not an error")
            .is_none());
        assert_eq!(sock.stats().empty_polls, before + 1);
    }
}
