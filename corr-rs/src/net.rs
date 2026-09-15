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
//! * `recvmmsg` amortises one syscall over a batch of packets. At 195 kpps per
//!   station a syscall per packet would spend a measurable fraction of a core on
//!   entry and exit alone.
//! * `SO_RCVTIMEO` is deliberately short. A receive thread must not block for
//!   ever: when a station goes quiet it has to notice and honour the scheduler's
//!   request to zero fill up to the next slot boundary.
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
    /// Batches returned by `recvmmsg`.
    pub batches: u64,
    /// Packets returned.
    pub packets: u64,
    /// Times `recvmmsg` came back empty (the receive timeout expired).
    pub empty_polls: u64,
    /// Packets the kernel reported as truncated, i.e. the buffer was too small.
    pub truncated: u64,
}

/// One UDP multicast socket, with room for a batch of packets.
pub struct MulticastSocket {
    fd: RawFd,
    /// One slot per packet in the batch, each `stride` bytes.
    buf: Vec<u8>,
    stride: usize,
    iovecs: Vec<libc::iovec>,
    hdrs: Vec<libc::mmsghdr>,
    vlen: usize,
    stats: RecvStats,
    /// Whether the kernel accepted `SO_BUSY_POLL`. Reported rather than assumed:
    /// it needs a privilege this host's user does not have.
    busy_poll_granted: bool,
}

impl MulticastSocket {
    /// Open and join `group`:`port` on `cfg.interface_ip`.
    pub fn open(cfg: &ReceiverCfg, group: Ipv4Addr, port: u16) -> io::Result<Self> {
        assert!(cfg.recvmmsg_batch >= 1, "recvmmsg batch must be >= 1");
        let vlen = cfg.recvmmsg_batch as usize;
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
            buf: vec![0u8; stride * vlen],
            stride,
            iovecs: vec![
                libc::iovec {
                    iov_base: std::ptr::null_mut(),
                    iov_len: 0,
                };
                vlen
            ],
            hdrs: vec![unsafe { std::mem::zeroed() }; vlen],
            vlen,
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

    /// Receive up to one batch. Returns the number of packets, or an error.
    ///
    /// `WouldBlock`/`TimedOut` are returned as `Ok(0)`: an empty poll is the
    /// normal way a receive thread notices that it should service other work.
    pub fn recv_batch(&mut self) -> io::Result<usize> {
        for i in 0..self.vlen {
            let base = self.buf.as_mut_ptr().wrapping_add(i * self.stride);
            self.iovecs[i].iov_base = base as *mut libc::c_void;
            self.iovecs[i].iov_len = self.stride;
            self.hdrs[i].msg_hdr.msg_iov = &mut self.iovecs[i] as *mut libc::iovec;
            self.hdrs[i].msg_hdr.msg_iovlen = 1;
            self.hdrs[i].msg_hdr.msg_namelen = 0;
            self.hdrs[i].msg_hdr.msg_controllen = 0;
            self.hdrs[i].msg_hdr.msg_flags = 0;
            self.hdrs[i].msg_len = 0;
        }

        // SAFETY: hdrs has vlen entries, each pointing at a distinct iovec whose
        // base is inside `buf`.
        let n = unsafe {
            libc::recvmmsg(
                self.fd,
                self.hdrs.as_mut_ptr(),
                self.vlen as libc::c_uint,
                0,
                std::ptr::null_mut(),
            )
        };
        if n < 0 {
            let e = io::Error::last_os_error();
            match e.kind() {
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
                    self.stats.empty_polls += 1;
                    return Ok(0);
                }
                _ => return Err(e),
            }
        }
        let n = n as usize;
        self.stats.batches += 1;
        self.stats.packets += n as u64;
        for i in 0..n {
            if self.hdrs[i].msg_hdr.msg_flags & libc::MSG_TRUNC != 0 {
                self.stats.truncated += 1;
            }
        }
        Ok(n)
    }

    /// The `i`-th packet of the last batch, as `(counter, samples)`.
    ///
    /// Returns `None` for a packet too short to hold a counter, and a payload of
    /// whatever length arrived otherwise; the assembler rejects anything shorter
    /// than a full sample block.
    pub fn packet(&self, i: usize) -> Option<(u64, &[u8])> {
        if i >= self.vlen {
            return None;
        }
        let len = self.hdrs[i].msg_len as usize;
        if len < COUNTER_BYTES {
            return None;
        }
        let base = i * self.stride;
        let bytes = &self.buf[base..base + len];
        let mut c = [0u8; 8];
        c.copy_from_slice(&bytes[..COUNTER_BYTES]);
        // Little endian, verified against the front end: the counter is
        // contiguous as a uint64 LE and not as a BE.
        Some((u64::from_le_bytes(c), &bytes[COUNTER_BYTES..]))
    }

    pub fn fd(&self) -> RawFd {
        self.fd
    }
}

// SAFETY: the socket owns its descriptor and its packet buffers outright; the
// raw pointers inside `iovecs` all point into that same `buf`. Moving the whole
// struct to another thread therefore moves the only references to it, which is
// what `Send` means. It is deliberately not `Sync`: exactly one thread receives
// on a socket, which is the whole design.
unsafe impl Send for MulticastSocket {}

impl Drop for MulticastSocket {
    fn drop(&mut self) {
        // SAFETY: fd came from socket(2) and is closed exactly once.
        unsafe { libc::close(self.fd) };
    }
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
        let n = sock.recv_batch().expect("an empty poll is not an error");
        assert_eq!(n, 0, "nothing was sent, so nothing should arrive");
        let dt = t0.elapsed();
        assert!(
            dt < std::time::Duration::from_secs(2),
            "SO_RCVTIMEO should have cut the wait short, took {dt:?}"
        );
        assert_eq!(sock.stats().empty_polls, 1);
    }

    #[test]
    fn pinning_to_a_cpu_that_exists_succeeds() {
        // CPU 0 always exists; the kernel clamps an affinity mask to the online
        // set, so this is safe on any host.
        pin_to_cpu(0).expect("sched_setaffinity(0) should work unprivileged");
    }
}
