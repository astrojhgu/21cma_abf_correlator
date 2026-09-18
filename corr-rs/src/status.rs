//! The read-only status endpoint: `GET /status.json`, one JSON object, served
//! from a thread of its own.
//!
//! Why a server inside the correlator, and why that is not a problem:
//!
//! * **It is not in the data path.** The sampler (`monitor.rs`) publishes into a
//!   [`Cell`] whether or not anybody is listening; this module reads that cell and
//!   serialises it. Nothing in the receive, assemble, schedule or write path
//!   touches a socket, a JSON buffer or a lock this file owns.
//! * **It is off unless asked for**: `monitoring.http.enabled`, default false.
//! * **It owns exactly one thread and one listening socket**, and both go away
//!   with the process.
//! * **It cannot take the run down.** The release profile is `panic = "abort"`, so
//!   a panic here would abort the whole run: every error path below is a
//!   `match`/`let else` that drops one connection, and no `unwrap`, `expect` or
//!   slice indexing is reachable. A failed bind (port taken, no permission) is a
//!   warning and a run that carries on -- receiving is the point, monitoring is
//!   not.
//!
//! The interface is deliberately tiny: one path, one method, no query
//! parameters, no keep-alive, no compression, and no dependency beyond
//! `std::net` and `serde_json`. A monitor program polls it once a second
//! (`tools/corr-web.py`); `docs/DESIGN-status-http.md` has the field table.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::autospectra::Cell as AutospectraCell;
use crate::config::HttpCfg;
use crate::monitor::{Cell, SCHEMA};

/// How long a reader may take to send its request and to read the answer. A
/// monitor that goes away mid-request must not pin this thread.
const READ_TIMEOUT: Duration = Duration::from_secs(5);
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// A request line longer than this is not a request we answer.
const MAX_REQUEST_BYTES: usize = 2048;

/// Start serving, if the configuration asks for it.
///
/// Called once at startup from the thread that is about to run the pipeline.
/// Returns whether a server is now running. `cell` is `None` when the endpoint is
/// off, and then this does nothing at all.
pub(crate) fn serve(
    cfg: &HttpCfg,
    cell: Option<Arc<Cell>>,
    autospec: Option<Arc<AutospectraCell>>,
) -> bool {
    if !cfg.enabled {
        return false;
    }
    let Some(cell) = cell else {
        // The caller builds a cell whenever the endpoint is on, so reaching here
        // means the two disagree -- a bug worth saying out loud rather than
        // serving a page that never updates.
        eprintln!(
            "status     : monitoring.http.enabled is set but no status cell was created; \
             not serving"
        );
        return false;
    };
    let listener = match TcpListener::bind(&cfg.bind) {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!(
                "status     : cannot listen on {}: {e}; the run continues without a status \
                 endpoint",
                cfg.bind
            );
            return false;
        }
    };
    let addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| cfg.bind.clone());
    match spawn(listener, cell, autospec) {
        Ok(_thread) => {
            eprintln!(
                "status     : http://{addr}/status.json and /autospectra.json \
                 (schema {SCHEMA}, read-only)"
            );
            true
        }
        Err(e) => {
            eprintln!(
                "status     : cannot spawn the serving thread: {e}; the run continues without it"
            );
            false
        }
    }
}

/// The first request line as `"GET /path HTTP/1.1"`, or `None` if the peer sent
/// nothing usable -- the normal case for a port probe or a scanner.
fn read_request_line(stream: &mut TcpStream) -> Option<String> {
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    let mut chunk = [0u8; 512];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                // Only the request line is needed: nothing here looks at headers.
                if buf.contains(&b'\n') || buf.len() >= MAX_REQUEST_BYTES {
                    break;
                }
            }
            Err(_) => return None,
        }
    }
    let text = String::from_utf8_lossy(&buf);
    let line = text.lines().next()?.trim().to_string();
    if line.is_empty() {
        None
    } else {
        Some(line)
    }
}

/// The current sample, or the fact that there is not one yet.
///
/// `ready: false` is a normal answer, not an error: it is what a monitor that
/// connects between the socket being opened and the first sample landing sees.
fn body(cell: &Cell) -> Vec<u8> {
    match cell.current() {
        Some(snap) => serde_json::to_vec(&*snap).unwrap_or_else(|_| not_ready()),
        None => not_ready(),
    }
}

fn not_ready() -> Vec<u8> {
    format!("{{\"schema\":{SCHEMA},\"ready\":false}}").into_bytes()
}

/// The newest row's autocorrelations, or `ready: false` until a row has been
/// written (the first one lands about one integration period after startup).
fn autospec_body(cell: Option<&AutospectraCell>) -> Vec<u8> {
    match cell.and_then(|c| c.current()) {
        Some(snap) => serde_json::to_vec(&*snap).unwrap_or_else(|_| not_ready()),
        None => not_ready(),
    }
}

/// One request, one response, then close.
///
/// Serial, deliberately: a monitor polls once a second, and a second reader
/// waiting a few milliseconds behind the first is a smaller problem than the
/// bookkeeping a thread pool would add to a process whose job is receiving.
fn handle(mut stream: TcpStream, cell: &Cell, autospec: Option<&AutospectraCell>) {
    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));

    let Some(request) = read_request_line(&mut stream) else {
        return;
    };
    let mut parts = request.split_whitespace();
    let method = parts.next().unwrap_or("");
    // A query string is accepted and ignored, so a monitor can add a cache
    // buster without this side having to parse one.
    let target = parts.next().unwrap_or("");
    let path = target.split('?').next().unwrap_or("");

    let (status, body) = match (method, path) {
        ("GET", "/status.json") => ("200 OK", body(cell)),
        ("GET", "/autospectra.json") => ("200 OK", autospec_body(autospec)),
        // `/` lists what exists, so a human who guessed the address is told where
        // to look instead of getting a bare 404. Still JSON: this endpoint never
        // serves HTML.
        ("GET", "/") => (
            "200 OK",
            format!(
                "{{\"schema\":{SCHEMA},\"endpoints\":[\"/status.json\",\"/autospectra.json\"],\
                 \"note\":\"one JSON object per request, no query parameters\"}}"
            )
            .into_bytes(),
        ),
        _ => (
            "404 Not Found",
            format!(
                "{{\"schema\":{SCHEMA},\"error\":\"no such endpoint\",\
                 \"endpoints\":[\"/status.json\",\"/autospectra.json\"]}}"
            )
            .into_bytes(),
        ),
    };
    let head = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Both);
}

/// The serving thread. Detached on purpose: it ends when the process does.
fn spawn(
    listener: TcpListener,
    cell: Arc<Cell>,
    autospec: Option<Arc<AutospectraCell>>,
) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("corr-status".to_string())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => handle(stream, &cell, autospec.as_deref()),
                    // A failed accept is a per-connection condition (a descriptor
                    // limit, a connection reset between SYN and accept). The short
                    // sleep keeps a persistent one from spinning this thread.
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bind port 0 and drive a real request through a real socket: the point of
    /// this module is that it speaks HTTP, so the test speaks HTTP.
    fn get(addr: std::net::SocketAddr, target: &str) -> (String, String) {
        let mut s = TcpStream::connect(addr).expect("connect");
        s.set_read_timeout(Some(Duration::from_secs(5))).expect("timeout");
        write!(s, "GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("send");
        let mut text = String::new();
        s.read_to_string(&mut text).expect("read");
        let (head, body) = text.split_once("\r\n\r\n").expect("headers then body");
        (
            head.lines().next().unwrap_or_default().to_string(),
            body.to_string(),
        )
    }

    fn serving(cell: Arc<Cell>) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        // Dropped handle: the thread is detached and dies with the test process.
        drop(spawn(listener, cell, None).expect("spawn"));
        addr
    }

    #[test]
    fn a_published_snapshot_is_served_as_json() {
        let cell = Cell::new();
        cell.publish(crate::monitor::tests::minimal_snapshot(1234));
        let addr = serving(cell);

        let (status, body) = get(addr, "/status.json");
        assert_eq!(status, "HTTP/1.1 200 OK");
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["schema"], SCHEMA);
        assert_eq!(v["ready"], true);
        assert_eq!(v["ts_unix_ms"], 1234);
        // The verdict the log line prints is the one the snapshot carries.
        assert_eq!(v["capacity"]["bottleneck"], "ok");
    }

    #[test]
    fn an_unpublished_cell_says_ready_false_and_unknown_paths_are_404() {
        let addr = serving(Cell::new());

        let (status, body) = get(addr, "/status.json");
        assert_eq!(status, "HTTP/1.1 200 OK");
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["ready"], false);
        assert_eq!(v["schema"], SCHEMA);

        let (status, _) = get(addr, "/nope");
        assert_eq!(status, "HTTP/1.1 404 Not Found");

        // A query string is ignored rather than refused.
        let (status, _) = get(addr, "/status.json?cachebust=1");
        assert_eq!(status, "HTTP/1.1 200 OK");

        // `/` names the endpoints instead of 404ing.
        let (status, body) = get(addr, "/");
        assert_eq!(status, "HTTP/1.1 200 OK");
        assert!(body.contains("/status.json"), "{body}");
        assert!(body.contains("/autospectra.json"), "{body}");

        // The autocorrelation endpoint answers `ready: false` until a row has been
        // written, which is a normal state one integration after startup.
        let (status, body) = get(addr, "/autospectra.json");
        assert_eq!(status, "HTTP/1.1 200 OK");
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["ready"], false);
    }
}
