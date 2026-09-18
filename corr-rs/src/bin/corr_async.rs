//! `corr-async` -- the correlator with a readiness-driven receive path.
//!
//! Three lines, deliberately: the backend, the CLI and the startup order all live
//! in `corr_rs::async_recv`, inside the library, so that everything they are built
//! on -- the staging ring, the frame assembler, the scheduler, the GPU hand-off,
//! the writer -- is the same code the thread-per-station `corr` runs, with no
//! re-export and no second copy.
//!
//! Built by default (`cargo build --release`); `--no-default-features` drops it
//! along with the `async-recv` feature and the `tokio` dependency.

use std::process::ExitCode;

fn main() -> ExitCode {
    corr_rs::async_recv::main()
}
