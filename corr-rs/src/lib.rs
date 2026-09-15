//! corr-rs -- host side of the real-time correlator.
//!
//! Receives the front-end's UDP multicast streams, assembles them into FFT
//! frames, schedules integration blocks, hands them to the GPU side through the
//! `corr_cuda` C ABI, and writes one `complex64[ntime][n_freq]` file per
//! baseline.
//!
//! The algorithmic decisions this crate is built on are frozen in HANDOFF.md
//! section 4 and mirrored in `config/correlator.yaml`. In particular:
//!
//! * one FFT per station per frame, never per baseline;
//! * `Y_j * conj(Y_k)` for every `j <= k`, accumulated in complex64;
//! * R2C with the Nyquist bin dropped, no `fftshift`, no FFT normalisation;
//! * `n_cum` counts FFT frames, not packets, and the block sum is divided by
//!   `n_cum` regardless of how many frames actually arrived;
//! * a frame whose packets did not all arrive is zero filled, and the count of
//!   valid frames is recorded so the amplitude bias is explainable downstream.
//!
//! Module map, and what each one owns:
//!
//! | module | responsibility |
//! |---|---|
//! | [`config`]   | parse and validate `correlator.yaml`, derive the run plan |
//! | [`frame`]    | packet counter -> frame coordinates, missing-frame tracking |
//! | [`assemble`] | per-station packet -> staging bytes, and slot ownership |
//! | [`sched`]    | watermark, slot submission and row boundaries |
//! | [`net`]      | the multicast sockets and their options |
//! | [`run`]      | the receive path: sockets, assemblers, scheduler loop |
//! | [`writer`]   | the output files, and resuming a run that was interrupted |
//! | [`gpu`]      | safe wrapper over the `corr_cuda` C ABI |
//! | [`ffi`]      | the raw C ABI declarations |
//!
//! The receive threads, the block scheduler and the writer are not here yet:
//! this build establishes the contract between the two halves and the pieces
//! that can be verified without hardware or traffic.

pub mod assemble;
pub mod config;
pub mod ffi;
pub mod frame;
pub mod gpu;
pub mod net;
pub mod pipeline;
pub mod receiver;
pub mod report;
pub mod run;
pub mod sched;
pub mod stop;
pub mod writer;

pub use config::{Config, RunPlan};
pub use gpu::Correlator;
