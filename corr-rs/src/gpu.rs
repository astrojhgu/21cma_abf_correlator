//! Safe wrapper around the `corr_cuda` context.
//!
//! The wrapper's job is to make two classes of mistake impossible from Rust:
//! building a `corr_config` whose `struct_size` or `abi_version` is wrong, and
//! forgetting to destroy a context. Everything else about ownership of slot
//! state stays the caller's responsibility, as the header documents.

use std::os::raw::c_void;

use anyhow::{bail, Context, Result};

use crate::config::RunPlan;
use crate::ffi::{self, corr_status};

/// Turn a failing status plus the library's message into an error.
fn check(st: corr_status, what: &str) -> Result<()> {
    if st.is_ok() {
        return Ok(());
    }
    let msg = ffi::last_error();
    if msg.is_empty() {
        bail!("{what} failed: {}", st.name());
    }
    bail!("{what} failed: {} ({msg})", st.name())
}

/// An owned correlator instance.
///
/// Not `Sync`: the intended use is that exactly one scheduler thread drives a
/// context. The receive threads only hold the raw staging pointers obtained from
/// [`Correlator::staging`].
pub struct Correlator {
    ctx: *mut c_void,
    cfg: ffi::corr_config,
    /// Frames per output row. Not part of the ABI: the library is told the frame
    /// count when a row is closed, precisely so that the integration length is a
    /// host-side decision.
    plan_n_cum: u64,
}

// SAFETY: the context is an opaque heap allocation with no thread affinity, and
// ownership can move to the scheduler thread. It is deliberately not Sync so
// that two threads cannot drive one stream by accident.
unsafe impl Send for Correlator {}

impl Correlator {
    /// Create a context over `devices` for the given run plan.
    ///
    /// One device is the single-card path and is exactly what the shipped
    /// baseline was measured on; several devices split the pair space between
    /// them, and the library is the only thing that knows how many there are.
    /// An empty slice would leave the library guessing, so it is rejected here
    /// rather than silently falling back to device 0.
    pub fn new(devices: &[i32], split_mode: i32, plan: &RunPlan) -> Result<Self> {
        if devices.is_empty() {
            bail!("no GPU device was given; gpu.devices must list at least one");
        }
        if devices.len() > ffi::CORR_MAX_DEVICES {
            bail!(
                "gpu.devices lists {} devices but the ABI carries at most {}; \
                 raise CORR_MAX_DEVICES in corr_cuda.h and ffi.rs together",
                devices.len(),
                ffi::CORR_MAX_DEVICES
            );
        }
        let mut device_list = [0i32; ffi::CORR_MAX_DEVICES];
        device_list[..devices.len()].copy_from_slice(devices);
        let cfg = ffi::corr_config {
            abi_version: ffi::CORR_ABI_VERSION,
            struct_size: std::mem::size_of::<ffi::corr_config>() as u32,
            // Left at 0: the list below is what selects the devices, and n_devices
            // is what says how many of them to use.
            device_id: 0,
            n_devices: devices.len() as i32,
            devices: device_list,
            split_mode,
            n_station: i32::try_from(plan.n_station).context("n_station overflows i32")?,
            n_freq: i32::try_from(plan.n_freq).context("n_freq overflows i32")?,
            frames_per_slot: i32::try_from(plan.frames_per_slot)
                .context("frames_per_slot overflows i32")?,
            chunk_frames: i32::try_from(plan.chunk_frames)
                .context("chunk_frames overflows i32")?,
            block_slots: i32::try_from(plan.block_slots).context("block_slots overflows i32")?,
            rows_per_fetch: i32::try_from(plan.rows_per_fetch)
                .context("rows_per_fetch overflows i32")?,
        };

        let mut ctx: *mut c_void = std::ptr::null_mut();
        // SAFETY: cfg is a fully initialised local of the exact layout the
        // header declares, and ctx is a valid out-parameter.
        let st = unsafe { ffi::corr_create(&cfg, &mut ctx) };
        check(st, "corr_create")?;
        if ctx.is_null() {
            bail!("corr_create reported success but returned a NULL context");
        }
        Ok(Correlator { ctx, cfg, plan_n_cum: plan.n_cum as u64 })
    }

    pub fn config(&self) -> &ffi::corr_config {
        &self.cfg
    }

    /// How many devices the library actually created. Asked of the library
    /// rather than assumed from the configuration: if the two ever disagree, the
    /// run is not doing what the plan says it is.
    pub fn n_devices(&self) -> Result<i32> {
        // SAFETY: ctx is a live handle owned by self.
        Ok(unsafe { ffi::corr_n_devices(self.ctx) })
    }

    /// One entry per device: which stations it holds and which baselines it owns.
    pub fn placement(&self) -> Result<Vec<ffi::corr_device_plan>> {
        let n = self.n_devices()?;
        if n < 0 {
            bail!("corr_n_devices reported {n}; the context is not usable");
        }
        let mut out = Vec::with_capacity(n as usize);
        for d in 0..n {
            let mut plan = ffi::corr_device_plan::default();
            // SAFETY: ctx is live and plan is a valid out-parameter.
            let st = unsafe { ffi::corr_device_plan_get(self.ctx, d, &mut plan) };
            check(st, "corr_device_plan_get")?;
            out.push(plan);
        }
        Ok(out)
    }

    /// The global baselines one device owns, in its own row-slice order.
    pub fn device_baselines(&self, device_index: i32, n_baseline: i32) -> Result<Vec<i32>> {
        if n_baseline < 0 {
            bail!("n_baseline is negative");
        }
        let mut v = vec![0i32; n_baseline as usize];
        // SAFETY: ctx is live and the buffer is sized for exactly what the
        // library says it will write.
        let st = unsafe {
            ffi::corr_device_baselines(self.ctx, device_index, v.as_mut_ptr(), n_baseline)
        };
        check(st, "corr_device_baselines")?;
        Ok(v)
    }

    /// Cumulative traffic through one device since the context was created.
    pub fn device_stats(&self, device_index: i32) -> Result<ffi::corr_device_stats> {
        let mut out = ffi::corr_device_stats::default();
        // SAFETY: ctx is live and out is a valid out-parameter.
        let st = unsafe { ffi::corr_device_stats_get(self.ctx, device_index, &mut out) };
        check(st, "corr_device_stats_get")?;
        Ok(out)
    }

    /// The global station indices one device holds a copy of, in its local order.
    pub fn device_stations(&self, device_index: i32, n_resident: i32) -> Result<Vec<i32>> {
        if n_resident < 0 {
            bail!("n_resident is negative");
        }
        let mut v = vec![0i32; n_resident as usize];
        // SAFETY: ctx is live and the buffer is sized for exactly what the
        // library says it will write.
        let st = unsafe {
            ffi::corr_device_stations(self.ctx, device_index, v.as_mut_ptr(), n_resident)
        };
        check(st, "corr_device_stations")?;
        Ok(v)
    }

    pub fn n_station(&self) -> i32 {
        self.cfg.n_station
    }

    pub fn n_freq(&self) -> i32 {
        self.cfg.n_freq
    }

    pub fn frames_per_slot(&self) -> i32 {
        self.cfg.frames_per_slot
    }

    pub fn block_slots(&self) -> i32 {
        self.cfg.block_slots
    }

    pub fn rows_per_fetch(&self) -> i32 {
        self.cfg.rows_per_fetch
    }

    /// Where the receive thread for `station` writes frames.
    pub fn staging(&self, station: i32) -> Result<ffi::corr_staging> {
        let mut out = ffi::corr_staging {
            base: std::ptr::null_mut(),
            frame_stride: 0,
            slot_stride: 0,
            total_bytes: 0,
            validity_offset: 0,
            validity_bytes_per_slot: 0,
        };
        // SAFETY: ctx is live and out is a valid pointer.
        let st = unsafe { ffi::corr_staging_info(self.ctx, station, &mut out) };
        check(st, "corr_staging_info")?;
        Ok(out)
    }

    /// Stage info for every station, in station index order.
    pub fn staging_all(&self) -> Result<Vec<ffi::corr_staging>> {
        (0..self.cfg.n_station).map(|s| self.staging(s)).collect()
    }

    /// Hand one filled staging slot to the GPU. `frames` is how many frames the
    /// caller actually wrote, which is what makes a partial last slot safe.
    /// Hand the frames in staging slot `slot_seq % block_slots` over, adding their
    /// correlation into the row being accumulated.
    ///
    /// The *sequence* is passed, not the physical slot: the ring reuses a slot
    /// every `block_slots` sequences, so the sequence is the identity of these
    /// frames, and under the time-slice split it is also what decides which
    /// device the slot belongs to.
    pub fn row_accumulate(&self, slot_seq: u64, frames: i32) -> Result<()> {
        // SAFETY: ctx is live.
        let st = unsafe { ffi::corr_row_accumulate(self.ctx, slot_seq as i64, frames) };
        check(st, "corr_row_accumulate")
    }

    /// Wait until every device that reads `slot_seq` may have its staging bytes
    /// written again.
    pub fn row_slot_free(&self, slot_seq: u64) -> Result<()> {
        // SAFETY: ctx is live.
        let st = unsafe { ffi::corr_row_slot_free(self.ctx, slot_seq as i64) };
        check(st, "corr_row_slot_free")
    }

    /// Close the current integration row, dividing it by `frames`.
    pub fn row_end(&self, frames: u64) -> Result<()> {
        // SAFETY: ctx is live.
        let st = unsafe { ffi::corr_row_end(self.ctx, frames) };
        check(st, "corr_row_end")
    }

    /// Block until every closed row is complete.
    pub fn rows_wait(&self) -> Result<()> {
        // SAFETY: ctx is live.
        let st = unsafe { ffi::corr_rows_wait(self.ctx) };
        check(st, "corr_rows_wait")
    }

    /// How many complete rows are waiting to be fetched.
    pub fn rows_pending(&self) -> Result<i32> {
        let mut n = 0i32;
        // SAFETY: ctx is live and n is a valid out-parameter.
        let st = unsafe { ffi::corr_rows_pending(self.ctx, &mut n) };
        check(st, "corr_rows_pending")?;
        Ok(n)
    }

    /// Copy every pending row into `out` in one transfer, returning how many
    /// rows were written.
    pub fn rows_fetch(&self, out: &mut [u8]) -> Result<i32> {
        let need = self.fetch_bytes()?;
        if out.len() < need {
            bail!("rows_fetch needs a {}-byte buffer, got {}", need, out.len());
        }
        let mut rows = 0i32;
        // SAFETY: ctx is live, out is a valid mutable slice of at least the
        // required length, and rows is a valid out-parameter.
        let st = unsafe {
            ffi::corr_rows_fetch(
                self.ctx,
                out.as_mut_ptr() as *mut c_void,
                out.len(),
                &mut rows,
            )
        };
        check(st, "corr_rows_fetch")?;
        Ok(rows)
    }

    /// Bytes one `rows_fetch` call can write: `rows_per_fetch` rows of
    /// `n_baseline * n_freq` complex64.
    pub fn fetch_bytes(&self) -> Result<usize> {
        let nbl = Self::n_baseline(self.cfg.n_station)?;
        Ok(nbl * self.cfg.n_freq as usize * 8 * self.cfg.rows_per_fetch as usize)
    }

    /// Bytes of one row's spectrum.
    pub fn row_bytes(&self) -> Result<usize> {
        let nbl = Self::n_baseline(self.cfg.n_station)?;
        Ok(nbl * self.cfg.n_freq as usize * 8)
    }

    /// Frames per output row, from the run plan rather than the ABI, because the
    /// integration length is a host-side decision.
    pub fn frames_per_row(&self) -> u64 {
        self.plan_n_cum
    }

    /// Bytes of one block's spectrum. `n_station` must be >= 1.
    pub fn n_baseline(n_station: i32) -> Result<usize> {
        // SAFETY: pure function of its argument.
        let n = unsafe { ffi::corr_n_baseline(n_station) };
        if n < 0 {
            bail!("corr_n_baseline({n_station}) rejected its argument");
        }
        Ok(n as usize)
    }

    /// `(j, k)` of every baseline with `j <= k`, row major, as defined by the
    /// library rather than reimplemented here.
    pub fn baseline_pairs(n_station: i32) -> Result<Vec<(i32, i32)>> {
        let nbl = Self::n_baseline(n_station)?;
        let mut buf = vec![0i32; 2 * nbl];
        // SAFETY: buf has exactly the length the library requires.
        let st = unsafe {
            ffi::corr_baseline_pairs(n_station, buf.as_mut_ptr(), buf.len() as i32)
        };
        check(st, "corr_baseline_pairs")?;
        Ok(buf.chunks_exact(2).map(|c| (c[0], c[1])).collect())
    }
}

impl Drop for Correlator {
    fn drop(&mut self) {
        // SAFETY: ctx came from corr_create and is destroyed exactly once. A
        // failure here cannot be reported usefully; it can only mean a leak or a
        // stuck transfer, and panicking in Drop would abort the run.
        let st = unsafe { ffi::corr_destroy(self.ctx) };
        if !st.is_ok() {
            eprintln!(
                "corr-rs: corr_destroy failed: {} ({})",
                st.name(),
                ffi::last_error()
            );
        }
    }
}
