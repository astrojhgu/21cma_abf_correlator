//! A row buffer in page-locked host memory.
//!
//! `corr_rows_fetch()` copies a row into whatever the caller hands it, and the
//! rate depends on what that is: measured on this machine, 1.31 MiB goes back at
//! 22.5 GB/s into page-locked memory and 12 GB/s into ordinary pageable memory,
//! because the pageable case is staged through a bounce buffer inside the
//! driver. At a one-second integration this is noise -- the fetch is 0.2% of the
//! run -- but a 40.96 us row is mostly fetch, and the difference decides how
//! short a row this program can do at all.
//!
//! The whole point is that the *same* buffer is then handed to the writer, so
//! [`crate::writer::Writer::take_row_buffer`] returns one of these and the
//! writer threads write straight out of it. Page-locked memory is ordinary
//! cached memory to the CPU; the only thing to remember is that a large
//! page-locked allocation counts against `RLIMIT_MEMLOCK`, so `Drop` is what
//! gives it back and a leaked buffer is not recovered by the allocator.

use std::ffi::c_void;

use anyhow::{bail, Result};

/// A page-locked byte buffer of a fixed length.
///
/// Deliberately not a `Vec`: the allocation comes from CUDA, not from the Rust
/// allocator, and `Drop` has to put it back there.
pub struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
}

// The allocation belongs to this process and nothing in it is thread-affine; the
// writer pool hands these between the pipeline thread and the writer threads.
unsafe impl Send for PinnedBuf {}

// Sound for the same reason as `Send` plus one more: the only way to get a
// `&mut` to the bytes is `&mut self`, so a shared reference can only ever hand
// out `&[u8]`. Sharing one batch buffer between the rows of that batch is
// exactly what `RowBuf` is shared for.
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    /// Allocate `len` page-locked bytes, zeroed.
    pub fn new(len: usize) -> Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-parameter and `len` is checked non-zero.
        let st = unsafe { crate::ffi::corr_host_alloc(len, &mut raw) };
        if !st.is_ok() || raw.is_null() {
            bail!(
                "cudaHostAlloc({len} bytes) for a row buffer failed ({}); \
                 page-locked memory counts against RLIMIT_MEMLOCK",
                st.name()
            );
        }
        // Deliberately not zeroed: the allocation is only ever read after a fetch
        // has written every byte the caller is told about (`corr_rows_fetch`
        // reports how many rows it wrote, and only those are handed on), and
        // zeroing 1.31 MB costs about as much as the fetch itself -- measured as a
        // 94 us per row gap at one frame per row before this was removed. Page
        // locked memory is populated by the driver at allocation, so nothing is
        // deferred to the first fetch either.
        Ok(PinnedBuf {
            ptr: raw as *mut u8,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` owns `len` initialised bytes for as long as `self` lives.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` rules out aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl std::ops::Deref for PinnedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for PinnedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from corr_host_alloc and is freed exactly once.
        unsafe { crate::ffi::corr_host_free(self.ptr as *mut c_void) };
    }
}

impl std::fmt::Debug for PinnedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PinnedBuf({} bytes)", self.len)
    }
}

/// One row's buffer, page-locked or ordinary.
///
/// Page-locking only pays when the fetch lands in the buffer -- one row per
/// fetch, which is the setting short rows need. With `rows_per_fetch > 1` the
/// fetch goes into the caller's own buffer and is split out afterwards, so a
/// pinned row buffer would buy nothing and would still count against
/// `RLIMIT_MEMLOCK`: at 24 stations one row is 39 MB and the queue holds tens of
/// them.
pub enum RowBuf {
    Pinned(PinnedBuf),
    Heap(Vec<u8>),
}

impl RowBuf {
    pub fn new(len: usize, pinned: bool) -> Result<Self> {
        if pinned {
            Ok(RowBuf::Pinned(PinnedBuf::new(len)?))
        } else {
            Ok(RowBuf::Heap(vec![0u8; len]))
        }
    }

    pub fn len(&self) -> usize {
        match self {
            RowBuf::Pinned(b) => b.len(),
            RowBuf::Heap(b) => b.len(),
        }
    }

    pub fn is_pinned(&self) -> bool {
        matches!(self, RowBuf::Pinned(_))
    }

    // Inherent, not only through `Deref`: without these, `buf.as_mut_slice()`
    // resolves to an unstable method on `str` somewhere up the auto-deref chain.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            RowBuf::Pinned(b) => b.as_slice(),
            RowBuf::Heap(b) => b.as_slice(),
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            RowBuf::Pinned(b) => b.as_mut_slice(),
            RowBuf::Heap(b) => b.as_mut_slice(),
        }
    }
}

impl std::ops::Deref for RowBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            RowBuf::Pinned(b) => b.as_slice(),
            RowBuf::Heap(b) => b.as_slice(),
        }
    }
}

impl std::ops::DerefMut for RowBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            RowBuf::Pinned(b) => b.as_mut_slice(),
            RowBuf::Heap(b) => b.as_mut_slice(),
        }
    }
}
