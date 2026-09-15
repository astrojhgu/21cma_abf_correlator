//! Integration test: the shipped configuration, the Rust view of it, and the
//! real `libcorr_cuda.so` must all agree.
//!
//! These checks are the ones `corr --check` performs interactively, plus the
//! failure paths a manual run would not bother to trigger, plus a full row
//! driven end to end through the staging ring. They talk to the shared object for
//! real, so a broken CMake build or a stale `.so` fails here rather than at 3 a.m.
//! during an observing run.

use std::path::PathBuf;

use corr_rs::config::Config;
use corr_rs::ffi;
use corr_rs::frame::FrameMath;
use corr_rs::gpu::Correlator;

fn shipped_config() -> (Config, corr_rs::RunPlan) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("config/correlator.yaml");
    let cfg = Config::from_yaml_file(&path).expect("shipped config must load");
    let plan = cfg.compile().expect("shipped config must compile");
    (cfg, plan)
}

#[test]
fn the_library_reports_the_abi_this_crate_was_built_against() {
    let banner = ffi::abi_banner();
    assert!(!banner.is_empty(), "empty ABI banner");
    assert!(
        banner.contains(&format!("abi={}", ffi::CORR_ABI_VERSION)),
        "banner {banner:?} does not advertise abi={}",
        ffi::CORR_ABI_VERSION
    );
}

#[test]
fn baseline_numbering_is_defined_by_the_library() {
    for n in 1..=24i32 {
        let expected = n * (n + 1) / 2;
        assert_eq!(
            Correlator::n_baseline(n).unwrap(),
            expected as usize,
            "n_station={n}"
        );

        let pairs = Correlator::baseline_pairs(n).unwrap();
        assert_eq!(pairs.len(), expected as usize, "n_station={n}");
        // Row major over j <= k, and every pair is a valid station index.
        let mut idx = 0;
        for j in 0..n {
            for k in j..n {
                assert_eq!(pairs[idx], (j, k), "n_station={n} baseline {idx}");
                idx += 1;
            }
        }
    }

    assert!(
        Correlator::n_baseline(0).is_err(),
        "n_station=0 must be rejected"
    );
    assert!(
        Correlator::n_baseline(-1).is_err(),
        "negative n_station must be rejected"
    );
}

#[test]
fn a_context_can_be_created_from_the_shipped_config() {
    let (_cfg, plan) = shipped_config();
    let corr = Correlator::new(&[0], corr_rs::config::SplitMode::BaselineRange.abi_value(), &plan).expect("corr_create must succeed on the shipped config");

    assert_eq!(corr.n_station(), plan.n_station as i32);
    assert_eq!(corr.n_freq(), plan.n_freq as i32);
    assert_eq!(corr.frames_per_slot(), plan.frames_per_slot as i32);
    assert_eq!(corr.rows_per_fetch(), plan.rows_per_fetch as i32);
    assert_eq!(corr.frames_per_row(), plan.n_cum as u64);

    let staging = corr.staging_all().expect("staging info");
    assert_eq!(staging.len(), plan.n_station);

    let frame_stride = 2 * plan.n_freq as usize;
    let slot_stride = frame_stride * plan.frames_per_slot as usize;
    let station_bytes = slot_stride * plan.block_slots as usize;

    for st in &staging {
        assert_eq!(st.frame_stride, frame_stride);
        assert_eq!(st.slot_stride, slot_stride);
        // base is already advanced to this station, so total_bytes is the extent
        // of that station's region, not the size of the whole allocation.
        assert_eq!(st.total_bytes, station_bytes);
        assert!(!st.base.is_null());
    }
    // Stations are contiguous, so the scheduler can index them arithmetically.
    for w in staging.windows(2) {
        // SAFETY: both pointers are inside the same allocation.
        assert_eq!(w[1].base, unsafe { w[0].base.add(station_bytes) });
    }
}

#[test]
fn frame_coordinates_address_the_staging_bytes_the_library_reports() {
    let (cfg, plan) = shipped_config();
    let corr = Correlator::new(&cfg.gpu.devices, cfg.gpu.split.abi_value(), &plan).unwrap();

    let fm = FrameMath::new(plan.packets_per_frame);
    assert_eq!(fm.packets_per_frame(), 2, "n_freq=4096 spans two packets");

    let st = corr.staging(0).unwrap();
    let fps = plan.frames_per_slot as u64;

    // Walk two full passes around the staging ring and check that every packet
    // lands on a distinct, 4096-byte aligned offset. The slot index has to be
    // part of the address: the ring is exactly what makes two frames with the
    // same index in different slots coexist.
    let first_counter = 500 * fps * fm.packets_per_frame();
    let mut touched = vec![false; st.total_bytes];

    for counter in first_counter..first_counter + 2 * fps * fm.packets_per_frame() {
        let frame_id = fm.frame_id(counter);
        let frame_in_slot = frame_id % fps;
        let slot = (frame_id / fps) % cfg.gpu.block_slots as u64;
        let payload_off = fm.slot_in_frame(counter) * 4096;

        let off = (slot * st.slot_stride as u64
            + frame_in_slot * st.frame_stride as u64
            + payload_off) as usize;

        assert!(
            off + 4096 <= st.total_bytes,
            "counter {counter} addresses beyond the station region: off={off}"
        );
        assert_eq!(off % 4096, 0, "packets must be 4096-byte aligned");
        assert!(
            !touched[off],
            "counter {counter} would overwrite an earlier packet at {off}"
        );
        touched[off] = true;
    }

    assert_eq!(
        touched.iter().filter(|t| **t).count(),
        (2 * fps * fm.packets_per_frame()) as usize,
        "every packet must map to a distinct byte offset"
    );
}

#[test]
fn a_full_row_runs_through_the_staging_ring() {
    let (cfg, plan) = shipped_config();
    let corr = Correlator::new(&cfg.gpu.devices, cfg.gpu.split.abi_value(), &plan).unwrap();

    let fps = plan.frames_per_slot as i32;
    let row_frames = plan.n_cum as i64;
    let slots = corr.block_slots();
    assert_eq!(slots, plan.block_slots as i32);

    // A short row, so the test exercises the ring and the partial last slot
    // without pushing a whole second of frames through the device.
    let short_row = row_frames.min(3 * fps as i64) as i32;

    // The library is told the slot *sequence*: it derives the physical slot from
    // it, and under the time-slice split the sequence is what routes the slot to
    // a device.
    let mut submits = 0u64;
    let mut done = 0i32;
    while done < short_row {
        let ft = (short_row - done).min(fps);
        if submits >= slots as u64 {
            corr.row_slot_free(submits - slots as u64)
                .expect("releasing a used slot");
        }
        corr.row_accumulate(submits, ft).expect("accumulating a slot");
        // The frame count must be the caller's, not the slot size: the staging
        // past `ft` still holds the previous batch's bytes.
        assert!(
            corr.row_accumulate(submits, ft).is_err(),
            "a slot already handed over must not be accepted twice"
        );
        submits += 1;
        done += ft;
    }

    corr.row_end(short_row as u64).expect("closing the row");
    assert_eq!(corr.rows_pending().unwrap(), 1);

    // A second row must be refused: rows_per_fetch is 1 and the first has not
    // been fetched, so closing another would overwrite it.
    let _ = corr.row_end(short_row as u64);
    assert!(
        corr.row_end(short_row as u64).is_err(),
        "closing a row with an unfetched row on the stack must be refused"
    );

    corr.rows_wait().expect("waiting for the row");

    let mut out = vec![0u8; corr.fetch_bytes().unwrap()];
    let rows = corr.rows_fetch(&mut out).expect("fetching the row");
    assert_eq!(rows, 1, "one row was closed, so one row must come back");
    assert_eq!(corr.rows_pending().unwrap(), 0, "fetch must clear the stack");

    // The staging was zeroed at create and nothing wrote to it, so the spectrum
    // of silence is exactly zero. Anything else means uninitialised device
    // memory reached the output.
    let nonzero = out.iter().filter(|b| **b != 0).count();
    assert_eq!(nonzero, 0, "zero input produced {nonzero} nonzero output bytes");

    // Release the slots still held so the ring is clean for the next row.
    let first_held = submits.saturating_sub(slots as u64);
    for i in (first_held..submits).rev() {
        corr.row_slot_free(i).expect("final release");
    }

    // A too-small fetch buffer must be reported before anything is copied.
    let mut tiny = vec![0u8; 8];
    assert!(corr.rows_fetch(&mut tiny).is_err());
}

#[test]
fn bad_slots_and_frame_counts_are_rejected() {
    let (cfg, plan) = shipped_config();
    let corr = Correlator::new(&cfg.gpu.devices, cfg.gpu.split.abi_value(), &plan).unwrap();

    assert!(
        corr.row_accumulate(0, 0).is_err(),
        "frames=0 must be rejected"
    );
    assert!(
        corr.row_accumulate(0, plan.frames_per_slot as i32 + 1).is_err(),
        "more frames than a slot holds must be rejected"
    );
    assert!(
        corr.row_accumulate(0, 0).is_err(),
        "zero frames must be rejected"
    );
    assert!(
        corr.row_accumulate(0, plan.frames_per_slot as i32 + 1).is_err(),
        "more frames than a slot holds must be rejected"
    );
    assert!(corr.row_end(0).is_err(), "a row of zero frames must be rejected");
    assert!(
        corr.row_slot_free(0).is_err(),
        "releasing a slot that was never submitted must be an error"
    );
}

#[test]
fn destroying_a_context_repeatedly_is_safe() {
    let (cfg, plan) = shipped_config();
    for _ in 0..3 {
        let corr = Correlator::new(&cfg.gpu.devices, cfg.gpu.split.abi_value(), &plan).unwrap();
        drop(corr);
    }
}
