//! Build script: drive the CMake project in ../corr-cuda and link its C ABI.
//!
//! One `cargo build` is meant to build both halves, so the CUDA side is not a
//! separate manual step that can silently go stale. The shared object lands in
//! two places on the link line:
//!
//!   * the OUT_DIR copy, baked in as an absolute rpath, so `cargo run` and
//!     `cargo test` work with no environment setup at all;
//!   * `$ORIGIN`, so a deployed layout of `corr` next to `libcorr_cuda.so`
//!     works without the build directory existing.
//!
//! The absolute dev rpath is a build artefact, not a deployment decision: a
//! release tarball ships the .so beside the binary and relies on `$ORIGIN`.

use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cuda_src = manifest_dir
        .parent()
        .expect("corr-rs must live next to corr-cuda")
        .join("corr-cuda");

    if !cuda_src.join("CMakeLists.txt").is_file() {
        panic!(
            "cannot find the CUDA project at {}; corr-rs and corr-cuda must be siblings",
            cuda_src.display()
        );
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let lib_out = out_dir.join("corr-cuda-lib");
    std::fs::create_dir_all(&lib_out).expect("create corr-cuda-lib");

    // Building the .so in release mode regardless of the cargo profile is
    // deliberate: a debug-built .so would be compared against a release-built
    // host in any throughput measurement, which is a trap.
    let dst = cmake::Config::new(&cuda_src)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_LIBRARY_OUTPUT_DIRECTORY", &lib_out)
        .define("CMAKE_RUNTIME_OUTPUT_DIRECTORY", &lib_out)
        .build_target("corr_cuda")
        .build();

    let lib_path = find_library(&lib_out).unwrap_or_else(|| {
        panic!(
            "libcorr_cuda.so was not produced in {} (cmake prefix {})",
            lib_out.display(),
            dst.display()
        )
    });

    println!("cargo:rustc-link-search=native={}", lib_out.display());
    println!("cargo:rustc-link-lib=dylib=corr_cuda");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_out.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    // Re-run when anything on the CUDA side changes. Watching the directory
    // would also fire on the build/ tree, so list the inputs instead.
    println!("cargo:rerun-if-changed={}", cuda_src.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", cuda_src.join("include").display());
    println!("cargo:rerun-if-changed={}", cuda_src.join("src").display());
    println!("cargo:rerun-if-changed=build.rs");

    println!(
        "cargo:warning=corr-rs: linked {}",
        lib_path.display()
    );
}

/// Accept the versioned name CMake actually creates when SOVERSION is set.
fn find_library(dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libcorr_cuda.so"))
                .unwrap_or(false)
        })
        .collect();
    candidates.sort();
    // Prefer the plain name if the symlink exists, else the versioned file.
    candidates
        .iter()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("libcorr_cuda.so"))
        .cloned()
        .or_else(|| candidates.into_iter().next())
}
