cmake_minimum_required(VERSION 3.20)

project(corr_cuda
        VERSION 0.1.0
        DESCRIPTION "GPU side of the real-time correlator (C ABI shared library)"
        LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# corr-rs/build.rs drives this project with -DCMAKE_BUILD_TYPE=Release, but make
# a plain `cmake ..` default to Release too: a Debug build of this library is
# useless for the 12 GB/s path and easy to mistake for a performance bug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# L20 is compute capability 8.9. Pinning the architecture keeps the build
# reproducible and stops nvcc from emitting a fat binary for GPUs this machine
# does not have.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# libcorr_cuda.so -- the single artifact corr-rs links against.
#
# corr_cuda.cpp is the C ABI layer (validation, context, pinned staging, slot
# state machine); kernels.cu is the device work. They talk through
# src/gpu_internal.h.
# ---------------------------------------------------------------------------
add_library(corr_cuda SHARED
  src/corr_cuda.cpp
  src/kernels.cu
)

add_library(corr_cuda::corr_cuda ALIAS corr_cuda)

target_include_directories(corr_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(corr_cuda PRIVATE CUDA::cudart CUDA::cufft)

# -Wpedantic is for our C++ only. Applying it to CUDA sources makes nvcc's own
# generated line directives warn ("style of line directive is a GCC extension"),
# which buries real warnings under hundreds of lines of noise.
target_compile_options(corr_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

set_target_properties(corr_cuda PROPERTIES
  VERSION   ${PROJECT_VERSION}
  SOVERSION 1
)

# Ask the linker to record where libcudart lives, so that corr-rs does not have
# to guess a CUDA path at run time.
set_target_properties(corr_cuda PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS corr_cuda LIBRARY DESTINATION lib)
install(FILES include/corr_cuda.h DESTINATION include)

# ---------------------------------------------------------------------------
# C-side ABI test. Deliberately independent of corr-rs: if this passes and the
# Rust integration test fails, the shared object is fine and the bug is in the
# Rust binding.
# ---------------------------------------------------------------------------
enable_testing()

add_executable(abi_smoke tests/abi_smoke.c)
target_link_libraries(abi_smoke PRIVATE corr_cuda)
target_compile_options(abi_smoke PRIVATE -Wall -Wextra)

add_test(NAME abi_smoke COMMAND abi_smoke)

# The multi-device path: one correct answer out of N cards. It skips itself with a
# message when fewer than two devices are visible, so on a single-card machine it
# says so instead of passing vacuously.
add_executable(multi_device tests/multi_device.c)
target_link_libraries(multi_device PRIVATE corr_cuda CUDA::cudart)
target_compile_options(multi_device PRIVATE -Wall -Wextra)
add_test(NAME multi_device COMMAND multi_device)

# Throughput harness. Not a test: it answers "can the GPU keep up", which is a
# question for a human reading numbers, not a pass/fail.
# The multi-card host-link measurement. Not a test either: it answers "what does
# the machine's aggregate H2D do when N cards pull the same staging", which is
# the number the multi-card split has to fit into.
add_executable(multi_h2d_bw tests/multi_h2d_bw.c)
target_link_libraries(multi_h2d_bw PRIVATE corr_cuda CUDA::cudart pthread)
target_compile_options(multi_h2d_bw PRIVATE -Wall -Wextra)

add_executable(bench tests/bench.c)
target_link_libraries(bench PRIVATE corr_cuda)
target_compile_options(bench PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# Numerical test against an independent numpy reference.
#
# gen_golden.py writes the input frames and the expected spectra; golden.c drives
# the public ABI and compares. Two configurations are covered: one whose last
# chunk is partial (300 = 128+128+44, the case a naive loop gets wrong) and one
# at the real 15-station deployment size.
#
# The interpreter is probed for numpy rather than merely for being Python.
# find_package(Python3) picks /usr/bin/python3.11 on this host, which has no
# numpy, while /usr/bin/python3 does -- so "a python exists" is the wrong test.
# ---------------------------------------------------------------------------
find_program(CORR_PYTHON3 NAMES python3 python)
set(CORR_HAVE_NUMPY FALSE)
if(CORR_PYTHON3)
  execute_process(COMMAND ${CORR_PYTHON3} -c "import numpy"
                  RESULT_VARIABLE CORR_NUMPY_RC
                  OUTPUT_QUIET ERROR_QUIET)
  if(CORR_NUMPY_RC EQUAL 0)
    set(CORR_HAVE_NUMPY TRUE)
  endif()
endif()

if(CORR_HAVE_NUMPY)
  add_executable(golden tests/golden.c)
  # libm is not linked by default on this toolchain; sqrt() needs it.
  target_link_libraries(golden PRIVATE corr_cuda m)
  target_compile_options(golden PRIVATE -Wall -Wextra)

  function(corr_add_golden name n_station n_freq n_cum frames_per_slot rows)
    # Not under ${CMAKE_CURRENT_BINARY_DIR}/golden: that path is the test
    # executable itself, not a directory.
    set(datadir ${CMAKE_CURRENT_BINARY_DIR}/golden_data/${name})
    add_test(NAME gen_${name}
             COMMAND ${CORR_PYTHON3}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/gen_golden.py
                     ${n_station} ${n_freq} ${n_cum} ${frames_per_slot} ${rows}
                     ${datadir})
    add_test(NAME ${name} COMMAND golden ${datadir})
    set_tests_properties(${name} PROPERTIES DEPENDS gen_${name})
  endfunction()

  # 300 frames = 128+128+44, so a row spans three staging slots and the last one
  # is partial; two rows share the device stack and leave in one transfer.
  corr_add_golden(golden_partial_slot 4 2048 300 128 2)
  # 15 stations, the deployment size, with 4096-point frames.
  corr_add_golden(golden_15st 15 4096 200 64 2)
  message(STATUS "numpy golden test enabled, using ${CORR_PYTHON3}")
else()
  message(WARNING
    "no python3 with numpy found (looked at: ${CORR_PYTHON3}); the numerical "
    "golden test against numpy will NOT be registered. Install numpy or set "
    "-DCORR_PYTHON3=/path/to/python3.")
endif()

# Pure PCIe H2D bandwidth probe. Answers whether the ~15 GB/s pipeline ceiling
# is the link or the access pattern.
add_executable(h2d_bw tests/h2d_bw.c)
target_link_libraries(h2d_bw PRIVATE CUDA::cudart)
target_compile_options(h2d_bw PRIVATE -Wall -Wextra)
