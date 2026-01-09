#
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (
# INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

include(FetchContent)

#
# Fetch Google Benchmark dependency
#
# Creates benchmark and benchmark_main targets
# FetchContent automatically handles caching - only fetches once
# Requires gtest to be fetched first (benchmark dependency)
#
function(fetch_benchmark)
    MESSAGE(WARNING "By enabling Google Benchmark, you agree to its license terms: https://github.com/google/benchmark/blob/main/LICENSE")

    # Ensure gtest is available first (benchmark has gtest dependency)
    fetch_gtest()

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.4
    )

    # Disable unnecessary benchmark components to speed up build
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)

    # FetchContent_MakeAvailable handles fetching and caching automatically
    FetchContent_MakeAvailable(benchmark)
endfunction()

function(create_bench_config)
    # Parameters
    set(config_dir ${ARGV0})
    set(config_input_header ${ARGV1})
    set(config_output_header ${ARGV2})

    # Current CONFIG DIR
    set(BENCH_CONFIG_DIR ${config_dir})

    # Configure the file
    configure_file(${config_input_header} ${config_output_header})
endfunction()

#
# Add a benchmark executable with optional object library reuse
#
# This function creates a benchmark executable, linking it with necessary dependencies
# and optionally reusing pre-compiled object libraries to avoid redundant compilation.
#
# Parameters:
#   NAME            - Name of the benchmark executable
#   SOURCES         - Benchmark-specific source files
#   OBJECT_LIBS     - Pre-compiled object libraries to link (optional)
#   INCLUDE_DIRS    - Additional include directories
#   DEPENDS         - Additional dependencies
#   DISABLED        - If present, marks benchmark as disabled
#
# Example:
#   dlp_add_benchmark(
#       NAME bench_gemm
#       SOURCES bench_gemm.cc
#       OBJECT_LIBS bench_framework_obj test_adaptors_obj
#       DEPENDS yaml-cpp numa
#   )
#
function(dlp_add_benchmark)
    # Parse arguments for the benchmark function
    set(options DISABLED)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES OBJECT_LIBS DEPENDS INCLUDE_DIRS)

    cmake_parse_arguments(DLP_BENCH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Check if required arguments are provided
    if(NOT DLP_BENCH_NAME)
        message(FATAL_ERROR "dlp_add_benchmark: NAME argument is required")
    endif()

    if(NOT DLP_BENCH_SOURCES)
        message(FATAL_ERROR "dlp_add_benchmark: SOURCES argument is required")
    endif()

    # Create benchmark executable with sources and object libraries
    if(DLP_BENCH_OBJECT_LIBS)
        # Build list of object library contents using generator expressions
        set(OBJECT_LIB_CONTENTS "")
        foreach(obj_lib ${DLP_BENCH_OBJECT_LIBS})
            list(APPEND OBJECT_LIB_CONTENTS $<TARGET_OBJECTS:${obj_lib}>)
        endforeach()

        add_executable(${DLP_BENCH_NAME} ${DLP_BENCH_SOURCES} ${OBJECT_LIB_CONTENTS})
    else()
        # Fallback to old behavior if no object libraries specified
        add_executable(${DLP_BENCH_NAME} ${DLP_BENCH_SOURCES})
    endif()

    # Choose library target based on static linking preference
    if(DLP_BENCHMARKS_LINK_STATIC)
        set(DLP_LIBRARY_TARGET ${PROJECT_NAME}_static)
    else()
        set(DLP_LIBRARY_TARGET ${PROJECT_NAME})
    endif()

    # Link with Google Benchmark and the main project library
    # Note: For static library, whole-archive is automatically applied via INTERFACE_LINK_OPTIONS
    target_link_libraries(${DLP_BENCH_NAME}
        PRIVATE
            benchmark
            ${DLP_LIBRARY_TARGET}
            ${DLP_BENCH_DEPENDS}
    )

    # Add include directories - follow modern CMake target-based approach
    target_include_directories(${DLP_BENCH_NAME}
        PRIVATE
            ${DLP_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/include
            ${BENCH_INCLUDE_DIRS}
    )

    # Set compiler flags
    dlp_set_global_compile_flags(${DLP_BENCH_NAME})

endfunction()

# Example of adding a specific test with additional dependencies
# dlp_add_benchmark(
#     NAME specific_feature_test
#     SOURCES specific_feature_test.cc
#     DEPENDS some_additional_dependency
#     INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
# )

# Example of a disabled test
# dlp_add_benchmark(
#     NAME disabled_test
#     SOURCES disabled_test.cc
#     INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
#     DISABLED
# )

function(dlp_define_benchmarking_options)
    # Benchmarking infrastructure options
    option(BUILD_BENCHMARKS "Build benchmark programs" OFF)
    option(DLP_BENCHMARKS_LINK_STATIC "Link benchmarks with static AOCL-DLP library for better performance" OFF)

    # Propagate variables back to the caller
    set(BUILD_BENCHMARKS ${BUILD_BENCHMARKS} PARENT_SCOPE)
    set(DLP_BENCHMARKS_LINK_STATIC ${DLP_BENCHMARKS_LINK_STATIC} PARENT_SCOPE)
endfunction()
