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

# Create test config header with path to test YAML files
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bench_test_config.hh.in
    ${CMAKE_CURRENT_BINARY_DIR}/bench_test_config.hh
    @ONLY
)

# Fetch google benchmark (needed for bench_metrics.cc compilation)
fetch_benchmark()

# Bench infrastructure sources (reuse from bench directory)
set(BENCH_INFRA_SOURCES
    ${PROJECT_SOURCE_DIR}/bench/src/bench_types.cc
    ${PROJECT_SOURCE_DIR}/bench/src/bench_metrics.cc
)

# Additional include directories for bench tests
set(BENCH_TEST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/bench/include
    ${PROJECT_BINARY_DIR}/bench  # For generated bench_config.hh
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated bench_test_config.hh
)

# Collect test source files
dlp_glob_sources(OUTPUT BENCH_TEST_SOURCES DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# Add each test file as a separate test
foreach(test_source ${BENCH_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    dlp_add_test(
        NAME ${test_name}
        SOURCES ${test_source} ${BENCH_INFRA_SOURCES} ${FRAMEWORK_SOURCES} ${UTILS_SOURCES} ${ADAPTORS_SOURCES}
        INCLUDE_DIRS ${TEST_INCLUDE_DIRS} ${BENCH_TEST_INCLUDE_DIRS}
        DEPENDS yaml-cpp benchmark::benchmark
    )
    dlp_set_global_compile_flags(${test_name} VISIBILITY PUBLIC)
endforeach()
