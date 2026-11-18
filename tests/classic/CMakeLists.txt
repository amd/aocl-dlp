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

# Create test config
create_test_config(${CMAKE_CURRENT_SOURCE_DIR}/configs ${CMAKE_CURRENT_SOURCE_DIR}/../include/test_config.hh.in ${CMAKE_CURRENT_BINARY_DIR}/test_config.hh)

# Collect test source files
dlp_glob_sources(OUTPUT TEST_SOURCES DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

#
# Add each test executable
# Reuse parent test framework and adaptor object libraries to eliminate duplication
# Each test only compiles its own source file, then links with pre-compiled objects
# Note: dlp::onednn provides compile definitions and linking automatically (or is empty if disabled)
#
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    # test_tiny_route includes internal headers and is added explicitly.
    if(test_name STREQUAL "test_tiny_route")
        continue()
    endif()
    dlp_add_test(
        NAME ${test_name}
        SOURCES ${test_source}
        OBJECT_LIBS test_framework_obj test_adaptors_obj
        INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
        DEPENDS yaml-cpp dlp::onednn
    )
    if(DLP_ENABLE_OPENMP)
        target_link_libraries(${test_name} PRIVATE dlp::openmp)
    endif()
    dlp_set_global_compile_flags(${test_name} VISIBILITY PUBLIC)
endforeach()

# Internal-header test: routes through the tiny header, which needs src/include.
dlp_add_test(
    NAME test_tiny_route
    SOURCES test_tiny_route.cc
    OBJECT_LIBS test_framework_obj test_adaptors_obj
    INCLUDE_DIRS ${TEST_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/include
    DEPENDS yaml-cpp dlp::onednn
)
if(DLP_ENABLE_OPENMP)
    target_link_libraries(test_tiny_route PRIVATE dlp::openmp)
endif()
dlp_set_global_compile_flags(test_tiny_route VISIBILITY PUBLIC)

# Direct Tiny F32 leaves run only for single-thread calls. The sweep is one
# CTest entry; the binary checks its own thread precondition.
dlp_add_test(
    NAME test_gemm_f32_tiny_sgemm
    SOURCES test_gemm.cc
    OBJECT_LIBS test_framework_obj test_adaptors_obj
    INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
    DEPENDS yaml-cpp dlp::onednn
    SINGLE_TEST
    ENVIRONMENT "DLP_NUM_THREADS=1;OMP_NUM_THREADS=1;DLP_IC_NT=1;DLP_JC_NT=1;DLP_REQUIRE_SINGLE_THREAD=1"
    EXTRA_ARGS -f ${CMAKE_CURRENT_SOURCE_DIR}/configs/f32_tiny_sgemm_test_config.yaml --gtest_filter=YamlConfigurations*
)
if(DLP_ENABLE_OPENMP)
    target_link_libraries(test_gemm_f32_tiny_sgemm PRIVATE dlp::openmp)
endif()
dlp_set_global_compile_flags(test_gemm_f32_tiny_sgemm VISIBILITY PUBLIC)

# The same sweep pinned to the AVX2 tier. AVX2 pinning selects Zen3 and the
# AVX2 YMM preference, so the AVX2 leaves run on an AVX-512 host too.
add_test(NAME test_gemm_f32_tiny_sgemm_avx2
    COMMAND test_gemm_f32_tiny_sgemm -f ${CMAKE_CURRENT_SOURCE_DIR}/configs/f32_tiny_sgemm_test_config.yaml --gtest_filter=YamlConfigurations*)
set_tests_properties(test_gemm_f32_tiny_sgemm_avx2 PROPERTIES
    ENVIRONMENT "DLP_NUM_THREADS=1;OMP_NUM_THREADS=1;DLP_IC_NT=1;DLP_JC_NT=1;DLP_REQUIRE_SINGLE_THREAD=1;AOCL_DLP_ENABLE_INSTRUCTIONS=avx2"
    PASS_REGULAR_EXPRESSION "\\[  PASSED  \\] [1-9][0-9]* test"
    FAIL_REGULAR_EXPRESSION "\\[  FAILED  \\]|did not match any test")
