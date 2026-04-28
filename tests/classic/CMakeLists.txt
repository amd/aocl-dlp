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
#
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    dlp_add_test(
        NAME ${test_name}
        SOURCES ${test_source}
        OBJECT_LIBS test_framework_obj test_adaptors_obj
        INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
        DEPENDS yaml-cpp
    )
    if(DLP_ENABLE_OPENMP)
        target_link_libraries(${test_name} PRIVATE dlp::openmp)
    endif()
    dlp_set_global_compile_flags(${test_name} VISIBILITY PUBLIC)
endforeach()

#
# Runtime threading tests with specific env var combos for alias resolution.
# Each variant invokes the same test_runtime_threading binary under a
# different environment, exercising the EnvVarAliasTest cases that key off
# DLP_NUM_THREADS / MKL_NUM_THREADS / OMP_NUM_THREADS / AOCL_ENABLE_INSTRUCTIONS.
# CTest ENVIRONMENT property takes a CMake list (semicolon-separated) of
# KEY=VAL entries.
#
if(TARGET test_runtime_threading)
    add_test(NAME test_runtime_threading_omp_alias
             COMMAND test_runtime_threading)
    set_tests_properties(test_runtime_threading_omp_alias PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=7")

    add_test(NAME test_runtime_threading_mkl_alias
             COMMAND test_runtime_threading)
    set_tests_properties(test_runtime_threading_mkl_alias PROPERTIES
        ENVIRONMENT "MKL_NUM_THREADS=5")

    add_test(NAME test_runtime_threading_mkl_over_omp
             COMMAND test_runtime_threading)
    set_tests_properties(test_runtime_threading_mkl_over_omp PROPERTIES
        ENVIRONMENT "MKL_NUM_THREADS=8;OMP_NUM_THREADS=16")

    add_test(NAME test_runtime_threading_primary_wins
             COMMAND test_runtime_threading)
    set_tests_properties(test_runtime_threading_primary_wins PROPERTIES
        ENVIRONMENT "DLP_NUM_THREADS=2;MKL_NUM_THREADS=8;OMP_NUM_THREADS=16")

    add_test(NAME test_runtime_threading_instr_alias
             COMMAND test_runtime_threading)
    set_tests_properties(test_runtime_threading_instr_alias PROPERTIES
        ENVIRONMENT "AOCL_ENABLE_INSTRUCTIONS=avx2")
endif()
