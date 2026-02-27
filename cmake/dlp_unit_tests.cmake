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

function (add_unit_tests_paths)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/jit/amdzen/unit_tests)
endfunction()

# Function to add unit tests from a directory
# Parameters:
#   INCLUDE_DIRS - List of include directories for the tests
#   DEPENDS - List of dependencies (libraries) for the tests (e.g aocl_dlp_classic/aocl_dlp_plus)
function (dlp_add_unit_tests)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs INCLUDE_DIRS DEPENDS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Collect test source files from current directory
    dlp_glob_sources(OUTPUT TEST_SOURCES DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    # Add each test file as a separate test
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)

        dlp_add_test(
            NAME ${test_name}
            SOURCES ${test_source}
            INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
            DEPENDS ${ARG_DEPENDS}
        )
        dlp_set_global_compile_flags(${test_name} VISIBILITY PUBLIC)
    endforeach()
endfunction()
