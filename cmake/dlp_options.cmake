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

function(dlp_parse_threading_model DLP_THREADING_MODEL)
    set(THREADING_MODELS ${DLP_THREADING_MODEL})
    set(DLP_ENABLE_OPENMP OFF)
    set(DLP_ENABLE_PTHREAD OFF)

    if(NOT THREADING_MODELS)
        set(THREADING_MODELS "none")
    endif()

    foreach(THREADING_MODEL IN LISTS THREADING_MODELS)
        if (THREADING_MODEL STREQUAL "none")
            # Do nothing, 'none' is a valid option but doesn't enable anything
        elseif (THREADING_MODEL STREQUAL "openmp")
            set(DLP_ENABLE_OPENMP ON)
        elseif (THREADING_MODEL STREQUAL "pthread")
            set(DLP_ENABLE_PTHREAD ON)
        else()
            message(FATAL_ERROR "Invalid threading model: ${THREADING_MODEL}. Valid options are 'none', 'openmp', or 'pthread'.")
        endif()
    endforeach()

    set(DLP_ENABLE_OPENMP ${DLP_ENABLE_OPENMP} PARENT_SCOPE)
    set(DLP_ENABLE_PTHREAD ${DLP_ENABLE_PTHREAD} PARENT_SCOPE)
endfunction()

function(dlp_define_options)
    # Here are the user configurable options
    option(BUILD_TESTING "Build tests" OFF)
    option(BUILD_EXAMPLES "Build examples" OFF)
    option(BUILD_BENCHMARKS "Build benchmarks" OFF)
    option(BUILD_DOXYGEN "Generate Doxygen documentation" OFF)
    option(BUILD_SPHINX "Generate Sphinx documentation" OFF)
    option(DLP_ENABLE_ASAN "Enable Address Sanitizer" OFF)
    option(DLP_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
    option(DLP_ENABLE_TSAN "Enable Thread Sanitizer" OFF)
    option(DLP_CTEST_DISABLED "Disable ctest detection" ON)
    option(DLP_ENABLE_HIGH_PRECISION_FLOAT "Enable high precision float (double) support" OFF)
    option(DLP_ENABLE_DETAILED_DEBUG "Enable detailed debug information" OFF)
    option(DLP_ENABLE_LOGGING "Enable logging for DLP APIs" OFF)

    # Threading model.
    option(DLP_THREADING_MODEL "Threading model to use" "none")
    dlp_parse_threading_model(${DLP_THREADING_MODEL})
    set(DLP_ENABLE_OPENMP ${DLP_ENABLE_OPENMP} PARENT_SCOPE)
    set(DLP_ENABLE_PTHREAD ${DLP_ENABLE_PTHREAD} PARENT_SCOPE)

    # We propagte variables back to the caller
    set(BUILD_TESTING ${BUILD_TESTING} PARENT_SCOPE)
    set(BUILD_EXAMPLES ${BUILD_EXAMPLES} PARENT_SCOPE)
    set(BUILD_BENCHMARKS ${BUILD_BENCHMARKS} PARENT_SCOPE)
    set(BUILD_DOXYGEN ${BUILD_DOXYGEN} PARENT_SCOPE)
    set(BUILD_SPHINX ${BUILD_SPHINX} PARENT_SCOPE)
    set(DLP_THREADING_MODEL ${DLP_THREADING_MODEL} PARENT_SCOPE)
    set(DLP_ENABLE_ASAN ${DLP_ENABLE_ASAN} PARENT_SCOPE)
    set(DLP_ENABLE_UBSAN ${DLP_ENABLE_UBSAN} PARENT_SCOPE)
    set(DLP_ENABLE_TSAN ${DLP_ENABLE_TSAN} PARENT_SCOPE)
    set(DLP_CTEST_DISABLED ${DLP_TEST_DISABLED} PARENT_SCOPE)
    set(DLP_ENABLE_HIGH_PRECISION_FLOAT ${DLP_ENABLE_HIGH_PRECISION_FLOAT} PARENT_SCOPE)
    set(DLP_ENABLE_DETAILED_DEBUG ${DLP_ENABLE_DETAILED_DEBUG} PARENT_SCOPE)
	set(DLP_ENABLE_LOGGING ${DLP_ENABLE_LOGGING} PARENT_SCOPE)
endfunction()
