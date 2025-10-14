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

function(dlp_define_core_options)
    # Core library options that affect runtime behavior
    option(DLP_ENABLE_LOGGING "Enable logging for DLP APIs" OFF)
    option(DLP_ENABLE_JIT_DEBUGGING "Enable JIT debugging help for DLP APIs" OFF)

    # Threading model configuration
    option(DLP_THREADING_MODEL "Threading model to use" "none")
    dlp_parse_threading_model(${DLP_THREADING_MODEL})

    # Propagate variables back to the caller
    set(DLP_ENABLE_LOGGING ${DLP_ENABLE_LOGGING} PARENT_SCOPE)
    set(DLP_ENABLE_JIT_DEBUGGING ${DLP_ENABLE_JIT_DEBUGGING} PARENT_SCOPE)
    set(DLP_THREADING_MODEL ${DLP_THREADING_MODEL} PARENT_SCOPE)
    set(DLP_ENABLE_OPENMP ${DLP_ENABLE_OPENMP} PARENT_SCOPE)
    set(DLP_ENABLE_PTHREAD ${DLP_ENABLE_PTHREAD} PARENT_SCOPE)
endfunction()
