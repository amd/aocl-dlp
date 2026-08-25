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

# Clang's -fsanitize=undefined bundles the "function" check, which verifies a
# signature stored in the eight bytes preceding every indirect call target. The
# JIT emits kernels into raw executable pages that carry no such prologue, so
# the check reads off the front of the code page and faults. Drop just that one
# check; GCC has no equivalent and rejects the option, hence the guard. Resolved
# on call rather than on include, so the compiler id is already known.
function(dlp_get_ubsan_flags out_var)
    set(_flags -fsanitize=undefined)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang"
       OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND _flags -fno-sanitize=function)
    endif()
    set(${out_var} ${_flags} PARENT_SCOPE)
endfunction()

# GCC 13 through 16.1 enable detect_stack_use_after_return by default. The ASan
# fake stack is only 32-byte aligned (GCC PR120201), which SIGSEGVs AVX-512
# kernels that spill 64-byte-aligned __m512i locals. Disable only that mode;
# all other ASan checks remain enabled. Clang/AOCC, GCC 12.x, and GCC 16.2+
# are unaffected.
function(dlp_get_asan_flags out_var)
    set(_flags -fsanitize=address)
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "13"
       AND CMAKE_C_COMPILER_VERSION VERSION_LESS "16.2")
        list(APPEND _flags --param=asan-use-after-return=0)
    endif()
    set(${out_var} ${_flags} PARENT_SCOPE)
endfunction()

# Given a target, add the appropriate sanitizer flags to the targets
function(dlp_add_asan)
    if(DLP_ENABLE_ASAN)
        dlp_get_asan_flags(_asan_flags)
        # Add sanitizer flags to target and all dependencies.
        foreach(target ${ARGV})
            target_compile_options(${target} PUBLIC ${_asan_flags})
            target_link_options(${target} PUBLIC ${_asan_flags})
        endforeach()
    endif()
endfunction()

function(dlp_add_ubsan)
    if(DLP_ENABLE_UBSAN)
        dlp_get_ubsan_flags(_ubsan_flags)
        # Add sanitizer flags to target and all dependencies.
        foreach(target ${ARGV})
            target_compile_options(${target} PUBLIC ${_ubsan_flags})
            target_link_options(${target} PUBLIC ${_ubsan_flags})
        endforeach()
    endif()
endfunction()

function(dlp_add_tsan)
    if(DLP_ENABLE_TSAN)
        # Add sanitizer flags to target and all dependencies.
        foreach(target ${ARGV})
            target_compile_options(${target} PUBLIC -fsanitize=thread)
            target_link_options(${target} PUBLIC -fsanitize=thread)
        endforeach()
    endif()
endfunction()


function(dlp_add_all_sanitizers)
    dlp_add_asan(${ARGV})
    dlp_add_ubsan(${ARGV})
    dlp_add_tsan(${ARGV})
endfunction()

# Route the sanitizer flags through the dlp_compiler_flags interface target so
# they reach every translation unit, including the OBJECT libraries that hold
# the library implementation. Applying them only to the final library targets
# instruments nothing, because those targets compile no sources of their own on
# a UNIX build. Mirrors how dlp_compiler_flags_coverage carries both compile
# and link flags. Must be called after dlp_define_build_options().
#
function(dlp_setup_sanitizers)
    set(_san_flags "")
    if(DLP_ENABLE_ASAN)
        dlp_get_asan_flags(_asan_flags)
        list(APPEND _san_flags ${_asan_flags} -fno-omit-frame-pointer)
    endif()
    if(DLP_ENABLE_UBSAN)
        dlp_get_ubsan_flags(_ubsan_flags)
        list(APPEND _san_flags ${_ubsan_flags})
    endif()
    if(DLP_ENABLE_TSAN)
        list(APPEND _san_flags -fsanitize=thread)
    endif()

    if(_san_flags AND TARGET dlp_compiler_flags)
        target_compile_options(dlp_compiler_flags INTERFACE ${_san_flags})
        target_link_options(dlp_compiler_flags INTERFACE ${_san_flags})
        message(STATUS "Sanitizers enabled for all DLP targets: ${_san_flags}")
    endif()
endfunction()
