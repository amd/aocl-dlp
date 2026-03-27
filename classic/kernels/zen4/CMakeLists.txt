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

# CMake file for Zen4 kernels

# Get all source files in the zen4 directory recursively
dlp_glob_sources_recursive(ZEN4_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

# Note: FP16 kernel sources (pack/reorder) are always compiled - they use
# portable uint16_t type and standard AVX-512 integer intrinsics (not FP16-specific).
# JIT handles native FP16 compute instructions at runtime.

# Create the zen4 kernel object library
add_library(dlp_kernel_zen4 OBJECT ${ZEN4_SOURCES})

if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/s8s8s32/dlp_gemm_6x64rowmajor_s8_grp_amd512vnni.c
        ${CMAKE_CURRENT_SOURCE_DIR}/s8s8s32/dlp_gemm_m_fringe_s8_grp_amd512vnni.c
        ${CMAKE_CURRENT_SOURCE_DIR}/s8s8s32/dlp_gemm_mn_fringe_s8_grp_amd512vnni.c
        ${CMAKE_CURRENT_SOURCE_DIR}/s8s8s32/dlp_gemm_n_fringe_s8_grp_amd512vnni.c
        PROPERTIES COMPILE_FLAGS "-fno-tree-partial-pre"
    )
endif()

# Set include directories for zen4 kernels
set(ZEN4_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/bf16bf16f32
    ${CMAKE_CURRENT_SOURCE_DIR}/f32f32f32
    ${CMAKE_CURRENT_SOURCE_DIR}/fp16fp16fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/f32f16f32
    ${CMAKE_CURRENT_SOURCE_DIR}/s8s8s32
    ${CMAKE_CURRENT_SOURCE_DIR}/u8s8s32
)

# Make all include directories public
target_include_directories(dlp_kernel_zen4
    PRIVATE
        ${ZEN4_INCLUDES}
        ${CLASSIC_INCLUDES}
)

# Apply compiler flags using our centralized functions
dlp_set_global_compile_flags(dlp_kernel_zen4 VISIBILITY PRIVATE)
dlp_set_arch_flags(dlp_kernel_zen4 zen4)

# Output the found sources (optional, for debugging)
# message(STATUS "Zen4 kernel sources: ${ZEN4_SOURCES}")

# Export the variables to parent scope
set(ZEN4_SOURCES ${ZEN4_SOURCES} PARENT_SCOPE)
set(ZEN4_INCLUDES ${ZEN4_INCLUDES} PARENT_SCOPE)
