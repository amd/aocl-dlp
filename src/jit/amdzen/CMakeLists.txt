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

# Get all source files in the amdzen directory recursively
dlp_glob_sources_recursive(AMDZEN_JIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

# Create object library for AMD Zen JIT code
add_library(aocl_dlp_jit_amdzen OBJECT ${AMDZEN_JIT_SOURCES})

# Enable position-independent code for the object library
set_property(TARGET aocl_dlp_jit_amdzen PROPERTY POSITION_INDEPENDENT_CODE ON)

# Apply global compiler flags (generic, release/debug, etc.)
dlp_set_global_compile_flags(aocl_dlp_jit_amdzen)

# Apply JIT-specific flags (includes -mcmodel=medium for x86)
dlp_set_jit_flags(aocl_dlp_jit_amdzen)

# Set include directories - inherit from parent
target_include_directories(aocl_dlp_jit_amdzen PRIVATE
    ${DLP_PLUS_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../xbyak  # Access to xbyak headers
)

# Propagate any JIT-specific definitions if needed
if(DLP_ENABLE_JIT_DEBUGGING)
    target_compile_definitions(aocl_dlp_jit_amdzen PRIVATE DLP_JIT_DEBUG=1)
endif()

# Optional: Log what we're building
message(STATUS "Building AMD Zen JIT object library (aocl_dlp_jit_amdzen)")
