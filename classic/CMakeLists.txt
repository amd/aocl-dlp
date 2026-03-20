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

# Get all source files in the current directory
dlp_glob_sources(CLASSIC_SOURCES)

set(CLASSIC_INCLUDES ${CLASSIC_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/include ${KERNELS_INCLUDES})
# DLP Plus dependencies.
set(CLASSIC_INCLUDES ${CLASSIC_INCLUDES} ${PROJECT_SOURCE_DIR}/src/include)

# FP16 code always compiles - it uses uint16_t and standard AVX-512 intrinsics.
# Runtime CPU detection determines if FP16 operations can execute.
# The -mavx512fp16 compiler flag (GCC 12+) is handled in dlp_compiler_flags_linux.cmake
# as an optional optimization flag added to znver4_extra_flags when available.

# Add the frame subdirectory
add_subdirectory(frame)
add_subdirectory(JIT)
add_subdirectory(kernels)
add_subdirectory(config)

# Output the found sources (optional, for debugging)
# message(STATUS "Classic sources: ${CLASSIC_SOURCES}")
# message(STATUS "Combined sources (without kernels): ${CLASSIC_SOURCES} ${FRAME_SOURCES}")

# Create the classic library without kernel sources (they're in the object libraries)
add_library(aocl_dlp_classic OBJECT ${CLASSIC_SOURCES} ${FRAME_SOURCES} ${CONFIG_SOURCES})

# Enable position-independent code for the object library
set_property(TARGET aocl_dlp_classic PROPERTY POSITION_INDEPENDENT_CODE ON)

# Set include directories for the target
target_include_directories(aocl_dlp_classic
    PUBLIC
        ${CLASSIC_INCLUDES}
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> # For aocl_dlp_config.h
)

# Link with OpenMP if enabled
if(DLP_ENABLE_OPENMP)
    target_link_libraries(aocl_dlp_classic PRIVATE dlp::openmp)
endif()

if(DLP_ENABLE_LOGGING)
    target_compile_definitions(aocl_dlp_classic PRIVATE AOCL_DLP_LOGGER_SUPPORT=1)
endif()


set(CLASSIC_TARGETS
    aocl_dlp_classic
    ${KERNEL_TARGETS}
    PARENT_SCOPE
)

# Apply centralized compiler flags
dlp_set_global_compile_flags(aocl_dlp_classic)

# Install the classic library
# dlp_install(aocl_dlp_classic)
