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
dlp_glob_sources(DLP_PLUS_SOURCES)

set(DLP_PLUS_INCLUDES ${PROJECT_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the frame subdirectory
add_subdirectory(frame)
add_subdirectory(jit)

# Output the found sources (optional, for debugging)
# message(STATUS "DLP Plus sources: ${DLP_PLUS_SOURCES}")

# Create the DLP Plus library WITHOUT amdzen object files
# amdzen will be linked separately to preserve static constructors
add_library(aocl_dlp_plus OBJECT
    ${DLP_PLUS_SOURCES}
    ${DLP_PLUS_FRAME_SOURCES}
)

# Enable position-independent code for the object library
set_property(TARGET aocl_dlp_plus PROPERTY POSITION_INDEPENDENT_CODE ON)

# Set include directories for the target
target_include_directories(aocl_dlp_plus
    PUBLIC
    ${DLP_PLUS_INCLUDES}
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> # For aocl_dlp_config.h
)

# Link with OpenMP for threading support
if(DLP_ENABLE_OPENMP)
    target_link_libraries(aocl_dlp_plus PRIVATE dlp::openmp)
endif()

if(DLP_ENABLE_LOGGING)
    target_compile_definitions(aocl_dlp_plus PRIVATE AOCL_DLP_GEMM_LOGGER_SUPPORT=1)
endif()


set(DLP_PLUS_TARGETS
    aocl_dlp_plus
    PARENT_SCOPE
)

# Apply centralized compiler flags
dlp_set_global_compile_flags(aocl_dlp_plus)

# Install the plus library
# dlp_install(aocl_dlp_plus)
