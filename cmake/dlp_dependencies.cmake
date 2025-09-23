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

# This file will be responsible for bringing in all dependencies

#[==[
Function to set up OpenMP dependency using modern CMake best practices.
This function:
1. Uses CMake's built-in FindOpenMP module properly
2. Creates a clean interface target for consistent usage
3. Handles both C and C++ OpenMP targets
4. Provides fallback options for custom installations

Usage: dlp_setup_openmp()

Variables affecting behavior:
- DLP_ENABLE_OPENMP: Set to ON/OFF to enable/disable OpenMP support
- DLP_OPENMP_ROOT: Custom path to OpenMP installation root
]==]
function(dlp_setup_openmp)
    # Check if OpenMP is disabled first
    if(NOT DLP_ENABLE_OPENMP)
        message(STATUS "OpenMP support explicitly disabled")
        # Create a dummy interface target for consistency
        add_library(dlp_openmp_interface INTERFACE)
        add_library(dlp::openmp ALIAS dlp_openmp_interface)
        return()
    endif()

    # Allow user to specify a custom OpenMP root
    set(DLP_OPENMP_ROOT "" CACHE PATH "Path to custom OpenMP installation")

    # Store original CMAKE_PREFIX_PATH to restore later if needed
    set(_original_prefix_path "${CMAKE_PREFIX_PATH}")

    # If custom OpenMP root is specified, add it to prefix path temporarily
    if(DLP_OPENMP_ROOT)
        message(STATUS "Using custom OpenMP installation from: ${DLP_OPENMP_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH "${DLP_OPENMP_ROOT}")
    endif()

    # Use CMake's built-in FindOpenMP module
    find_package(OpenMP REQUIRED COMPONENTS C CXX)

    # Restore original prefix path
    if(DLP_OPENMP_ROOT)
        set(CMAKE_PREFIX_PATH "${_original_prefix_path}")
    endif()

    if(OpenMP_FOUND)
        message(STATUS "Found OpenMP: C flags=${OpenMP_C_FLAGS}, CXX flags=${OpenMP_CXX_FLAGS}")

        # Create our own interface target that combines both C and CXX OpenMP
        add_library(dlp_openmp_interface INTERFACE)

        # Link to the official OpenMP targets
        target_link_libraries(dlp_openmp_interface INTERFACE
            OpenMP::OpenMP_C
            OpenMP::OpenMP_CXX
        )

        # Make the interface library globally accessible
        add_library(dlp::openmp ALIAS dlp_openmp_interface)

        # Set export name for the interface library
        set_target_properties(dlp_openmp_interface PROPERTIES EXPORT_NAME openmp)

        # Install the interface target so it can be exported
        install(TARGETS dlp_openmp_interface
            EXPORT AoclDlpTargets
            INCLUDES DESTINATION include
        )

        message(STATUS "OpenMP integration configured successfully")
    else()
        message(FATAL_ERROR "OpenMP was requested but not found. Please install OpenMP or set DLP_ENABLE_OPENMP=OFF")
    endif()
endfunction()
