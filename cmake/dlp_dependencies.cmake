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
Function to set up OpenMP dependency with flexibility for different environments.
This function:
1. Tries to find the standard OpenMP package first
2. Allows using LLVM OpenMP even with GCC compiler
3. Respects custom OpenMP installations via environment variables
   (LIBRARY_PATH, LD_LIBRARY_PATH, CMAKE_PREFIX_PATH, etc.)
4. Creates an interface target for consistent usage throughout the project

Usage: dlp_setup_openmp()

Variables affecting behavior:
- DLP_ENABLE_OPENMP: Set to ON/OFF to enable/disable OpenMP support (default: ON)
- DLP_USE_LLVM_OPENMP: Set to ON to force using LLVM OpenMP even with GCC
- DLP_OPENMP_ROOT: Custom path to OpenMP installation root
]==]
function(dlp_setup_openmp)
    # Create an interface library for OpenMP
    add_library(dlp_openmp_dep INTERFACE)

    # Option for enabling/disabling OpenMP support
    option(DLP_ENABLE_OPENMP "Enable OpenMP support for multithreading" ON)

    # Check if OpenMP is disabled first
    if(NOT DLP_ENABLE_OPENMP)
        message(STATUS "OpenMP support explicitly disabled")
        # Make the interface library globally accessible
        add_library(dlp::openmp ALIAS dlp_openmp_dep)
        return()
    endif()

    # Find and configure OpenMP with multiple options
    include(FindOpenMP)

    # Options for controlling OpenMP behavior
    option(DLP_USE_LLVM_OPENMP "Force using LLVM OpenMP implementation even with GCC" OFF)

    # Allow user to specify a custom OpenMP root
    set(DLP_OPENMP_ROOT "" CACHE PATH "Path to custom OpenMP installation")

    # Store original CMAKE_PREFIX_PATH to restore later if needed
    set(_original_prefix_path "${CMAKE_PREFIX_PATH}")

    # If custom OpenMP root is specified, add it to prefix path temporarily
    if(DLP_OPENMP_ROOT)
        message(STATUS "Using custom OpenMP installation from: ${DLP_OPENMP_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH "${DLP_OPENMP_ROOT}")
    endif()

    # Standard approach: try to find OpenMP using CMake's built-in module
    find_package(OpenMP)

    # Restore original prefix path
    if(DLP_OPENMP_ROOT)
        set(CMAKE_PREFIX_PATH "${_original_prefix_path}")
    endif()

    # Check if we need to handle special cases
    if(OpenMP_FOUND AND NOT DLP_USE_LLVM_OPENMP)
        # Standard case: OpenMP was found and we don't need LLVM OpenMP
        message(STATUS "Using standard OpenMP implementation")

        # Explicitly ensure OpenMP compiler flags are included
        target_compile_options(dlp_openmp_dep INTERFACE ${OpenMP_C_FLAGS})
        target_link_libraries(dlp_openmp_dep INTERFACE OpenMP::OpenMP_C)

    else()
        # Either OpenMP wasn't found or user explicitly requested LLVM OpenMP

        if(DLP_USE_LLVM_OPENMP)
            message(STATUS "Attempting to use LLVM OpenMP implementation as requested")
        else()
            message(STATUS "Standard OpenMP not found, trying LLVM OpenMP as fallback")
        endif()

        # Try to find LLVM OpenMP library
        set(_openmp_lib_names "omp" "libomp" "libiomp")

        # Look in common locations and environment variables
        find_library(LLVM_OPENMP_LIBRARY
            NAMES ${_openmp_lib_names}
            PATHS
                ${DLP_OPENMP_ROOT}
                ENV LIBRARY_PATH
                ENV LD_LIBRARY_PATH
                /usr/lib
                /usr/local/lib
                /opt/llvm/lib
            PATH_SUFFIXES lib lib64
            DOC "LLVM OpenMP runtime library"
        )

        # Find OpenMP header
        find_path(LLVM_OPENMP_INCLUDE
            NAMES omp.h
            PATHS
                ${DLP_OPENMP_ROOT}
                ENV CPATH
                ENV C_INCLUDE_PATH
                ENV CPLUS_INCLUDE_PATH
                /usr/include
                /usr/local/include
                /opt/llvm/include
            PATH_SUFFIXES include
            DOC "LLVM OpenMP header"
        )

        if(LLVM_OPENMP_LIBRARY AND LLVM_OPENMP_INCLUDE)
            message(STATUS "Found LLVM OpenMP: ${LLVM_OPENMP_LIBRARY}")
            target_link_libraries(dlp_openmp_dep INTERFACE ${LLVM_OPENMP_LIBRARY})
            target_include_directories(dlp_openmp_dep INTERFACE ${LLVM_OPENMP_INCLUDE})

            # When using LLVM OpenMP with GCC, we may need to link to the pthread library too
            if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
                find_package(Threads REQUIRED)
                target_link_libraries(dlp_openmp_dep INTERFACE Threads::Threads)
                target_link_libraries(dlp_openmp_dep INTERFACE ${CMAKE_DL_LIBS})
                message(STATUS "Using LLVM OpenMP with GCC - adding additional dependencies")
            endif()

        else()
            # No OpenMP implementation found
            message(WARNING "No OpenMP implementation found. Multithreading will be disabled.")
        endif()
        # Explicitly add OpenMP compiler flag
        target_compile_options(dlp_openmp_dep INTERFACE -fopenmp)
    endif()

    # Make the interface library globally accessible and exportable
    add_library(dlp::openmp ALIAS dlp_openmp_dep)
    # Set export name for the interface library
    set_target_properties(dlp_openmp_dep PROPERTIES EXPORT_NAME openmp)
endfunction()
