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

# Define compiler flags using interface libraries
# This is the modern CMake way to handle compiler flags

# Include the common compiler flags file to test compiler for flags
include(${CMAKE_CURRENT_LIST_DIR}/dlp_compiler_flags_common.cmake)

# Architecture-specific flags defined for convenience
set(DLP_GENERIC_FLAGS -Wall
                      -Wno-unused
                      -fPIC
)

# FIXME: Enable -Werror after all warnings are fixed
set(DLP_RELEASE_FLAGS -O3
                    #   -Werror
)

set(DLP_COVERAGE_FLAGS -g
                       -O0
                       -fprofile-arcs
                       -ftest-coverage
                       -fprofile-abs-path
)

set(DLP_COVERAGE_LINK_FLAGS --coverage
)

# Function to check support for ZnVer flags (znver1 through znver5)
# Sets variables:
#   DLP_ZNVER1_SUPPORTED, DLP_ZNVER2_SUPPORTED, etc. - Whether each flag is supported
#   DLP_HIGHEST_ZNVER - The highest supported ZnVer flag (e.g., "znver4")
#   DLP_ZNVER_FLAGS_MAP - A map of ZnVer names to their associated flags
function(dlp_check_znver_support)
    # Check support for each ZnVer flag
    dlp_check_compiler_flag("-march=znver1" DLP_ZNVER1_SUPPORTED)
    dlp_check_compiler_flag("-march=znver2" DLP_ZNVER2_SUPPORTED)
    dlp_check_compiler_flag("-march=znver3" DLP_ZNVER3_SUPPORTED)
    dlp_check_compiler_flag("-march=znver4" DLP_ZNVER4_SUPPORTED)
    dlp_check_compiler_flag("-march=znver5" DLP_ZNVER5_SUPPORTED)

    # Export the support variables to parent scope
    set(DLP_ZNVER1_SUPPORTED ${DLP_ZNVER1_SUPPORTED} PARENT_SCOPE)
    set(DLP_ZNVER2_SUPPORTED ${DLP_ZNVER2_SUPPORTED} PARENT_SCOPE)
    set(DLP_ZNVER3_SUPPORTED ${DLP_ZNVER3_SUPPORTED} PARENT_SCOPE)
    set(DLP_ZNVER4_SUPPORTED ${DLP_ZNVER4_SUPPORTED} PARENT_SCOPE)
    set(DLP_ZNVER5_SUPPORTED ${DLP_ZNVER5_SUPPORTED} PARENT_SCOPE)

    # Determine the highest supported ZnVer
    set(highest_znver "none")
    if(DLP_ZNVER1_SUPPORTED)
        set(highest_znver "znver1")
    endif()
    if(DLP_ZNVER2_SUPPORTED)
        set(highest_znver "znver2")
    endif()
    if(DLP_ZNVER3_SUPPORTED)
        set(highest_znver "znver3")
    endif()
    if(DLP_ZNVER4_SUPPORTED)
        set(highest_znver "znver4")
    endif()
    if(DLP_ZNVER5_SUPPORTED)
        set(highest_znver "znver5")
    endif()

    set(DLP_HIGHEST_ZNVER ${highest_znver} PARENT_SCOPE)
    set(DLP_HIGHEST_ZNVER ${highest_znver})

    # Define fallback flags for ZnVer versions
    # These are the additional flags needed to approximate higher ZnVer capabilities
    # when a specific znver flag isn't available

    # znver2 specific features over znver1
    set(znver2_extra_flags "-mavx2;-mfma")

    # znver3 specific features over znver2
    set(znver3_extra_flags "-mavx2;-mfma;-mvaes")

    # znver4 specific features over znver3
    set(znver4_extra_flags "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mavx512ifma;-mavx512vbmi;-mavx512vbmi2;-mavx512vnni;-mavx512bf16;-mvaes")

    # znver5 placeholder (update when specifications are available)
    set(znver5_extra_flags "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mavx512ifma;-mavx512vbmi;-mavx512vbmi2;-mavx512vnni;-mavx512bf16;-mvaes;-mavx10.1-256;-mavx10.1-512")

    # Create a map of znver targets to their flags (primary or fallback)
    # This will be used to get the optimal flags for each ZnVer target

    # Set Zen1 flags or the fallback
    if(DLP_ZNVER1_SUPPORTED)
        set(znver1_flags "-march=znver1")
    else()
        set(znver1_flags "-march=x86-64;-msse4.2")
    endif()


    # Set Zen2 flags or the fallback
    if(DLP_ZNVER2_SUPPORTED)
        set(znver2_flags "-march=znver2")
    elseif(DLP_ZNVER1_SUPPORTED)
        set(znver2_flags "-march=znver1;${znver2_extra_flags}")
    else()
        set(znver2_flags "-march=x86-64;-msse4.2;${znver2_extra_flags}")
    endif()


    # Set Zen3 flags or the fallback
    if(DLP_ZNVER3_SUPPORTED)
        set(znver3_flags "-march=znver3")
    elseif(DLP_ZNVER2_SUPPORTED)
        set(znver3_flags "-march=znver2;${znver3_extra_flags}")
    elseif(DLP_ZNVER1_SUPPORTED)
        set(znver3_flags "-march=znver1;${znver3_extra_flags}")
    else()
        set(znver3_flags "-march=x86-64;-msse4.2;${znver3_extra_flags}")
    endif()


    # Set Zen4 flags or the fallback
    if(DLP_ZNVER4_SUPPORTED)
        set(znver4_flags "-march=znver4")
    elseif(DLP_ZNVER3_SUPPORTED)
        set(znver4_flags "-march=znver3;${znver4_extra_flags}")
    elseif(DLP_ZNVER2_SUPPORTED)
        set(znver4_flags "-march=znver2;${znver4_extra_flags}")
    elseif(DLP_ZNVER1_SUPPORTED)
        set(znver4_flags "-march=znver1;${znver4_extra_flags}")
    else()
        set(znver4_flags "-march=x86-64;-msse4.2;${znver4_extra_flags}")
    endif()

    # Set Zen5 flags or the fallback
    if(DLP_ZNVER5_SUPPORTED)
        set(znver5_flags "-march=znver5")
    elseif(DLP_ZNVER4_SUPPORTED)
        set(znver5_flags "-march=znver4;${znver5_extra_flags}")
    elseif(DLP_ZNVER3_SUPPORTED)
        set(znver5_flags "-march=znver3;${znver5_extra_flags}")
    elseif(DLP_ZNVER2_SUPPORTED)
        set(znver5_flags "-march=znver2;${znver5_extra_flags}")
    elseif(DLP_ZNVER1_SUPPORTED)
        set(znver5_flags "-march=znver1;${znver5_extra_flags}")
    else()
        set(znver5_flags "-march=x86-64;-msse4.2;${znver5_extra_flags}")
    endif()

    # Export the flags map
    set(DLP_ZNVER1_FLAGS "${znver1_flags}" PARENT_SCOPE)
    set(DLP_ZNVER2_FLAGS "${znver2_flags}" PARENT_SCOPE)
    set(DLP_ZNVER3_FLAGS "${znver3_flags}" PARENT_SCOPE)
    set(DLP_ZNVER4_FLAGS "${znver4_flags}" PARENT_SCOPE)
    set(DLP_ZNVER5_FLAGS "${znver5_flags}" PARENT_SCOPE)

    # Log the highest supported ZnVer
    message(STATUS "Highest supported ZnVer: ${DLP_HIGHEST_ZNVER}")
    if(NOT DLP_HIGHEST_ZNVER STREQUAL "none")
    message(STATUS "ZnVer Support: znver1=${DLP_ZNVER1_SUPPORTED}, znver2=${DLP_ZNVER2_SUPPORTED}, znver3=${DLP_ZNVER3_SUPPORTED}, znver4=${DLP_ZNVER4_SUPPORTED}, znver5=${DLP_ZNVER5_SUPPORTED}")
    else()
    message(STATUS "No ZnVer flags are supported by the compiler")
    endif()
endfunction()

# Function to get optimal flags for a specific ZnVer target
# Parameters:
#   target_znver - The target ZnVer (znver1, znver2, etc.)
#   output_var - The variable to store the result in
function(dlp_get_znver_flags target_znver output_var)
    string(TOUPPER ${target_znver} upper_target)
    set(${output_var} "${DLP_${upper_target}_FLAGS}" PARENT_SCOPE)
endfunction()

# Define Zen architecture flags using the ZnVer detection results
dlp_get_znver_flags("znver2" DLP_ARCH_ZEN_FLAGS)
dlp_get_znver_flags("znver4" DLP_ARCH_ZEN4_FLAGS)

# Create interface libraries for different flag sets
add_library(dlp_compiler_flags INTERFACE)
add_library(dlp_compiler_flags_release INTERFACE)
add_library(dlp_compiler_flags_debug INTERFACE)
add_library(dlp_compiler_flags_coverage INTERFACE)

# Set default compiler flags
target_compile_options(dlp_compiler_flags INTERFACE ${DLP_GENERIC_FLAGS})

# Set release-specific compiler flags
target_compile_options(dlp_compiler_flags_release INTERFACE ${DLP_RELEASE_FLAGS})

# Set coverage-specific compiler and link flags
target_compile_options(dlp_compiler_flags_coverage INTERFACE ${DLP_COVERAGE_FLAGS})
target_link_options(dlp_compiler_flags_coverage INTERFACE ${DLP_COVERAGE_LINK_FLAGS})

# No additional debug-specific flags, but we could add them here
# target_compile_options(dlp_compiler_flags_debug INTERFACE -g)

# Define Coverage build type with proper CMAKE variables
string(JOIN " " CMAKE_C_FLAGS_COVERAGE_STRING ${DLP_COVERAGE_FLAGS})
string(JOIN " " CMAKE_CXX_FLAGS_COVERAGE_STRING ${DLP_COVERAGE_FLAGS})
string(JOIN " " CMAKE_EXE_LINKER_FLAGS_COVERAGE_STRING ${DLP_COVERAGE_LINK_FLAGS})
string(JOIN " " CMAKE_SHARED_LINKER_FLAGS_COVERAGE_STRING ${DLP_COVERAGE_LINK_FLAGS})

set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_COVERAGE_STRING}" CACHE STRING "C flags for Coverage build type" FORCE)
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE_STRING}" CACHE STRING "CXX flags for Coverage build type" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_COVERAGE_STRING}" CACHE STRING "Linker flags for Coverage build type" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_COVERAGE_STRING}" CACHE STRING "Shared linker flags for Coverage build type" FORCE)

# Mark Coverage as a valid build type
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES};Coverage" CACHE STRING "Available build types" FORCE)

# Function to apply global compiler flags to a target
# Parameters:
#   target - The target to apply flags to
#   visibility - Optional: The visibility level (INTERFACE, PUBLIC, PRIVATE) - defaults to PUBLIC
function(dlp_set_global_compile_flags target)
    # Parse arguments to allow specifying visibility
    set(options "")
    set(oneValueArgs VISIBILITY)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default visibility to PUBLIC if not specified
    if(NOT ARG_VISIBILITY)
        set(ARG_VISIBILITY PUBLIC)
    endif()

    # Apply compiler flags with specified visibility
    target_link_libraries(${target} ${ARG_VISIBILITY}
        dlp_compiler_flags
        $<$<CONFIG:Release>:dlp_compiler_flags_release>
        $<$<CONFIG:RelWithDebInfo>:dlp_compiler_flags_release>
        $<$<CONFIG:Debug>:dlp_compiler_flags_debug>
        $<$<CONFIG:Coverage>:dlp_compiler_flags_coverage>
    )
endfunction()

# Function to set architecture-specific flags for a target
# Parameters:
#   target - The target to apply flags to
#   arch - The architecture (zen, zen2, zen3, zen4, zen5)
#   visibility - Optional: The visibility level (defaults to PRIVATE)
function(dlp_set_arch_flags target arch)
    set(options "")
    set(oneValueArgs VISIBILITY)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default visibility to PRIVATE if not specified
    if(NOT ARG_VISIBILITY)
        set(ARG_VISIBILITY PRIVATE)
    endif()

    # Translate arch names to ZnVer names
    set(znver_name "")
    if(arch STREQUAL "zen")
        set(znver_name "znver1")
    elseif(arch STREQUAL "zen2")
        set(znver_name "znver2")
    elseif(arch STREQUAL "zen3")
        set(znver_name "znver3")
    elseif(arch STREQUAL "zen4")
        set(znver_name "znver4")
    elseif(arch STREQUAL "zen5")
        set(znver_name "znver5")
    else()
        message(WARNING "Unknown architecture: ${arch}")
        return()
    endif()

    # Get the optimal flags for the target architecture
    dlp_get_znver_flags(${znver_name} target_flags)

    # Apply the appropriate architecture flags
    target_compile_options(${target} ${ARG_VISIBILITY}
        $<$<COMPILE_LANGUAGE:C>:${target_flags}>
        $<$<COMPILE_LANGUAGE:CXX>:${target_flags}>
    )

    # Log the flags being used
    string(REPLACE ";" " " target_flags_str "${target_flags}")
    message(STATUS "Setting ${arch} flags for target ${target}: ${target_flags_str}")
endfunction()

function(dlp_set_platform_options)
    # Set binary name for static library
    if(DLP_OS_LINUX)
        #FIXME: For windows we might need to do something similar
        set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
    endif()

    # CRITICAL: Set INTERFACE_LINK_OPTIONS to automatically apply whole-archive for static library
    # This ensures all static constructors (JIT registration, kernel registration) are included
    # when any target links against aocl-dlp_static. This eliminates the need for users to
    # manually specify whole-archive flags.
    # Linux/macOS: Use --whole-archive
    # This flag will be applied to the target which is being linked against aocl-dlp_static library.
    target_link_options(${PROJECT_NAME}_static INTERFACE
        "LINKER:--whole-archive,$<TARGET_FILE:$<TARGET_NAME:${PROJECT_NAME}_static>>,--no-whole-archive"
    )
endfunction()
