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
# This is the modern CMake way to handle compiler flags for Windows/MSVC

# Generic flags for MSVC compiler
set(DLP_GENERIC_FLAGS_MSVC
    /W4             # Warning level 4
    /permissive-    # Standards conformance
    /Zc:inline      # Remove unreferenced functions during optimization
    /Zc:wchar_t     # wchar_t is a native type
)

# Release flags for MSVC compiler
set(DLP_RELEASE_FLAGS_MSVC
    /O2             # Optimize for speed
    /GL             # Whole program optimization
    /Gy             # Function-level linking
    /Oi             # Generate intrinsic functions
   # /WX             # Treat warnings as errors (equivalent to -Werror)
)

# Debug flags for MSVC compiler
set(DLP_DEBUG_FLAGS_MSVC
    /Od             # Disable optimization
    /Zi             # Generate complete debugging information
    /RTC1           # Run-time error checks
)

# Architecture-specific flags for MSVC
set(DLP_ARCH_ZEN_FLAGS_MSVC /arch:AVX2)   # AVX2 for Zen architecture
set(DLP_ARCH_ZEN4_FLAGS_MSVC /arch:AVX512) # AVX512 for Zen4 architecture

# Generic flags for non-MSVC compilers (MinGW/Clang) when used on Windows
set(DLP_GENERIC_FLAGS_OTHER -Wall)
set(DLP_RELEASE_FLAGS_OTHER -O3)
set(DLP_DEBUG_FLAGS_OTHER -g -O0)
set(DLP_ARCH_ZEN_FLAGS_OTHER -mavx2 -mfma)
set(DLP_ARCH_ZEN4_FLAGS_OTHER -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512bf16 -mvaes)

# Create interface libraries for different flag sets
add_library(dlp_compiler_flags INTERFACE)
add_library(dlp_compiler_flags_release INTERFACE)
add_library(dlp_compiler_flags_debug INTERFACE)

# Set default compiler flags based on compiler
target_compile_options(dlp_compiler_flags INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:${DLP_GENERIC_FLAGS_MSVC}>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${DLP_GENERIC_FLAGS_OTHER}>
)

# Set release-specific compiler flags
target_compile_options(dlp_compiler_flags_release INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:${DLP_RELEASE_FLAGS_MSVC}>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${DLP_RELEASE_FLAGS_OTHER}>
)

# Set debug-specific compiler flags
target_compile_options(dlp_compiler_flags_debug INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:${DLP_DEBUG_FLAGS_MSVC}>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${DLP_DEBUG_FLAGS_OTHER}>
)

# Additional MSVC-specific settings
target_compile_options(dlp_compiler_flags INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/MP>  # Enable multi-processor compilation
)

# Disable specific warnings for MSVC that might be too noisy.
# These are MSVC-only /W4 pedantry that GCC and Clang accept silently; none of
# them flags a real defect in the vectorized kernels. Suppressing them here
# (MSVC-only) keeps the Windows build readable without changing any code or
# touching the Linux/GCC/Clang flags. Genuinely suspicious codes (e.g. C4334,
# C4245, C4133) are deliberately NOT suppressed and are handled separately.
target_compile_options(dlp_compiler_flags INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4996>  # Deprecation (CRT/POSIX names): intentional cross-platform usage
    $<$<CXX_COMPILER_ID:MSVC>:/wd4305>  # double->float truncation: benign _mm*_set1_ps(literal) idioms missing an 'f' suffix
    $<$<CXX_COMPILER_ID:MSVC>:/wd4146>  # unary minus on unsigned: benign mask/sign idioms in the SIMD math macros
    $<$<CXX_COMPILER_ID:MSVC>:/wd4244>  # narrowing conversion (double->float / int): benign kernel arithmetic
    $<$<CXX_COMPILER_ID:MSVC>:/wd4324>  # padding due to alignment specifier: INTENTIONAL cache-line alignment of structs
)

# Force-include the centralized portability header for all MSVC compilation units.
# This ensures NOMINMAX, __builtin_clz, DLP_INF, DLP_ATOMIC_*, etc. are available
# everywhere without requiring explicit #include in each source file.
if(MSVC)
    # Use DLP_SOURCE_DIR (the DLP project root), not CMAKE_SOURCE_DIR, which
    # points at a parent project when DLP is built via add_subdirectory/
    # FetchContent.
    target_compile_options(dlp_compiler_flags INTERFACE
        "/FI${DLP_SOURCE_DIR}/include/classic/dlp_compat.h"
    )
endif()

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
    )

    # MSVC-specific link options
    if(MSVC)
        # Enable Link Time Optimization in release builds
        set_target_properties(${target} PROPERTIES
            LINK_FLAGS_RELEASE "/LTCG"
            LINK_FLAGS_RELWITHDEBINFO "/LTCG"
        )
    endif()
endfunction()

# Function to set architecture-specific flags for a target
# Parameters:
#   target - The target to apply flags to
#   arch - The architecture (zen, zen4)
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

    # Apply the appropriate architecture flags based on compiler
    if(arch STREQUAL "zen")
        target_compile_options(${target} ${ARG_VISIBILITY}
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN_FLAGS_OTHER}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN_FLAGS_OTHER}>
        )
    elseif(arch STREQUAL "zen4")
        target_compile_options(${target} ${ARG_VISIBILITY}
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN4_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${DLP_ARCH_ZEN4_FLAGS_MSVC}>
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN4_FLAGS_OTHER}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${DLP_ARCH_ZEN4_FLAGS_OTHER}>
        )
    else()
        message(WARNING "Unknown architecture: ${arch}")
    endif()
endfunction()

# Define Windows-specific functions for build types
function(dlp_set_release_build_flags)
    # Set global release build flags
    if(MSVC)
        set(CMAKE_C_FLAGS_RELEASE "/MD" PARENT_SCOPE)  # Multi-threaded DLL runtime
        set(CMAKE_CXX_FLAGS_RELEASE "/MD" PARENT_SCOPE)
    else()
        set(CMAKE_C_FLAGS_RELEASE "" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_RELEASE "" PARENT_SCOPE)
    endif()
endfunction()

function(dlp_set_debug_build_flags)
    # Set global debug build flags
    if(MSVC)
        set(CMAKE_C_FLAGS_DEBUG "/MDd" PARENT_SCOPE)  # Multi-threaded Debug DLL runtime
        set(CMAKE_CXX_FLAGS_DEBUG "/MDd" PARENT_SCOPE)
    else()
        set(CMAKE_C_FLAGS_DEBUG "" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_DEBUG "" PARENT_SCOPE)
    endif()
endfunction()

function(dlp_set_relwithdebinfo_build_flags)
    # Set global relwithdebinfo build flags
    if(MSVC)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "/MD /Zi" PARENT_SCOPE)  # Multi-threaded DLL runtime with debug info
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi" PARENT_SCOPE)
    else()
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" PARENT_SCOPE)
    endif()
endfunction()
# Function to check support for ZnVer flags (znver1 through znver5) on Windows
# Sets variables:
#   DLP_ZNVER1_SUPPORTED, DLP_ZNVER2_SUPPORTED, etc. - Whether each flag is supported
#   DLP_HIGHEST_ZNVER - The highest supported ZnVer flag (e.g., "znver4")
#   DLP_ZNVER1_FLAGS, DLP_ZNVER2_FLAGS, ... - The flags to use for each ZnVer
function(dlp_check_znver_support)
    # On MSVC, only AVX2/AVX512 are supported, not -march=znver*
    if(MSVC)
        set(DLP_ZNVER1_SUPPORTED FALSE PARENT_SCOPE)
        set(DLP_ZNVER2_SUPPORTED FALSE PARENT_SCOPE)
        set(DLP_ZNVER3_SUPPORTED FALSE PARENT_SCOPE)
        set(DLP_ZNVER4_SUPPORTED TRUE  PARENT_SCOPE)  # AVX512 is closest to znver4
        set(DLP_ZNVER5_SUPPORTED FALSE PARENT_SCOPE)
        set(DLP_ZNVER6_SUPPORTED FALSE PARENT_SCOPE)
        set(DLP_HIGHEST_ZNVER "znver4" PARENT_SCOPE)
        set(DLP_ZNVER1_FLAGS "${DLP_ARCH_ZEN_FLAGS_MSVC}" PARENT_SCOPE)
        set(DLP_ZNVER2_FLAGS "${DLP_ARCH_ZEN_FLAGS_MSVC}" PARENT_SCOPE)
        set(DLP_ZNVER3_FLAGS "${DLP_ARCH_ZEN_FLAGS_MSVC}" PARENT_SCOPE)
        set(DLP_ZNVER4_FLAGS "${DLP_ARCH_ZEN4_FLAGS_MSVC}" PARENT_SCOPE)
        set(DLP_ZNVER5_FLAGS "${DLP_ARCH_ZEN4_FLAGS_MSVC}" PARENT_SCOPE)
        set(DLP_ZNVER6_FLAGS "${DLP_ARCH_ZEN4_FLAGS_MSVC}" PARENT_SCOPE)
        message(STATUS "ZnVer support (MSVC): Only AVX2/AVX512 mapped. Highest: znver4")
    else()
        # For MinGW/Clang, check for -march=znver* support (simple version)
        include(CheckCCompilerFlag)
        set(DLP_HIGHEST_ZNVER "none")
        set(DLP_ZNVER1_SUPPORTED FALSE)
        set(DLP_ZNVER2_SUPPORTED FALSE)
        set(DLP_ZNVER3_SUPPORTED FALSE)
        set(DLP_ZNVER4_SUPPORTED FALSE)
        set(DLP_ZNVER5_SUPPORTED FALSE)
        set(DLP_ZNVER6_SUPPORTED FALSE)
        foreach(ver RANGE 1 6)
            set(flag "-march=znver${ver}")
            string(TOUPPER "DLP_ZNVER${ver}_SUPPORTED" var)
            check_c_compiler_flag(${flag} ${var})
            if(${var})
                set(DLP_HIGHEST_ZNVER "znver${ver}")
                set(DLP_ZNVER${ver}_SUPPORTED TRUE)
            endif()
            set(${var} ${${var}} PARENT_SCOPE)
        endforeach()
        set(DLP_HIGHEST_ZNVER ${DLP_HIGHEST_ZNVER} PARENT_SCOPE)
        set(DLP_ZNVER1_FLAGS "${DLP_ARCH_ZEN_FLAGS_OTHER}" PARENT_SCOPE)
        set(DLP_ZNVER2_FLAGS "${DLP_ARCH_ZEN_FLAGS_OTHER}" PARENT_SCOPE)
        set(DLP_ZNVER3_FLAGS "${DLP_ARCH_ZEN_FLAGS_OTHER}" PARENT_SCOPE)
        set(DLP_ZNVER4_FLAGS "${DLP_ARCH_ZEN4_FLAGS_OTHER}" PARENT_SCOPE)
        set(DLP_ZNVER5_FLAGS "${DLP_ARCH_ZEN4_FLAGS_OTHER}" PARENT_SCOPE)
        set(DLP_ZNVER6_FLAGS "${DLP_ARCH_ZEN4_FLAGS_OTHER}" PARENT_SCOPE)
        message(STATUS "ZnVer support (MinGW/Clang): Highest: ${DLP_HIGHEST_ZNVER}")
    endif()
endfunction()

# Function to apply JIT-specific compiler flags to a target
# Uses lazy initialization to ensure CMAKE_SYSTEM_PROCESSOR is available
# Parameters:
#   target - The target to apply JIT flags to
#   visibility - Optional: The visibility level (defaults to PRIVATE)
function(dlp_set_jit_flags target)
    set(options "")
    set(oneValueArgs VISIBILITY)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default visibility to PRIVATE if not specified
    if(NOT ARG_VISIBILITY)
        set(ARG_VISIBILITY PRIVATE)
    endif()

    # Lazy initialization: Determine JIT flags on first call
    if(NOT DEFINED DLP_JIT_FLAGS_INITIALIZED)
        set(DLP_JIT_FLAGS_MSVC "")
        set(DLP_JIT_FLAGS_OTHER "")

        # MSVC: /bigobj allows more sections in object files (for large JIT-generated code)
        # /FI force-includes MSVC shims (__builtin_clz, NOMINMAX, DLP_FLOAT_INF) for all JIT TUs
        if(MSVC)
            list(APPEND DLP_JIT_FLAGS_MSVC /bigobj)
            list(APPEND DLP_JIT_FLAGS_MSVC "/FI${DLP_SOURCE_DIR}/src/jit/amdzen/dlp_msvc_compat.h")
            message(STATUS "JIT flags (MSVC) initialized: /bigobj, /FI dlp_msvc_compat.h")
        endif()

        # MinGW/Clang on Windows: Use same flags as Linux
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
            list(APPEND DLP_JIT_FLAGS_OTHER -mcmodel=medium)
            message(STATUS "JIT flags (MinGW/Clang) initialized for ${CMAKE_SYSTEM_PROCESSOR}: -mcmodel=medium")
        endif()

        # Mark as initialized (cache in parent scope)
        set(DLP_JIT_FLAGS_INITIALIZED TRUE CACHE INTERNAL "JIT flags have been initialized")
        set(DLP_JIT_FLAGS_MSVC "${DLP_JIT_FLAGS_MSVC}" CACHE INTERNAL "MSVC JIT flags")
        set(DLP_JIT_FLAGS_OTHER "${DLP_JIT_FLAGS_OTHER}" CACHE INTERNAL "MinGW/Clang JIT flags")
    endif()

    # Apply JIT-specific flags based on compiler
    if(MSVC)
        if(DLP_JIT_FLAGS_MSVC)
            target_compile_options(${target} ${ARG_VISIBILITY} ${DLP_JIT_FLAGS_MSVC})
            string(REPLACE ";" " " jit_flags_str "${DLP_JIT_FLAGS_MSVC}")
            message(STATUS "Applying JIT flags to ${target}: ${jit_flags_str}")
        endif()
    else()
        # MinGW or Clang on Windows
        if(DLP_JIT_FLAGS_OTHER)
            target_compile_options(${target} ${ARG_VISIBILITY} ${DLP_JIT_FLAGS_OTHER})
            string(REPLACE ";" " " jit_flags_str "${DLP_JIT_FLAGS_OTHER}")
            message(STATUS "Applying JIT flags to ${target}: ${jit_flags_str}")
        endif()
    endif()
endfunction()

# Function to setup atomic support for C11 _Atomic and C++ std::atomic operations.
# See dlp_compiler_flags_linux.cmake for detailed rationale.
function(dlp_setup_atomic_support)
    if(MSVC)
        # MSVC-like (cl.exe and clang-cl) have built-in atomic support via the
        # MSVC runtime / Interlocked intrinsics; libatomic is a GCC/Linux-only
        # runtime and does not exist for this toolchain.
        message(STATUS "MSVC-like: atomic support built-in, no additional linking needed")
    elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
        # MinGW/Clang/IntelLLVM on Windows — uses GCC's libatomic runtime
        find_library(ATOMIC_LIBRARY NAMES atomic
            HINTS
                ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
                ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}
                ENV LIBRARY_PATH
        )

        if(ATOMIC_LIBRARY)
            target_link_libraries(${PROJECT_NAME} PRIVATE ${ATOMIC_LIBRARY})
            target_link_libraries(${PROJECT_NAME}_static PRIVATE ${ATOMIC_LIBRARY})
            message(STATUS "Linking with libatomic: ${ATOMIC_LIBRARY}")
        else()
            message(FATAL_ERROR "libatomic is required but was not found. "
                "Ensure libatomic is installed and LIBRARY_PATH is set correctly.")
        endif()
    elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
        # Intel Classic (icc/icpc) bundles atomic support in libirc
        message(STATUS "Intel Classic compiler: atomic support provided by libirc")
    else()
        message(WARNING "Unknown compiler '${CMAKE_C_COMPILER_ID}': atomic support may require manual configuration")
    endif()
endfunction()

function(dlp_set_platform_options)
    # On MSVC, define DLP_IS_BUILDING_LIBRARY for all object libraries
    # so that DLP_CLASSIC_EXPORT expands to __declspec(dllexport) in the DLL build.
    if(MSVC)
        foreach(classic_target IN LISTS CLASSIC_TARGETS DLP_PLUS_TARGETS)
            target_compile_definitions(${classic_target} PRIVATE DLP_IS_BUILDING_LIBRARY)
        endforeach()
        target_compile_definitions(aocl_dlp_jit_amdzen PRIVATE DLP_IS_BUILDING_LIBRARY)
        target_compile_definitions(${PROJECT_NAME} PRIVATE DLP_IS_BUILDING_LIBRARY)
    elseif(BUILD_SHARED_LIBS)
        foreach(classic_target IN LISTS CLASSIC_TARGETS)
            target_compile_definitions(${classic_target} PUBLIC DLP_IS_BUILDING_LIBRARY)
        endforeach()
    endif()

    # CRITICAL: Set INTERFACE_LINK_OPTIONS to automatically apply whole-archive for static library
    # This ensures all static constructors (JIT registration, kernel registration) are included
    # when any target links against aocl-dlp_static. This eliminates the need for users to
    # manually specify whole-archive flags.
    if(MSVC)
        # MSVC: Use /WHOLEARCHIVE
        target_link_options(${PROJECT_NAME}_static INTERFACE
            "/WHOLEARCHIVE:$<TARGET_FILE:${PROJECT_NAME}_static>"
        )
    else()
        # MinGW/Clang on Windows: Use --whole-archive
        target_link_options(${PROJECT_NAME}_static INTERFACE
            "LINKER:--whole-archive,$<TARGET_FILE:${PROJECT_NAME}_static>,--no-whole-archive"
        )
    endif()
endfunction()
