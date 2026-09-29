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

function(dlp_openmp_flags_match_runtime flags runtime frontend out_var)
    set(_dlp_matches FALSE)
    if("${runtime}" STREQUAL "LLVM")
        if("${frontend}" STREQUAL "CLANG_CL"
           AND ("${flags}" MATCHES "[-/]openmp:llvm"
                OR "${flags}" MATCHES
                   "(^|[ ;])-fopenmp((=|:)libomp)?($|[ ;])"))
            set(_dlp_matches TRUE)
        elseif("${frontend}" STREQUAL "MSVC"
               AND "${flags}" MATCHES "[-/]openmp:llvm")
            set(_dlp_matches TRUE)
        endif()
    elseif("${runtime}" STREQUAL "VCOMP")
        if(("${frontend}" STREQUAL "MSVC"
            OR "${frontend}" STREQUAL "CLANG_CL")
           AND "${flags}" MATCHES "[-/]openmp($|[^:])")
            set(_dlp_matches TRUE)
        endif()
    endif()
    set(${out_var} ${_dlp_matches} PARENT_SCOPE)
endfunction()

function(dlp_setup_openmp)
    set(DLP_OPENMP_ROOT "" CACHE PATH
        "Path to custom OpenMP installation root"
    )
    set_property(CACHE DLP_OPENMP_ROOT PROPERTY TYPE PATH)

    # The private capability header is generated for both enabled and disabled
    # builds.  This keeps the wrapper usable without making OpenMP a package
    # requirement for a serial build.
    if(DLP_ENABLE_OPENMP)
        set(DLP_OPENMP_ENABLED 1)
    else()
        set(DLP_OPENMP_ENABLED 0)
    endif()
    set(DLP_OPENMP_HAS_ACTIVE_LEVELS 0)
    set(DLP_OPENMP_HAS_NESTING 0)
    set(DLP_OPENMP_HAS_PLACES 0)
    set(DLP_OPENMP_RUNTIME_VCOMP 0)
    set(DLP_OPENMP_RUNTIME_LLVM 0)
    set(DLP_OPENMP_WIN_RUNTIME_SELECTED "NONE")
    set(DLP_OPENMP_WIN_FRONTEND_SELECTED "COMPILER")

    if(NOT DLP_ENABLE_OPENMP)
        message(STATUS "OpenMP support explicitly disabled")
        add_library(dlp_openmp_interface INTERFACE)
        target_include_directories(dlp_openmp_interface INTERFACE
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/private>
            $<BUILD_INTERFACE:${DLP_SOURCE_DIR}/classic/include>
        )
        add_library(dlp::openmp ALIAS dlp_openmp_interface)
        configure_file(
            "${DLP_SOURCE_DIR}/cmake/dlp_openmp_config.h.in"
            "${CMAKE_BINARY_DIR}/include/private/dlp_openmp_config.h"
            @ONLY
        )
        set(DLP_OPENMP_ENABLED ${DLP_OPENMP_ENABLED} PARENT_SCOPE)
        set(DLP_OPENMP_HAS_ACTIVE_LEVELS ${DLP_OPENMP_HAS_ACTIVE_LEVELS} PARENT_SCOPE)
        set(DLP_OPENMP_HAS_NESTING ${DLP_OPENMP_HAS_NESTING} PARENT_SCOPE)
        set(DLP_OPENMP_HAS_PLACES ${DLP_OPENMP_HAS_PLACES} PARENT_SCOPE)
        set(DLP_OPENMP_RUNTIME_VCOMP ${DLP_OPENMP_RUNTIME_VCOMP} PARENT_SCOPE)
        set(DLP_OPENMP_RUNTIME_LLVM ${DLP_OPENMP_RUNTIME_LLVM} PARENT_SCOPE)
        set(DLP_OPENMP_WIN_RUNTIME_SELECTED
            ${DLP_OPENMP_WIN_RUNTIME_SELECTED} PARENT_SCOPE
        )
        set(DLP_OPENMP_WIN_FRONTEND_SELECTED
            ${DLP_OPENMP_WIN_FRONTEND_SELECTED} PARENT_SCOPE
        )
        return()
    endif()

    # DLP_OPENMP_WIN_RUNTIME is intentionally Windows-only.  Linux and other
    # Unix builds retain the compiler-selected OpenMP runtime.
    if(WIN32)
        set(DLP_OPENMP_WIN_RUNTIME "AUTO" CACHE STRING
            "Windows OpenMP runtime policy (AUTO, VCOMP, or LLVM)"
        )
        set_property(CACHE DLP_OPENMP_WIN_RUNTIME PROPERTY STRINGS AUTO VCOMP LLVM)
        string(TOUPPER "${DLP_OPENMP_WIN_RUNTIME}" _dlp_requested_runtime)
        if(NOT _dlp_requested_runtime MATCHES "^(AUTO|VCOMP|LLVM)$")
            message(FATAL_ERROR
                "DLP_OPENMP_WIN_RUNTIME must be AUTO, VCOMP, or LLVM; got "
                "'${DLP_OPENMP_WIN_RUNTIME}'."
            )
        endif()

        # A policy change in an existing build directory is unsafe because
        # FindOpenMP may have cached flags and imported targets for the old
        # runtime.  Require a clean configure instead of silently flipping.
        if(DEFINED DLP_OPENMP_CONFIGURED_REQUEST
           AND NOT "${DLP_OPENMP_CONFIGURED_REQUEST}" STREQUAL
                   "${_dlp_requested_runtime}")
            message(FATAL_ERROR
                "DLP_OPENMP_WIN_RUNTIME changed from "
                "'${DLP_OPENMP_CONFIGURED_REQUEST}' to "
                "'${_dlp_requested_runtime}' in an existing build directory. "
                "Please configure a fresh build directory."
            )
        endif()
        set(DLP_OPENMP_CONFIGURED_REQUEST "${_dlp_requested_runtime}"
            CACHE INTERNAL "DLP OpenMP runtime policy used by this build" FORCE
        )

        set(_dlp_is_clang_cl FALSE)
        if((CMAKE_C_COMPILER_ID MATCHES "Clang"
            OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
           AND (CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
                OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
            set(_dlp_is_clang_cl TRUE)
        endif()
        set(_dlp_is_cl FALSE)
        if(CMAKE_C_COMPILER_ID STREQUAL "MSVC"
           OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(_dlp_is_cl TRUE)
        endif()

        if(_dlp_is_clang_cl)
            if(_dlp_requested_runtime STREQUAL "VCOMP")
                message(FATAL_ERROR
                    "DLP_OPENMP_WIN_RUNTIME=VCOMP is not supported for "
                    "clang-cl. Use AUTO or LLVM so DLP can validate LLVM "
                    "libomp for the clang-cl frontend."
                )
            endif()
            set(DLP_OPENMP_WIN_RUNTIME_SELECTED "LLVM")
            set(DLP_OPENMP_WIN_FRONTEND_SELECTED "CLANG_CL")
        elseif(_dlp_is_cl)
            if(_dlp_requested_runtime STREQUAL "LLVM")
                if(CMAKE_VERSION VERSION_LESS "3.30")
                    message(FATAL_ERROR
                        "DLP_OPENMP_WIN_RUNTIME=LLVM for cl.exe requires "
                        "CMake 3.30 or newer; this project otherwise supports "
                        "CMake 3.26+. Use VCOMP/AUTO or a newer CMake."
                    )
                endif()
                set(DLP_OPENMP_WIN_RUNTIME_SELECTED "LLVM")
            else()
                set(DLP_OPENMP_WIN_RUNTIME_SELECTED "VCOMP")
            endif()
            set(DLP_OPENMP_WIN_FRONTEND_SELECTED "MSVC")
        else()
            # MinGW and non-MSVC Windows toolchains keep their compiler-selected
            # runtime.  The policy remains visible in the cache but does not
            # pretend that VCOMP is available for those frontends.
            set(DLP_OPENMP_WIN_RUNTIME_SELECTED "COMPILER")
            set(DLP_OPENMP_WIN_FRONTEND_SELECTED "COMPILER")
        endif()
    else()
        set(DLP_OPENMP_WIN_RUNTIME_SELECTED "COMPILER")
        set(DLP_OPENMP_WIN_FRONTEND_SELECTED "COMPILER")
    endif()

    if(_dlp_is_clang_cl)
        set(_dlp_openmp_frontend "CLANG_CL")
    elseif(_dlp_is_cl)
        set(_dlp_openmp_frontend "MSVC")
    else()
        set(_dlp_openmp_frontend "COMPILER")
    endif()

    # Do not allow stale FindOpenMP cache entries to choose a different
    # Windows runtime.  These are discovery results, not user-facing DLP
    # policy, but an already cached incompatible flag is evidence that this
    # build directory was configured for another runtime.
    if(WIN32 AND (_dlp_is_cl OR _dlp_is_clang_cl))
        if(DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "VCOMP")
            foreach(_dlp_flag_var OpenMP_C_FLAGS OpenMP_CXX_FLAGS)
                dlp_openmp_flags_match_runtime(
                    "${${_dlp_flag_var}}"
                    "LLVM"
                    "${_dlp_openmp_frontend}"
                    _dlp_stale_llvm_flags
                )
                if(DEFINED ${_dlp_flag_var} AND _dlp_stale_llvm_flags)
                    message(FATAL_ERROR
                        "The existing build directory contains LLVM OpenMP "
                        "flags (${_dlp_flag_var}) but the selected DLP policy "
                        "is VCOMP. Configure a fresh build directory."
                    )
                endif()
            endforeach()
        elseif(DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "LLVM")
            foreach(_dlp_flag_var OpenMP_C_FLAGS OpenMP_CXX_FLAGS)
                dlp_openmp_flags_match_runtime(
                    "${${_dlp_flag_var}}"
                    "VCOMP"
                    "${_dlp_openmp_frontend}"
                    _dlp_stale_vcomp_flags
                )
                if(DEFINED ${_dlp_flag_var} AND _dlp_stale_vcomp_flags)
                    message(FATAL_ERROR
                        "The existing build directory contains VCOMP OpenMP "
                        "flags (${_dlp_flag_var}) but the selected DLP policy "
                        "is LLVM. Configure a fresh build directory."
                    )
                endif()
            endforeach()
        endif()
    endif()

    foreach(_dlp_openmp_cache_var
            OpenMP_C_FLAGS
            OpenMP_C_LIB_NAMES
            OpenMP_C_LIBRARIES
            OpenMP_C_INCLUDE_DIRS
            OpenMP_CXX_FLAGS
            OpenMP_CXX_LIB_NAMES
            OpenMP_CXX_LIBRARIES
            OpenMP_CXX_INCLUDE_DIRS
            OpenMP_FOUND
            OpenMP_C_FOUND
            OpenMP_CXX_FOUND
            OpenMP_RUNTIME_MSVC)
        unset(${_dlp_openmp_cache_var} CACHE)
        unset(${_dlp_openmp_cache_var})
    endforeach()

    # FindOpenMP uses OpenMP_RUNTIME_MSVC only for the explicit LLVM mode.
    # In particular, VCOMP must clear an inherited "llvm" setting.
    if(DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "LLVM"
       AND WIN32 AND (_dlp_is_cl OR _dlp_is_clang_cl))
        set(OpenMP_RUNTIME_MSVC "llvm")
    else()
        unset(OpenMP_RUNTIME_MSVC CACHE)
        unset(OpenMP_RUNTIME_MSVC)
    endif()

    set(_dlp_original_prefix_path "${CMAKE_PREFIX_PATH}")
    if(DLP_OPENMP_ROOT)
        message(STATUS "Using custom OpenMP installation from: ${DLP_OPENMP_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH "${DLP_OPENMP_ROOT}")
    endif()

    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    set(CMAKE_PREFIX_PATH "${_dlp_original_prefix_path}")

    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR
            "OpenMP was requested but was not found. Install OpenMP or set "
            "DLP_THREADING_MODEL=none."
        )
    endif()

    # FindOpenMP versions before the runtime-selection support may ignore
    # OpenMP_RUNTIME_MSVC.  Validate the flags it actually selected so an LLVM
    # build cannot be mislabeled as LLVM while using plain /openmp.
    if(WIN32 AND (_dlp_is_cl OR _dlp_is_clang_cl))
        dlp_openmp_flags_match_runtime(
            "${OpenMP_C_FLAGS}"
            "${DLP_OPENMP_WIN_RUNTIME_SELECTED}"
            "${_dlp_openmp_frontend}"
            _dlp_c_runtime_flags_match
        )
        dlp_openmp_flags_match_runtime(
            "${OpenMP_CXX_FLAGS}"
            "${DLP_OPENMP_WIN_RUNTIME_SELECTED}"
            "${_dlp_openmp_frontend}"
            _dlp_cxx_runtime_flags_match
        )
        if(NOT _dlp_c_runtime_flags_match OR NOT _dlp_cxx_runtime_flags_match)
            message(FATAL_ERROR
                "FindOpenMP did not select the requested "
                "${DLP_OPENMP_WIN_RUNTIME_SELECTED} runtime flags. "
                "CMake may not support OpenMP_RUNTIME_MSVC for this compiler; "
                "use a fresh build with a compatible CMake version."
            )
        endif()
    endif()

    message(STATUS "Found OpenMP: C flags=${OpenMP_C_FLAGS}, CXX flags=${OpenMP_CXX_FLAGS}")

    # Compile and link against the exact imported targets used by DLP.  The
    # directive-bearing probes prevent a header-only API check from accepting
    # an ABI or runtime whose required surface cannot compile and link.
    # CMake's try_compile() does not execute these probe programs.
    set(_dlp_probe_identity
        "${DLP_OPENMP_WIN_RUNTIME_SELECTED};${OpenMP_C_FLAGS};"
        "${OpenMP_CXX_FLAGS};${OpenMP_C_LIBRARIES};${OpenMP_CXX_LIBRARIES};"
        "${CMAKE_MSVC_RUNTIME_LIBRARY}"
    )
    string(MD5 _dlp_probe_key "${_dlp_probe_identity}")
    set(_dlp_probe_dir
        "${CMAKE_BINARY_DIR}/CMakeFiles/dlp_openmp_probes-${_dlp_probe_key}"
    )
    file(MAKE_DIRECTORY "${_dlp_probe_dir}")
    set(_dlp_probe_cmake_flags)
    if(DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        list(APPEND _dlp_probe_cmake_flags
            "-DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW"
            "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        )
    endif()
    set(_dlp_probe_c_flags "${OpenMP_C_FLAGS}")
    set(_dlp_probe_cxx_flags "${OpenMP_CXX_FLAGS}")
    set(_dlp_probe_compile_definitions)
    if(WIN32
       AND _dlp_is_clang_cl
       AND DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "LLVM")
        # VS clang-cl uses the MSVC omp.h, which hides the LLVM API surface
        # unless this vendor macro is explicitly supplied.
        string(APPEND _dlp_probe_c_flags " -D_OPENMP_LLVM_RUNTIME")
        string(APPEND _dlp_probe_cxx_flags " -D_OPENMP_LLVM_RUNTIME")
        list(APPEND _dlp_probe_compile_definitions /D_OPENMP_LLVM_RUNTIME)
    endif()
    file(WRITE "${_dlp_probe_dir}/baseline.c" [=[
#include <omp.h>

int main(void)
{
    int sum  = 0;
    int team = 0;
#pragma omp parallel reduction(+ : sum)
    {
        sum += 1;
#pragma omp single
        team = omp_get_num_threads();
    }
    omp_set_num_threads(1);
    (void)omp_get_max_threads();
    (void)omp_get_num_procs();
    (void)omp_get_num_threads();
    (void)omp_get_thread_num();
    (void)omp_in_parallel();
    return (sum > 0 && team > 0) ? 0 : 1;
}
]=])
    file(WRITE "${_dlp_probe_dir}/baseline.cc" [=[
#include <omp.h>

int main()
{
    int sum  = 0;
    int team = 0;
#pragma omp parallel reduction(+ : sum)
    {
        sum += 1;
#pragma omp single
        team = omp_get_num_threads();
    }
    omp_set_num_threads(1);
    (void)omp_get_max_threads();
    (void)omp_get_num_procs();
    (void)omp_get_num_threads();
    (void)omp_get_thread_num();
    (void)omp_in_parallel();
    return (sum > 0 && team > 0) ? 0 : 1;
}
]=])
    file(WRITE "${_dlp_probe_dir}/nesting.c" [=[
#include <omp.h>

int main(void)
{
    int in_parallel = 0;
#pragma omp parallel reduction(+ : in_parallel)
    {
        in_parallel += omp_in_parallel();
    }
    omp_set_nested(0);
    return in_parallel > 0 ? 0 : 1;
}
]=])
    file(WRITE "${_dlp_probe_dir}/nesting.cc" [=[
#include <omp.h>

int main()
{
    int in_parallel = 0;
#pragma omp parallel reduction(+ : in_parallel)
    {
        in_parallel += omp_in_parallel();
    }
    omp_set_nested(0);
    return in_parallel > 0 ? 0 : 1;
}
]=])
    file(WRITE "${_dlp_probe_dir}/active.c" [=[
#include <omp.h>

int main(void)
{
    (void)omp_get_active_level();
    (void)omp_get_max_active_levels();
    omp_set_max_active_levels(1);
    return 0;
}
]=])
    file(WRITE "${_dlp_probe_dir}/active.cc" [=[
#include <omp.h>

int main()
{
    (void)omp_get_active_level();
    (void)omp_get_max_active_levels();
    omp_set_max_active_levels(1);
    return 0;
}
]=])
    file(WRITE "${_dlp_probe_dir}/places.c" [=[
#include <omp.h>

int main(void)
{
    int place = omp_get_place_num();
    int count = omp_get_place_num_procs(place);
    int ids[1] = { 0 };
    if (count > 0)
        omp_get_place_proc_ids(place, ids);
    return 0;
}
]=])
    file(WRITE "${_dlp_probe_dir}/places.cc" [=[
#include <omp.h>

int main()
{
    int place = omp_get_place_num();
    int count = omp_get_place_num_procs(place);
    int ids[1] = { 0 };
    if (count > 0)
        omp_get_place_proc_ids(place, ids);
    return 0;
}
]=])

    try_compile(
        _dlp_baseline_c
        "${_dlp_probe_dir}/baseline-c"
        SOURCES "${_dlp_probe_dir}/baseline.c"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_C
        CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=${_dlp_probe_c_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_baseline_c_output
    )
    try_compile(
        _dlp_baseline_cxx
        "${_dlp_probe_dir}/baseline-cxx"
        SOURCES "${_dlp_probe_dir}/baseline.cc"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_CXX
        CMAKE_FLAGS "-DCMAKE_CXX_FLAGS:STRING=${_dlp_probe_cxx_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_baseline_cxx_output
    )
    if(NOT _dlp_baseline_c OR NOT _dlp_baseline_cxx)
        message(FATAL_ERROR
            "The selected OpenMP runtime failed the required C/C++ "
            "compile-and-link baseline probe. C output:\n${_dlp_baseline_c_output}\n"
            "C++ output:\n${_dlp_baseline_cxx_output}"
        )
    else()
        message(STATUS
            "OpenMP baseline probe: C/C++ compile-and-link succeeded"
        )
    endif()

    try_compile(
        _dlp_nesting_c
        "${_dlp_probe_dir}/nesting-c"
        SOURCES "${_dlp_probe_dir}/nesting.c"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_C
        CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=${_dlp_probe_c_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_nesting_c_output
    )
    try_compile(
        _dlp_nesting_cxx
        "${_dlp_probe_dir}/nesting-cxx"
        SOURCES "${_dlp_probe_dir}/nesting.cc"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_CXX
        CMAKE_FLAGS "-DCMAKE_CXX_FLAGS:STRING=${_dlp_probe_cxx_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_nesting_cxx_output
    )
    if(_dlp_nesting_c AND _dlp_nesting_cxx)
        set(DLP_OPENMP_HAS_NESTING 1)
        message(STATUS
            "OpenMP nesting probe: C/C++ compile-and-link succeeded"
        )
    elseif(DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "VCOMP")
        message(FATAL_ERROR
            "The selected VCOMP runtime failed the required nesting "
            "compile-and-link probe. C output:\n${_dlp_nesting_c_output}\n"
            "C++ output:\n${_dlp_nesting_cxx_output}"
        )
    else()
        message(STATUS
            "OpenMP nesting probe unavailable; the wrapper will use the "
            "active-level path when available."
        )
    endif()

    try_compile(
        _dlp_active_c
        "${_dlp_probe_dir}/active-c"
        SOURCES "${_dlp_probe_dir}/active.c"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_C
        CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=${_dlp_probe_c_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_active_c_output
    )
    try_compile(
        _dlp_active_cxx
        "${_dlp_probe_dir}/active-cxx"
        SOURCES "${_dlp_probe_dir}/active.cc"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_CXX
        CMAKE_FLAGS "-DCMAKE_CXX_FLAGS:STRING=${_dlp_probe_cxx_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_active_cxx_output
    )
    if(_dlp_active_c AND _dlp_active_cxx)
        set(DLP_OPENMP_HAS_ACTIVE_LEVELS 1)
        message(STATUS
            "OpenMP active-level probe: C/C++ compile-and-link succeeded"
        )
    elseif(DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "VCOMP")
        message(STATUS
            "VCOMP active-level probe unavailable; using the conservative "
            "omp_in_parallel() predicate (C=${_dlp_active_c}, "
            "CXX=${_dlp_active_cxx})."
        )
    else()
        message(STATUS
            "OpenMP active-level probe unavailable; using the conservative "
            "omp_in_parallel() predicate (C=${_dlp_active_c}, "
            "CXX=${_dlp_active_cxx})."
        )
    endif()

    try_compile(
        _dlp_places_c
        "${_dlp_probe_dir}/places-c"
        SOURCES "${_dlp_probe_dir}/places.c"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_C
        CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=${_dlp_probe_c_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_places_c_output
    )
    try_compile(
        _dlp_places_cxx
        "${_dlp_probe_dir}/places-cxx"
        SOURCES "${_dlp_probe_dir}/places.cc"
        COMPILE_DEFINITIONS ${_dlp_probe_compile_definitions}
        LINK_LIBRARIES OpenMP::OpenMP_CXX
        CMAKE_FLAGS "-DCMAKE_CXX_FLAGS:STRING=${_dlp_probe_cxx_flags}"
                    ${_dlp_probe_cmake_flags}
        OUTPUT_VARIABLE _dlp_places_cxx_output
    )
    if(_dlp_places_c AND _dlp_places_cxx)
        set(DLP_OPENMP_HAS_PLACES 1)
        message(STATUS
            "OpenMP places probe: C/C++ compile-and-link succeeded"
        )
    else()
        message(STATUS
            "OpenMP places probe unavailable (C=${_dlp_places_c}, "
            "CXX=${_dlp_places_cxx}, WIN32=${WIN32})"
        )
    endif()

    if(WIN32 AND (_dlp_is_cl OR _dlp_is_clang_cl)
       AND _dlp_baseline_c AND _dlp_baseline_cxx
       AND _dlp_nesting_c AND _dlp_nesting_cxx)
        if(DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "VCOMP"
           AND _dlp_c_runtime_flags_match
           AND _dlp_cxx_runtime_flags_match)
            set(DLP_OPENMP_RUNTIME_VCOMP 1)
        elseif(DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "LLVM"
               AND _dlp_c_runtime_flags_match
               AND _dlp_cxx_runtime_flags_match)
            set(DLP_OPENMP_RUNTIME_LLVM 1)
        endif()
    endif()

    configure_file(
        "${DLP_SOURCE_DIR}/cmake/dlp_openmp_config.h.in"
        "${CMAKE_BINARY_DIR}/include/private/dlp_openmp_config.h"
        @ONLY
    )

    add_library(dlp_openmp_interface INTERFACE)
    target_include_directories(dlp_openmp_interface INTERFACE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/private>
        $<BUILD_INTERFACE:${DLP_SOURCE_DIR}/classic/include>
    )
    target_link_libraries(dlp_openmp_interface INTERFACE
        OpenMP::OpenMP_C
        OpenMP::OpenMP_CXX
    )
    if(WIN32
       AND _dlp_is_clang_cl
       AND DLP_OPENMP_WIN_RUNTIME_SELECTED STREQUAL "LLVM")
        target_compile_definitions(dlp_openmp_interface INTERFACE
            _OPENMP_LLVM_RUNTIME
        )
    endif()
    add_library(dlp::openmp ALIAS dlp_openmp_interface)
    set_target_properties(dlp_openmp_interface PROPERTIES EXPORT_NAME openmp)
    install(TARGETS dlp_openmp_interface
        EXPORT AoclDlpTargets
        INCLUDES DESTINATION include
    )

    set(DLP_OPENMP_ENABLED ${DLP_OPENMP_ENABLED} PARENT_SCOPE)
    set(DLP_OPENMP_HAS_ACTIVE_LEVELS ${DLP_OPENMP_HAS_ACTIVE_LEVELS} PARENT_SCOPE)
    set(DLP_OPENMP_HAS_NESTING ${DLP_OPENMP_HAS_NESTING} PARENT_SCOPE)
    set(DLP_OPENMP_HAS_PLACES ${DLP_OPENMP_HAS_PLACES} PARENT_SCOPE)
    set(DLP_OPENMP_RUNTIME_VCOMP ${DLP_OPENMP_RUNTIME_VCOMP} PARENT_SCOPE)
    set(DLP_OPENMP_RUNTIME_LLVM ${DLP_OPENMP_RUNTIME_LLVM} PARENT_SCOPE)
    set(DLP_OPENMP_WIN_RUNTIME_SELECTED
        ${DLP_OPENMP_WIN_RUNTIME_SELECTED} PARENT_SCOPE
    )
    set(DLP_OPENMP_WIN_FRONTEND_SELECTED
        ${DLP_OPENMP_WIN_FRONTEND_SELECTED} PARENT_SCOPE
    )
    message(STATUS
        "OpenMP integration configured: runtime=${DLP_OPENMP_WIN_RUNTIME_SELECTED}, "
        "active-levels=${DLP_OPENMP_HAS_ACTIVE_LEVELS}, "
        "nesting=${DLP_OPENMP_HAS_NESTING}, "
        "places=${DLP_OPENMP_HAS_PLACES}"
    )
endfunction()
