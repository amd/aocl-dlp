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

# Profile-Guided Optimization (PGO) support for AOCL-DLP
#
# PGO builds happen in two phases:
# Phase 1 (Generate): Build with instrumentation, run workload, collect profile data
# Phase 2 (Use): Build with profile data for optimized code generation
#
# Usage:
#   Phase 1: cmake -DDLP_ENABLE_PGO=GENERATE -DDLP_PGO_PROFILE_DIR=<profile_dir> ..
#            make
#            # Run your representative workload (benchmarks/tests)
#            ./build/bin/bench_gemm --config configs/your_workload.yaml
#
#   Phase 2: cmake -DDLP_ENABLE_PGO=USE -DDLP_PGO_PROFILE_DIR=<profile_dir> ..
#            make
# Note: LLVM has a different approach, refer below comments.

function(dlp_define_pgo_options)
    # PGO mode: OFF, GENERATE, or USE
    set(DLP_ENABLE_PGO "OFF" CACHE STRING "Enable Profile-Guided Optimization (OFF/GENERATE/USE)")
    set_property(CACHE DLP_ENABLE_PGO PROPERTY STRINGS "OFF" "GENERATE" "USE")

    # Directory where profile data will be stored/read from
    set(DLP_PGO_PROFILE_DIR "${CMAKE_BINARY_DIR}/pgo_profiles" CACHE PATH
        "Directory for PGO profile data")

    # Propagate to parent scope
    set(DLP_ENABLE_PGO ${DLP_ENABLE_PGO} PARENT_SCOPE)
    set(DLP_PGO_PROFILE_DIR ${DLP_PGO_PROFILE_DIR} PARENT_SCOPE)
endfunction()

function(dlp_setup_pgo)
    if(DLP_ENABLE_PGO STREQUAL "OFF")
        return()
    endif()

    message(STATUS "PGO Mode: ${DLP_ENABLE_PGO}")
    message(STATUS "PGO Profile Directory: ${DLP_PGO_PROFILE_DIR}")

    # Use pre-detected compiler macros
    if(NOT DLP_COMPILER_GCC AND NOT DLP_COMPILER_CLANG)
        message(FATAL_ERROR "PGO is only supported with GCC or Clang")
    endif()

    # Create profile directory for GENERATE phase
    if(DLP_ENABLE_PGO STREQUAL "GENERATE")
        file(MAKE_DIRECTORY ${DLP_PGO_PROFILE_DIR})

        if(DLP_COMPILER_GCC)
            # GCC PGO flags
            set(PGO_COMPILE_FLAGS "-fprofile-generate=${DLP_PGO_PROFILE_DIR}")
            set(PGO_LINK_FLAGS "-fprofile-generate=${DLP_PGO_PROFILE_DIR}")
            message(STATUS "Configuring GCC PGO instrumentation phase")
        elseif(DLP_COMPILER_CLANG)
            # Clang PGO flags
            # Note: For Clang, you MUST set LLVM_PROFILE_FILE environment variable when running
            # Use -fprofile-instr-generate which automatically links the profile runtime
            set(PGO_COMPILE_FLAGS "-fprofile-instr-generate")
            set(PGO_LINK_FLAGS "-fprofile-instr-generate")
            message(STATUS "Configuring Clang PGO instrumentation phase")
        endif()

        message(STATUS "")
        message(STATUS "========================================")
        message(STATUS "PGO GENERATE Phase - Next Steps:")
        message(STATUS "1. Build the library: make -j")
        message(STATUS "2. Run representative workload:")
        message(STATUS "   cd ${CMAKE_BINARY_DIR}/bin")
        message(STATUS "   ./bench_gemm --config ../configs/your_workload.yaml")
        message(STATUS "3. Profile data will be in: ${DLP_PGO_PROFILE_DIR}")
        if(DLP_COMPILER_CLANG)
            message(STATUS "4. For Clang, set environment variable before running:")
            message(STATUS "   export LLVM_PROFILE_FILE=${DLP_PGO_PROFILE_DIR}/default_%m.profraw")
            message(STATUS "   ./bench")
            message(STATUS "5. Merge profiles:")
            message(STATUS "   llvm-profdata merge -output=${DLP_PGO_PROFILE_DIR}/default.profdata ${DLP_PGO_PROFILE_DIR}/*.profraw")
            message(STATUS "6. Reconfigure for USE phase:")
        else()
            message(STATUS "4. Reconfigure for USE phase:")
        endif()
        message(STATUS "   cmake -DDLP_ENABLE_PGO=USE -DDLP_PGO_PROFILE_DIR=${DLP_PGO_PROFILE_DIR} ..")
        message(STATUS "========================================")
        message(STATUS "")

    elseif(DLP_ENABLE_PGO STREQUAL "USE")
        # Verify profile directory exists
        if(NOT EXISTS ${DLP_PGO_PROFILE_DIR})
            message(FATAL_ERROR "PGO profile directory does not exist: ${DLP_PGO_PROFILE_DIR}")
        endif()

        if(DLP_COMPILER_GCC)
            # GCC: Check for .gcda files
            file(GLOB PROFILE_FILES "${DLP_PGO_PROFILE_DIR}/*.gcda")
            if(NOT PROFILE_FILES)
                message(FATAL_ERROR "No GCC profile data (.gcda files) found in ${DLP_PGO_PROFILE_DIR}")
            endif()

            set(PGO_COMPILE_FLAGS "-fprofile-use=${DLP_PGO_PROFILE_DIR}" "-fprofile-correction")
            set(PGO_LINK_FLAGS "-fprofile-use=${DLP_PGO_PROFILE_DIR}")
            list(LENGTH PROFILE_FILES profile_count)
            message(STATUS "Configuring GCC PGO optimization phase with ${profile_count} profile files")
        elseif(DLP_COMPILER_CLANG)
            # Clang: Check for merged .profdata file
            set(PROFDATA_FILE "${DLP_PGO_PROFILE_DIR}/default.profdata")
            if(NOT EXISTS ${PROFDATA_FILE})
                message(FATAL_ERROR "Merged profile data not found: ${PROFDATA_FILE}")
                message(FATAL_ERROR "Run: llvm-profdata merge -output=${PROFDATA_FILE} ${DLP_PGO_PROFILE_DIR}/default_*.profraw")
            endif()

            set(PGO_COMPILE_FLAGS "-fprofile-instr-use=${PROFDATA_FILE}")
            set(PGO_LINK_FLAGS "-fprofile-instr-use=${PROFDATA_FILE}")
            message(STATUS "Configuring Clang PGO optimization phase with profile: ${PROFDATA_FILE}")
        endif()

        message(STATUS "")
        message(STATUS "========================================")
        message(STATUS "PGO USE Phase - Building optimized library")
        message(STATUS "Profile-guided optimizations will be applied")
        message(STATUS "========================================")
        message(STATUS "")
    endif()

    # Export flags to parent scope AND cache for use by other functions
    set(DLP_PGO_COMPILE_FLAGS ${PGO_COMPILE_FLAGS} PARENT_SCOPE)
    set(DLP_PGO_LINK_FLAGS ${PGO_LINK_FLAGS} PARENT_SCOPE)
endfunction()

function(dlp_apply_pgo_to_target target_name)
    if(NOT DLP_ENABLE_PGO OR DLP_ENABLE_PGO STREQUAL "OFF")
        return()
    endif()

    # Get flags from parent scope (set by dlp_setup_pgo)
    if(DEFINED DLP_PGO_COMPILE_FLAGS AND DLP_PGO_COMPILE_FLAGS)
        target_compile_options(${target_name} PRIVATE ${DLP_PGO_COMPILE_FLAGS})
        message(STATUS "Applied PGO compile flags to ${target_name}: ${DLP_PGO_COMPILE_FLAGS}")
    else()
        message(WARNING "DLP_PGO_COMPILE_FLAGS not set for target ${target_name}")
    endif()

    if(DEFINED DLP_PGO_LINK_FLAGS AND DLP_PGO_LINK_FLAGS)
        target_link_options(${target_name} PRIVATE ${DLP_PGO_LINK_FLAGS})
        message(STATUS "Applied PGO link flags to ${target_name}: ${DLP_PGO_LINK_FLAGS}")
    else()
        message(WARNING "DLP_PGO_LINK_FLAGS not set for target ${target_name}")
    endif()
endfunction()
