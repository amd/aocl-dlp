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

#
# Add a C example executable
#
# This function creates an example executable from C sources, links it with the
# project library and the dependencies every example needs, applies the
# centralized compiler flags, and optionally installs it.
#
# The examples are pure C while AOCL-DLP is C++, so CMake would derive the C
# driver from their sources. That driver contributes only the C half of the
# compiler's sanitizer runtime, whereas a sanitizer-instrumented shared library
# deliberately leaves the C++ half's handlers (vptr, dynamic type cache)
# undefined for the final executable to supply - so a C-driver link either fails
# to resolve them or produces a binary that aborts at load. Naming the C++
# driver here is what makes a C example link correctly against the library.
#
# Parameters:
#   NAME                - Name of the example executable
#   SOURCES             - Example source files
#   INCLUDE_DIRS        - Additional include directories (optional)
#   DEPENDS             - Additional dependencies (optional)
#   INSTALL_DESTINATION - Install directory for the executable (optional)
#
# Example:
#   dlp_add_c_example(
#       NAME simple_gemm_f32
#       SOURCES simple_gemm_f32.c
#       INSTALL_DESTINATION examples/classic
#   )
#
function(dlp_add_c_example)
    # Parse arguments for the example function
    set(options "")
    set(oneValueArgs NAME INSTALL_DESTINATION)
    set(multiValueArgs SOURCES DEPENDS INCLUDE_DIRS)

    cmake_parse_arguments(DLP_EXAMPLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Check if required arguments are provided
    if(NOT DLP_EXAMPLE_NAME)
        message(FATAL_ERROR "dlp_add_c_example: NAME argument is required")
    endif()

    if(NOT DLP_EXAMPLE_SOURCES)
        message(FATAL_ERROR "dlp_add_c_example: SOURCES argument is required")
    endif()

    add_executable(${DLP_EXAMPLE_NAME} ${DLP_EXAMPLE_SOURCES})

    # Link with the C++ driver; see the note above this function
    set_target_properties(${DLP_EXAMPLE_NAME} PROPERTIES LINKER_LANGUAGE CXX)

    # Choose library target based on static linking preference
    if(DLP_EXAMPLES_LINK_STATIC)
        set(DLP_LIBRARY_TARGET ${PROJECT_NAME}_static)
        message(STATUS "Linking example ${DLP_EXAMPLE_NAME} with static AOCL-DLP library (whole-archive automatic)")
    else()
        set(DLP_LIBRARY_TARGET ${PROJECT_NAME})
        message(STATUS "Linking example ${DLP_EXAMPLE_NAME} with shared AOCL-DLP library")
    endif()

    # Link with the main project library and the math library on Unix/Linux
    # Note: For static library, whole-archive is automatically applied via INTERFACE_LINK_OPTIONS
    target_link_libraries(${DLP_EXAMPLE_NAME}
        PRIVATE
            ${DLP_LIBRARY_TARGET}
            ${DLP_EXAMPLE_DEPENDS}
            $<$<BOOL:${UNIX}>:m>
    )

    # Add include directories - follow modern CMake target-based approach
    target_include_directories(${DLP_EXAMPLE_NAME}
        PRIVATE
            ${DLP_SOURCE_DIR}/include
            ${DLP_EXAMPLE_INCLUDE_DIRS}
    )

    # Link with OpenMP if available
    if(OpenMP_C_FOUND)
        target_link_libraries(${DLP_EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_C)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${DLP_EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(OpenMP_C_FOUND OR OpenMP_CXX_FOUND)
        target_compile_definitions(${DLP_EXAMPLE_NAME} PRIVATE DLP_EXAMPLE_ENABLE_OPENMP)
    endif()

    # Set compiler flags
    dlp_set_global_compile_flags(${DLP_EXAMPLE_NAME})

    if(DLP_EXAMPLE_INSTALL_DESTINATION)
        install(TARGETS ${DLP_EXAMPLE_NAME}
            RUNTIME DESTINATION ${DLP_EXAMPLE_INSTALL_DESTINATION}
        )
    endif()

endfunction()
