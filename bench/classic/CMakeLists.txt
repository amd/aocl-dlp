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

# Use project's glob function to find all source files
dlp_glob_sources(BENCH_SOURCES)
list(REMOVE_DUPLICATES BENCH_SOURCES)

# Filter sources to get benchmark executables (exclude helper files)
list(FILTER BENCH_SOURCES INCLUDE REGEX "bench_[a-z_]+\\.c$")

# Create list of bench helper files
set(BENCH_HELPER_SOURCES "")
foreach(source ${BENCH_SOURCES})
    if(source MATCHES "bench_lpgemm_.*helpers.*\\.c$")
        list(APPEND BENCH_HELPER_SOURCES ${source})
    endif()
endforeach()

# Create an executable for each benchmark source file
foreach(source ${BENCH_SOURCES})
    # Get the filename without extension
    get_filename_component(exec_name ${source} NAME_WE)

    # Check if this is a helper file that shouldn't be built as a standalone executable
    list(FIND BENCH_HELPER_SOURCES ${source} HELPER_INDEX)
    if(${HELPER_INDEX} EQUAL -1)
        # Create the executable
        add_executable(${exec_name} ${source} ${BENCH_HELPER_SOURCES})

        # Set include directories for the benchmark executable
        target_include_directories(${exec_name}
            PRIVATE
                ${CLASSIC_INCLUDES}
                ${CMAKE_BINARY_DIR}/include  # For aocl_dlp_config.h
        )

        # Link against the AOCL DLP library
        target_link_libraries(${exec_name}
            PRIVATE
                aocl-dlp
                $<$<BOOL:${UNIX}>:m>  # Link math library for any math functions (Unix-like platforms)
        )

        # Link with OpenMP if it's enabled (benchmarks may use OpenMP functions directly)
        if(TARGET OpenMP::OpenMP_C AND DLP_THREADING_MODEL STREQUAL "openmp")
            target_link_libraries(${exec_name} PRIVATE OpenMP::OpenMP_C)
        endif()

        # Set compile flags (ensuring all executables use -fPIC for consistency)
        set_target_properties(${exec_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
        )

        # Install the benchmark executables using the project's install function if available
        install(TARGETS ${exec_name}
            RUNTIME DESTINATION bin/bench
        )
    endif()
endforeach()

# Copy bench input files to build directory for easy testing
set(TEXT_EXTENSIONS ".txt")
set(BENCH_INPUT_FILES "")
foreach(ext ${TEXT_EXTENSIONS})
    file(GLOB txt_files "*${ext}")
    list(APPEND BENCH_INPUT_FILES ${txt_files})
endforeach()

foreach(input_file ${BENCH_INPUT_FILES})
    get_filename_component(file_name ${input_file} NAME)
    configure_file(${input_file} ${CMAKE_CURRENT_BINARY_DIR}/${file_name} COPYONLY)
endforeach()

message(STATUS "Classic bench executables configured")
