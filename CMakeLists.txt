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

cmake_minimum_required(VERSION 3.26)

# Set CMake policy
cmake_policy(SET CMP0010 NEW)

# Set CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include CMake modules
include(cmake/dlp_variables.cmake)
include(cmake/dlp_install.cmake)
include(cmake/dlp_extensions.cmake)
include(cmake/dlp_options.cmake)
include(cmake/dlp_dependencies.cmake)
include(cmake/dlp_documentation.cmake)
include(cmake/dlp_testing.cmake)
include(cmake/dlp_benchmark.cmake)
include(cmake/dlp_sanitizers.cmake)

# OS specific CMake modules goes here
if(UNIX)
    include(cmake/dlp_compiler_flags_linux.cmake)
elseif(WIN32)
    include(cmake/dlp_compiler_flags_windows.cmake)
endif()

# Project configuration - using PROJECT_NAME from dlp_variables.cmake
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

include(cmake/dlp_platform.cmake)

# Define file extensions
dlp_define_extensions()
# Define Project Options
dlp_define_options()
# Initialize Project dependant variables
dlp_init_project_dep_var()
# Add capability of openmp
dlp_setup_openmp()
# Run the ZnVer support detection
dlp_check_znver_support()

# Add classic directory
add_subdirectory(classic)
add_subdirectory(src)

set(DLP_SRCS "")

# Create the library target
add_library(${PROJECT_NAME} SHARED ${DLP_SRCS})

# Create the static library target
add_library(${PROJECT_NAME}_static STATIC ${DLP_SRCS})

# Set OS dependant definitions/target properties dlp_compiler_flags_<platform>.cmake
dlp_set_platform_options()

# Link with the classic object library and dependencies
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${CLASSIC_TARGETS}
        ${DLP_PLUS_TARGETS}
    PUBLIC
        dlp::openmp
)

# Link with the classic object static library and dependencies
target_link_libraries(${PROJECT_NAME}_static
    PUBLIC
        ${CLASSIC_TARGETS}
    PUBLIC
        dlp::openmp
)

# Add all sanitizers to the target
dlp_add_all_sanitizers(${PROJECT_NAME} ${PROJECT_NAME}_static)

# Set include directories for the target shared library
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Set include directories for the target static library
target_include_directories(${PROJECT_NAME}_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Apply centralized compiler flags
dlp_set_global_compile_flags(${PROJECT_NAME})

# Apply installation rules from dlp_install.cmake
dlp_install(${PROJECT_NAME})

# Enable testing
if(BUILD_TESTING)
    message(STATUS "Enabling testing")
    enable_testing()
    add_subdirectory(tests)
endif()

# Add benchmarking
if(BUILD_BENCHMARKS)
    message(STATUS "Enabling benchmarks")
    add_subdirectory(bench)
else()
    message(STATUS "Disabling benchmarks")
endif()

# Add example build option
if(BUILD_EXAMPLES)
    message(STATUS "Enabling examples")
    add_subdirectory(examples)
endif()

# Generate Doxygen documentation
if(BUILD_DOXYGEN)
    dlp_generate_doxygen_config()
    dlp_generate_doxygen_docs()
endif()

# Generate Sphinx documentation
if(BUILD_SPHINX)
    dlp_generate_sphinx_docs()
endif()

# Generate config file with version info
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/aocl_dlp_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/aocl_dlp_config.h
)
