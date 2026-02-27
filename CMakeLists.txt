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
# Define DLP_SOURCE_DIR to support FetchContent usage
# (CMAKE_SOURCE_DIR points to parent project when used as subdirectory)
set(DLP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "AOCL-DLP source directory")

# Set CMake module path (must be done before including dlp_variables.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${DLP_SOURCE_DIR}/cmake")

# Include dlp_variables.cmake first to get CMAKE_MINIMUM_REQUIRED_VERSION
include(cmake/dlp_variables.cmake)

# Now use the variable for cmake_minimum_required
cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# Set CMake policy
cmake_policy(SET CMP0010 NEW)

# Include remaining CMake modules
include(cmake/dlp_install.cmake)
include(cmake/dlp_extensions.cmake)
include(cmake/dlp_core_options.cmake)
include(cmake/dlp_build_options.cmake)
include(cmake/dlp_dependencies.cmake)
include(cmake/dlp_documentation.cmake)
include(cmake/dlp_testing.cmake)
include(cmake/dlp_benchmark.cmake)
include(cmake/dlp_sanitizers.cmake)
include(cmake/dlp_pgo.cmake)
include(cmake/dlp_unit_tests.cmake)

# OS specific CMake modules goes here
if(UNIX)
    include(cmake/dlp_compiler_flags_linux.cmake)
elseif(WIN32)
    include(cmake/dlp_compiler_flags_windows.cmake)
endif()

# Project configuration - using PROJECT_NAME from dlp_variables.cmake
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

include(cmake/dlp_platform.cmake)

# Find pthread library for bench/example compilation in Milan machines
find_package(Threads REQUIRED)

# Define file extensions
dlp_define_extensions()
# Define Project Options (modular approach)
dlp_define_core_options()
dlp_define_testing_options()
dlp_define_benchmarking_options()
dlp_define_build_options()
dlp_define_documentation_options()
dlp_define_pgo_options()
# Initialize Project dependant variables
dlp_init_project_dep_var()
# Add capability of openmp
dlp_setup_openmp()
# Setup Profile-Guided Optimization
dlp_setup_pgo()
# Run the ZnVer support detection
dlp_check_znver_support()

# Add classic directory
add_subdirectory(classic)
add_subdirectory(src)

set(DLP_SRCS "")

# Create the library target
add_library(${PROJECT_NAME} SHARED ${DLP_SRCS})

# Create the static library target including object libraries
add_library(${PROJECT_NAME}_static STATIC
    ${DLP_SRCS}
    $<TARGET_OBJECTS:aocl_dlp_classic>
    $<TARGET_OBJECTS:aocl_dlp_plus>
    $<TARGET_OBJECTS:aocl_dlp_jit_amdzen>
    $<TARGET_OBJECTS:dlp_kernel_zen>
    $<TARGET_OBJECTS:dlp_kernel_zen4>
)

# Set OS dependant definitions/target properties dlp_compiler_flags_<platform>.cmake
dlp_set_platform_options()

# Link with the classic object library and dependencies
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${CLASSIC_TARGETS}
        ${DLP_PLUS_TARGETS}
        $<$<PLATFORM_ID:Linux>:-Wl,--whole-archive>
        aocl_dlp_jit_amdzen
        $<$<PLATFORM_ID:Linux>:-Wl,--no-whole-archive>
    PUBLIC
        Threads::Threads
)

# Link with OpenMP if enabled
if(DLP_ENABLE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PRIVATE dlp::openmp)
endif()

# Link with dependencies for static library (no object libraries needed as they're included)
target_link_libraries(${PROJECT_NAME}_static
    PUBLIC
        Threads::Threads
)

# Link with OpenMP if enabled
if(DLP_ENABLE_OPENMP)
    target_link_libraries(${PROJECT_NAME}_static PRIVATE dlp::openmp)
endif()

# Setup atomic support for 16-byte atomic structs (platform-specific)
if(COMMAND dlp_setup_atomic_support)
    dlp_setup_atomic_support()
endif()

# Add all sanitizers to the target
dlp_add_all_sanitizers(${PROJECT_NAME} ${PROJECT_NAME}_static)

# Apply PGO to library targets (both final libraries and object libraries)
dlp_apply_pgo_to_target(${PROJECT_NAME})
dlp_apply_pgo_to_target(${PROJECT_NAME}_static)
dlp_apply_pgo_to_target(aocl_dlp_classic)
dlp_apply_pgo_to_target(aocl_dlp_plus)

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

# Apply centralized compiler flags directly to avoid interface target export issues
# Get the generic compiler flags and apply them directly
get_target_property(GENERIC_FLAGS dlp_compiler_flags INTERFACE_COMPILE_OPTIONS)
if(GENERIC_FLAGS)
    target_compile_options(${PROJECT_NAME} PRIVATE ${GENERIC_FLAGS})
    target_compile_options(${PROJECT_NAME}_static PRIVATE ${GENERIC_FLAGS})
endif()

# Apply release-specific flags for Release builds
get_target_property(RELEASE_FLAGS dlp_compiler_flags_release INTERFACE_COMPILE_OPTIONS)
if(RELEASE_FLAGS AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${PROJECT_NAME} PRIVATE ${RELEASE_FLAGS})
    target_compile_options(${PROJECT_NAME}_static PRIVATE ${RELEASE_FLAGS})
endif()

# Apply installation rules from dlp_install.cmake
dlp_install(${PROJECT_NAME})
dlp_install(${PROJECT_NAME}_static)

# Set up CMake package configuration variables
set(AOCL_DLP_REQUIRED_COMPONENTS "aocl-dlp;aocl-dlp_static")
set(AOCL_DLP_AVAILABLE_COMPONENTS "aocl-dlp;aocl-dlp_static")

# Generate config header
dlp_generate_config_header()
# Install CMake package configuration files
dlp_install_package_config()

# Enable testing
if(BUILD_TESTING)
    message(STATUS "Enabling testing")
    enable_testing()
    fetch_gtest()
    add_subdirectory(tests)
    add_unit_tests_paths()
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
