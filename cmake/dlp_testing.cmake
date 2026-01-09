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

include(FetchContent)
include(GoogleTest)

#
# Fetch Google Test dependency
#
# Creates gtest and gtest_main targets
# FetchContent automatically handles caching - only fetches once
#
function(fetch_gtest)
    MESSAGE(WARNING "By enabling Google Test, you agree to its license terms: https://github.com/google/googletest/blob/main/LICENSE")

    FetchContent_Declare(
        gtest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
    )

    # Disable unnecessary gtest components to speed up build
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # FetchContent_MakeAvailable handles fetching and caching automatically
    FetchContent_MakeAvailable(gtest)
endfunction()

function(create_test_config)
    # Parameters
    set(config_dir ${ARGV0})
    set(config_input_header ${ARGV1})
    set(config_output_header ${ARGV2})

    # Current CONFIG DIR
    set(TEST_CONFIG_DIR ${config_dir})

    # Configure the file
    configure_file(${config_input_header} ${config_output_header})
endfunction()

#
# Fetch yaml-cpp dependency
#
# Creates yaml-cpp target
# FetchContent automatically handles caching - only fetches once
#
function(fetch_yaml_cpp)
    MESSAGE(STATUS "Setting up yaml-cpp...")

    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        2f86d13775d119edbb69af52e5f566fd65c6953b
    )

    # Disable unnecessary yaml-cpp components
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

    # FetchContent_MakeAvailable handles fetching and caching automatically
    FetchContent_MakeAvailable(yaml-cpp)
endfunction()

#
# Create a test object library from source files
#
# This creates an OBJECT library that can be reused across multiple tests,
# eliminating redundant compilation of common test infrastructure code.
#
# Parameters:
#   NAME         - Name of the object library target
#   SOURCES      - Source files to compile into the object library
#   INCLUDE_DIRS - Include directories for the object library
#   DEPENDS      - Dependencies for the object library
#
# Example:
#   dlp_create_test_object_library(
#       NAME test_framework_obj
#       SOURCES ${FRAMEWORK_SOURCES}
#       INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
#       DEPENDS yaml-cpp
#   )
#
function(dlp_create_test_object_library)
    # Parse arguments
    set(options "")
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES INCLUDE_DIRS DEPENDS)

    cmake_parse_arguments(OBJ "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT OBJ_NAME)
        message(FATAL_ERROR "dlp_create_test_object_library: NAME argument is required")
    endif()

    if(NOT OBJ_SOURCES)
        message(FATAL_ERROR "dlp_create_test_object_library: SOURCES argument is required")
    endif()

    # Create OBJECT library (compiled once, linked many times)
    add_library(${OBJ_NAME} OBJECT ${OBJ_SOURCES})

    # Set position-independent code for the object library
    set_property(TARGET ${OBJ_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

    # Add include directories
    target_include_directories(${OBJ_NAME}
        PRIVATE
            ${DLP_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/include
            ${OBJ_INCLUDE_DIRS}
    )

    # Link dependencies if provided
    if(OBJ_DEPENDS)
        target_link_libraries(${OBJ_NAME} PRIVATE ${OBJ_DEPENDS})
    endif()

    # Apply global compiler flags
    dlp_set_global_compile_flags(${OBJ_NAME} VISIBILITY PUBLIC)

    message(STATUS "Created test object library: ${OBJ_NAME} (${list_length} source files)")
endfunction()

#
# Add a test executable with optional object library reuse
#
# This function creates a test executable, linking it with necessary dependencies
# and optionally reusing pre-compiled object libraries to avoid redundant compilation.
#
# Parameters:
#   NAME            - Name of the test executable
#   SOURCES         - Test-specific source files
#   OBJECT_LIBS     - Pre-compiled object libraries to link (optional)
#   INCLUDE_DIRS    - Additional include directories
#   DEPENDS         - Additional dependencies
#   DISABLED        - If present, marks test as disabled
#
# Example:
#   dlp_add_test(
#       NAME my_test
#       SOURCES my_test.cc
#       OBJECT_LIBS test_framework_obj test_adaptors_obj
#       INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
#       DEPENDS yaml-cpp
#   )
#
function(dlp_add_test)
    # Parse arguments for the test function
    set(options DISABLED)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES OBJECT_LIBS DEPENDS INCLUDE_DIRS)

    cmake_parse_arguments(DLP_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Check if required arguments are provided
    if(NOT DLP_TEST_NAME)
        message(FATAL_ERROR "dlp_add_test: NAME argument is required")
    endif()

    if(NOT DLP_TEST_SOURCES)
        message(FATAL_ERROR "dlp_add_test: SOURCES argument is required")
    endif()

    # Create test executable with sources and object libraries
    if(DLP_TEST_OBJECT_LIBS)
        # Build list of object library contents using generator expressions
        set(OBJECT_LIB_CONTENTS "")
        foreach(obj_lib ${DLP_TEST_OBJECT_LIBS})
            list(APPEND OBJECT_LIB_CONTENTS $<TARGET_OBJECTS:${obj_lib}>)
        endforeach()

        add_executable(${DLP_TEST_NAME} ${DLP_TEST_SOURCES} ${OBJECT_LIB_CONTENTS})
    else()
        # Fallback to old behavior if no object libraries specified
        add_executable(${DLP_TEST_NAME} ${DLP_TEST_SOURCES})
    endif()

    # Choose library target based on static linking preference
    if(DLP_TESTING_LINK_STATIC)
        set(DLP_LIBRARY_TARGET ${PROJECT_NAME}_static)
    else()
        set(DLP_LIBRARY_TARGET ${PROJECT_NAME})
    endif()

    # Link with Google Test and the main project library
    # Note: For static library, whole-archive is automatically applied via INTERFACE_LINK_OPTIONS
    target_link_libraries(${DLP_TEST_NAME}
        PRIVATE
            gtest
            gtest_main
            ${DLP_LIBRARY_TARGET}
            ${DLP_TEST_DEPENDS}
    )

    # Add include directories - follow modern CMake target-based approach
    target_include_directories(${DLP_TEST_NAME}
        PRIVATE
            ${DLP_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/include
            ${DLP_TEST_INCLUDE_DIRS}
    )

    # Set compiler flags
    dlp_set_global_compile_flags(${DLP_TEST_NAME})

    # Add test to CTest
    if(DLP_TEST_DISABLED OR DLP_TESTING_CTEST_DISABLED)
        # Traditional Testing
        add_test(NAME ${DLP_TEST_NAME} COMMAND ${DLP_TEST_NAME})
    else()
        # Gtest aware testing
        gtest_discover_tests(${DLP_TEST_NAME})
    endif()

endfunction()

# Example of adding a specific test with additional dependencies
# dlp_add_test(
#     NAME specific_feature_test
#     SOURCES specific_feature_test.cc
#     DEPENDS some_additional_dependency
#     INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
# )

# Example of a disabled test
# dlp_add_test(
#     NAME disabled_test
#     SOURCES disabled_test.cc
#     INCLUDE_DIRS ${TEST_INCLUDE_DIRS}
#     DISABLED
# )

function(dlp_define_testing_options)
    # Testing infrastructure options
    option(BUILD_TESTING "Build test programs" OFF)
    option(DLP_TESTING_CTEST_DISABLED "Disable CTest integration (use traditional testing)" ON)
    option(DLP_TESTING_LINK_STATIC "Link tests with static AOCL-DLP library for better performance" OFF)

    # Propagate variables back to the caller
    set(BUILD_TESTING ${BUILD_TESTING} PARENT_SCOPE)
    set(DLP_TESTING_CTEST_DISABLED ${DLP_TESTING_CTEST_DISABLED} PARENT_SCOPE)
    set(DLP_TESTING_LINK_STATIC ${DLP_TESTING_LINK_STATIC} PARENT_SCOPE)
endfunction()
