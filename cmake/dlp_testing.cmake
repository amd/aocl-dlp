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

function(fetch_gtest)
    if(NOT gtest_FETCHED)
        MESSAGE(WARNING "By enabling Google Test, you agree to its license terms: https://github.com/google/googletest/blob/main/LICENSE")
    endif()

        FetchContent_Declare(
            gtest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.17.0
        )

        FetchContent_MakeAvailable(gtest)
        set(gtest_FETCHED TRUE CACHE INTERNAL "gtest already fetched")
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

function(fetch_yaml_cpp)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        2f86d13775d119edbb69af52e5f566fd65c6953b
        CMAKE_ARGS     -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF
    )

    FetchContent_MakeAvailable(yaml-cpp)
endfunction()

function(dlp_add_test)
    # Parse arguments for the test function
    set(options DISABLED)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES DEPENDS INCLUDE_DIRS)

    cmake_parse_arguments(DLP_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Check if required arguments are provided
    if(NOT DLP_TEST_NAME)
        message(FATAL_ERROR "dlp_add_test: NAME argument is required")
    endif()

    if(NOT DLP_TEST_SOURCES)
        message(FATAL_ERROR "dlp_add_test: SOURCES argument is required")
    endif()

    # Create test executable
    add_executable(${DLP_TEST_NAME} ${DLP_TEST_SOURCES})

    # Link with Google Test and the main project library
    target_link_libraries(${DLP_TEST_NAME}
        PRIVATE
            gtest
            gtest_main
            ${PROJECT_NAME}
            ${DLP_TEST_DEPENDS}
    )

    # Add include directories - follow modern CMake target-based approach
    target_include_directories(${DLP_TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/include
            ${DLP_TEST_INCLUDE_DIRS}
    )

    # Set compiler flags
    dlp_set_global_compile_flags(${DLP_TEST_NAME})

    # Add test to CTest
    if(DLP_TEST_DISABLED OR DLP_CTEST_DISABLED)
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
