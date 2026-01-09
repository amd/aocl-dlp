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

# Doxygen documentation
set(DOXYGEN_CONFIG ${CMAKE_BINARY_DIR}/docs/doxygen/doxygen.conf)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_CONFIG_IN ${DLP_SOURCE_DIR}/docs/doxygen/doxygen.conf.in)
set(DOXYGEN_CONFIG_OUT ${CMAKE_BINARY_DIR}/docs/doxygen/doxygen.conf)

function(dlp_generate_doxygen_config)

    configure_file(${DOXYGEN_CONFIG_IN} ${DOXYGEN_CONFIG_OUT})
endfunction()

function(dlp_generate_doxygen_docs)
    find_package(Doxygen REQUIRED)

    add_custom_target(doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG} > ${PROJECT_BINARY_DIR}/docs/doxygen/doxygen.log 2>&1
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT "Generating Doxygen documentation"
        VERBATIM)

    message(STATUS "Doxygen log: ${PROJECT_BINARY_DIR}/doxygen/doxygen.log")

    set(DOXYGEN_OUTPUT_DIR ${DOXYGEN_OUTPUT_DIR} PARENT_SCOPE)
endfunction()

# Sphinx documentation
function(dlp_generate_sphinx_docs)
    if(NOT BUILD_DOXYGEN)
        message(FATAL_ERROR "Sphinx depends on Doxygen. Please enable it with -DBUILD_DOXYGEN=ON")
    endif()

    find_package(Sphinx REQUIRED)
    set(SPHINX_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs/sphinx)

    add_custom_target(sphinx ALL
        COMMAND ${SPHINX_EXECUTABLE} -b html -Dbreathe_projects.aocl-dlp=${DOXYGEN_OUTPUT_DIR}/xml ${DLP_SOURCE_DIR}/docs/sphinx ${SPHINX_OUTPUT_DIR} > ${PROJECT_BINARY_DIR}/docs/sphinx/sphinx.log 2>&1
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating Sphinx documentation"
        VERBATIM
        DEPENDS doxygen)

    # Create docs/sphinx directory if it doesn't exist
    if(NOT EXISTS ${PROJECT_BINARY_DIR}/docs/sphinx)
        file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/docs/sphinx)
    endif()

    message(STATUS "Sphinx log: ${PROJECT_BINARY_DIR}/sphinx/sphinx.log")

    set(SPHINX_OUTPUT_DIR ${SPHINX_OUTPUT_DIR} PARENT_SCOPE)
endfunction()

function(dlp_define_documentation_options)
    # Documentation generation options
    option(BUILD_DOXYGEN "Generate Doxygen documentation" OFF)
    option(BUILD_SPHINX "Generate Sphinx documentation" OFF)

    # Propagate variables back to the caller
    set(BUILD_DOXYGEN ${BUILD_DOXYGEN} PARENT_SCOPE)
    set(BUILD_SPHINX ${BUILD_SPHINX} PARENT_SCOPE)
endfunction()
