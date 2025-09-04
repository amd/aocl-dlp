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


function(dlp_install target_name)
    # Installation rules with export set
    install(TARGETS ${target_name}
        EXPORT AoclDlpTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endfunction()

# Function to install CMake package configuration files
function(dlp_install_package_config)
    include(CMakePackageConfigHelpers)

    # Set up the package configuration directory
    set(AOCL_DLP_CMAKE_CONFIG_DIR "lib/cmake/AoclDlp")

        # Configure the main config file
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/AoclDlpConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/AoclDlpConfig.cmake"
        INSTALL_DESTINATION "${AOCL_DLP_CMAKE_CONFIG_DIR}"
        PATH_VARS CMAKE_INSTALL_PREFIX
    )

    # Configure the version file using our custom template
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/AoclDlpConfigVersion.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/AoclDlpConfigVersion.cmake"
        @ONLY
    )

    # Install the export targets file
    install(EXPORT AoclDlpTargets
        FILE AoclDlpTargets.cmake
        NAMESPACE AoclDlp::
        DESTINATION "${AOCL_DLP_CMAKE_CONFIG_DIR}"
    )

    # Install the configured config files
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/AoclDlpConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/AoclDlpConfigVersion.cmake"
        DESTINATION "${AOCL_DLP_CMAKE_CONFIG_DIR}"
    )

    # Config header installation is handled in dlp_platform.cmake
endfunction()
