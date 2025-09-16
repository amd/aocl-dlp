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

# This file defines platform-specific variables to enable source code to detect and adapt to the target platform.

# Detect OS
if(UNIX)
  set(DLP_OS_UNIX 1)
  if(APPLE)
    set(DLP_OS_MACOS 1)
  else()
    set(DLP_OS_LINUX 1)

    # Detect Linux distribution
    execute_process(
      COMMAND lsb_release -si
      OUTPUT_VARIABLE DLP_LINUX_DISTRIBUTION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
      COMMAND lsb_release -sr
      OUTPUT_VARIABLE DLP_LINUX_DISTRIBUTION_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Handle case when lsb_release is not available
    if(NOT DLP_LINUX_DISTRIBUTION)
      if(EXISTS "/etc/os-release")
        execute_process(
          COMMAND cat /etc/os-release
          COMMAND grep "^ID="
          COMMAND sed "s/ID=\\(.*\\)/\\1/"
          COMMAND tr -d "\"'"
          OUTPUT_VARIABLE DLP_LINUX_DISTRIBUTION
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        execute_process(
          COMMAND cat /etc/os-release
          COMMAND grep "^VERSION_ID="
          COMMAND sed "s/VERSION_ID=\\(.*\\)/\\1/"
          COMMAND tr -d "\"'"
          OUTPUT_VARIABLE DLP_LINUX_DISTRIBUTION_VERSION
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
      endif()
    endif()

    message(STATUS "Linux Distribution: ${DLP_LINUX_DISTRIBUTION} ${DLP_LINUX_DISTRIBUTION_VERSION}")
  endif()
elseif(WIN32)
  set(DLP_OS_WINDOWS 1)

  # Determine Windows version
  if(CMAKE_SYSTEM_VERSION)
    set(DLP_WINDOWS_VERSION ${CMAKE_SYSTEM_VERSION})

    # Map version numbers to names
    if(CMAKE_SYSTEM_VERSION VERSION_EQUAL "10.0")
      set(DLP_WINDOWS_NAME "Windows 10/11")
    elseif(CMAKE_SYSTEM_VERSION VERSION_EQUAL "6.3")
      set(DLP_WINDOWS_NAME "Windows 8.1")
    elseif(CMAKE_SYSTEM_VERSION VERSION_EQUAL "6.2")
      set(DLP_WINDOWS_NAME "Windows 8")
    elseif(CMAKE_SYSTEM_VERSION VERSION_EQUAL "6.1")
      set(DLP_WINDOWS_NAME "Windows 7")
    else()
      set(DLP_WINDOWS_NAME "Windows")
    endif()

    message(STATUS "Windows Version: ${DLP_WINDOWS_NAME} (${DLP_WINDOWS_VERSION})")
  endif()
else()
  message(WARNING "Unsupported OS detected")
endif()

# Detect Compiler
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  set(DLP_COMPILER_GCC 1)
  set(DLP_COMPILER_NAME "GCC")
  set(DLP_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS "11.2")
    message(FATAL_ERROR "Unsupported GCC version: ${CMAKE_C_COMPILER_VERSION}. AOCL-DLP requires GCC 11.2 or newer.")
  endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(DLP_COMPILER_CLANG 1)
  set(DLP_COMPILER_NAME "Clang")
  set(DLP_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
  set(DLP_COMPILER_INTEL 1)
  set(DLP_COMPILER_NAME "Intel")
  set(DLP_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
elseif(MSVC)
  set(DLP_COMPILER_MSVC 1)
  set(DLP_COMPILER_NAME "MSVC")
  set(DLP_COMPILER_VERSION ${MSVC_VERSION})

  # Map MSVC version numbers to year
  if(MSVC_VERSION GREATER_EQUAL 1930)
    set(DLP_COMPILER_MSVC_YEAR "2022")
  elseif(MSVC_VERSION GREATER_EQUAL 1920)
    set(DLP_COMPILER_MSVC_YEAR "2019")
  elseif(MSVC_VERSION GREATER_EQUAL 1910)
    set(DLP_COMPILER_MSVC_YEAR "2017")
  elseif(MSVC_VERSION GREATER_EQUAL 1900)
    set(DLP_COMPILER_MSVC_YEAR "2015")
  elseif(MSVC_VERSION GREATER_EQUAL 1800)
    set(DLP_COMPILER_MSVC_YEAR "2013")
  else()
    set(DLP_COMPILER_MSVC_YEAR "Unknown")
  endif()
else()
  set(DLP_COMPILER_NAME "${CMAKE_C_COMPILER_ID}")
  set(DLP_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
  message(WARNING "Unsupported compiler: ${DLP_COMPILER_NAME} ${DLP_COMPILER_VERSION}")
endif()

message(STATUS "Compiler: ${DLP_COMPILER_NAME} ${DLP_COMPILER_VERSION}")

# Detect Endian
include(TestBigEndian)
TEST_BIG_ENDIAN(DLP_IS_BIG_ENDIAN)
if(DLP_IS_BIG_ENDIAN)
  set(DLP_ENDIAN "big")
  message(STATUS "Endianness: Big Endian")
else()
  set(DLP_ENDIAN "little")
  message(STATUS "Endianness: Little Endian")
endif()

function(dlp_generate_config_header)
  # Add compiler and platform information to config header
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/aocl_dlp_config.h.in"
    "${CMAKE_BINARY_DIR}/include/aocl_dlp_config.h"
  )

  # Install the generated config header
  install(
    FILES "${CMAKE_BINARY_DIR}/include/aocl_dlp_config.h"
    DESTINATION include
  )
endfunction()
