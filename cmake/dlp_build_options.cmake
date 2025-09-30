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

function(dlp_define_build_options)
    # Build target options - additional components that can be built
    option(BUILD_EXAMPLES "Build example programs" OFF)
    option(DLP_EXAMPLES_LINK_STATIC "Link examples with static AOCL-DLP library for better performance" OFF)

    # Sanitizer options for debugging and development
    option(DLP_ENABLE_ASAN "Enable Address Sanitizer" OFF)
    option(DLP_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
    option(DLP_ENABLE_TSAN "Enable Thread Sanitizer" OFF)

    # Propagate variables back to the caller
    set(BUILD_EXAMPLES ${BUILD_EXAMPLES} PARENT_SCOPE)
    set(DLP_EXAMPLES_LINK_STATIC ${DLP_EXAMPLES_LINK_STATIC} PARENT_SCOPE)
    set(DLP_ENABLE_ASAN ${DLP_ENABLE_ASAN} PARENT_SCOPE)
    set(DLP_ENABLE_UBSAN ${DLP_ENABLE_UBSAN} PARENT_SCOPE)
    set(DLP_ENABLE_TSAN ${DLP_ENABLE_TSAN} PARENT_SCOPE)
endfunction()
