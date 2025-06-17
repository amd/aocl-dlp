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

# Given a target, add the appropriate sanitizer flags to the targets
function(dlp_add_asan targets)
    if(DLP_ENABLE_ASAN)
        # Add sanitizer flags to target and all dependencies.
        foreach(target ${targets})
            target_compile_options(${target} PUBLIC -fsanitize=address)
            target_link_options(${target} PUBLIC -fsanitize=address)
        endforeach()
    endif()
endfunction()

function(dlp_add_ubsan targets)
    if(DLP_ENABLE_UBSAN)
        # Add sanitizer flags to target and all dependencies.
        foreach(target ${targets})
            target_compile_options(${target} PUBLIC -fsanitize=undefined)
            target_link_options(${target} PUBLIC -fsanitize=undefined)
        endforeach()
    endif()
endfunction()

function(dlp_add_tsan targets)
    if(DLP_ENABLE_TSAN)
        # Add sanitizer flags to target and all dependencies.
        foreach(target ${targets})
            target_compile_options(${target} PUBLIC -fsanitize=thread)
            target_link_options(${target} PUBLIC -fsanitize=thread)
        endforeach()
    endif()
endfunction()


function(dlp_add_all_sanitizers targets)
    dlp_add_asan(${targets})
    dlp_add_ubsan(${targets})
    dlp_add_tsan(${targets})
endfunction()
