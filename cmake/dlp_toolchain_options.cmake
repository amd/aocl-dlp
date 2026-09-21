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

function(dlp_define_toolchain_options)
    # Pins Clang/AOCC to a specific GCC install (--gcc-install-dir=) instead
    # of whatever GCC it finds by default. See dlp_clang_gcc_toolchain() in
    # dlp_compiler_flags_linux.cmake/windows.cmake for how this is applied.
    set(DLP_GCC_TOOLCHAIN "" CACHE PATH
        "Exact GCC install directory (e.g. /usr/lib/gcc/x86_64-pc-linux-gnu/13.4.1) for Clang/AOCC to use via --gcc-install-dir=. Leave empty to use the compiler's default toolchain detection.")

    # Propagate to the caller for consistency with the project's other
    # dlp_define_*_options() functions.
    set(DLP_GCC_TOOLCHAIN ${DLP_GCC_TOOLCHAIN} PARENT_SCOPE)
endfunction()
