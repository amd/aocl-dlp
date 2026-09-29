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

# Basic examples
set(EXAMPLE_SOURCES
    simple_gemm_f32.c
    multi_instance_gemm_f32.c
    multi_instance_gemm_u8s8.c
    simple_gemm_bf16.c
    simple_gemm_s8.c
    simple_gemm_u8s8_12x32.c
    simple_gemm_with_bias.c
    simple_gemm_with_relu.c
    simple_gemm_with_mish.c
    matrix_reorder.c
    matrix_unreorder.c
    matrix_reorder_reference.c
    batch_gemm.c
    quantization.c
    simple_gemm_bf16s8.c
    simple_gemm_s8_sym_quant.c
    simple_gemm_s8s4.c
    simple_gemm_bf16s4.c
    simple_gemm_bf16u4.c
    simple_gemm_f32s8.c
    simple_gemm_per_token_quant.c
    post_ops_combinations.c
    eltwise_ops.c
    simple_gemm_with_glu_swiglu_and_mul_bf16.c
    version.c
)

# Create an executable for each example source file
foreach(source ${EXAMPLE_SOURCES})
    # Extract the filename without extension
    get_filename_component(example_name ${source} NAME_WE)

    dlp_add_c_example(
        NAME ${example_name}
        SOURCES ${source}
        INSTALL_DESTINATION examples/classic
    )
endforeach()

# Copy example source files to the installation directory.
# dlp_example_utils.h is a header shared only by the examples (it is not part
# of the installed public API); it must sit next to the sources so the
# standalone example build can resolve the "" include.
install(FILES ${EXAMPLE_SOURCES} dlp_example_utils.h
    DESTINATION examples/classic/src
)

# Configure and install the standalone CMakeLists.txt for building examples after installation
# Note: CMAKE_MINIMUM_REQUIRED_VERSION is defined in cmake/dlp_variables.cmake
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt
    DESTINATION examples/classic
)

# Install a readme file with building instructions
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md.in
    ${CMAKE_CURRENT_BINARY_DIR}/README.md
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/README.md
    DESTINATION examples/classic
)
